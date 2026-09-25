# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Stage 0: per-file geometry and physical-unit window conversion.

Every later stage of the seabed detector works in physical units (metres,
pulse lengths, seconds) and converts to samples and pings through the
``PingGeometry`` built here from the Sv dataset's own metadata.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
import xarray as xr

from aa_si_utils.utils import _resolve_channel_index, haversine_distance

# Metres past the first sample searched when no r_min is given; skips the
# transmit pulse and near-field ringing, which are loud and phase-stable.
DEFAULT_R_MIN_OFFSET_M = 10.0


class EmptyWindowError(ValueError):
    """No ping has a search window with any range in it."""


@dataclass
class PingGeometry:
    """Geometry of one channel of an Sv dataset in physical units.

    Attributes:
        channel: Primary channel label.
        channel_index: Position of the primary channel on ``channel``.
        range_var: ``"depth"`` when the dataset carries depth, else
            ``"echo_range"``. The returned line shares this reference.
        range0: Range of ``range_sample`` 0 per ping, metres, shape (P,).
            Taken as the transducer face, as ``get_transducer_depth`` does,
            so range from the transducer is ``range - range0``; a dataset
            whose range samples were cropped upstream breaks that.
        dr: Range sample spacing in metres.
        n_valid: Number of samples with a finite range on this channel.
        pulse_length_m: Pulse extent in range, c tau / 2, metres.
        beamwidth_deg: Nominal full beamwidth, degrees.
        dt_ping: Interval to the next ping in seconds, shape (P,).
        dx_ping: Along-track distance to the next ping in metres, shape
            (P,); NaN where neither GPS nor a vessel speed is available.
        r_min: Per-ping lower search bound in metres, shape (P,).
        r_max: Per-ping upper search bound in metres, shape (P,).
        i_min: Per-ping lower search bound as an absolute sample index.
        i_max: Per-ping upper search bound as an absolute sample index,
            exclusive.
        i_lo: Start of the sample crop that contains every search window.
        i_hi: End of that crop, exclusive.
        empty: Pings with no search window, shape (P,): the window's bounds
            crossed (a prior outside the recorded range) or the ping has no
            valid range. They get no candidates and no seabed.
        has_angles: Whether split-beam angles are available on the channel.
        sound_speed: Sound speed used for the pulse length, m/s.
        tau_s: Pulse duration in seconds.
        absorption_db_per_m: Sound absorption used to remove TVG from the
            background noise estimate; 0 when the dataset has none.
    """

    channel: str
    channel_index: int
    range_var: str
    range0: np.ndarray
    dr: float
    n_valid: int
    pulse_length_m: float
    beamwidth_deg: float
    dt_ping: np.ndarray
    dx_ping: np.ndarray
    r_min: np.ndarray
    r_max: np.ndarray
    i_min: np.ndarray
    i_max: np.ndarray
    i_lo: int
    i_hi: int
    empty: np.ndarray
    has_angles: bool
    sound_speed: float
    tau_s: float
    absorption_db_per_m: float = 0.0

    @property
    def n_ping(self):
        return int(self.range0.size)

    @property
    def n_crop(self):
        return int(self.i_hi - self.i_lo)

    @property
    def pulse_samples(self):
        """Samples in one pulse length, at least 1."""
        return self.pulse_lengths_to_samples(1.0)

    def metres_to_samples(self, w_m):
        """Range window in metres to a whole number of samples, at least 1."""
        return max(1, int(math.ceil(float(w_m) / self.dr)))

    def pulse_lengths_to_samples(self, w_tau):
        """Range window in pulse lengths to samples, at least 1."""
        return self.metres_to_samples(float(w_tau) * self.pulse_length_m)

    def seconds_to_pings(self, w_s):
        """Along-track window in seconds to pings, using the median interval."""
        dt = float(np.nanmedian(self.dt_ping))
        if not np.isfinite(dt) or dt <= 0:
            return 1
        return max(1, int(math.ceil(float(w_s) / dt)))

    def sample_to_range(self, sample_idx):
        """Absolute sample index per ping to metres on ``range_var``.

        Args:
            sample_idx: Array of shape (P,), float so NaN marks no pick.

        Returns:
            np.ndarray: Range in metres, NaN where the index is NaN.
        """
        idx = np.asarray(sample_idx, dtype=float)
        return self.range0 + idx * self.dr

    def range_to_sample(self, range_m):
        """Metres on ``range_var`` per ping to a fractional sample index."""
        return (np.asarray(range_m, dtype=float) - self.range0) / self.dr

    def crop_range_axis(self):
        """Range in metres of every cropped sample, shape (P, n_crop)."""
        idx = np.arange(self.i_lo, self.i_hi, dtype=float)
        return self.range0[:, None] + idx[None, :] * self.dr

    def window_mask(self):
        """Boolean (P, n_crop) mask of the per-ping search window."""
        idx = np.arange(self.i_lo, self.i_hi)
        return (idx[None, :] >= self.i_min[:, None]) & (idx[None, :] < self.i_max[:, None])


def _channel_scalar(ds_Sv, name, channel_index, default=None):
    """Mean finite value of a per-channel parameter, or ``default``."""
    if name not in ds_Sv:
        return default
    da = ds_Sv[name]
    if "channel" in da.dims:
        da = da.isel(channel=channel_index)
    values = np.asarray(da.compute().values if hasattr(da, "compute") else da.values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(finite.mean())


def resolve_channel(ds_Sv, channel):
    """Index of ``channel`` given as a label, Hz, or kHz."""
    if isinstance(channel, (float, np.floating)) and float(channel).is_integer():
        channel = int(channel)
    return _resolve_channel_index(channel, ds_Sv["channel"].values, ds_Sv=ds_Sv)


def _ping_intervals(ping_time):
    """Seconds between consecutive pings, last value repeated, shape (P,)."""
    t = np.asarray(ping_time, dtype="datetime64[ns]").astype("int64") / 1e9
    if t.size < 2:
        return np.full(t.size, np.nan)
    dt = np.diff(t)
    return np.append(dt, dt[-1])


def _along_track_distances(ds_Sv, dt_ping, vessel_speed_m_s):
    """Metres between consecutive pings from GPS, else speed, else NaN."""
    if "latitude" in ds_Sv and "longitude" in ds_Sv:
        lat = np.asarray(ds_Sv["latitude"].values, dtype=float)
        lon = np.asarray(ds_Sv["longitude"].values, dtype=float)
        if lat.ndim == 1 and lat.size == dt_ping.size and np.isfinite(lat).sum() > 1:
            dx = np.full(dt_ping.size, np.nan)
            for p in range(dt_ping.size - 1):
                if np.isfinite([lat[p], lon[p], lat[p + 1], lon[p + 1]]).all():
                    dx[p] = haversine_distance(lat[p], lon[p], lat[p + 1], lon[p + 1])
            if dt_ping.size > 1:
                dx[-1] = dx[-2]
            if np.isfinite(dx).any():
                return dx
    if vessel_speed_m_s is not None:
        return float(vessel_speed_m_s) * dt_ping
    return np.full(dt_ping.size, np.nan)


def _prior_bounds(ds_Sv, z_prior, sigma_prior_m, n_ping):
    """Per-ping (r_min, r_max) from a bathymetry prior, or None."""
    if z_prior is None:
        return None
    if isinstance(z_prior, xr.DataArray):
        z = z_prior
        if "ping_time" in z.dims:
            z = z.interp(ping_time=ds_Sv["ping_time"])
            z = z.interpolate_na("ping_time").bfill("ping_time").ffill("ping_time")
        z = np.asarray(z.values, dtype=float)
        if z.ndim == 0:
            z = np.full(n_ping, float(z))
    else:
        z = np.full(n_ping, float(z_prior))
    if sigma_prior_m is None:
        sigma = 0.02 * z
    elif isinstance(sigma_prior_m, xr.DataArray):
        s = sigma_prior_m
        if "ping_time" in s.dims:
            s = s.interp(ping_time=ds_Sv["ping_time"]).bfill("ping_time").ffill("ping_time")
        sigma = np.asarray(s.values, dtype=float)
        if sigma.ndim == 0:
            sigma = np.full(n_ping, float(sigma))
    elif np.ndim(sigma_prior_m) == 1:
        sigma = np.asarray(sigma_prior_m, dtype=float)
    else:
        sigma = np.full(n_ping, float(sigma_prior_m))
    return z - 3.0 * sigma, z + 3.0 * sigma


def build_geometry(
    ds_Sv,
    channel,
    r_min=None,
    r_max=None,
    z_prior=None,
    sigma_prior_m=None,
    vessel_speed_m_s=None,
    pulse_length_s=None,
    beamwidth_deg=None,
):
    """Build the ``PingGeometry`` of one channel of an Sv dataset.

    Args:
        ds_Sv: Calibrated Sv dataset with ``echo_range`` and, ideally,
            ``depth``, ``tau_effective``, ``sound_speed`` and the beamwidths
            that ``compute_Sv`` attaches.
        channel: Channel label, Hz, or kHz of the primary channel.
        r_min: Fixed lower search bound in metres on the range reference.
            Defaults to 10 m past the first sample.
        r_max: Fixed upper search bound in metres. Defaults to the end of
            the channel's valid range.
        z_prior: Expected seabed depth: a scalar or a ``(ping_time,)``
            DataArray. Sets the per-ping window to z_prior +/- 3 sigma.
        sigma_prior_m: Prior uncertainty in metres, a scalar or a
            ``(ping_time,)`` array; default 2 % of z_prior.
        vessel_speed_m_s: Speed used for along-track distance when the
            dataset carries no latitude/longitude.
        pulse_length_s: Pulse duration override when ``tau_effective`` is
            not on the dataset.
        beamwidth_deg: Beamwidth override when the dataset has none.

    Returns:
        PingGeometry: Geometry of the primary channel.
    """
    ci = resolve_channel(ds_Sv, channel)
    channel_label = str(ds_Sv["channel"].values[ci])
    range_var = "depth" if "depth" in ds_Sv else "echo_range"

    # Only the first sample of every ping and the first ping's full row are
    # read, so a whole-survey dataset costs no more than one file here.
    range_da = ds_Sv[range_var].isel(channel=ci)
    first_row = range_da.isel(ping_time=0)
    first_col = range_da.isel(range_sample=0)
    if hasattr(first_row, "compute"):
        first_row = first_row.compute()
        first_col = first_col.compute()
    first_row = np.asarray(first_row.values, dtype=float)
    range0 = np.asarray(first_col.values, dtype=float)
    n_ping = range0.size
    n_valid = int(np.isfinite(first_row).sum())
    if n_valid < 3:
        raise ValueError(f"channel {channel_label!r} has fewer than 3 finite range samples")
    dr = float(np.median(np.diff(first_row[np.isfinite(first_row)])))
    if dr <= 0:
        raise ValueError(f"range spacing on channel {channel_label!r} is not positive")

    sound_speed = _channel_scalar(ds_Sv, "sound_speed", ci, default=None)
    if sound_speed is None:
        sound_speed = 1500.0
        warnings.warn("ds_Sv has no sound_speed; using 1500 m/s for the pulse length")
    if pulse_length_s is None:
        tau_s = _channel_scalar(ds_Sv, "tau_effective", ci, default=None)
        if tau_s is None:
            raise KeyError(
                "ds_Sv has no tau_effective; pass pulse_length_s so windows in "
                "pulse lengths can be converted to samples"
            )
    else:
        tau_s = float(pulse_length_s)
    pulse_length_m = sound_speed * tau_s / 2.0

    if beamwidth_deg is None:
        widths = [
            _channel_scalar(ds_Sv, name, ci, default=None)
            for name in ("beamwidth_alongship", "beamwidth_athwartship")
        ]
        widths = [w for w in widths if w is not None]
        if widths:
            beamwidth_deg = float(np.mean(widths))
        else:
            beamwidth_deg = 7.0
            warnings.warn("ds_Sv has no beamwidth; using 7 degrees")

    dt_ping = _ping_intervals(ds_Sv["ping_time"].values)
    dx_ping = _along_track_distances(ds_Sv, dt_ping, vessel_speed_m_s)

    valid_end = range0 + (n_valid - 1) * dr
    prior = _prior_bounds(ds_Sv, z_prior, sigma_prior_m, n_ping)
    if prior is not None:
        lo, hi = prior
    else:
        lo = np.full(n_ping, range0 + DEFAULT_R_MIN_OFFSET_M) if r_min is None else np.full(n_ping, float(r_min))
        hi = valid_end.copy() if r_max is None else np.full(n_ping, float(r_max))
    if r_min is not None and prior is not None:
        lo = np.maximum(lo, float(r_min))
    if r_max is not None and prior is not None:
        hi = np.minimum(hi, float(r_max))
    # The near field is never searched unless r_min says so explicitly: the
    # transmit pulse and ringing are loud and phase-stable and would seed a
    # detector handed a prior window that reaches the transducer.
    lo = np.maximum(lo, range0 + (0.0 if r_min is not None else DEFAULT_R_MIN_OFFSET_M))
    hi = np.minimum(hi, valid_end)
    # NaN bounds (a ping with no valid range) fail this test too.
    empty = ~(hi > lo)
    if empty.all():
        raise EmptyWindowError("search window is empty on every ping; check r_min, r_max, z_prior")

    i_min = np.zeros(n_ping, dtype=int)
    i_max = np.zeros(n_ping, dtype=int)
    ok = ~empty
    i_min[ok] = np.clip(np.floor((lo[ok] - range0[ok]) / dr).astype(int), 0, n_valid - 1)
    i_max[ok] = np.clip(np.ceil((hi[ok] - range0[ok]) / dr).astype(int) + 1, i_min[ok] + 1, n_valid)
    i_lo = int(i_min[ok].min())
    i_hi = int(i_max[ok].max())
    # Empty pings get a zero-width window at the crop start.
    i_min[empty] = i_lo
    i_max[empty] = i_lo

    absorption = _channel_scalar(ds_Sv, "sound_absorption", ci, default=0.0)

    has_angles = False
    if "angle_alongship" in ds_Sv and "angle_athwartship" in ds_Sv:
        sample = ds_Sv["angle_alongship"].isel(channel=ci, ping_time=slice(0, 10))
        sample = sample.compute() if hasattr(sample, "compute") else sample
        has_angles = bool(np.isfinite(np.asarray(sample.values, dtype=float)).any())

    return PingGeometry(
        channel=channel_label,
        channel_index=ci,
        range_var=range_var,
        range0=range0,
        dr=dr,
        n_valid=n_valid,
        pulse_length_m=float(pulse_length_m),
        beamwidth_deg=float(beamwidth_deg),
        dt_ping=dt_ping,
        dx_ping=dx_ping,
        r_min=lo,
        r_max=hi,
        i_min=i_min,
        i_max=i_max,
        i_lo=i_lo,
        i_hi=i_hi,
        empty=empty,
        has_angles=has_angles,
        sound_speed=float(sound_speed),
        tau_s=float(tau_s),
        absorption_db_per_m=float(absorption),
    )


def crop_block(ds_Sv, geom, name):
    """One variable of the primary channel on the cropped sample block.

    Args:
        ds_Sv: The Sv dataset.
        geom: Geometry from :func:`build_geometry`.
        name: Variable name, e.g. ``"Sv"`` or ``"angle_alongship"``.

    Returns:
        np.ndarray: float64 array of shape (P, n_crop).
    """
    da = ds_Sv[name].isel(channel=geom.channel_index, range_sample=slice(geom.i_lo, geom.i_hi))
    da = da.transpose("ping_time", "range_sample")
    if hasattr(da, "compute"):
        da = da.compute()
    return np.asarray(da.values, dtype=float)
