# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Mode 0: fast seabed estimation from sampled slabs, used as a prior.

A slab is a few consecutive pings averaged in the linear domain and binned
in range to two pulse lengths. Slabs are scored with a reduced Mode 1 score
and joined by the same dynamic-programming line search as the full
detector. Mode 0b places half a budget of slabs evenly and spends the rest
bisecting gaps whose picks disagree by more than the break-point, so flat
terrain converges at once and rough terrain gets the extra slabs. The
result is an expected seabed depth and uncertainty per ping, which bounds
the full detector's search window without any external line.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import xarray as xr

from aa_si_utils.seabed.dp import Candidates, run_dp, transition_params
from aa_si_utils.seabed.geometry import EmptyWindowError, build_geometry
from aa_si_utils.seabed.phase import angle_variance

LINEAR_EPS = 1e-12


def _read_slab(ds_Sv, geom, name, a, b):
    da = ds_Sv[name].isel(
        channel=geom.channel_index, ping_time=slice(a, b), range_sample=slice(geom.i_lo, geom.i_hi)
    ).transpose("ping_time", "range_sample")
    if hasattr(da, "compute"):
        da = da.compute()
    return np.asarray(da.values, dtype=float)


def slab_profile(ds_Sv, geom, center, slab_pings, n_bin, use_angles):
    """One slab's binned Sv profile and, with angles, its phase activity.

    Args:
        ds_Sv: The Sv dataset (lazy or loaded).
        geom: ``PingGeometry`` of the primary channel on the search crop.
        center: Centre ping index of the slab.
        slab_pings: Pings averaged per slab.
        n_bin: Samples per range bin (two pulse lengths).
        use_angles: Read the angle arrays and compute the phase term.

    Returns:
        tuple: ``(sv_db, f2_deg, bin_range_m)`` with one value per bin;
        ``f2_deg`` is None without angles.
    """
    half = slab_pings // 2
    a, b = max(0, center - half), min(geom.n_ping, center - half + slab_pings)
    sv = _read_slab(ds_Sv, geom, "Sv", a, b)
    n_bins = sv.shape[1] // n_bin
    sv = sv[:, : n_bins * n_bin]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        linear = 10.0 ** (sv / 10.0)
        binned = np.nanmean(np.nanmean(linear.reshape(sv.shape[0], n_bins, n_bin), axis=2), axis=0)
        sv_db = 10.0 * np.log10(np.where(binned > 0, binned, np.nan))
    f2 = None
    if use_angles and geom.has_angles:
        theta = _read_slab(ds_Sv, geom, "angle_alongship", a, b)[:, : n_bins * n_bin]
        phi = _read_slab(ds_Sv, geom, "angle_athwartship", a, b)[:, : n_bins * n_bin]
        total = np.zeros(n_bins)
        for angle in (theta, phi):
            block = angle.reshape(angle.shape[0], n_bins, n_bin)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(block, axis=(0, 2))
                mean_sq = np.nanmean(block**2, axis=(0, 2))
            total += np.maximum(mean_sq - mean**2, 0.0)
        f2 = np.sqrt(total)
    bin_range = geom.range0[center] + (geom.i_lo + (np.arange(n_bins) + 0.5) * n_bin) * geom.dr
    return sv_db, f2, bin_range


def slab_score(sv_db, f2, beamwidth_deg):
    """Reduced Mode 1 score on a slab profile (the spec's Lambda_0).

    Amplitude against the slab's own 90th percentile, the cumulative
    energy gradient across one bin, and, when available, the phase
    activity term.
    """
    finite = np.isfinite(sv_db)
    if finite.sum() < 3:
        return np.full(sv_db.size, np.nan)
    sv90 = np.nanpercentile(sv_db, 90)
    score = np.tanh((sv_db - sv90) / 10.0)
    linear = np.where(finite, 10.0 ** (sv_db / 10.0), 0.0)
    cum = np.cumsum(linear)
    idx = np.arange(sv_db.size)
    ahead = cum[np.minimum(idx + 1, sv_db.size - 1)]
    behind = cum[np.maximum(idx - 1, 0)]
    with np.errstate(divide="ignore", invalid="ignore"):
        f6 = 10.0 * np.log10(ahead / (behind + LINEAR_EPS))
    score = score + np.tanh((f6 - 10.0) / 5.0)
    if f2 is not None:
        score = score - np.tanh(f2 / (beamwidth_deg / 2.0))
    score[~finite] = np.nan
    return score


def _slab_candidates(score, bin_range, n_candidates, min_score, sv_db=None, min_sv_db=None):
    """Top-N local maxima of a slab score, optionally only loud enough ones."""
    s = np.where(np.isfinite(score), score, -np.inf)
    if sv_db is not None and min_sv_db is not None:
        s = np.where(np.isfinite(sv_db) & (sv_db >= min_sv_db), s, -np.inf)
    left = np.concatenate([[-np.inf], s[:-1]])
    right = np.concatenate([s[1:], [-np.inf]])
    local = (s >= left) & (s >= right) & (s >= min_score) & np.isfinite(s)
    cols = np.nonzero(local)[0]
    cols = cols[np.argsort(-s[cols], kind="stable")][:n_candidates]
    idx = np.full(n_candidates, -1, dtype=int)
    sc = np.full(n_candidates, np.nan)
    rng = np.full(n_candidates, np.nan)
    idx[: cols.size] = cols
    sc[: cols.size] = s[cols]
    rng[: cols.size] = bin_range[cols]
    return idx, sc, rng


def _dp_over_slabs(centers, slabs, geom, n_candidates, max_slope_deg, tolerance_pulses):
    """Viterbi across slabs; returns picks (m), margins, delta per gap."""
    k = len(centers)
    idx = np.vstack([slabs[c]["idx"] for c in centers])
    score = np.vstack([slabs[c]["score"] for c in centers])
    range_m = np.vstack([slabs[c]["range"] for c in centers])
    cand = Candidates(idx, score, range_m, (idx >= 0).any(axis=1))
    dx = np.full(k, np.nan)
    for i in range(k - 1):
        seg = geom.dx_ping[centers[i] : centers[i + 1]]
        dx[i] = float(np.nansum(seg)) if np.isfinite(seg).any() else np.nan
    if k > 1:
        dx[-1] = dx[-2]
    slab_geom = SimpleNamespace(
        dr=geom.dr,
        pulse_length_m=geom.pulse_length_m,
        beamwidth_deg=geom.beamwidth_deg,
        range0=geom.range0[centers],
        dx_ping=dx,
        r_min=geom.r_min[centers],
        r_max=geom.r_max[centers],
    )
    params = transition_params(cand, slab_geom, max_slope_deg=max_slope_deg)
    result = run_dp(cand, slab_geom, params, tolerance_pulses=tolerance_pulses)
    picks = np.full(k, np.nan)
    ok = result.chosen >= 0
    picks[ok] = range_m[np.nonzero(ok)[0], result.chosen[ok]]
    return picks, result.margin, params.delta


def estimate_prior(
    ds_Sv,
    channel,
    r_min=None,
    r_max=None,
    slab_pings=5,
    spacing_s=30.0,
    budget=64,
    adaptive=True,
    n_candidates=5,
    min_score=0.0,
    max_slope_deg=30.0,
    use_angles=True,
    sigma_fraction=0.05,
    sigma_floor_m=5.0,
    min_seabed_sv_db=-50.0,
    **geometry_kwargs,
):
    """Mode 0 seabed prior from sampled slabs.

    Args:
        ds_Sv: Sv dataset; only the sampled slabs are read.
        channel: Primary channel label, Hz or kHz.
        r_min: Optional lower bound of the prior's own search range.
        r_max: Optional upper bound; default the full recorded range.
        slab_pings: Pings averaged per slab.
        spacing_s: Mode 0a slab spacing in seconds; used when
            ``adaptive`` is False.
        budget: Mode 0b slab budget; half placed evenly, the rest spent
            bisecting inconsistent gaps.
        adaptive: Mode 0b (True) or 0a (False).
        n_candidates: Score maxima kept per slab.
        min_score: Candidates below this are dropped.
        max_slope_deg: Maximum-slope prior for the break-point.
        use_angles: Include the phase term when angles are available.
        sigma_fraction: Floor on the uncertainty as a fraction of depth.
        sigma_floor_m: Absolute floor on the uncertainty in metres.
        min_seabed_sv_db: A slab bin quieter than this cannot be the seabed,
            whatever its score; the score is relative to the slab itself, so
            without it a slab with no seabed in range picks its loudest
            scattering layer. None disables the floor.
        **geometry_kwargs: ``vessel_speed_m_s``, ``pulse_length_s``,
            ``beamwidth_deg`` for :func:`build_geometry`.

    Returns:
        xr.Dataset or None: ``z_prior`` and ``sigma_prior`` per ping plus
        the slab picks, or None when fewer than two slabs found a seabed.
    """
    try:
        geom = build_geometry(ds_Sv, channel, r_min=r_min, r_max=r_max, **geometry_kwargs)
    except EmptyWindowError:
        return None
    n_bin = geom.pulse_lengths_to_samples(2.0)
    n_ping = geom.n_ping
    if adaptive:
        n_even = max(2, min(budget // 2, n_ping // max(1, slab_pings)))
    else:
        step = max(slab_pings, geom.seconds_to_pings(spacing_s))
        n_even = max(2, n_ping // step)
        budget = n_even
    centers = sorted(set(np.linspace(slab_pings // 2, n_ping - 1 - slab_pings // 2, n_even).astype(int).tolist()))
    slabs = {}

    def ensure(c):
        if c not in slabs:
            sv_db, f2, bin_range = slab_profile(ds_Sv, geom, c, slab_pings, n_bin, use_angles)
            score = slab_score(sv_db, f2, geom.beamwidth_deg)
            idx, sc, rng = _slab_candidates(
                score, bin_range, n_candidates, min_score, sv_db=sv_db, min_sv_db=min_seabed_sv_db
            )
            slabs[c] = {"idx": idx, "score": sc, "range": rng}

    for c in centers:
        ensure(c)
    inconsistent = []
    for _ in range(64):
        picks, margins, delta = _dp_over_slabs(centers, slabs, geom, n_candidates, max_slope_deg, 2.0)
        inconsistent = [
            i
            for i in range(len(centers) - 1)
            if np.isfinite(picks[i]) and np.isfinite(picks[i + 1]) and abs(picks[i + 1] - picks[i]) > delta[i]
        ]
        if not adaptive or not inconsistent or len(centers) >= budget:
            break
        added = False
        for i in inconsistent:
            if len(centers) >= budget:
                break
            mid = (centers[i] + centers[i + 1]) // 2
            if centers[i + 1] - centers[i] > slab_pings and mid not in slabs:
                ensure(mid)
                centers.append(mid)
                added = True
        if not added:
            break
        centers.sort()

    centers = np.asarray(centers)
    finite = np.isfinite(picks)
    if finite.sum() < 2:
        return None
    pings = np.arange(n_ping)
    z = np.interp(pings, centers[finite], picks[finite])
    # Uncertainty per gap: the change actually seen across it (a gap the
    # bisection could not settle gets the full break-point instead), with
    # floors of a fraction of depth, an absolute floor and two pulse
    # lengths. The spec's delta alone is the largest plausible change over
    # the gap, which on a fast-pinging shelf survey is hundreds of metres
    # and would hand the detector the whole recording as its window.
    gap_sigma = np.full(len(centers), np.nan)
    for i in range(len(centers) - 1):
        if i in inconsistent or not (np.isfinite(picks[i]) and np.isfinite(picks[i + 1])):
            gap_sigma[i] = delta[i]
        else:
            gap_sigma[i] = abs(picks[i + 1] - picks[i])
    gap_sigma[-1] = gap_sigma[-2] if len(centers) > 1 else geom.pulse_length_m
    which = np.clip(np.searchsorted(centers, pings, side="right") - 1, 0, len(centers) - 1)
    sigma = np.maximum(gap_sigma[which], sigma_fraction * z)
    sigma = np.maximum(sigma, max(sigma_floor_m, 2.0 * geom.pulse_length_m))
    return xr.Dataset(
        {
            "z_prior": (("ping_time",), z, {"units": "m", "reference": geom.range_var}),
            "sigma_prior": (("ping_time",), sigma, {"units": "m"}),
            "slab_ping": (("slab",), centers),
            "slab_pick": (("slab",), picks, {"units": "m"}),
            "slab_margin": (("slab",), margins),
        },
        coords={"ping_time": ds_Sv["ping_time"], "slab": np.arange(len(centers))},
        attrs={
            "mode": "0b" if adaptive else "0a",
            "n_slabs": int(len(centers)),
            "n_inconsistent_gaps": int(len(inconsistent)),
            "coverage": float(finite.mean()),
            "channel": geom.channel,
            "has_angles": int(geom.has_angles and use_angles),
        },
    )
