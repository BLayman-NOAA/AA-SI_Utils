# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Stage 1 provisional background estimate and Stage 3 point features.

Point features depend on a small window around each pixel. Echo-shape
windows are in pulse lengths, neighbourhood windows in metres or seconds.
"""

import warnings

import numpy as np

from aa_si_utils.seabed.phase import nan_mean_filter

LINEAR_EPS = 1e-12


def _to_linear(sv_db):
    lin = 10.0 ** (sv_db / 10.0)
    lin[~np.isfinite(sv_db)] = np.nan
    return lin


def _to_db(lin):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(lin)


def background_estimate(sv, geom, alias=None, seeds=None, cell_m=5.0, cell_s=20.0):
    """Provisional background noise after De Robertis and Higginbottom (2007).

    Time-varied gain is removed, the power is averaged in cells, the
    minimum over range is taken per ping block, and the gain is restored so
    the estimate can be subtracted from Sv sample by sample. Cells in the
    alias mask and at or below the top of the seed mask are excluded.

    Args:
        sv: Sv in dB on the cropped block, shape (P, S).
        geom: ``PingGeometry``; supplies absorption and the range axis.
        alias: Alias mask to exclude, or None.
        seeds: Seed mask; everything at or below its top per ping is
            excluded. None excludes nothing.
        cell_m: Cell height in metres.
        cell_s: Cell width in seconds.

    Returns:
        np.ndarray: Background Sv in dB, shape (P, S).
    """
    n_ping, n_sample = sv.shape
    r = geom.crop_range_axis() - geom.range0[:, None]
    r = np.maximum(r, geom.dr)
    tvg = 20.0 * np.log10(r) + 2.0 * geom.absorption_db_per_m * r
    power = _to_linear(sv - tvg)

    exclude = np.zeros(sv.shape, dtype=bool)
    if alias is not None:
        exclude |= alias
    if seeds is not None:
        has_seed = seeds.any(axis=1)
        seed_top = np.where(has_seed, seeds.argmax(axis=1), n_sample)
        exclude |= np.arange(n_sample)[None, :] >= seed_top[:, None]
    power[exclude] = np.nan

    ns = geom.metres_to_samples(cell_m)
    npg = geom.seconds_to_pings(cell_s)
    n_pb = -(-n_ping // npg)
    n_sb = -(-n_sample // ns)
    padded = np.full((n_pb * npg, n_sb * ns), np.nan)
    padded[:n_ping, :n_sample] = power
    cells = padded.reshape(n_pb, npg, n_sb, ns)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cell_mean = np.nanmean(np.nanmean(cells, axis=3), axis=1)
        block_min = np.nanmin(cell_mean, axis=1)
    fallback = np.nanmin(block_min) if np.isfinite(block_min).any() else np.nanmin(power)
    block_min = np.where(np.isfinite(block_min), block_min, fallback)
    per_ping = np.repeat(block_min, npg)[:n_ping]
    return _to_db(per_ping[:, None]) + tvg


def _trailing_min(x, n):
    """Minimum over the n samples before each sample, +inf at the edge."""
    out = np.full(x.shape, np.inf)
    for i in range(1, n + 1):
        out[:, i:] = np.fmin(out[:, i:], x[:, :-i])
    return out


def _leading_min(x, n):
    """Minimum over the n samples after each sample, +inf at the edge."""
    out = np.full(x.shape, np.inf)
    for i in range(1, n + 1):
        out[:, :-i] = np.fmin(out[:, :-i], x[:, i:])
    return out


def echo_asymmetry(sv, n_k, eps_db=0.1):
    """Rise-versus-decay asymmetry f3 over +/- n_k samples, in [-1, 1]."""
    rise = np.maximum(sv - _trailing_min(sv, n_k), 0.0)
    decay = np.maximum(sv - _leading_min(sv, n_k), 0.0)
    rise[~np.isfinite(rise)] = 0.0
    decay[~np.isfinite(decay)] = 0.0
    return (rise - decay) / (rise + decay + eps_db)


def angle_stability(theta, phi, n_k):
    """Sum of the alongship and athwartship angle std over +/- n_k samples."""
    size = (1, 2 * n_k + 1)
    out = np.zeros(theta.shape)
    for angle in (theta, phi):
        mean = nan_mean_filter(angle, size)
        mean_sq = nan_mean_filter(angle**2, size)
        out += np.sqrt(np.maximum(mean_sq - mean**2, 0.0))
    return out


def energy_gradient(sv, window, n_m):
    """Cumulative-energy ratio f6 in dB across +/- n_m samples.

    The cumulative sum starts at each ping's search window so that energy
    above the window does not dilute the jump at the leading edge.
    """
    lin = _to_linear(sv)
    lin = np.where(window & np.isfinite(lin), lin, 0.0)
    cum = np.cumsum(lin, axis=1)
    n_sample = sv.shape[1]
    idx = np.arange(n_sample)
    ahead = cum[:, np.minimum(idx + n_m, n_sample - 1)]
    behind = cum[:, np.maximum(idx - n_m, 0)]
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(ahead / (behind + LINEAR_EPS))


def _window_mean(x, start, length):
    """NaN-aware mean of ``x[:, i + start : i + start + length]`` per sample.

    Windows are clipped at the array edges and averaged over the samples
    they keep; a window with no finite sample gives NaN.
    """
    n_sample = x.shape[1]
    valid = np.isfinite(x)
    values = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(np.where(valid, x, 0.0), axis=1)], axis=1)
    counts = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(valid, axis=1)], axis=1)
    idx = np.arange(n_sample)
    lo = np.clip(idx + start, 0, n_sample)
    hi = np.clip(idx + start + length, 0, n_sample)
    total = values[:, hi] - values[:, lo]
    count = counts[:, hi] - counts[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count <= 0] = np.nan
    return out


def shape_features(sv, geom, window_m, activity=None):
    """Stage 3b local shape features f8 to f10 (Modes 2 and 3).

    The above window covers ``window_m`` metres ending at the candidate; the
    below window covers the same length starting one pulse length below it,
    so the leading edge itself is excluded. Both are at least two pulse
    lengths long.

    Args:
        sv: Sv in dB on the cropped block, shape (P, S).
        geom: ``PingGeometry``.
        window_m: Window length in metres (regime preset).
        activity: ``(M_theta, M_phi)`` on the seed window, or None to skip
            f10.

    Returns:
        tuple: ``(features, attrs)`` with ``f8`` (below minus above mean
        Sv, dB), ``f9`` (below Sv std, dB) and, with angles, ``f10`` (below
        mean of sqrt(M), deg).
    """
    n_w = max(geom.metres_to_samples(window_m), geom.pulse_lengths_to_samples(2.0))
    n_k = geom.pulse_samples
    above = _window_mean(sv, -n_w, n_w)
    below = _window_mean(sv, n_k, n_w)
    below_sq = _window_mean(sv**2, n_k, n_w)
    features = {
        "f8": below - above,
        "f9": np.sqrt(np.maximum(below_sq - below**2, 0.0)),
    }
    window_attrs = {"window_m": float(window_m), "window_samples": int(n_w), "below_offset_pulse_lengths": 1.0}
    attrs = {
        "f8": {"long_name": "below minus above Sv contrast", "units": "dB", **window_attrs},
        "f9": {"long_name": "below-region Sv std", "units": "dB", **window_attrs},
    }
    if activity is not None:
        features["f10"] = _window_mean(np.sqrt(activity[0] + activity[1]), n_k, n_w)
        attrs["f10"] = {"long_name": "below-region phase activity", "units": "deg", **window_attrs}
    return features, attrs


def point_features(sv, geom, background, theta=None, phi=None, activity=None, alias=None):
    """Stage 3 point features f1 to f6 on the cropped block.

    Args:
        sv: Sv in dB, shape (P, S).
        geom: ``PingGeometry``.
        background: Background Sv estimate in dB, shape (P, S).
        theta: Alongship physical angle, or None.
        phi: Athwartship physical angle, or None.
        activity: ``(M_theta, M_phi)`` on the seed window from Stage 2b,
            or None to recompute (or skip without angles).
        alias: Alias mask; pixels in it are set to NaN in every feature.

    Returns:
        tuple: ``(features, attrs)`` where ``features`` maps ``f1`` to
        ``f6`` (``f2`` and ``f5`` only with angles) to (P, S) arrays and
        ``attrs`` records the window each was computed with.
    """
    n_k = geom.pulse_samples
    window = geom.window_mask()
    features = {
        "f1": sv.copy(),
        "f3": echo_asymmetry(sv, n_k),
        "f4": sv - background,
        "f6": energy_gradient(sv, window, n_k),
    }
    attrs = {
        "f1": {"long_name": "Sv", "units": "dB"},
        "f3": {"long_name": "echo asymmetry", "window_pulse_lengths": 1.0},
        "f4": {"long_name": "local SNR", "units": "dB"},
        "f6": {"long_name": "leading-edge energy gradient", "units": "dB", "window_pulse_lengths": 1.0},
    }
    if theta is not None and phi is not None and geom.has_angles:
        if activity is None:
            from aa_si_utils.seabed.phase import phase_activity

            activity = phase_activity(theta, phi, n_k, geom.seconds_to_pings(5.0))
        features["f2"] = np.sqrt(activity[0] + activity[1])
        features["f5"] = angle_stability(theta, phi, n_k)
        attrs["f2"] = {"long_name": "local phase activity", "units": "deg", "window_pulse_lengths": 1.0, "window_s": 5.0}
        attrs["f5"] = {"long_name": "phase stability along range", "units": "deg", "window_pulse_lengths": 1.0}
    if alias is not None and alias.any():
        for arr in features.values():
            arr[alias] = np.nan
    return features, attrs
