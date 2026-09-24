# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Stage 5c leading-edge refinement and Stage 6 Ona-Mitson backstep."""

import numpy as np
from scipy import ndimage

# Ratio of effective to nominal beamwidth observed on an ES38B (12 deg
# effective against 7 deg nominal); used when no effective value is given.
EFFECTIVE_BEAMWIDTH_RATIO = 12.0 / 7.0


def leading_edge(sv, r_star_idx, geom, walkback_db=10.0):
    """Walk each pick back to the sample just above the leading edge.

    From the pick, the peak Sv within one pulse length below is found and
    the walk moves toward the transducer until Sv falls ``walkback_db``
    below that peak, stopping after one pulse length.

    Args:
        sv: Sv in dB on the cropped block, shape (P, S).
        r_star_idx: Crop-relative pick per ping, NaN where skipped.
        geom: ``PingGeometry``.
        walkback_db: Drop below the peak that marks the edge.

    Returns:
        tuple: ``(le_idx, bound_hit)``; ``le_idx`` is crop-relative and NaN
        where skipped, ``bound_hit`` marks pings that kept the raw pick
        because the walk reached its bound without crossing the threshold.
    """
    n_ping, n_sample = sv.shape
    n_k = geom.pulse_samples
    le = np.full(n_ping, np.nan)
    bound_hit = np.zeros(n_ping, dtype=bool)
    for p in range(n_ping):
        if not np.isfinite(r_star_idx[p]):
            continue
        i = int(r_star_idx[p])
        below = sv[p, i : min(n_sample, i + n_k + 1)]
        if not np.isfinite(below).any():
            le[p] = i
            bound_hit[p] = True
            continue
        threshold = np.nanmax(below) - walkback_db
        found = False
        for j in range(i, max(-1, i - n_k - 1), -1):
            if sv[p, j] < threshold:
                le[p] = j
                found = True
                break
        if not found:
            le[p] = i
            bound_hit[p] = True
    return le, bound_hit


def local_slope(r_le_m, geom, smooth_pings=5):
    """Seabed slope per ping in radians from the line's finite difference.

    The line is mean filtered over ``smooth_pings`` first, since a pulse
    length of pick jitter between pings tens of metres apart reads as
    several degrees of slope. Uses the along-track distances actually
    travelled; zero where they are unknown or a neighbour is missing.
    """
    n_ping = r_le_m.size
    slope = np.zeros(n_ping)
    line = r_le_m
    if smooth_pings > 1 and n_ping >= smooth_pings:
        finite = np.isfinite(r_le_m)
        filled = np.interp(np.arange(n_ping), np.nonzero(finite)[0], r_le_m[finite]) if finite.any() else r_le_m
        line = np.where(finite, ndimage.uniform_filter1d(filled, size=smooth_pings, mode="nearest"), np.nan)
    dx = geom.dx_ping
    for p in range(1, n_ping - 1):
        run = dx[p - 1] + dx[p]
        rise = line[p + 1] - line[p - 1]
        if np.isfinite(run) and run > 0 and np.isfinite(rise):
            slope[p] = np.arctan(rise / run)
    return slope


def backstep(r_le_m, geom, effective_beamwidth_deg=None, slope_correction=True, cap_fraction=0.15):
    """Ona-Mitson dead-zone backstep from the leading edge to the integration line.

    Args:
        r_le_m: Leading-edge range in metres on ``geom.range_var``, NaN
            where skipped.
        geom: ``PingGeometry``.
        effective_beamwidth_deg: Full effective beamwidth; defaults to the
            nominal beamwidth scaled by the ES38B ratio.
        slope_correction: Add the local slope to the half beamwidth (Patel
            et al. 2009).
        cap_fraction: Cap on the geometric term as a fraction of the range
            from the transducer.

    Returns:
        np.ndarray: Integration line in metres on the same reference.
    """
    if effective_beamwidth_deg is None:
        effective_beamwidth_deg = geom.beamwidth_deg * EFFECTIVE_BEAMWIDTH_RATIO
    r_t = np.maximum(r_le_m - geom.range0, 0.0)
    angle = np.radians(effective_beamwidth_deg / 2.0)
    if slope_correction:
        angle = angle + np.abs(local_slope(r_le_m, geom))
    geometric = np.minimum(r_t * (1.0 - np.cos(angle)), cap_fraction * r_t)
    return r_le_m - geom.pulse_length_m - geometric
