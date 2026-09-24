# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Stage 2: phase-activity statistic, alias candidates and seabed seeds.

One sign convention: water column and aliased seabed show high angle
activity, true seabed under the beam shows low activity. Seeds and the f2
feature measure activity as the variance of the physical angles within the
window rather than the spec's mean of squares. On HB1603 at 38 kHz the
seabed centroid sat about 2 degrees off axis (vessel roll and seabed slope),
which put the mean of squares at 11.5 deg2 against a 3 deg2 threshold, while
the variance within a pulse length was 2.9 deg2 on the seabed and 30 deg2 in
the water column. The alias detector keeps the mean of squares, which is
what the cited Blackwell statistic responds to.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

EIGHT_CONNECTED = ndimage.generate_binary_structure(2, 2)


def nan_mean_filter(x, size):
    """NaN-aware moving mean over a (pings, samples) window.

    Args:
        x: Array of shape (P, S).
        size: Window as ``(pings, samples)``.

    Returns:
        np.ndarray: Mean of the finite values in each window, NaN where the
        window holds none. Edges repeat the nearest value.
    """
    valid = np.isfinite(x)
    num = ndimage.uniform_filter(np.where(valid, x, 0.0), size=size, mode="nearest")
    den = ndimage.uniform_filter(valid.astype(float), size=size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den <= 0] = np.nan
    return out


def angle_variance(x, size):
    """Variance of an angle field within each window, degrees squared."""
    mean = nan_mean_filter(x, size)
    return np.maximum(nan_mean_filter(x**2, size) - mean**2, 0.0)


def phase_activity(theta, phi, window_samples, window_pings):
    """Variance of the alongship and athwartship angles over a window.

    Insensitive to a steady off-axis bearing from roll, pitch or seabed
    slope; it measures whether the angle estimates are stable.

    Args:
        theta: Alongship physical angle in degrees, shape (P, S).
        phi: Athwartship physical angle in degrees, shape (P, S).
        window_samples: Window length in range samples.
        window_pings: Window length in pings.

    Returns:
        tuple: ``(M_theta, M_phi)`` in degrees squared, shape (P, S).
    """
    size = (int(window_pings), int(window_samples))
    return angle_variance(theta, size), angle_variance(phi, size)


def seed_phase_threshold(beamwidth_deg):
    """Default seed threshold on the summed angle variance, degrees squared.

    (beamwidth / 2)^2: for a 7 degree beam 12.3 deg2, about 2.5 degrees of
    angle noise per axis, which passed 78 % of the HB1603 38 kHz seabed at
    1750 m and 15 % of its water column.
    """
    return (float(beamwidth_deg) / 2.0) ** 2


def _linear_median_db(sv_db):
    """Median of Sv taken in the linear domain, returned in dB."""
    finite = sv_db[np.isfinite(sv_db)]
    if finite.size == 0:
        return np.nan
    return float(10.0 * np.log10(np.median(10.0 ** (finite / 10.0))))


def alias_candidates(
    sv,
    theta,
    phi,
    geom,
    threshold_theta_deg2=70.0,
    threshold_phi_deg2=28.0,
    window_theta=(5.0, 30.0),
    window_phi=(10.0, 50.0),
    sv_floor_db=-70.0,
):
    """Aliased-seabed candidate mask after Blackwell et al. (2020).

    Pixels whose angle activity exceeds the thresholds are phase-active;
    connected regions of Sv above an adaptive threshold that touch a
    phase-active pixel are alias candidates. Thresholds are in physical
    degrees squared on the mean-of-squares statistic; Blackwell's published
    values were derived on smoothed electrical angles, so rescale them by
    the transducer's angle sensitivity if you carry them over directly.

    Off by default in the pipeline: on HB1603 (8 s pinging, no aliasing
    possible) the water-column angle noise alone exceeded the thresholds and
    the resulting mask covered the true seabed. Enable it only with
    thresholds checked against a survey that does alias.

    Args:
        sv: Sv in dB on the cropped block, shape (P, S).
        theta: Alongship physical angle, shape (P, S), or None.
        phi: Athwartship physical angle, shape (P, S), or None.
        geom: ``PingGeometry`` for unit conversion.
        threshold_theta_deg2: Alongship activity threshold.
        threshold_phi_deg2: Athwartship activity threshold.
        window_theta: Alongship window as ``(metres, seconds)``.
        window_phi: Athwartship window as ``(metres, seconds)``.
        sv_floor_db: Floor for the adaptive Sv threshold.

    Returns:
        np.ndarray: Boolean alias candidate mask, shape (P, S). All False
        without angles.
    """
    if theta is None or phi is None or not geom.has_angles:
        return np.zeros(sv.shape, dtype=bool)
    m_theta = nan_mean_filter(
        theta**2,
        (geom.seconds_to_pings(window_theta[1]), geom.metres_to_samples(window_theta[0])),
    )
    m_phi = nan_mean_filter(
        phi**2,
        (geom.seconds_to_pings(window_phi[1]), geom.metres_to_samples(window_phi[0])),
    )
    active = (m_theta > threshold_theta_deg2) | (m_phi > threshold_phi_deg2)
    active &= np.isfinite(sv)
    if not active.any():
        return np.zeros(sv.shape, dtype=bool)
    threshold = max(_linear_median_db(sv[active]), sv_floor_db)
    loud = np.isfinite(sv) & (sv > threshold)
    labels, n_labels = ndimage.label(loud, structure=EIGHT_CONNECTED)
    if n_labels == 0:
        return np.zeros(sv.shape, dtype=bool)
    hit = np.zeros(n_labels + 1, dtype=bool)
    hit[np.unique(labels[active & loud])] = True
    hit[0] = False
    # Noise leaves holes inside a region that is above the median only on
    # average; fill them so the whole region is excluded downstream.
    return ndimage.binary_fill_holes(hit[labels])


@dataclass
class SeedResult:
    """Seabed seeds and the references derived from them.

    Attributes:
        seeds: Boolean seed mask S, shape (P, S).
        uncertain: Boolean mask U of components too small to be seeds.
        sv_threshold_db: Seed-median Sv reference T_Sv,S in dB.
        detection_range_m: R_det, the 95th percentile range of seed pixels,
            NaN when there are no seeds.
        n_components: Number of seed components kept.
        phase_activity: ``(M_theta, M_phi)`` on the seed window, or None.
    """

    seeds: np.ndarray
    uncertain: np.ndarray
    sv_threshold_db: float
    detection_range_m: float
    n_components: int
    phase_activity: tuple = None


def seed_mask(
    sv,
    theta,
    phi,
    geom,
    sv_floor_db=-50.0,
    alias=None,
    window_seconds=5.0,
    min_extent_s=20.0,
    min_extent_pulses=1.0,
    phase_threshold_deg2=None,
):
    """High-precision seabed seed regions (Stage 2b).

    Args:
        sv: Sv in dB on the cropped block, shape (P, S).
        theta: Alongship physical angle, shape (P, S), or None.
        phi: Athwartship physical angle, shape (P, S), or None.
        geom: ``PingGeometry``; supplies the search window and beamwidth.
        sv_floor_db: Regime floor for the adaptive seed Sv threshold.
        alias: Alias candidate mask to exclude, or None.
        window_seconds: Along-track extent of the seed phase window; its
            range extent is one pulse length.
        min_extent_s: Minimum along-track extent of a seed component.
        min_extent_pulses: Minimum median per-ping range extent of a seed
            component, in pulse lengths.
        phase_threshold_deg2: Threshold on the summed angle variance below
            which a pixel is phase-stable; default
            :func:`seed_phase_threshold`.

    Returns:
        SeedResult: Seeds, uncertain components and references.
    """
    window = geom.window_mask() & np.isfinite(sv)
    if alias is not None:
        window &= ~alias

    activity = None
    if theta is not None and phi is not None and geom.has_angles:
        activity = phase_activity(theta, phi, geom.pulse_samples, geom.seconds_to_pings(window_seconds))
        if phase_threshold_deg2 is None:
            phase_threshold_deg2 = seed_phase_threshold(geom.beamwidth_deg)
        stable = (activity[0] + activity[1]) < phase_threshold_deg2
        low_activity = window & stable
    else:
        warnings.warn(
            "no split-beam angles: seeds are selected on Sv and extent alone, "
            "so a dense aggregation can seed the detector"
        )
        low_activity = window

    threshold = sv_floor_db
    if low_activity.any():
        threshold = max(_linear_median_db(sv[low_activity]), sv_floor_db)
    candidate = low_activity & (sv > threshold)

    labels, n_labels = ndimage.label(candidate, structure=EIGHT_CONNECTED)
    seeds = np.zeros(sv.shape, dtype=bool)
    uncertain = np.zeros(sv.shape, dtype=bool)
    n_kept = 0
    if n_labels > 0:
        min_pings = geom.seconds_to_pings(min_extent_s)
        min_samples = geom.pulse_lengths_to_samples(min_extent_pulses)
        keep = _components_with_extent(labels, n_labels, min_pings, min_samples)
        seeds = keep[labels]
        uncertain = candidate & ~seeds
        n_kept = int(keep.sum())

    if seeds.any():
        sv_ref = _linear_median_db(sv[seeds])
        ranges = geom.crop_range_axis()[seeds]
        r_det = float(np.percentile(ranges, 95))
    else:
        sv_ref = threshold
        r_det = np.nan
    return SeedResult(seeds, uncertain, float(sv_ref), r_det, n_kept, activity)


def _components_with_extent(labels, n_labels, min_pings, min_samples):
    """Which labels span enough pings and enough samples per ping."""
    rows, cols = np.nonzero(labels)
    lab = labels[rows, cols]
    per_ping = np.zeros((n_labels + 1, labels.shape[0]), dtype=np.int32)
    np.add.at(per_ping, (lab, rows), 1)
    n_pings = (per_ping > 0).sum(axis=1)
    masked = np.where(per_ping > 0, per_ping, np.nan).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_samples = np.nanmedian(masked, axis=1)
    keep = (n_pings >= min_pings) & (median_samples >= min_samples)
    keep[0] = False
    return keep
