# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Stage 5c leading-edge walk-back and Stage 6 backstep."""

import numpy as np
import pytest

from aa_si_utils.seabed.geometry import build_geometry, crop_block
from aa_si_utils.seabed.refine import backstep, leading_edge, local_slope
from seabed_synthetic import make_synthetic_sv


def test_walkback_lands_just_above_the_threshold_crossing():
    ds, truth = make_synthetic_sv(n_ping=20, seabed_depth_m=100.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    sv = crop_block(ds, geom, "Sv")
    edge = truth["edge_idx"] - geom.i_lo
    # Picks sitting inside the plateau, half a pulse below the edge.
    picks = edge.astype(float) + geom.pulse_samples // 2

    le, bound = leading_edge(sv, picks, geom, walkback_db=10.0)

    assert not bound.any()
    assert np.all(le <= picks)
    assert np.abs(le - (edge - 1)).max() <= 1
    rows = np.arange(20)
    peak = np.array([sv[p, int(picks[p]) : int(picks[p]) + geom.pulse_samples + 1].max() for p in rows])
    assert np.all(sv[rows, le.astype(int)] < peak - 10.0)


def test_walkback_keeps_pick_and_flags_when_bound_is_hit():
    ds, truth = make_synthetic_sv(n_ping=5, seabed_depth_m=100.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    sv = crop_block(ds, geom, "Sv")
    edge = truth["edge_idx"] - geom.i_lo
    picks = edge.astype(float) + 3 * geom.pulse_samples
    picks[2] = np.nan

    le, bound = leading_edge(sv, picks, geom, walkback_db=10.0)

    assert bound[[0, 1, 3, 4]].all()
    np.testing.assert_allclose(le[[0, 1, 3, 4]], picks[[0, 1, 3, 4]])
    assert np.isnan(le[2]) and not bound[2]


def test_backstep_is_above_edge_and_capped():
    ds, _ = make_synthetic_sv(n_ping=10, seabed_depth_m=100.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    r_le = np.full(10, 100.0)

    r_int = backstep(r_le, geom, effective_beamwidth_deg=12.0, slope_correction=False)

    expected = 100.0 - geom.pulse_length_m - 95.0 * (1 - np.cos(np.radians(6.0)))
    np.testing.assert_allclose(r_int, expected)

    capped = backstep(r_le, geom, effective_beamwidth_deg=90.0, cap_fraction=0.1, slope_correction=False)
    np.testing.assert_allclose(capped, 100.0 - geom.pulse_length_m - 9.5)

    default = backstep(r_le, geom, slope_correction=False)
    assert np.all(default < r_le)


def test_slope_correction_widens_dead_zone_on_a_slope():
    ds, _ = make_synthetic_sv(n_ping=10, seabed_depth_m=100.0, vessel_speed_m_s=5.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    r_le = 100.0 + 5.0 * np.arange(10)  # 45 degree slope at 5 m per ping

    slope = local_slope(r_le, geom)
    assert slope[0] == 0.0 and slope[-1] == 0.0
    np.testing.assert_allclose(slope[3:-3], np.radians(45.0), rtol=0.02)

    jitter = 100.0 + np.random.default_rng(3).normal(0.0, 2.0, 10)
    raw = np.abs(local_slope(jitter, geom, smooth_pings=1)).max()
    assert raw > np.radians(10.0)
    assert np.abs(local_slope(jitter, geom)).max() < raw

    flat = backstep(r_le, geom, effective_beamwidth_deg=12.0, slope_correction=False, cap_fraction=1.0)
    sloped = backstep(r_le, geom, effective_beamwidth_deg=12.0, slope_correction=True, cap_fraction=1.0)
    assert np.all(sloped[3:-3] < flat[3:-3])

    r_le[4] = np.nan
    assert np.isnan(backstep(r_le, geom)[4])
