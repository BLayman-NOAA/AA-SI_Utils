# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Stage 2: phase activity, alias candidates and seeds on synthetic data."""

import numpy as np
import pytest

from aa_si_utils.seabed.geometry import build_geometry, crop_block
from aa_si_utils.seabed.phase import alias_candidates, nan_mean_filter, phase_activity, seed_mask, seed_phase_threshold
from seabed_synthetic import make_synthetic_sv


def _blocks(ds, geom):
    return (
        crop_block(ds, geom, "Sv"),
        crop_block(ds, geom, "angle_alongship"),
        crop_block(ds, geom, "angle_athwartship"),
    )


def test_nan_mean_filter_ignores_nan():
    x = np.ones((5, 5))
    x[2, 2] = np.nan
    out = nan_mean_filter(x, (3, 3))
    np.testing.assert_allclose(out, 1.0)
    assert np.isnan(nan_mean_filter(np.full((3, 3), np.nan), (3, 3))).all()


def test_phase_activity_separates_water_from_seabed():
    ds, truth = make_synthetic_sv(n_ping=40, seabed_depth_m=100.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    sv, theta, phi = _blocks(ds, geom)

    m_theta, m_phi = phase_activity(theta, phi, geom.pulse_samples, 5)
    m = m_theta + m_phi
    edge = truth["edge_idx"][0] - geom.i_lo
    water = m[:, : edge - 2 * geom.pulse_samples]
    plateau = m[:, edge + 1 : edge + 2 * geom.pulse_samples]
    threshold = seed_phase_threshold(geom.beamwidth_deg)
    assert threshold == pytest.approx(12.25)
    assert np.median(water) > threshold
    assert np.median(plateau) < threshold


def test_phase_activity_ignores_a_steady_offset():
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.3, (20, 50))
    m_theta, _ = phase_activity(noise + 2.5, noise, 5, 3)
    assert np.median(m_theta) < 0.2


def test_seeds_land_on_plateau_only():
    ds, truth = make_synthetic_sv(n_ping=60, seabed_depth_m=100.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    sv, theta, phi = _blocks(ds, geom)

    result = seed_mask(sv, theta, phi, geom, sv_floor_db=-50.0)

    assert result.n_components >= 1
    assert result.seeds.any()
    rows, cols = np.nonzero(result.seeds)
    edge = truth["edge_idx"] - geom.i_lo
    assert (cols >= edge[rows] - geom.pulse_samples).all()
    assert -40.0 < result.sv_threshold_db < -25.0
    assert 100.0 <= result.detection_range_m <= 150.0
    assert result.phase_activity is not None


def test_school_does_not_seed_but_seabed_does():
    school = {"ping_start": 10, "ping_end": 50, "top_m": 70.0, "bottom_m": 80.0}
    ds, truth = make_synthetic_sv(n_ping=60, seabed_depth_m=100.0, school=school)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    sv, theta, phi = _blocks(ds, geom)

    result = seed_mask(sv, theta, phi, geom, sv_floor_db=-50.0)

    school_rows = slice(10, 50)
    school_cols = slice(int((70 - 5) / 0.5) - geom.i_lo, int((80 - 5) / 0.5) - geom.i_lo)
    assert not result.seeds[school_rows, school_cols].any()
    assert result.seeds.any()


def test_small_components_are_uncertain():
    ds, _ = make_synthetic_sv(n_ping=10, seabed_depth_m=100.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    sv, theta, phi = _blocks(ds, geom)

    result = seed_mask(sv, theta, phi, geom, sv_floor_db=-50.0, min_extent_s=20.0)

    assert result.n_components == 0
    assert not result.seeds.any()
    assert result.uncertain.any()
    assert np.isnan(result.detection_range_m)
    assert result.sv_threshold_db >= -50.0


def test_without_angles_seeds_come_from_sv_with_warning():
    ds, _ = make_synthetic_sv(n_ping=60, seabed_depth_m=100.0, angles=False)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    sv = crop_block(ds, geom, "Sv")

    with pytest.warns(UserWarning, match="no split-beam angles"):
        result = seed_mask(sv, None, None, geom, sv_floor_db=-50.0)

    assert result.seeds.any()
    assert result.phase_activity is None
    assert not alias_candidates(sv, None, None, geom).any()


def test_alias_patch_is_flagged_and_seabed_is_not():
    alias = {"ping_start": 5, "ping_end": 55, "top_m": 60.0, "bottom_m": 75.0}
    ds, truth = make_synthetic_sv(n_ping=60, seabed_depth_m=100.0, alias=alias)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    sv, theta, phi = _blocks(ds, geom)

    a = alias_candidates(sv, theta, phi, geom)

    patch_rows = slice(5, 55)
    patch_cols = slice(int((60 - 5) / 0.5) - geom.i_lo, int((75 - 5) / 0.5) - geom.i_lo)
    assert a[patch_rows, patch_cols].mean() > 0.9
    edge = truth["edge_idx"] - geom.i_lo
    seabed = np.zeros_like(a)
    for p in range(60):
        seabed[p, edge[p] : edge[p] + 3 * geom.pulse_samples] = True
    assert a[seabed].mean() < 0.05

    result = seed_mask(sv, theta, phi, geom, sv_floor_db=-50.0, alias=a)
    assert not (result.seeds & a).any()
    assert result.seeds.any()


def test_clean_echogram_has_no_alias_candidates():
    ds, _ = make_synthetic_sv(n_ping=40, seabed_depth_m=100.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    sv, theta, phi = _blocks(ds, geom)
    assert not alias_candidates(sv, theta, phi, geom).any()
