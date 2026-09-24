# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Stage 5 candidates and Viterbi line, Stage 8 confidence."""

import numpy as np
import pytest

from aa_si_utils.seabed.dp import (
    emission_cost,
    run_dp,
    select_candidates,
    transition_cost,
    transition_params,
)
from aa_si_utils.seabed.features import background_estimate, point_features
from aa_si_utils.seabed.geometry import build_geometry, crop_block
from aa_si_utils.seabed.phase import seed_mask
from aa_si_utils.seabed.scoring import PRESETS, reference_table, score_mode1
from seabed_synthetic import make_synthetic_sv


def _score(ds, geom):
    sv = crop_block(ds, geom, "Sv")
    theta = crop_block(ds, geom, "angle_alongship") if geom.has_angles else None
    phi = crop_block(ds, geom, "angle_athwartship") if geom.has_angles else None
    seeds = seed_mask(sv, theta, phi, geom)
    bg = background_estimate(sv, geom, seeds=seeds.seeds)
    features, _ = point_features(sv, geom, bg, theta, phi, activity=seeds.phase_activity)
    refs = reference_table(PRESETS["shelf"], seeds.sv_threshold_db, geom.beamwidth_deg)
    return score_mode1(features, refs)[0], sv


def test_candidates_are_separated_and_sorted():
    ds, truth = make_synthetic_sv(n_ping=30, seabed_depth_m=100.0, multiple=True, n_sample=500)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=240.0)
    score, _ = _score(ds, geom)

    cand = select_candidates(score, geom, n_candidates=5, min_score=None)

    assert cand.idx.shape == (30, 5)
    assert cand.valid.all()
    for p in range(30):
        cols = cand.idx[p][cand.idx[p] >= 0]
        assert len(cols) >= 2
        assert np.all(np.diff(cand.score[p][: len(cols)]) <= 0)
        assert np.min(np.abs(np.subtract.outer(cols, cols))[~np.eye(len(cols), dtype=bool)]) > geom.pulse_samples
    edge = truth["edge_idx"] - geom.i_lo
    assert np.abs(cand.idx[:, 0] - edge).max() <= geom.pulse_samples


def test_min_score_skips_pings():
    ds, _ = make_synthetic_sv(n_ping=10, seabed_depth_m=100.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    score, _ = _score(ds, geom)
    cand = select_candidates(score, geom, n_candidates=3, min_score=1e9)
    assert not cand.valid.any()
    assert np.isnan(cand.range_m).all()


def test_transition_costs():
    assert transition_cost(np.array([1.0]), 2.0, 4.0, "huber")[0] == pytest.approx(1.0)
    assert transition_cost(np.array([3.0]), 2.0, 4.0, "huber")[0] == pytest.approx(8.0)
    assert transition_cost(np.array([10.0]), 2.0, 1.0, "truncated")[0] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="unknown transition"):
        transition_cost(np.array([1.0]), 1.0, 1.0, "cubic")


def test_transition_params_scale_with_distance_and_pulse():
    ds, _ = make_synthetic_sv(n_ping=20, seabed_depth_m=100.0, vessel_speed_m_s=5.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    score, _ = _score(ds, geom)
    cand = select_candidates(score, geom)

    params = transition_params(cand, geom, max_slope_deg=30.0, alpha=1.0, beta_fraction=0.1)

    # 5 m per ping at 30 degrees is 2.9 m, so the 2 m pulse floor is exceeded.
    np.testing.assert_allclose(params.delta, 5.0 * np.tan(np.radians(30.0)), rtol=0.02)
    assert params.lam > 0
    assert params.beta >= 0
    assert params.temperature > 0

    ds_station, _ = make_synthetic_sv(n_ping=20, seabed_depth_m=100.0, gps=False)
    geom_station = build_geometry(ds_station, "38000", r_min=50.0, r_max=150.0)
    params_station = transition_params(cand, geom_station)
    np.testing.assert_allclose(params_station.delta, geom_station.pulse_length_m)


def test_emission_prefers_shallower_of_equal_candidates():
    ds, _ = make_synthetic_sv(n_ping=5, seabed_depth_m=100.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    score, _ = _score(ds, geom)
    cand = select_candidates(score, geom, n_candidates=2, min_score=None)
    cand.score[:] = np.where(cand.idx >= 0, 1.0, np.nan)
    cost = emission_cost(cand, geom, beta=0.5)
    shallow = np.argmin(cand.range_m, axis=1)
    assert np.all(cost[np.arange(5), shallow] <= np.nanmin(np.where(cand.idx >= 0, cost, np.inf), axis=1) + 1e-12)
    assert np.isinf(cost[cand.idx < 0]).all()


def test_dp_follows_slope_and_step():
    depth = np.concatenate([np.linspace(90.0, 110.0, 30), np.full(30, 130.0)])
    ds, truth = make_synthetic_sv(n_ping=60, seabed_depth_m=depth)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=170.0)
    score, _ = _score(ds, geom)
    cand = select_candidates(score, geom)
    params = transition_params(cand, geom, max_slope_deg=30.0)

    dp = run_dp(cand, geom, params)

    edge = truth["edge_idx"] - geom.i_lo
    assert np.isfinite(dp.r_star_idx).all()
    assert np.abs(dp.r_star_idx - edge).max() <= geom.pulse_samples
    assert np.isfinite(dp.confidence).all()
    assert dp.confidence.min() >= 0.0 and dp.confidence.max() <= 1.0
    assert np.isfinite(dp.total_cost)


def test_dp_prefers_seabed_over_multiple():
    ds, truth = make_synthetic_sv(n_ping=40, seabed_depth_m=100.0, multiple=True, n_sample=500)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=240.0)
    score, _ = _score(ds, geom)
    cand = select_candidates(score, geom)
    dp = run_dp(cand, geom, transition_params(cand, geom))
    edge = truth["edge_idx"] - geom.i_lo
    assert np.abs(dp.r_star_idx - edge).max() <= geom.pulse_samples


def test_dp_prefers_seabed_over_school():
    school = {"ping_start": 5, "ping_end": 35, "top_m": 80.0, "bottom_m": 92.0}
    ds, truth = make_synthetic_sv(n_ping=40, seabed_depth_m=100.0, school=school)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    score, _ = _score(ds, geom)
    cand = select_candidates(score, geom)
    dp = run_dp(cand, geom, transition_params(cand, geom))
    edge = truth["edge_idx"] - geom.i_lo
    assert np.abs(dp.r_star_idx - edge).max() <= geom.pulse_samples


def test_dp_bridges_skipped_pings_and_flags_ambiguity():
    ds, truth = make_synthetic_sv(n_ping=30, seabed_depth_m=100.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    score, _ = _score(ds, geom)
    score[10:13] = np.nan
    cand = select_candidates(score, geom)
    assert not cand.valid[10:13].any()

    dp = run_dp(cand, geom, transition_params(cand, geom))

    assert np.isnan(dp.r_star_idx[10:13]).all()
    assert np.isnan(dp.confidence[10:13]).all()
    edge = truth["edge_idx"] - geom.i_lo
    ok = cand.valid
    assert np.abs(dp.r_star_idx[ok] - edge[ok]).max() <= geom.pulse_samples

    # A second, identical candidate well outside the tolerance on one ping
    # halves the posterior and removes the margin there; one inside the
    # tolerance counts as the same pick and changes neither.
    far = 6 * geom.pulse_samples
    cand.idx[20, 1] = cand.idx[20, 0] + far
    cand.score[20, 1] = cand.score[20, 0]
    cand.range_m[20, 1] = cand.range_m[20, 0] + far * geom.dr
    dp2 = run_dp(cand, geom, transition_params(cand, geom, beta_fraction=0.0))
    assert dp2.confidence[20] < dp.confidence[20]
    assert dp2.margin[20] < dp.margin[20]

    near = geom.pulse_samples
    cand.idx[20, 1] = cand.idx[20, 0] + near
    cand.range_m[20, 1] = cand.range_m[20, 0] + near * geom.dr
    dp3 = run_dp(cand, geom, transition_params(cand, geom, beta_fraction=0.0), tolerance_pulses=2.0)
    assert dp3.confidence[20] == pytest.approx(dp.confidence[20], abs=0.05)

    # A far alternative inside the same loud run (the echo tail) is also
    # the same pick once the loud mask is supplied.
    cand.idx[20, 1] = cand.idx[20, 0] + far
    cand.range_m[20, 1] = cand.range_m[20, 0] + far * geom.dr
    loud = np.zeros(score.shape, dtype=bool)
    loud[20, cand.idx[20, 0] : cand.idx[20, 1] + 1] = True
    dp4 = run_dp(cand, geom, transition_params(cand, geom, beta_fraction=0.0), loud=loud)
    assert dp4.confidence[20] == pytest.approx(dp.confidence[20], abs=0.05)
    assert dp4.margin[20] > dp2.margin[20]


def test_dp_with_no_candidates_at_all():
    ds, _ = make_synthetic_sv(n_ping=5, seabed_depth_m=100.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    score, _ = _score(ds, geom)
    cand = select_candidates(score, geom, min_score=1e9)
    dp = run_dp(cand, geom, transition_params(cand, geom))
    assert np.isnan(dp.r_star_idx).all()
    assert np.isinf(dp.total_cost)
