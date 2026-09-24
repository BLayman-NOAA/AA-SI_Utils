# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Stage 4 Mode 1 scoring and regime presets."""

import numpy as np
import pytest

from aa_si_utils.seabed.features import background_estimate, point_features
from aa_si_utils.seabed.geometry import build_geometry, crop_block
from aa_si_utils.seabed.phase import seed_mask
from aa_si_utils.seabed.scoring import PRESETS, reference_table, score_mode1, select_preset
from seabed_synthetic import make_synthetic_sv


def test_preset_selection_by_depth_and_name():
    assert select_preset("auto", 50.0)[0] == "shelf"
    assert select_preset("auto", 500.0)[0] == "slope"
    assert select_preset("auto", 3000.0)[0] == "deep"
    assert select_preset("deep", 50.0)[0] == "deep"
    with pytest.raises(ValueError, match="unknown regime"):
        select_preset("abyssal", 50.0)
    for preset in PRESETS.values():
        assert preset["version"]


def test_reference_table_uses_seed_median_and_beamwidth():
    refs = reference_table(PRESETS["slope"], seed_sv_db=-33.0, beamwidth_deg=8.0, seed_f2_deg=1.5, seed_f5_deg=2.5)
    assert refs["f1"] == (-33.0, 10.0, 1.0)
    assert refs["f2"] == (1.5, 4.0, -1.0)
    assert refs["f5"] == (2.5, 4.0, -1.0)
    assert refs["f4"][0] == 20.0
    assert refs["f5"][2] == -1.0
    assert set(refs) == {"f1", "f2", "f3", "f4", "f5", "f6"}

    refs2 = reference_table(PRESETS["deep"], -33.0, 8.0, mode=2)
    assert refs2["f8"] == (15.0, 10.0, 1.0)
    assert refs2["f9"] == (5.0, 3.0, -1.0)
    assert refs2["f10"] == (4.0, 4.0, -1.0)
    with pytest.raises(ValueError, match="mode"):
        reference_table(PRESETS["deep"], -33.0, 8.0, mode=3)


def test_scores_are_bounded_and_weighted():
    refs = reference_table(PRESETS["shelf"], -30.0, 7.0)
    features = {"f1": np.array([[-30.0, 100.0, -200.0]]), "f3": np.array([[0.5, 10.0, -10.0]])}
    lam, used = score_mode1(features, refs)
    assert used == ["f1", "f3"]
    np.testing.assert_allclose(lam[0, 0], 0.0, atol=1e-12)
    assert lam[0, 1] == pytest.approx(2.0, abs=1e-3)
    assert lam[0, 2] == pytest.approx(-2.0, abs=1e-3)
    lam_w, _ = score_mode1(features, refs, weights={"f3": 2.0})
    assert lam_w[0, 1] == pytest.approx(3.0, abs=1e-3)
    with pytest.raises(ValueError, match="no features"):
        score_mode1({}, refs)


def test_lambda_peaks_within_a_pulse_of_the_edge():
    ds, truth = make_synthetic_sv(n_ping=60, seabed_depth_m=100.0)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    sv = crop_block(ds, geom, "Sv")
    theta = crop_block(ds, geom, "angle_alongship")
    phi = crop_block(ds, geom, "angle_athwartship")
    seeds = seed_mask(sv, theta, phi, geom)
    bg = background_estimate(sv, geom, seeds=seeds.seeds)
    features, _ = point_features(sv, geom, bg, theta, phi, activity=seeds.phase_activity)
    refs = reference_table(PRESETS["shelf"], seeds.sv_threshold_db, geom.beamwidth_deg)

    lam, used = score_mode1(features, refs)

    assert len(used) == 6
    edge = truth["edge_idx"] - geom.i_lo
    peak = np.nanargmax(np.where(geom.window_mask(), lam, -np.inf), axis=1)
    assert np.abs(peak - edge).max() <= geom.pulse_samples
