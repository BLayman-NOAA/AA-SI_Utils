# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Pure parts of the inter-method comparison harness."""

import numpy as np
import pytest

from aa_si_utils.seabed.compare import agreement, max_sv_line, sanity_metrics
from aa_si_utils.seabed.geometry import build_geometry, crop_block
from seabed_synthetic import make_synthetic_sv


def test_max_sv_baseline_finds_the_edge():
    ds, truth = make_synthetic_sv(n_ping=30, seabed_depth_m=100.0)
    line = max_sv_line(ds, 38, 50.0, 150.0)
    assert line.dims == ("ping_time",)
    np.testing.assert_allclose(line.values, truth["edge_m"], atol=2.0)


def test_sanity_metrics_separate_a_good_line_from_a_bad_one():
    ds, truth = make_synthetic_sv(n_ping=30, seabed_depth_m=100.0)
    geom = build_geometry(ds, 38, r_min=50.0, r_max=150.0)
    sv = crop_block(ds, geom, "Sv")

    good = sanity_metrics(truth["edge_m"].astype(float), sv, geom)
    bad = sanity_metrics(truth["edge_m"] - 30.0, sv, geom)

    assert good["coverage"] == 1.0
    assert good["contrast_db"] > 30.0
    assert good["below_sv_db"] > -45.0
    assert good["quiet_below_fraction"] == 0.0
    assert good["jump_fraction"] == 0.0
    assert bad["quiet_below_fraction"] == 1.0
    assert bad["contrast_db"] < 10.0

    jumpy = truth["edge_m"] + np.where(np.arange(30) % 2 == 0, 15.0, 0.0)
    assert sanity_metrics(jumpy, sv, geom)["jump_fraction"] > 0.9
    with_gaps = truth["edge_m"].astype(float)
    with_gaps[:10] = np.nan
    assert sanity_metrics(with_gaps, sv, geom)["coverage"] == pytest.approx(20 / 30)


def test_agreement_matrix_and_consensus():
    base = np.linspace(100.0, 110.0, 50)
    lines = {
        "bot": base + 1.0,
        "phase_m1": base,
        "phase_m2": base + 0.2,
        "phase_amp": base + 0.1,
        "ep_basic": base + 40.0,
    }
    pairwise, consensus = agreement(lines)

    assert pairwise.loc["phase_m1", "bot"] == pytest.approx(1.0)
    assert pairwise.loc["bot", "phase_m1"] == pytest.approx(1.0)
    assert pairwise.loc["ep_basic", "bot"] == pytest.approx(39.0)
    assert pairwise.loc["bot", "ep_basic"] == 0.0
    # Consensus is the median of bot, phase_m1 and ep_basic only; the
    # phase variants do not vote, so it lands on bot's line.
    assert consensus["bot"]["consensus_mad_m"] == pytest.approx(0.0)
    assert consensus["phase_m1"]["consensus_mad_m"] == pytest.approx(1.0)
    assert consensus["ep_basic"]["consensus_within_5m"] == 0.0
    assert consensus["phase_m2"]["consensus_mad_m"] == pytest.approx(0.8)


def test_agreement_handles_nan_and_single_line():
    a = np.array([1.0, np.nan, 3.0])
    pairwise, consensus = agreement({"x": a, "y": a + 2.0})
    assert pairwise.loc["y", "x"] == pytest.approx(2.0)
    assert consensus["x"]["consensus_mad_m"] == pytest.approx(1.0)
