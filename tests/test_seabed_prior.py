# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Mode 0 slab prior and the self-contained op."""

import numpy as np
import pytest

from aa_si_utils.seabed import detect_seafloor_phase, estimate_prior
from aa_si_utils.seabed.prior import slab_score
from seabed_synthetic import make_synthetic_sv


def test_slab_score_peaks_at_a_loud_edge():
    sv = np.full(100, -90.0)
    sv[60:66] = -30.0
    f2 = np.full(100, 5.0)
    f2[60:66] = 1.0
    score = slab_score(sv, f2, 7.0)
    assert int(np.nanargmax(score)) in (59, 60)
    assert np.nanmax(score) > 1.0
    assert np.nanmedian(score[:50]) < 0.0
    assert np.isnan(slab_score(np.full(100, np.nan), None, 7.0)).all()


def test_prior_finds_the_seabed_over_the_full_range():
    ds, truth = make_synthetic_sv(n_ping=120, n_sample=600, seabed_depth_m=100.0, multiple=True)

    prior = estimate_prior(ds, 38)

    assert prior is not None
    assert prior.attrs["mode"] == "0b"
    assert prior["z_prior"].dims == ("ping_time",)
    assert np.all(np.abs(prior["z_prior"].values - truth["edge_m"]) < 5.0)
    assert prior["sigma_prior"].min() >= 2 * 2.0
    assert prior.attrs["n_inconsistent_gaps"] == 0
    assert prior.attrs["n_slabs"] <= 64


def test_prior_bisects_a_step():
    depth = np.concatenate([np.full(60, 90.0), np.full(60, 140.0)])
    ds, truth = make_synthetic_sv(n_ping=120, n_sample=600, seabed_depth_m=depth)

    prior = estimate_prior(ds, 38, budget=24)
    even = estimate_prior(ds, 38, budget=24, adaptive=False, spacing_s=10.0)

    assert prior.attrs["n_slabs"] > 12
    slabs = prior["slab_ping"].values
    assert np.any((slabs > 50) & (slabs < 70))
    # Away from the step the prior is within a few metres of the edge.
    away = np.r_[0:50, 70:120]
    assert np.all(np.abs(prior["z_prior"].values[away] - truth["edge_m"][away]) < 6.0)
    assert even.attrs["mode"] == "0a"


def test_prior_returns_none_without_a_seabed():
    ds, _ = make_synthetic_sv(n_ping=40, n_sample=200, seabed_depth_m=500.0)
    assert estimate_prior(ds, 38) is None


def test_op_is_self_contained_and_avoids_the_multiple():
    ds, truth = make_synthetic_sv(n_ping=80, n_sample=600, seabed_depth_m=100.0, multiple=True)

    line = detect_seafloor_phase(ds)

    assert line.attrs["prior"] == "mode0"
    np.testing.assert_allclose(line.values, truth["edge_m"], atol=2.0)

    fixed = detect_seafloor_phase(ds, prior="none", r_min=50.0, r_max=150.0)
    assert fixed.attrs["prior"] == "none"
    np.testing.assert_allclose(fixed.values, line.values, atol=1.0)
    with pytest.raises(ValueError, match="prior must be"):
        detect_seafloor_phase(ds, prior="maybe")


def test_op_warns_and_falls_back_when_the_prior_fails():
    ds, _ = make_synthetic_sv(n_ping=40, n_sample=200, seabed_depth_m=500.0)
    with pytest.warns(UserWarning, match="Mode 0 prior found no seabed"):
        line = detect_seafloor_phase(ds, max_gap_s=None)
    assert line.attrs["prior"] == "none"


def test_missing_max_range_keeps_pings_without_a_seabed():
    ds, _ = make_synthetic_sv(n_ping=40, n_sample=200, seabed_depth_m=500.0)
    with pytest.warns(UserWarning):
        line = detect_seafloor_phase(ds, missing="max_range", max_gap_s=None)
    assert np.isfinite(line.values).all()
    assert line.attrs["missing"] == "max_range"
    assert line.attrs["ping_coverage"] < 1.0
    last = float(ds["depth"].isel(channel=0, ping_time=0).max())
    np.testing.assert_allclose(line.values, last)
    with pytest.raises(ValueError, match="missing must be"):
        detect_seafloor_phase(ds, missing="zero")
