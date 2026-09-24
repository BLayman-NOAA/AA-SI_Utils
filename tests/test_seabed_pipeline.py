# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""End-to-end behaviour of detect_seabed and the recipe op."""

import numpy as np
import pytest
import xarray as xr

from aa_si_utils import utils
from aa_si_utils.seabed import (
    FLAG_INTERPOLATED,
    FLAG_SKIPPED,
    detect_seabed,
    detect_seafloor_phase,
)
from seabed_synthetic import make_synthetic_sv


def test_diagnostics_dataset_carries_every_stage():
    ds, truth = make_synthetic_sv(n_ping=40, seabed_depth_m=100.0)

    diag = detect_seabed(ds, channel="38000", r_min=50.0, r_max=150.0)

    for name in ("alias", "seeds", "uncertain", "background", "score", "range_m",
                 "f1", "f2", "f3", "f4", "f5", "f6", "candidate_range_m", "candidate_score",
                 "r_star", "r_le", "r_int", "confidence", "margin", "delta", "flags"):
        assert name in diag, name
    assert diag["score"].dims == ("ping_time", "range_sample")
    assert diag.attrs["primary_channel"] == "38000"
    assert diag.attrs["regime"] == "shelf"
    assert diag.attrs["has_angles"] == 1
    assert diag.attrs["n_seed_components"] >= 1
    assert diag.attrs["features"] == "f1 f2 f3 f4 f5 f6"
    np.testing.assert_allclose(diag["r_le"].values, truth["edge_m"], atol=geom_tolerance(diag))
    assert np.all(diag["r_int"].values < diag["r_le"].values)
    assert np.all(diag["r_star"].values >= diag["r_le"].values)
    assert (diag["flags"].values & FLAG_SKIPPED).sum() == 0


def geom_tolerance(diag):
    return diag.attrs["pulse_length_m"]


def test_op_output_matches_the_shared_contract():
    ds, truth = make_synthetic_sv(n_ping=40, seabed_depth_m=100.0)

    line = detect_seafloor_phase(ds, r_min=50.0, r_max=150.0)

    assert isinstance(line, xr.DataArray)
    assert line.dims == ("ping_time",)
    assert line.name == "seafloor_depth"
    assert line.attrs["units"] == "m"
    assert line.attrs["vertical_reference"] == "surface"
    assert line.attrs["primary_channel"] == "38000"
    np.testing.assert_array_equal(line["ping_time"].values, ds["ping_time"].values)
    np.testing.assert_allclose(line.values, truth["edge_m"], atol=2.0)
    for value in line.attrs.values():
        assert isinstance(value, (str, int, float))


def test_line_is_a_drop_in_for_create_seafloor_mask():
    ds, truth = make_synthetic_sv(n_ping=30, seabed_depth_m=100.0)
    line = detect_seafloor_phase(ds, r_min=50.0, r_max=150.0)

    mask = utils.create_seafloor_mask(ds, line, seafloor_buffer_m=0.0)

    assert mask.dtype == bool
    assert mask.dims == ("channel", "ping_time", "range_sample")
    kept_depth = ds["depth"].where(mask).max(dim="range_sample").isel(channel=0).values
    assert np.all(kept_depth <= line.values)
    assert np.all(kept_depth > line.values - 1.0)


def test_transducer_reference_without_depth():
    ds, truth = make_synthetic_sv(n_ping=20, seabed_depth_m=100.0, depth=False)
    line = detect_seafloor_phase(ds, r_min=50.0, r_max=150.0)
    assert line.attrs["vertical_reference"] == "transducer"
    np.testing.assert_allclose(line.values, truth["edge_m"], atol=2.0)


def test_amplitude_only_runs_and_warns():
    ds, truth = make_synthetic_sv(n_ping=40, seabed_depth_m=100.0, angles=False)
    with pytest.warns(UserWarning, match="no split-beam angles"):
        line = detect_seafloor_phase(ds, r_min=50.0, r_max=150.0)
    assert line.attrs["has_angles"] == 0
    np.testing.assert_allclose(line.values, truth["edge_m"], atol=2.0)


def test_channel_auto_selection_prefers_lowest_frequency_with_seeds():
    ds, _ = make_synthetic_sv(n_ping=40, seabed_depth_m=100.0, frequencies_hz=(120000, 18000, 38000))
    line = detect_seafloor_phase(ds, r_min=50.0, r_max=150.0)
    assert line.attrs["primary_channel"] == "18000"
    line = detect_seafloor_phase(ds, channel=38, r_min=50.0, r_max=150.0)
    assert line.attrs["primary_channel"] == "38000"


def test_prior_line_from_another_detector():
    ds, truth = make_synthetic_sv(n_ping=40, seabed_depth_m=100.0)
    prior = xr.DataArray(truth["edge_m"] + 3.0, coords={"ping_time": ds["ping_time"]}, dims=["ping_time"])
    line = detect_seafloor_phase(ds, z_prior=prior, sigma_prior_m=5.0)
    np.testing.assert_allclose(line.values, truth["edge_m"], atol=2.0)


def test_gaps_are_interpolated_up_to_max_gap_and_ends_stay_nan(capsys):
    ds, truth = make_synthetic_sv(n_ping=40, seabed_depth_m=100.0)
    # Blank the seabed on a short interior run and on the tail.
    sv = ds["Sv"].values
    sv[:, 10:13, :] = -95.0
    sv[:, 34:, :] = -95.0
    ds["Sv"].values[:] = sv

    line = detect_seafloor_phase(ds, r_min=50.0, r_max=150.0, max_gap_s=5.0)

    assert np.isfinite(line.values[10:13]).all()
    assert line.attrs["n_interpolated"] == 3
    assert np.isnan(line.values[34:]).all()
    assert line.attrs["ping_coverage"] == pytest.approx(34 / 40)
    assert "WARNING" in capsys.readouterr().out

    strict = detect_seafloor_phase(ds, r_min=50.0, r_max=150.0, max_gap_s=1.0)
    assert np.isnan(strict.values[10:13]).all()


def test_diagnostics_written_to_zarr(tmp_path):
    ds, _ = make_synthetic_sv(n_ping=12, seabed_depth_m=100.0)
    path = tmp_path / "diag.zarr"
    detect_seafloor_phase(ds, r_min=50.0, r_max=150.0, diagnostics_path=str(path))
    diag = xr.open_zarr(path)
    assert "score" in diag and "r_int" in diag
    assert (diag["flags"].values & FLAG_INTERPOLATED).sum() == 0


def test_lazy_input_is_accepted():
    ds, truth = make_synthetic_sv(n_ping=20, seabed_depth_m=100.0)
    line = detect_seafloor_phase(ds.chunk({"ping_time": 7}), r_min=50.0, r_max=150.0)
    np.testing.assert_allclose(line.values, truth["edge_m"], atol=2.0)


def test_mode_2_adds_shape_features_and_keeps_the_line():
    ds, truth = make_synthetic_sv(n_ping=40, seabed_depth_m=100.0)

    diag = detect_seabed(ds, channel="38000", r_min=50.0, r_max=150.0, mode=2)

    assert diag.attrs["mode"] == 2
    assert diag.attrs["features"] == "f1 f2 f3 f4 f5 f6 f8 f9 f10"
    for name in ("f8", "f9", "f10"):
        assert name in diag
    np.testing.assert_allclose(diag["r_le"].values, truth["edge_m"], atol=geom_tolerance(diag))

    line = detect_seafloor_phase(ds, r_min=50.0, r_max=150.0, mode=2)
    assert line.attrs["mode"] == 2
    np.testing.assert_allclose(line.values, truth["edge_m"], atol=2.0)


def test_mode_2_rejects_a_school_on_the_bottom():
    """A dense aggregation touching the seabed is the case Mode 2 exists for."""
    school = {"ping_start": 5, "ping_end": 35, "top_m": 88.0, "bottom_m": 100.0, "sv_db": -38.0}
    ds, truth = make_synthetic_sv(n_ping=40, seabed_depth_m=100.0, school=school)

    line = detect_seafloor_phase(ds, r_min=50.0, r_max=150.0, mode=2)

    np.testing.assert_allclose(line.values, truth["edge_m"], atol=2.0)


def test_integration_line_option_returns_the_backstep():
    ds, truth = make_synthetic_sv(n_ping=30, seabed_depth_m=100.0)
    edge = detect_seafloor_phase(ds, r_min=50.0, r_max=150.0)
    integration = detect_seafloor_phase(ds, r_min=50.0, r_max=150.0, line="integration")

    assert edge.attrs["line"] == "leading_edge"
    assert integration.attrs["line"] == "integration"
    assert np.all(integration.values < edge.values)
    assert np.all(edge.values - integration.values < 5.0)
    with pytest.raises(ValueError, match="line must be"):
        detect_seafloor_phase(ds, r_min=50.0, r_max=150.0, line="raw")


def test_lazy_input_is_loaded_once_for_the_chosen_channel():
    from aa_si_utils.seabed.pipeline import _materialize

    ds, _ = make_synthetic_sv(n_ping=20, seabed_depth_m=100.0, frequencies_hz=(18000, 38000))
    lazy = ds.chunk({"ping_time": 5})
    lazy["unused"] = lazy["Sv"] * 2

    sub = _materialize(lazy, 38)

    assert sub.sizes["channel"] == 1
    assert str(sub["channel"].values[0]) == "38000"
    assert "unused" not in sub
    assert all(sub[v].chunks is None for v in sub.data_vars)
    assert _materialize(ds, None).sizes["channel"] == 2
