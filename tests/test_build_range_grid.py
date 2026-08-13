# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Unit tests for the resampling target grid builder."""

import numpy as np
import pytest
import xarray as xr

from aa_si_utils import utils


def _make_ds_sv(n_range=100, spacing=0.2, n_channels=2, n_pings=4):
    """Sv dataset whose channels share one evenly spaced echo_range."""
    echo_range = np.broadcast_to(
        np.arange(n_range, dtype=float) * spacing, (n_channels, n_pings, n_range)
    ).copy()
    return xr.Dataset(
        data_vars={
            "Sv": (
                ("channel", "ping_time", "range_sample"),
                np.zeros((n_channels, n_pings, n_range)),
            ),
            "echo_range": (("channel", "ping_time", "range_sample"), echo_range),
        },
        coords={
            "channel": np.array([f"ch{i}" for i in range(n_channels)]),
            "ping_time": np.arange(n_pings),
            "range_sample": np.arange(n_range),
        },
    )


def test_grid_shape_matches_resample_to_geometry_contract():
    """A 1-D range_sample DataArray named echo_range is what the resampler takes."""
    grid = utils.build_range_grid(_make_ds_sv(), spacing_m=1.0)

    assert isinstance(grid, xr.DataArray)
    assert grid.dims == ("range_sample",)
    assert grid.name == "echo_range"


def test_spacing_sets_the_axis_length():
    """Coarsening the spacing is the lever on range axis length."""
    ds_Sv = _make_ds_sv(n_range=100, spacing=0.2)  # covers 0 to 19.8 m

    fine = utils.build_range_grid(ds_Sv, spacing_m=1.0)
    coarse = utils.build_range_grid(ds_Sv, spacing_m=5.0)

    assert np.isclose(float(fine[1] - fine[0]), 1.0)
    assert np.isclose(float(coarse[1] - coarse[0]), 5.0)
    assert coarse.size < fine.size < ds_Sv.sizes["range_sample"]


def test_grid_spans_the_data_by_default():
    ds_Sv = _make_ds_sv(n_range=100, spacing=0.2)
    deepest = float(ds_Sv["echo_range"].max())

    grid = utils.build_range_grid(ds_Sv, spacing_m=1.0)

    assert float(grid[0]) == 0.0
    assert float(grid[-1]) >= deepest


def test_max_range_caps_the_grid():
    """Capping below the recorded extent is what drops unwanted deep samples."""
    grid = utils.build_range_grid(_make_ds_sv(), spacing_m=1.0, max_range_m=5.0)

    assert float(grid[-1]) >= 5.0
    assert float(grid[-1]) < 5.0 + 2 * 1.0


def test_ignores_nan_when_measuring_extent():
    ds_Sv = _make_ds_sv(n_range=20, spacing=1.0)
    ds_Sv["echo_range"][:, :, -5:] = np.nan

    grid = utils.build_range_grid(ds_Sv, spacing_m=1.0)

    # Deepest finite sample is index 14, at 14.0 m.
    assert float(grid[-1]) >= 14.0
    assert float(grid[-1]) < 19.0


def test_rejects_non_positive_spacing():
    with pytest.raises(ValueError, match="spacing_m must be positive"):
        utils.build_range_grid(_make_ds_sv(), spacing_m=0.0)


def test_rejects_missing_range_var():
    with pytest.raises(KeyError, match="not found in ds_Sv"):
        utils.build_range_grid(_make_ds_sv(), spacing_m=1.0, range_var="depth")


def test_rejects_all_nan_range():
    ds_Sv = _make_ds_sv()
    ds_Sv["echo_range"][:] = np.nan

    with pytest.raises(ValueError, match="no finite values"):
        utils.build_range_grid(ds_Sv, spacing_m=1.0)
