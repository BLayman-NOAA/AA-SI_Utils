# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Unit tests for ping_time window selection."""

import numpy as np
import pytest
import xarray as xr

from aa_si_utils import utils


def _make_ds_sv(n_pings=100, n_channels=2, n_range=6, start="2024-10-15T00:00:00"):
    """Sv dataset on a 1 s ping cadence."""
    ping_time = np.datetime64(start) + np.arange(n_pings, dtype="timedelta64[s]")
    shape = (n_channels, n_pings, n_range)
    return xr.Dataset(
        data_vars={
            "Sv": (("channel", "ping_time", "range_sample"), np.ones(shape)),
            "echo_range": (
                ("channel", "ping_time", "range_sample"),
                np.broadcast_to(np.arange(n_range, dtype=float) * 2.0, shape).copy(),
            ),
            "frequency_nominal": (("channel",), np.array([38000.0, 120000.0])),
        },
        coords={
            "channel": np.array([f"ch{i}" for i in range(n_channels)]),
            "ping_time": ping_time,
            "range_sample": np.arange(n_range),
        },
    )


def test_selects_the_requested_window():
    ds = _make_ds_sv(n_pings=100)
    out = utils.select_ping_time_range(
        ds, start="2024-10-15T00:00:10", end="2024-10-15T00:00:19"
    )
    assert out.sizes["ping_time"] == 10
    assert str(out["ping_time"].values[0]).startswith("2024-10-15T00:00:10")
    assert str(out["ping_time"].values[-1]).startswith("2024-10-15T00:00:19")


def test_bounds_are_inclusive():
    ds = _make_ds_sv(n_pings=10)
    out = utils.select_ping_time_range(
        ds, start="2024-10-15T00:00:00", end="2024-10-15T00:00:09"
    )
    assert out.sizes["ping_time"] == 10


def test_open_start_and_open_end():
    ds = _make_ds_sv(n_pings=100)
    assert utils.select_ping_time_range(
        ds, end="2024-10-15T00:00:09"
    ).sizes["ping_time"] == 10
    assert utils.select_ping_time_range(
        ds, start="2024-10-15T00:01:30"
    ).sizes["ping_time"] == 10


def test_no_bounds_is_a_no_op():
    ds = _make_ds_sv(n_pings=20)
    assert utils.select_ping_time_range(ds) is ds


def test_window_outside_the_data_raises():
    ds = _make_ds_sv(n_pings=20)
    with pytest.raises(ValueError, match="selects no pings"):
        utils.select_ping_time_range(
            ds, start="2024-10-20T00:00:00", end="2024-10-20T01:00:00"
        )


def test_reversed_bounds_raise():
    ds = _make_ds_sv(n_pings=20)
    with pytest.raises(ValueError, match="is after end"):
        utils.select_ping_time_range(
            ds, start="2024-10-15T00:00:10", end="2024-10-15T00:00:01"
        )


def test_other_variables_and_coords_survive():
    ds = _make_ds_sv(n_pings=100)
    out = utils.select_ping_time_range(
        ds, start="2024-10-15T00:00:10", end="2024-10-15T00:00:19"
    )
    np.testing.assert_array_equal(
        out["frequency_nominal"].values, ds["frequency_nominal"].values
    )
    assert out.sizes["channel"] == ds.sizes["channel"]
    assert out.sizes["range_sample"] == ds.sizes["range_sample"]


def test_dask_input_stays_lazy():
    ds = _make_ds_sv(n_pings=100).chunk({"ping_time": 10})
    out = utils.select_ping_time_range(
        ds, start="2024-10-15T00:00:10", end="2024-10-15T00:00:29"
    )
    assert out["Sv"].chunks is not None


def test_window_reads_only_the_chunks_it_touches():
    """The point of the step: on a chunked store, a narrow window must prune
    the dask graph rather than pulling the whole array."""
    ds = _make_ds_sv(n_pings=100).chunk({"ping_time": 10})
    before = ds["Sv"].data.npartitions
    out = utils.select_ping_time_range(
        ds, start="2024-10-15T00:00:10", end="2024-10-15T00:00:29"
    )
    assert out["Sv"].data.npartitions < before
    assert out["Sv"].data.npartitions == 2


def test_dask_and_numpy_agree():
    ds = _make_ds_sv(n_pings=100)
    kwargs = dict(start="2024-10-15T00:00:10", end="2024-10-15T00:00:29")
    eager = utils.select_ping_time_range(ds, **kwargs)
    lazy = utils.select_ping_time_range(ds.chunk({"ping_time": 10}), **kwargs)
    np.testing.assert_array_equal(
        eager["ping_time"].values, lazy["ping_time"].values
    )


def test_reports_the_window(capsys):
    ds = _make_ds_sv(n_pings=100)
    utils.select_ping_time_range(
        ds, start="2024-10-15T00:00:10", end="2024-10-15T00:00:19"
    )
    out = capsys.readouterr().out
    assert "100 -> 10 pings" in out
