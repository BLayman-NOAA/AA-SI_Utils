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


# ---------------------------------------------------------------------------
# allow_empty: mapping the window over a survey's per-file Sv
# ---------------------------------------------------------------------------
#
# Applied to one merged store, a window that selects nothing is a mistake and
# should fail. Mapped over a survey's per-file Sv it is the normal case: most
# files lie outside any one window, and the instances that miss have to drop out
# rather than take the run down with them.


def test_allow_empty_returns_none_instead_of_raising():
    ds = _make_ds_sv(n_pings=20, start="2016-06-27T00:00:00")

    out = utils.select_ping_time_range(
        ds, start="2016-07-07T00:00:00", end="2016-07-07T01:00:00",
        allow_empty=True,
    )

    assert out is None


def test_allow_empty_still_returns_the_slice_when_the_window_hits():
    ds = _make_ds_sv(n_pings=60, start="2016-06-27T00:00:00")

    out = utils.select_ping_time_range(
        ds, start="2016-06-27T00:00:10", end="2016-06-27T00:00:19",
        allow_empty=True,
    )

    assert out is not None
    assert out.sizes["ping_time"] == 10


def test_allow_empty_defaults_off():
    """A merged-store recipe must keep failing on a window that selects nothing."""
    ds = _make_ds_sv(n_pings=20, start="2016-06-27T00:00:00")

    with pytest.raises(ValueError, match="selects no pings"):
        utils.select_ping_time_range(
            ds, start="2016-07-07T00:00:00", end="2016-07-07T01:00:00"
        )


def test_a_mapped_window_collects_to_only_the_overlapping_files():
    """The whole point: fan out over the survey, fan in on what the window hits."""
    per_file = [
        _make_ds_sv(n_pings=20, start=f"2016-06-27T0{hour}:00:00")
        for hour in range(6)
    ]
    window = {"start": "2016-06-27T02:00:00", "end": "2016-06-27T03:00:19"}

    selected = [
        utils.select_ping_time_range(ds, window=window, allow_empty=True)
        for ds in per_file
    ]

    assert sum(s is None for s in selected) == 4, "non-overlapping files drop out"
    merged = utils.concat_datasets(selected, dim="ping_time")
    assert merged.sizes["ping_time"] == 40


# ---------------------------------------------------------------------------
# dim: windowing a binned product
# ---------------------------------------------------------------------------
#
# compute_per_cell_statistics stores its cells on cell_ping_time, holding each
# bin's left edge as a datetime. Windowing that to a dive needs the same slice
# against a differently named coordinate.


def _ds_cells(n_cells=12, start="2016-06-27T00:00:00", step_s=10):
    cell_ping_time = np.datetime64(start) + np.arange(
        0, n_cells * step_s, step_s, dtype="timedelta64[s]"
    )
    return xr.Dataset(
        {"cell_cv": (("channel", "cell_ping_time", "cell_echo_range"),
                     np.ones((1, n_cells, 4)))},
        coords={
            "channel": ["ch0"],
            "cell_ping_time": cell_ping_time,
            "cell_echo_range": np.arange(4) * 2.0,
        },
    )


def test_dim_windows_a_cell_grid():
    ds = _ds_cells(n_cells=12)

    out = utils.select_ping_time_range(
        ds, start="2016-06-27T00:00:20", end="2016-06-27T00:00:49",
        dim="cell_ping_time",
    )

    assert out.sizes["cell_ping_time"] == 3


def test_dim_combines_with_allow_empty():
    ds = _ds_cells(n_cells=12)

    out = utils.select_ping_time_range(
        ds, start="2016-07-07T00:00:00", end="2016-07-07T01:00:00",
        dim="cell_ping_time", allow_empty=True,
    )

    assert out is None


def test_a_missing_dim_is_an_error_not_a_silent_pass_through():
    """Naming the wrong coordinate must not quietly return everything."""
    ds = _ds_cells(n_cells=4)

    with pytest.raises(ValueError, match="no 'ping_time' dimension"):
        utils.select_ping_time_range(
            ds, start="2016-06-27T00:00:00", end="2016-06-27T00:00:20"
        )


# ---------------------------------------------------------------------------
# windows: the union of several windows in one call
# ---------------------------------------------------------------------------


def _w(start_s, end_s, label=None):
    w = {"start": f"2024-10-15T00:00:{start_s:02d}", "end": f"2024-10-15T00:00:{end_s:02d}"}
    if label:
        w["label"] = label
    return w


def test_union_of_windows_keeps_pings_from_each_in_time_order():
    ds = _make_ds_sv(n_pings=100)
    out = utils.select_ping_time_range(ds, windows=[_w(40, 44), _w(10, 12)])
    seconds = out["ping_time"].values.astype("datetime64[s]").astype(int) % 60
    assert list(seconds) == [10, 11, 12, 40, 41, 42, 43, 44]


def test_union_of_overlapping_windows_keeps_each_ping_once():
    ds = _make_ds_sv(n_pings=100)
    out = utils.select_ping_time_range(ds, windows=[_w(10, 20), _w(15, 25)])
    assert out.sizes["ping_time"] == 16
    assert not out.indexes["ping_time"].has_duplicates
    assert out.indexes["ping_time"].is_monotonic_increasing


def test_union_skips_windows_that_miss_and_returns_none_when_all_miss():
    ds = _make_ds_sv(n_pings=30)
    out = utils.select_ping_time_range(ds, windows=[_w(50, 55), _w(5, 6)])
    assert out.sizes["ping_time"] == 2
    assert utils.select_ping_time_range(ds, windows=[_w(50, 55)], allow_empty=True) is None
    with pytest.raises(ValueError, match="none of the 1 windows"):
        utils.select_ping_time_range(ds, windows=[_w(50, 55)])


def test_union_of_windows_is_lazy_on_dask_input():
    dask = pytest.importorskip("dask")
    ds = _make_ds_sv(n_pings=100).chunk({"ping_time": 25})
    out = utils.select_ping_time_range(ds, windows=[_w(10, 12), _w(40, 42)])
    assert dask.is_dask_collection(out["Sv"].data)
    assert out.sizes["ping_time"] == 6


def test_windows_is_exclusive_with_the_single_window_forms():
    ds = _make_ds_sv(n_pings=30)
    with pytest.raises(ValueError, match="windows alone"):
        utils.select_ping_time_range(ds, windows=[_w(1, 2)], window=_w(3, 4))
    with pytest.raises(ValueError, match="windows alone"):
        utils.select_ping_time_range(ds, windows=[_w(1, 2)], start="2024-10-15T00:00:03")


def test_empty_windows_list_means_not_given():
    """The op spec defaults windows to [], which must not shadow start/end."""
    ds = _make_ds_sv(n_pings=30)
    out = utils.select_ping_time_range(
        ds, windows=[], start="2024-10-15T00:00:03", end="2024-10-15T00:00:05"
    )
    assert out.sizes["ping_time"] == 3
