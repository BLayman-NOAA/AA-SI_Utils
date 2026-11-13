# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for compute_mvbs and the ping count it records.

MVBS computed per file and merged must equal MVBS computed on the whole
record, including in a time bin that a file boundary splits unevenly.
"""

import echopype as ep
import numpy as np
import pytest
import xarray as xr

from aa_si_utils import utils

T0 = np.datetime64("2024-01-01T00:00:00")


def _make_ds_sv(n_pings=30, n_range=8, seed=0):
    """Two-channel Sv on a 1 s ping cadence with 1 m samples."""
    rng = np.random.default_rng(seed)
    shape = (2, n_pings, n_range)
    echo_range = np.broadcast_to(np.arange(n_range, dtype=float), shape).copy()
    return xr.Dataset(
        data_vars={
            "Sv": (("channel", "ping_time", "range_sample"),
                   rng.uniform(-90.0, -50.0, size=shape)),
            "echo_range": (("channel", "ping_time", "range_sample"), echo_range),
            "frequency_nominal": ("channel", np.array([18e3, 38e3])),
        },
        coords={
            "channel": np.array(["ch0", "ch1"]),
            "ping_time": T0 + np.arange(n_pings) * np.timedelta64(1, "s"),
            "range_sample": np.arange(n_range),
        },
    )


def _mvbs(ds_Sv, **kwargs):
    return utils.compute_mvbs(
        ds_Sv, range_bin="4m", ping_time_bin="10s", range_var_max="8m", **kwargs
    )


def test_sv_matches_echopype():
    ds_Sv = _make_ds_sv()

    ours = _mvbs(ds_Sv)
    theirs = ep.commongrid.compute_MVBS(
        ds_Sv, range_bin="4m", ping_time_bin="10s", range_var_max="8m"
    )

    xr.testing.assert_identical(ours["Sv"], theirs["Sv"])
    assert set(ours.data_vars) - set(theirs.data_vars) == {"ping_count"}


def test_ping_count_holds_the_pings_in_each_bin():
    """27 pings at 1 s from the top of a minute fill 10 s bins as 10, 10, 7."""
    ds_MVBS = _mvbs(_make_ds_sv(n_pings=27))

    assert ds_MVBS["ping_count"].dims == ("ping_time",)
    assert list(ds_MVBS["ping_count"].values) == [10, 10, 7]
    assert ds_MVBS["ping_count"].dtype.kind == "i"


def test_a_bin_with_no_pings_counts_zero():
    """Two bursts with a silent bin between them, as two dive windows give."""
    ds_Sv = _make_ds_sv(n_pings=30)
    ds_Sv = ds_Sv.isel(ping_time=np.r_[0:10, 20:30])

    assert list(_mvbs(ds_Sv)["ping_count"].values) == [10, 0, 10]


def test_closed_right_counts_the_pings_echopype_bins():
    """With right-closed bins the ping on the first edge falls in no bin."""
    ds_MVBS = _mvbs(_make_ds_sv(n_pings=30), closed="right")

    assert list(ds_MVBS["ping_count"].values) == [10, 10, 9]


@pytest.mark.parametrize("split", [12, 15, 19])
def test_per_file_mvbs_merges_to_the_whole_record(split):
    """A boundary inside a bin must not change the merged MVBS."""
    ds_Sv = _make_ds_sv()
    whole = _mvbs(ds_Sv)

    parts = [_mvbs(ds_Sv.isel(ping_time=slice(None, split))),
             _mvbs(ds_Sv.isel(ping_time=slice(split, None)))]
    merged = utils.concat_datasets(parts, dim="ping_time", on_duplicate="mean")

    xr.testing.assert_allclose(merged["Sv"], whole["Sv"])
    np.testing.assert_array_equal(merged["ping_count"], whole["ping_count"])


def test_without_the_count_an_uneven_split_is_biased():
    """The case the count exists for: 2 pings on one side, 8 on the other."""
    ds_Sv = _make_ds_sv()
    whole = _mvbs(ds_Sv)

    parts = [_mvbs(ds_Sv.isel(ping_time=slice(None, 12))).drop_vars("ping_count"),
             _mvbs(ds_Sv.isel(ping_time=slice(12, None))).drop_vars("ping_count")]
    merged = utils.concat_datasets(parts, dim="ping_time", on_duplicate="mean")

    split_bin = T0 + np.timedelta64(10, "s")
    assert not np.allclose(merged["Sv"].sel(ping_time=split_bin),
                           whole["Sv"].sel(ping_time=split_bin))
