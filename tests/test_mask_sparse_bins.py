# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Unit tests for sparse-bin masking."""

import numpy as np
import pytest
import xarray as xr

from aa_si_utils import utils


def _make_ds_sv(n_pings=12, n_channels=2, n_range=8, seed=0):
    """Sv dataset on a 1 s ping cadence with a reproducible NaN pattern."""
    rng = np.random.default_rng(seed)
    channels = np.array([f"ch{i}" for i in range(n_channels)])
    ping_time = np.datetime64("2024-01-01T00:00:00") + np.arange(
        n_pings, dtype="timedelta64[s]"
    )
    range_sample = np.arange(n_range)
    echo_range_values = np.arange(n_range, dtype=float) * 2.0
    echo_range = np.broadcast_to(
        echo_range_values, (n_channels, n_pings, n_range)
    ).copy()

    sv = rng.normal(size=(n_channels, n_pings, n_range))
    # Deeper samples are mostly missing, so the lower bins cross the threshold
    # while the shallow ones stay.
    sv[:, :, n_range // 2:] = np.where(
        rng.random((n_channels, n_pings, n_range - n_range // 2)) < 0.9,
        np.nan,
        sv[:, :, n_range // 2:],
    )

    return xr.Dataset(
        data_vars={
            "Sv": (("channel", "ping_time", "range_sample"), sv),
            "echo_range": (("channel", "ping_time", "range_sample"), echo_range),
            "depth": (("channel", "ping_time", "range_sample"), echo_range + 5.0),
        },
        coords={
            "channel": channels,
            "ping_time": ping_time,
            "range_sample": range_sample,
        },
    )


def test_masks_only_bins_at_or_above_threshold():
    """One channel, two pings, four range samples: a 2s x 4m grid gives two
    bins of four samples each. The shallow bin is 50% NaN and survives; the
    deep bin is 75% NaN and has its one finite sample masked away."""
    ping_time = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T00:00:01"], dtype="datetime64[ns]"
    )
    echo_range = np.broadcast_to(
        np.array([0.0, 2.0, 4.0, 6.0]), (1, 2, 4)
    ).copy()
    sv = np.array([[[1.0, np.nan, np.nan, np.nan],
                    [1.0, np.nan, np.nan, 1.0]]])
    ds = xr.Dataset(
        data_vars={
            "Sv": (("channel", "ping_time", "range_sample"), sv),
            "echo_range": (("channel", "ping_time", "range_sample"), echo_range),
        },
        coords={
            "channel": np.array(["ch0"]),
            "ping_time": ping_time,
            "range_sample": np.arange(4),
        },
    )

    out = utils.mask_sparse_bins(
        ds, range_bin="4m", ping_time_bin="2s", nan_threshold=0.75
    )

    # Shallow bin (range 0-2 m): 2 of 4 NaN, below threshold, unchanged.
    # Deep bin (range 4-6 m): 3 of 4 NaN, at threshold, fully masked.
    expected = np.array([[[1.0, np.nan, np.nan, np.nan],
                          [1.0, np.nan, np.nan, np.nan]]])
    np.testing.assert_array_equal(
        np.isnan(out["Sv"].values), np.isnan(expected)
    )


def test_threshold_above_one_masks_nothing():
    ds = _make_ds_sv()
    out = utils.mask_sparse_bins(
        ds, range_bin="4m", ping_time_bin="2s", nan_threshold=1.1
    )
    np.testing.assert_array_equal(
        np.isnan(out["Sv"].values), np.isnan(ds["Sv"].values)
    )


@pytest.mark.parametrize("block_pings", [1, 5, 7, 12, 1000])
def test_block_size_does_not_change_result(block_pings):
    """Bins spanning a block boundary are counted across blocks, so the
    result must not depend on how the pings are blocked."""
    ds = _make_ds_sv()
    kwargs = dict(range_bin="4m", ping_time_bin="5s", nan_threshold=0.5)
    reference = utils.mask_sparse_bins(ds, block_pings=12, **kwargs)
    out = utils.mask_sparse_bins(ds, block_pings=block_pings, **kwargs)
    np.testing.assert_array_equal(
        np.isnan(out["Sv"].values), np.isnan(reference["Sv"].values)
    )


def test_input_is_not_mutated():
    ds = _make_ds_sv()
    before = ds["Sv"].values.copy()
    utils.mask_sparse_bins(
        ds, range_bin="4m", ping_time_bin="2s", nan_threshold=0.5
    )
    np.testing.assert_array_equal(ds["Sv"].values, before, strict=True)


def test_untouched_variables_are_shared_not_copied():
    ds = _make_ds_sv()
    out = utils.mask_sparse_bins(
        ds, range_bin="4m", ping_time_bin="2s", nan_threshold=0.5
    )
    assert out["depth"].values is ds["depth"].values
    assert out["echo_range"].values is ds["echo_range"].values


def test_dask_input_stays_lazy():
    ds = _make_ds_sv().chunk({"ping_time": 4})
    out = utils.mask_sparse_bins(
        ds, range_bin="4m", ping_time_bin="2s", nan_threshold=0.5
    )
    assert out["Sv"].chunks is not None


def test_dask_and_numpy_agree():
    ds = _make_ds_sv()
    kwargs = dict(range_bin="4m", ping_time_bin="2s", nan_threshold=0.5)
    eager = utils.mask_sparse_bins(ds, **kwargs)
    lazy = utils.mask_sparse_bins(ds.chunk({"ping_time": 5}), **kwargs)
    np.testing.assert_array_equal(
        np.isnan(lazy["Sv"].values), np.isnan(eager["Sv"].values)
    )


def test_single_channel():
    ds = _make_ds_sv(n_channels=1)
    out = utils.mask_sparse_bins(
        ds, range_bin="4m", ping_time_bin="2s", nan_threshold=0.5
    )
    assert out["Sv"].shape == ds["Sv"].shape


def test_all_nan_channel_is_fully_masked():
    ds = _make_ds_sv(n_channels=2)
    ds["Sv"].values[1, :, :] = np.nan
    out = utils.mask_sparse_bins(
        ds, range_bin="4m", ping_time_bin="2s", nan_threshold=0.9
    )
    assert np.isnan(out["Sv"].values[1]).all()


def test_time_bins_match_resample_ordinals():
    """The floor/factorize binning must agree with pandas resample, which is
    what compute_MVBS and compute_per_cell_statistics bin with."""
    import pandas as pd

    ds = _make_ds_sv(n_pings=30)
    ping_time_bin = "4s"
    expected = np.zeros(ds.sizes["ping_time"], dtype=int)
    for bin_idx, (_, group) in enumerate(ds.resample(ping_time=ping_time_bin)):
        mask = np.isin(
            ds["ping_time"].values, group.indexes["ping_time"].values
        )
        expected[mask] = bin_idx

    actual = pd.DatetimeIndex(
        ds["ping_time"].values
    ).floor(ping_time_bin).factorize()[0]
    np.testing.assert_array_equal(actual, expected)
