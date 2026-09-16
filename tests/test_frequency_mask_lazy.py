# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""The frequency mask must stay lazy on a lazy Sv.

The decision it encodes is one boolean per channel. Broadcasting that directly
against a dask-backed Sv returns a NUMPY array of the full three-dimensional
shape, so a handful of booleans becomes tens of megabytes - measured at 15.04
MiB on a 4-channel, 516-ping, 7677-sample file. That array then travels by value
inside every downstream dask graph and back to the client once per mapped
instance, which is what took the client process down partway through a 3322-file
survey.
"""

from __future__ import annotations

import pickle

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from aa_si_utils.utils import create_frequency_mask

FREQS_HZ = np.array([18000.0, 38000.0, 120000.0, 200000.0])


def _ds(lazy, n_ping=12, n_range=20):
    arr = np.arange(4 * n_ping * n_range, dtype="float64").reshape(4, n_ping, n_range)
    data = da.from_array(arr, chunks=arr.shape) if lazy else arr
    return xr.Dataset(
        {"Sv": (("channel", "ping_time", "range_sample"), data),
         "frequency_nominal": (("channel",), FREQS_HZ)},
        coords={"channel": [f"ch{i}" for i in range(4)],
                "ping_time": (np.datetime64("2016-06-27T00:00:00")
                              + np.arange(n_ping) * np.timedelta64(1, "s")),
                "range_sample": np.arange(n_range)},
    )


def test_a_lazy_sv_gives_a_lazy_mask():
    mask = create_frequency_mask(_ds(lazy=True), frequencies_to_mask=[120, 200])

    assert mask.chunks is not None


def test_the_graph_carries_the_decision_not_the_array():
    """4 booleans, not a materialized 3-D array."""
    big = _ds(lazy=True, n_ping=516, n_range=7677)

    mask = create_frequency_mask(big, frequencies_to_mask=[120, 200])
    graph_mib = len(pickle.dumps(mask.data.dask, protocol=5)) / 2**20

    assert graph_mib < 1.0, f"graph is {graph_mib:.2f} MiB; the array is embedded"


def test_an_eager_sv_still_gives_an_eager_mask():
    mask = create_frequency_mask(_ds(lazy=False), frequencies_to_mask=[120, 200])

    assert mask.chunks is None


@pytest.mark.parametrize("freqs,expected_kept", [
    (None, 4), ([], 4), ([120, 200], 2), ([18], 3), ([18, 38, 120, 200], 0),
])
@pytest.mark.parametrize("lazy", [False, True])
def test_which_channels_survive_is_unchanged(freqs, expected_kept, lazy):
    mask = np.asarray(create_frequency_mask(_ds(lazy), frequencies_to_mask=freqs))

    assert int(mask.any(axis=(1, 2)).sum()) == expected_kept
    # A kept channel is kept everywhere, a dropped one nowhere.
    assert set(np.unique(mask.all(axis=(1, 2)) == mask.any(axis=(1, 2)))) == {True}
