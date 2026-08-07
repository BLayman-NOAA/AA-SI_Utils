# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""combine_raw_stores leaves every group's chunking writable by zarr v2.

combine_echodata aligns its inputs with ``join="outer"``, so any dimension whose
length differs between raw files comes back reindexed into ragged blocks. That
happens on ``range_sample`` for surveys whose files record different numbers of
samples, and zarr v2 refuses to write it ("Zarr requires uniform chunk sizes
except for final chunk").
"""

from __future__ import annotations

import dask.array as da
import pytest
import xarray as xr

import aa_si_utils.utils as utils


class _FakeEchoData:
    """Group container with the bits combine_raw_stores touches."""

    sonar_model = "EK60"

    def __init__(self, groups):
        self._groups = dict(groups)
        self.group_paths = list(self._groups)

    def __getitem__(self, key):
        return self._groups[key]

    def __setitem__(self, key, value):
        self._groups[key] = value


def _ragged_beam_group():
    """A beam group shaped like a 4-file outer-join combine."""
    # range_sample blocks are the union of per-file lengths 2783/2795/2799/2806.
    arr = da.zeros(
        (2, 2500, 2806),
        chunks=((1, 1), (1200, 1300), (2783, 12, 4, 7)),
        dtype="float32",
    )
    dims = ("channel", "ping_time", "range_sample")
    return xr.Dataset({"angle_athwartship": (dims, arr)})


@pytest.fixture
def combined(monkeypatch):
    groups = {
        "Sonar/Beam_group1": _ragged_beam_group(),
        "Vendor_specific": xr.Dataset(),
    }
    echodata = _FakeEchoData(groups)
    monkeypatch.setattr(utils.ep, "open_converted", lambda *_a, **_k: echodata)
    return utils.combine_raw_stores(["combined.zarr"], ping_time_chunk=1000)


def _assert_zarr_writable(chunks):
    """Every block but the last equal, and the last no larger."""
    first = chunks[0]
    assert all(c == first for c in chunks[:-1])
    assert chunks[-1] <= first


def test_every_dim_is_zarr_writable(combined):
    for group_path in combined.group_paths:
        ds = combined[group_path]
        for var in ds.variables.values():
            if var.chunks is None:
                continue
            for dim_chunks in var.chunks:
                _assert_zarr_writable(dim_chunks)


def test_ragged_non_concat_dim_collapses_to_one_chunk(combined):
    beam = combined["Sonar/Beam_group1"]["angle_athwartship"]
    assert dict(zip(beam.dims, beam.chunks))["range_sample"] == (2806,)


def test_ping_time_still_honors_the_requested_chunk(combined):
    beam = combined["Sonar/Beam_group1"]["angle_athwartship"]
    assert dict(zip(beam.dims, beam.chunks))["ping_time"] == (1000, 1000, 500)
