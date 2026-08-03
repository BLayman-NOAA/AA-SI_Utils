# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for concat_datasets, the recipe-system fan-in (collect) reconsolidator."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from aa_si_utils.utils import concat_datasets


def _seg(times: list[int]) -> xr.Dataset:
    return xr.Dataset(
        {"Sv": ("ping_time", np.array(times, dtype=float))},
        coords={"ping_time": np.array(times)},
    )


def test_concat_multiple_along_ping_time():
    merged = concat_datasets([_seg([0, 1]), _seg([2, 3]), _seg([4, 5])])
    assert list(merged["ping_time"].values) == [0, 1, 2, 3, 4, 5]
    assert merged["Sv"].shape == (6,)


def test_single_item_passthrough():
    only = _seg([7, 8])
    # A one-element list and a bare Dataset both return the single dataset.
    assert concat_datasets([only]).equals(only)
    assert concat_datasets(only).equals(only)


def test_skips_none_entries():
    merged = concat_datasets([_seg([0]), None, _seg([1])])
    assert list(merged["ping_time"].values) == [0, 1]


def test_empty_raises():
    with pytest.raises(ValueError):
        concat_datasets([])


def _sv_seg(times: list[int], sound_speed: float = 1500.0) -> xr.Dataset:
    """Segment shaped like a real Sv dataset: per-ping data plus metadata
    variables that do NOT have a ping_time dimension."""
    return xr.Dataset(
        {
            "Sv": (("ping_time", "channel"), np.zeros((len(times), 3))),
            "frequency_nominal": ("channel", np.array([18e3, 38e3, 120e3])),
            "sound_speed": ((), sound_speed),
        },
        coords={
            "ping_time": np.array(times),
            "channel": ["ch1", "ch2", "ch3"],
        },
    )


def test_non_concat_dim_variables_keep_their_shape():
    """Variables lacking the concat dim must not be broadcast along it.

    xarray's default data_vars="all" would turn frequency_nominal into
    (ping_time, channel) and sound_speed into (ping_time,), which breaks
    consumers that expect per-channel metadata to stay 1-D (e.g. the
    echogram plotter's int(f/1000) over frequency_nominal).
    """
    merged = concat_datasets([_sv_seg([0, 1]), _sv_seg([2, 3]), _sv_seg([4, 5])])

    assert merged["Sv"].dims == ("ping_time", "channel")
    assert merged["Sv"].shape == (6, 3)
    # Unchanged by the concat:
    assert merged["frequency_nominal"].dims == ("channel",)
    assert merged["frequency_nominal"].shape == (3,)
    assert merged["sound_speed"].dims == ()
    # Still convertible to a Python scalar, as downstream consumers assume.
    assert float(merged["sound_speed"]) == 1500.0


def test_differing_metadata_takes_first_segment():
    """Segments calibrated with different env params still merge, taking the
    first segment's value rather than raising or growing a ping_time dim."""
    merged = concat_datasets(
        [_sv_seg([0, 1], sound_speed=1500.0), _sv_seg([2, 3], sound_speed=1480.0)]
    )
    assert merged["sound_speed"].dims == ()
    assert float(merged["sound_speed"]) == 1500.0


def _seg_with_sources(
    times: list[int],
    sources: list[str],
    as_coord: bool = False,
    indexed_dim: bool = False,
):
    """Segment carrying source_filenames on a 'filenames' dimension.

    ``indexed_dim`` reproduces echopype's real layout, where 'filenames' is an
    indexed dimension coordinate that restarts at 0 in every segment.
    """
    ds = _sv_seg(times)
    da = xr.DataArray(np.array(sources, dtype=object), dims=("filenames",))
    ds = ds.assign_coords(source_filenames=da) if as_coord else ds.assign(
        source_filenames=da
    )
    if indexed_dim:
        ds = ds.assign_coords(filenames=np.arange(len(sources)))
    return ds


def test_source_filenames_with_restarting_index():
    """Segments whose 'filenames' index restarts at 0 must still merge.

    Concatenating the DataArrays directly would build a duplicate-valued
    index ([0, 0, 0]) and xarray would refuse to align it back onto the
    merged dataset.
    """
    merged = concat_datasets(
        [
            _seg_with_sources([0, 1], ["a.raw"], indexed_dim=True),
            _seg_with_sources([2, 3], ["b.raw"], indexed_dim=True),
            _seg_with_sources([4, 5], ["c.raw"], indexed_dim=True),
        ]
    )
    assert list(merged["source_filenames"].values) == ["a.raw", "b.raw", "c.raw"]
    # The index is rebuilt positionally over the combined length.
    assert list(merged.coords["filenames"].values) == [0, 1, 2]


@pytest.mark.parametrize("as_coord", [False, True], ids=["data_var", "coord"])
def test_source_filenames_lists_every_segment(as_coord):
    """Provenance must name all contributing files, not just the first.

    data_vars="minimal" alone would keep only segment 0's source_filenames,
    making a survey merged from many raw files claim a single source.
    """
    merged = concat_datasets(
        [
            _seg_with_sources([0, 1], ["a.raw"], as_coord=as_coord),
            _seg_with_sources([2, 3], ["b.raw"], as_coord=as_coord),
            _seg_with_sources([4, 5], ["c.raw"], as_coord=as_coord),
        ]
    )
    assert merged["source_filenames"].dims == ("filenames",)
    assert list(merged["source_filenames"].values) == ["a.raw", "b.raw", "c.raw"]
    # The concat dim is unaffected by the provenance re-merge.
    assert list(merged["ping_time"].values) == [0, 1, 2, 3, 4, 5]
    assert ("source_filenames" in merged.coords) is as_coord


def test_concat_dataarrays():
    """Per-segment DataArrays must still concatenate.

    A collect fan-in over a DataArray-producing step (e.g. detect_seafloor's
    per-file seafloor_depth) hits this path, and xr.concat rejects the
    data_vars/coords kwargs used for Datasets.
    """
    segs = [
        xr.DataArray(np.array([0.0, 1.0]), dims=("ping_time",)),
        xr.DataArray(np.array([2.0, 3.0]), dims=("ping_time",)),
    ]
    merged = concat_datasets(segs, dim="ping_time")
    assert isinstance(merged, xr.DataArray)
    assert list(merged.values) == [0.0, 1.0, 2.0, 3.0]


def test_segments_without_source_filenames_still_merge():
    merged = concat_datasets([_sv_seg([0, 1]), _sv_seg([2, 3])])
    assert "source_filenames" not in merged.variables
    assert list(merged["ping_time"].values) == [0, 1, 2, 3]
