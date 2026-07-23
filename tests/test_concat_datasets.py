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
