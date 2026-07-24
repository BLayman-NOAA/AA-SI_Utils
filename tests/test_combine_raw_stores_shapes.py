# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""combine_raw_stores accepts both the whole-survey and per-file-parallel shapes.

When ``read_raw_files`` is parallelized with ``map_over``, each mapped instance
reads one raw file and returns a *one-element* list of stores, so the recipe's
``collect`` fan-in hands this function ``[[s0], [s1], ...]`` instead of a flat
``[s0, s1, ...]``. Both must open the same stores, in the same order.
"""

from __future__ import annotations

import pytest

import aa_si_utils.utils as utils


class _FakeEchoData:
    """Minimal stand-in: enough for the opened-store bookkeeping under test."""

    group_paths: list[str] = []


@pytest.fixture
def opened(monkeypatch):
    """Record the store paths combine_raw_stores actually opens."""
    seen: list[str] = []

    def _open_converted(path, **_kwargs):
        seen.append(str(path))
        return _FakeEchoData()

    monkeypatch.setattr(utils.ep, "open_converted", _open_converted)
    monkeypatch.setattr(utils.ep, "combine_echodata", lambda parts: _FakeEchoData())
    return seen


def _combine(raw_stores):
    # Everything past the open/combine step needs a real EchoData; this test is
    # only about which stores get opened, so stop there.
    try:
        utils.combine_raw_stores(raw_stores)
    except Exception:
        pass


def test_flat_list_opens_each_store(opened):
    _combine(["a.zarr", "b.zarr", "c.zarr"])
    assert opened == ["a.zarr", "b.zarr", "c.zarr"]


def test_nested_list_from_collect_is_flattened(opened):
    _combine([["a.zarr"], ["b.zarr"], ["c.zarr"]])
    assert opened == ["a.zarr", "b.zarr", "c.zarr"]


def test_flatten_preserves_order(opened):
    # combine_echodata requires time-ordered inputs, so the fan-in order that
    # the executor folds (instance order) must survive flattening.
    _combine([["t0.zarr"], ["t1.zarr"], ["t2.zarr"], ["t3.zarr"]])
    assert opened == ["t0.zarr", "t1.zarr", "t2.zarr", "t3.zarr"]


def test_mixed_shapes_are_tolerated(opened):
    _combine([["a.zarr"], "b.zarr"])
    assert opened == ["a.zarr", "b.zarr"]


def test_multi_store_instances_are_flattened(opened):
    # A mapped instance that read more than one file still flattens correctly.
    _combine([["a.zarr", "b.zarr"], ["c.zarr"]])
    assert opened == ["a.zarr", "b.zarr", "c.zarr"]


def test_empty_input_still_rejected():
    with pytest.raises(ValueError, match="No raw stores provided"):
        utils.combine_raw_stores([])


def test_all_empty_nested_input_rejected():
    with pytest.raises(ValueError, match="No raw stores provided"):
        utils.combine_raw_stores([[], []])
