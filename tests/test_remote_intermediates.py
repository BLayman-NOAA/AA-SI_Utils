# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for gs://-shaped (remote) exe_temp intermediate stores.

``memory://`` stands in for ``gs://`` — a non-local fsspec store needing no
credentials. echopype is faked so no real raw files or conversions are needed.
"""

from __future__ import annotations

from pathlib import Path

import fsspec
import pytest
import xarray as xr

from aa_si_utils import _storage, utils


@pytest.fixture(autouse=True)
def clear_memory_fs():
    mem = fsspec.filesystem("memory")
    mem.store.clear()
    mem.pseudo_dirs[:] = [""]
    yield
    mem.store.clear()
    mem.pseudo_dirs[:] = [""]


@pytest.fixture
def memory_temp_ctx():
    """Publish an ExecutionContext whose temp_dir is a memory:// location."""
    from aa_recipe_manager.executor.runtime_context import execution_context

    with execution_context(mode="direct", temp_dir="memory://scratch/exe_temp"):
        yield


class _StubEchoData:
    """Minimal EchoData: to_zarr writes a tiny real dataset to the given store."""

    def __init__(self, payload: int) -> None:
        self.payload = payload

    def to_zarr(self, save_path, zarr_format=2, compress=True,
                output_storage_options=None, **kwargs):
        ds = xr.Dataset({"value": ("x", [self.payload])})
        # save_path is a URL string for remote; storage_options is threaded.
        ds.to_zarr(
            str(save_path),
            mode="w",
            zarr_format=zarr_format,
            storage_options=(output_storage_options or None),
        )


# ---------------------------------------------------------------------------
# _storage helpers on memory://
# ---------------------------------------------------------------------------


def test_makedirs_and_remove_store_remote_roundtrip():
    url = "memory://scratch/exe_temp/data/file.zarr"
    _storage.makedirs("memory://scratch/exe_temp/data")
    xr.Dataset({"v": ("x", [1])}).to_zarr(url, mode="w", zarr_format=2)
    assert _storage.get_fs(url).exists(url)
    _storage.remove_store(url)
    assert not _storage.get_fs(url).exists(url)


def test_remove_store_missing_remote_is_noop():
    _storage.remove_store("memory://scratch/nope.zarr")  # must not raise


# ---------------------------------------------------------------------------
# read_raw_files_to_stores: remote zarr + netcdf guard
# ---------------------------------------------------------------------------


def test_netcdf_intermediate_on_remote_temp_raises(memory_temp_ctx):
    with pytest.raises(ValueError, match="netcdf intermediates require a local"):
        utils.read_raw_files_to_stores(
            ["dummy.raw"], sonar_model="EK80", intermediate_format="netcdf"
        )


def test_remote_zarr_intermediate_writes_to_bucket(memory_temp_ctx, monkeypatch):
    opened = []

    def fake_open_raw(raw_path, **kwargs):
        opened.append(Path(raw_path).stem)
        return _StubEchoData(payload=len(opened))

    monkeypatch.setattr(utils.ep, "open_raw", fake_open_raw)

    result = utils.read_raw_files_to_stores(
        ["D20160101-T000000.raw", "D20160101-T001000.raw"],
        sonar_model="EK80",
        intermediate_format="zarr",
    )

    assert result == [
        "memory://scratch/exe_temp/data/D20160101-T000000.zarr",
        "memory://scratch/exe_temp/data/D20160101-T001000.zarr",
    ]
    # Both stores actually landed in the memory (bucket) filesystem.
    fs = fsspec.filesystem("memory")
    for url in result:
        assert fs.exists(url)
        reloaded = xr.open_dataset(url, engine="zarr")
        assert "value" in reloaded


def test_combine_reads_remote_stores_with_storage_options(memory_temp_ctx, monkeypatch):
    """Remote store paths are opened via open_converted with storage_options."""
    seen = {}

    class _Sentinel(Exception):
        pass

    def fake_open_converted(path, chunks=None, storage_options="MISSING"):
        seen["path"] = path
        seen["storage_options"] = storage_options
        raise _Sentinel

    monkeypatch.setattr(utils.ep, "open_converted", fake_open_converted)

    with pytest.raises(_Sentinel):
        utils.combine_raw_stores(["memory://scratch/exe_temp/data/a.zarr"])

    assert seen["path"] == "memory://scratch/exe_temp/data/a.zarr"
    # storage_options was passed explicitly (None under memory://), not omitted.
    assert seen["storage_options"] is None
