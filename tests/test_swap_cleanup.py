# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Releasing echopype's swap zarr stores once an intermediate is written.

echopype spills a raw file's expanded backscatter to a swap zarr under the
system temp dir when ``use_swap`` fires, and frees it from ``EchoData.__del__``
only while ``converted_raw_path`` is None.  Its ``to_file`` sets that attribute
as soon as the object is written, so every file written to an intermediate
store keeps its swap directory for good.  The leak is per file, not per
concurrent instance, so a survey fills the temp disk and dies on ENOSPC a long
way from the cause.  ``_open_and_store`` therefore releases the swap explicitly,
at the point where the intermediate exists and nothing reads the EchoData again.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from aa_si_utils import utils


class _StubEchoData:
    """EchoData stand-in owning a real directory as its swap store."""

    def __init__(self, swap_dir: Path) -> None:
        self.swap_dir = swap_dir
        swap_dir.mkdir(parents=True, exist_ok=True)
        (swap_dir / "chunk.0.0").write_bytes(b"x" * 1024)
        self.cleanup_calls = 0

    def cleanup_swap_files(self) -> None:
        self.cleanup_calls += 1
        for child in self.swap_dir.glob("*"):
            child.unlink()
        self.swap_dir.rmdir()

    def to_netcdf(self, save_path, **kwargs):
        xr.Dataset({"v": ("x", [1])}).to_netcdf(str(save_path))

    def to_zarr(self, save_path, zarr_format=2, compress=True, **kwargs):
        xr.Dataset({"v": ("x", [1])}).to_zarr(
            str(save_path), mode="w", zarr_format=zarr_format
        )


@pytest.fixture
def raw_file(tmp_path: Path) -> Path:
    path = tmp_path / "HB1603_L1-D20160627-T142704.raw"
    path.write_bytes(b"\x00" * 64)
    return path


@pytest.fixture
def stub(tmp_path: Path, monkeypatch) -> _StubEchoData:
    """Install a stub ``open_raw`` returning an EchoData holding a swap dir."""
    ed = _StubEchoData(tmp_path / "ep-swap--abc123.zarr")
    monkeypatch.setattr(utils.ep, "open_raw", lambda *a, **k: ed)
    return ed


def _store(raw_file: Path, store_dir: Path, fmt: str):
    store_dir.mkdir(parents=True, exist_ok=True)
    return utils._open_and_store(
        raw_file, "EK60", False, "auto", fmt, store_dir, None, False
    )


# ---------------------------------------------------------------------------
# _release_swap_files
# ---------------------------------------------------------------------------


def test_release_calls_echopypes_cleanup(tmp_path):
    ed = _StubEchoData(tmp_path / "swap.zarr")

    utils._release_swap_files(ed, "f.raw")

    assert ed.cleanup_calls == 1
    assert not ed.swap_dir.exists()


def test_release_never_raises(tmp_path):
    """A converted file must not be failed by its own cleanup."""

    class _Exploding:
        def cleanup_swap_files(self):
            raise RuntimeError("no Sonar group")

    utils._release_swap_files(_Exploding(), "f.raw")  # must not raise


def test_release_is_a_no_op_without_swap():
    """Without swap echopype's cleanup finds no dask arrays and does nothing."""

    class _NoSwap:
        calls = 0

        def cleanup_swap_files(self):
            type(self).calls += 1

    utils._release_swap_files(_NoSwap(), "f.raw")
    assert _NoSwap.calls == 1


# ---------------------------------------------------------------------------
# _open_and_store
# ---------------------------------------------------------------------------


def test_netcdf_intermediate_releases_the_swap(tmp_path, raw_file, stub):
    out = _store(raw_file, tmp_path / "exe_temp", "netcdf")

    assert Path(out).exists(), "the intermediate must still be written"
    assert stub.cleanup_calls == 1
    assert not stub.swap_dir.exists()


def test_zarr_intermediate_releases_the_swap(tmp_path, raw_file, stub):
    out = _store(raw_file, tmp_path / "exe_temp", "zarr")

    assert Path(out).exists()
    assert stub.cleanup_calls == 1
    assert not stub.swap_dir.exists()


def test_a_failed_write_still_releases_the_swap(tmp_path, raw_file, stub, monkeypatch):
    """A write that dies must not strand the swap store it was spilling from."""
    def _boom(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(stub, "to_netcdf", _boom)

    with pytest.raises(OSError):
        _store(raw_file, tmp_path / "exe_temp", "netcdf")

    assert stub.cleanup_calls == 1
    assert not stub.swap_dir.exists()


def test_in_memory_mode_keeps_the_swap(tmp_path, raw_file, stub):
    """``none`` hands the EchoData to the caller, which still reads it.

    Its swap store is still live, and echopype's own ``__del__`` cleanup does
    apply here because nothing set ``converted_raw_path``.
    """
    ed = _store(raw_file, tmp_path / "exe_temp", "none")

    assert ed is stub
    assert stub.cleanup_calls == 0
    assert stub.swap_dir.exists()
