# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Permission handling and progress reporting around per-file store writes.

Regression tests for an incident where converting a raw file failed with
``[Errno 13] Permission denied`` naming a directory inside the store being
written, with nothing in the log to say which file it was on or why. Two
defects contributed: the rmtree error handler assigned ``stat.S_IWRITE``
(``0o200``) to the failing path, stripping read and execute from a directory
and guaranteeing EACCES on every later access; and a POSIX EACCES was retried
three times as though it were the transient Windows rename race.
"""

from __future__ import annotations

import os
import stat
import sys

import pytest

from aa_si_utils import utils

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX permission semantics"
)


# ---------------------------------------------------------------------------
# Removing a store whose permissions are hostile
# ---------------------------------------------------------------------------


@posix_only
def test_remove_existing_store_handles_unlistable_directory(tmp_path):
    store = tmp_path / "D20160725-T214425.zarr"
    array_dir = store / "Sonar" / "Beam_group1" / "backscatter_r"
    array_dir.mkdir(parents=True)
    (array_dir / "0.0.0").write_bytes(b"chunk")
    os.chmod(array_dir, 0o200)  # what the old handler left behind

    utils._remove_existing_store(store)

    assert not store.exists()


@posix_only
def test_remove_existing_store_handles_unwritable_parent(tmp_path):
    store = tmp_path / "sample.zarr"
    group = store / "Beam_group1"
    group.mkdir(parents=True)
    (group / "backscatter_r").write_bytes(b"chunk")
    os.chmod(group, 0o500)

    utils._remove_existing_store(store)

    assert not store.exists()


@posix_only
def test_grant_access_only_adds_bits(tmp_path):
    target = tmp_path / "group"
    target.mkdir()
    os.chmod(target, 0o750)

    utils._grant_access(target)

    mode = stat.S_IMODE(os.stat(target).st_mode)
    assert mode & stat.S_IRWXU == stat.S_IRWXU
    assert mode & stat.S_IRGRP  # group read survives


def test_remove_existing_store_missing_path_is_a_noop(tmp_path):
    utils._remove_existing_store(tmp_path / "absent.zarr")  # must not raise


def test_remove_existing_store_removes_a_plain_file(tmp_path):
    target = tmp_path / "intermediate.nc"
    target.write_bytes(b"payload")

    utils._remove_existing_store(target)

    assert not target.exists()


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def _posix_eacces(path):
    return PermissionError(13, "Permission denied", str(path))


def _windows_lock(path):
    exc = PermissionError(5, "Access is denied", str(path))
    exc.winerror = 5
    return exc


def test_posix_permission_error_is_not_retried(tmp_path, monkeypatch):
    store = tmp_path / "sample.zarr"
    attempts = []
    monkeypatch.setattr(utils.time, "sleep", lambda _s: pytest.fail("slept"))

    def always_denied():
        attempts.append(1)
        raise _posix_eacces(store / "Sonar" / "Beam_group1" / "backscatter_r")

    with pytest.raises(PermissionError) as excinfo:
        utils._write_store_with_retry(always_denied, store)

    # Retrying a deterministic EACCES only delays the real error by 3 seconds
    # and re-runs the removal that cannot succeed.
    assert len(attempts) == 1
    assert "store: " in str(excinfo.value)
    assert str(store) in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, PermissionError)


def test_posix_permission_error_message_reports_free_disk(tmp_path):
    store = tmp_path / "sample.zarr"

    def always_denied():
        raise _posix_eacces(store)

    with pytest.raises(PermissionError, match="free disk"):
        utils._write_store_with_retry(always_denied, store)


def test_windows_lock_is_retried(tmp_path, monkeypatch):
    store = tmp_path / "sample.zarr"
    attempts = []
    delays = []
    monkeypatch.setattr(utils.time, "sleep", lambda s: delays.append(s))

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise _windows_lock(store / ".zmetadata")
        store.mkdir(parents=True, exist_ok=True)

    utils._write_store_with_retry(flaky, store)

    assert len(attempts) == 3
    assert delays == [1.0, 2.0]


def test_windows_lock_reraises_when_retries_are_exhausted(tmp_path, monkeypatch):
    store = tmp_path / "sample.zarr"
    monkeypatch.setattr(utils.time, "sleep", lambda _s: None)

    def always_locked():
        raise _windows_lock(store)

    with pytest.raises(PermissionError) as excinfo:
        utils._write_store_with_retry(always_locked, store)

    assert getattr(excinfo.value, "winerror", None) == 5


def test_partial_store_is_cleared_before_each_attempt(tmp_path, monkeypatch):
    store = tmp_path / "sample.zarr"
    monkeypatch.setattr(utils.time, "sleep", lambda _s: None)
    seen = []

    def half_writes():
        seen.append(store.exists())
        store.mkdir(parents=True, exist_ok=True)
        (store / ".zmetadata").write_text("half", encoding="utf-8")
        if len(seen) < 2:
            raise _windows_lock(store / ".zmetadata")

    utils._write_store_with_retry(half_writes, store)

    # Every attempt starts from a clean slate, so a retry never mixes bytes
    # from the failed attempt into the good store.
    assert seen == [False, False]


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def test_read_raw_reports_progress_per_file(tmp_path, monkeypatch, capsys):
    from aa_recipe_manager.executor.runtime_context import execution_context

    class _StubEchoData:
        def to_zarr(self, save_path, **kwargs):
            import xarray as xr

            xr.Dataset({"value": ("x", [1])}).to_zarr(
                str(save_path), mode="w", zarr_format=2
            )

    raw_files = []
    for stamp in ("D20160725-T210000", "D20160725-T214425"):
        raw = tmp_path / f"{stamp}.raw"
        raw.write_bytes(b"0" * 2048)
        raw_files.append(str(raw))

    monkeypatch.setattr(utils.ep, "open_raw", lambda *a, **k: _StubEchoData())

    with execution_context(mode="direct", temp_dir=str(tmp_path / "exe_temp")):
        utils.read_raw_files_to_stores(
            raw_files, sonar_model="EK60", intermediate_format="zarr"
        )

    out = capsys.readouterr().out
    # Which file, in what order, and how far the loop got.
    assert "[1/2] D20160725-T210000.raw" in out
    assert "[2/2] D20160725-T214425.raw" in out
    assert out.count("open_raw ") == 2
    assert out.count("parsed ") == 2
    assert out.count("wrote ") == 2
    # The context that distinguishes a full disk from a permission problem.
    assert "free disk" in out
    assert "RAM " in out
