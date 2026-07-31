# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for gs://-shaped (remote) data *inputs* in aa_si_utils.

``memory://`` stands in for ``gs://`` — a non-local fsspec store needing no
credentials. echopype is faked so no real raw files or conversions are needed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import fsspec
import pandas as pd
import pytest
import xarray as xr

from aa_si_utils import _storage, utils
from aa_si_utils.data_retrieval import (
    filter_paths_by_file_time,
    parse_datetime_from_filename,
)


@pytest.fixture(autouse=True)
def clear_memory_fs():
    mem = fsspec.filesystem("memory")
    mem.store.clear()
    mem.pseudo_dirs[:] = [""]
    yield
    mem.store.clear()
    mem.pseudo_dirs[:] = [""]


class _StubEchoData:
    """to_zarr writes a tiny real dataset (local Path or remote URL)."""

    def __init__(self, payload: int) -> None:
        self.payload = payload

    def to_zarr(self, save_path, zarr_format=2, compress=True,
                output_storage_options=None, **kwargs):
        ds = xr.Dataset({"value": ("x", [self.payload])})
        ds.to_zarr(
            str(save_path),
            mode="w",
            zarr_format=zarr_format,
            storage_options=(output_storage_options or None),
        )


def _seed_raw(fs, folder, names):
    for name in names:
        fs.pipe_file(f"{folder}/{name}", b"raw-bytes")


# ---------------------------------------------------------------------------
# filter_paths_by_file_time
#
# These cover the name-based semantics on paths that do not exist on disk, so
# they pass verify_boundary=False wherever a file starts before the window.
# The byte-accurate boundary check has its own file-backed tests in
# test_raw_file_times.py.
# ---------------------------------------------------------------------------

_RAW_A = "D20160725-T205832.raw"   # 2016-07-25 20:58:32
_RAW_B = "D20160725-T210500.raw"   # 2016-07-25 21:05:00
_RAW_C = "D20160725-T213000.raw"   # 2016-07-25 21:30:00


def test_parse_datetime_from_filename():
    assert parse_datetime_from_filename(_RAW_A) == datetime(2016, 7, 25, 20, 58, 32)
    assert parse_datetime_from_filename("no_stamp.raw") is None


def test_filter_no_bounds_returns_all():
    paths = [_RAW_A, _RAW_B, _RAW_C]
    assert filter_paths_by_file_time(paths) == paths


def test_filter_inclusive_bounds():
    paths = [_RAW_A, _RAW_B, _RAW_C]
    # Bounds land exactly on A and B's stamps — both inclusive.
    kept = filter_paths_by_file_time(
        paths, "2016-07-25T20:58:32", "2016-07-25T21:05:00"
    )
    assert kept == [_RAW_A, _RAW_B]


def test_filter_accepts_datetime_bounds():
    paths = [_RAW_A, _RAW_B, _RAW_C]
    # A starts before 21:00 but records until B begins (21:05) — overlap keeps it.
    kept = filter_paths_by_file_time(
        paths, datetime(2016, 7, 25, 21, 0), None, verify_boundary=False
    )
    assert kept == [_RAW_A, _RAW_B, _RAW_C]


def test_filter_includes_file_straddling_window_start():
    paths = [_RAW_A, _RAW_B, _RAW_C]
    # A (20:58:32) starts before the window but records until B starts
    # (21:05:00), so it has in-window data and is kept.
    kept = filter_paths_by_file_time(
        paths, "2016-07-25T21:00", "2016-07-25T21:30:00", verify_boundary=False
    )
    assert kept == [_RAW_A, _RAW_B, _RAW_C]


def test_filter_excludes_file_ending_exactly_at_window_start():
    paths = [_RAW_A, _RAW_B, _RAW_C]
    # A's recording ends the instant B starts (21:05:00) — exactly at the
    # window start — so A has no in-window data.
    kept = filter_paths_by_file_time(
        paths, "2016-07-25T21:05:00", None, verify_boundary=False
    )
    assert kept == [_RAW_B, _RAW_C]


def test_filter_last_file_falls_back_to_own_stamp():
    # The chronologically last file has no next stamp to bound its end, so
    # only its own stamp decides: A alone, before the window, is excluded.
    assert filter_paths_by_file_time(
        [_RAW_A], "2016-07-25T21:00", None, verify_boundary=False
    ) == []


def test_filter_excludes_unparseable_names_when_bounded():
    paths = [_RAW_A, "misc_notes.raw"]
    assert filter_paths_by_file_time(paths, "2016-07-25T00:00", None) == [_RAW_A]


def test_filter_local_and_url_agree():
    local = [f"/data/{_RAW_A}", f"/data/{_RAW_B}"]
    urls = [f"memory://bkt/{_RAW_A}", f"memory://bkt/{_RAW_B}"]
    local_kept = filter_paths_by_file_time(local, None, "2016-07-25T21:00")
    url_kept = filter_paths_by_file_time(urls, None, "2016-07-25T21:00")
    assert [Path(p).name for p in local_kept] == [_storage.basename(u) for u in url_kept]
    assert local_kept == [f"/data/{_RAW_A}"]


def test_filter_ignores_dir_stamp_uses_basename():
    # A parent directory carrying a D..-T.. stamp must not affect the file's time.
    paths = [f"/D20160725-T210500/{_RAW_A}"]  # file is A (20:58:32)
    assert filter_paths_by_file_time(
        paths, "2016-07-25T21:00", None, verify_boundary=False
    ) == []


# ---------------------------------------------------------------------------
# initial_setup_and_validation: remote glob, subset, filter, BYO regression
# ---------------------------------------------------------------------------


def test_initial_setup_remote_glob(tmp_path):
    from aa_recipe_manager.executor.runtime_context import execution_context

    fs = fsspec.filesystem("memory")
    _seed_raw(fs, "/survey/raw", [_RAW_B, _RAW_A, "skip.txt"])

    with execution_context(mode="direct", artifacts_dir=tmp_path):
        result = utils.initial_setup_and_validation("memory://survey/raw")

    # Sorted full remote URLs (reopenable), only .raw files.
    paths = result["raw_file_paths"]
    assert [_storage.basename(p) for p in paths] == [_RAW_A, _RAW_B]
    assert all(p.startswith("memory://") for p in paths)
    assert paths == _storage.glob_url("memory://survey/raw", "*.raw")


def test_initial_setup_remote_subset(tmp_path):
    from aa_recipe_manager.executor.runtime_context import execution_context

    fs = fsspec.filesystem("memory")
    _seed_raw(fs, "/survey/raw", [_RAW_A, _RAW_B])

    with execution_context(mode="direct", artifacts_dir=tmp_path):
        result = utils.initial_setup_and_validation(
            "memory://survey/raw", raw_file_names=[_RAW_B]
        )
    assert result["raw_file_paths"] == ["memory://survey/raw/" + _RAW_B]


def test_initial_setup_remote_time_filter(tmp_path):
    from aa_recipe_manager.executor.runtime_context import execution_context

    fs = fsspec.filesystem("memory")
    _seed_raw(fs, "/survey/raw", [_RAW_A, _RAW_B, _RAW_C])

    with execution_context(mode="direct", artifacts_dir=tmp_path):
        result = utils.initial_setup_and_validation(
            "memory://survey/raw",
            file_time_start="2016-07-25T21:00",
            file_time_end="2016-07-25T21:10",
        )
    # A straddles the window start (records until B begins at 21:05);
    # C starts after the window end.
    assert [_storage.basename(p) for p in result["raw_file_paths"]] == [_RAW_A, _RAW_B]


def test_initial_setup_empty_remote_folder_raises(tmp_path):
    from aa_recipe_manager.executor.runtime_context import execution_context

    fsspec.filesystem("memory").makedirs("/survey/empty", exist_ok=True)
    with execution_context(mode="direct", artifacts_dir=tmp_path):
        with pytest.raises(FileNotFoundError, match="No raw files found"):
            utils.initial_setup_and_validation("memory://survey/empty")


def test_initial_setup_time_filter_empties_raises(tmp_path):
    from aa_recipe_manager.executor.runtime_context import execution_context

    fs = fsspec.filesystem("memory")
    _seed_raw(fs, "/survey/raw", [_RAW_A])
    with execution_context(mode="direct", artifacts_dir=tmp_path):
        with pytest.raises(FileNotFoundError, match="filename-time window"):
            utils.initial_setup_and_validation(
                "memory://survey/raw", file_time_start="2020-01-01T00:00"
            )


def test_initial_setup_local_regression(tmp_path):
    """BYO-local path: unchanged behavior — local absolute paths returned."""
    from aa_recipe_manager.executor.runtime_context import execution_context

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / _RAW_A).write_bytes(b"x")
    (raw_dir / _RAW_B).write_bytes(b"x")
    out_dir = tmp_path / "out"

    with execution_context(mode="direct", artifacts_dir=out_dir):
        result = utils.initial_setup_and_validation(str(raw_dir))

    assert result["raw_file_paths"] == [str(raw_dir / _RAW_A), str(raw_dir / _RAW_B)]
    # Calibration output dir resolved under the (local) artifacts dir.
    assert result["calibration_output_dir"] == str(out_dir / "calibration")


# ---------------------------------------------------------------------------
# read_raw_files_to_stores: download -> convert -> delete
# ---------------------------------------------------------------------------


def test_remote_raw_basename_preserved_and_local(monkeypatch, tmp_path):
    from aa_recipe_manager.executor.runtime_context import execution_context

    fs = fsspec.filesystem("memory")
    _seed_raw(fs, "/s/raw", [_RAW_A])
    urls = _storage.glob_url("memory://s/raw", "*.raw")

    def fake_open_raw(raw_path, **kwargs):
        p = Path(raw_path)
        assert p.exists()          # a real local copy during processing
        assert p.name == _RAW_A    # original basename preserved
        return _StubEchoData(1)

    monkeypatch.setattr(utils.ep, "open_raw", fake_open_raw)
    with execution_context(mode="direct", temp_dir=str(tmp_path)):
        utils.read_raw_files_to_stores(urls, sonar_model="EK80", intermediate_format="zarr")


def test_remote_raw_local_copy_deleted_between_files(monkeypatch, tmp_path):
    from aa_recipe_manager.executor.runtime_context import execution_context

    fs = fsspec.filesystem("memory")
    _seed_raw(fs, "/s/raw", [_RAW_A, _RAW_B])
    urls = _storage.glob_url("memory://s/raw", "*.raw")

    seen: list[str] = []

    def fake_open_raw(raw_path, **kwargs):
        p = Path(raw_path)
        assert p.exists()
        # Every previously-processed local copy is already gone.
        assert all(not Path(s).exists() for s in seen)
        seen.append(str(p))
        return _StubEchoData(len(seen))

    monkeypatch.setattr(utils.ep, "open_raw", fake_open_raw)
    with execution_context(mode="direct", temp_dir=str(tmp_path)):
        utils.read_raw_files_to_stores(urls, sonar_model="EK80", intermediate_format="zarr")

    # After the run, no local scratch remains, and the bucket is intact.
    assert all(not Path(s).exists() for s in seen)
    for name in [_RAW_A, _RAW_B]:
        assert fs.exists(f"/s/raw/{name}")


def test_remote_raw_bot_companion(monkeypatch, tmp_path):
    from aa_recipe_manager.executor.runtime_context import execution_context

    fs = fsspec.filesystem("memory")
    fs.pipe_file("/s/raw/" + _RAW_A, b"raw")
    fs.pipe_file("/s/raw/" + _RAW_A.replace(".raw", ".bot"), b"bot")
    urls = _storage.glob_url("memory://s/raw", "*.raw")

    def make_fake(expected_bot):
        def fake_open_raw(raw_path, include_bot=True, **kwargs):
            bot = Path(raw_path).with_suffix(".bot")
            assert bot.exists() is expected_bot
            return _StubEchoData(1)
        return fake_open_raw

    # include_bot=True downloads the .bot beside the raw file.
    monkeypatch.setattr(utils.ep, "open_raw", make_fake(True))
    with execution_context(mode="direct", temp_dir=str(tmp_path)):
        utils.read_raw_files_to_stores(
            urls, sonar_model="EK80", intermediate_format="zarr", include_bot=True
        )

    # include_bot=False leaves it in the bucket.
    monkeypatch.setattr(utils.ep, "open_raw", make_fake(False))
    with execution_context(mode="direct", temp_dir=str(tmp_path)):
        utils.read_raw_files_to_stores(
            urls, sonar_model="EK80", intermediate_format="zarr", include_bot=False
        )


def test_remote_raw_to_remote_zarr_end_to_end(monkeypatch):
    """Remote raw input + remote (memory://) temp + zarr writes to the bucket."""
    from aa_recipe_manager.executor.runtime_context import execution_context

    fs = fsspec.filesystem("memory")
    _seed_raw(fs, "/s/raw", [_RAW_A])
    urls = _storage.glob_url("memory://s/raw", "*.raw")

    monkeypatch.setattr(utils.ep, "open_raw", lambda raw_path, **k: _StubEchoData(7))
    with execution_context(mode="direct", temp_dir="memory://scratch/exe_temp"):
        result = utils.read_raw_files_to_stores(
            urls, sonar_model="EK80", intermediate_format="zarr"
        )

    assert result == ["memory://scratch/exe_temp/data/" + _RAW_A.replace(".raw", ".zarr")]
    assert fs.exists(result[0])


# ---------------------------------------------------------------------------
# add_dive_profile_to_dataset: remote CSV
# ---------------------------------------------------------------------------

_CSV = (
    "ClickTime_UTC,clickDepth_m,fit_line,lwr_CI_99,upr_CI_99\n"
    "2016-07-25_21:00:00,100,101,90,110\n"
    "2016-07-25_21:00:20,120,121,110,130\n"
)


def _mvbs_ds():
    times = pd.to_datetime(["2016-07-25T21:00:00", "2016-07-25T21:00:20"])
    return xr.Dataset(coords={"ping_time": times})


def test_remote_csv_matches_local(tmp_path):
    local = tmp_path / "dive.csv"
    local.write_text(_CSV)
    fsspec.filesystem("memory").pipe_file("/line/dive.csv", _CSV.encode())

    out_local = utils.add_dive_profile_to_dataset(_mvbs_ds(), str(local), "dive")
    out_remote = utils.add_dive_profile_to_dataset(
        _mvbs_ds(), "memory://line/dive.csv", "dive"
    )
    # source_file attr differs only by basename (identical here); compare values.
    assert "dive_depth" in out_remote
    xr.testing.assert_equal(out_local, out_remote)


def test_remote_csv_missing_raises():
    with pytest.raises(FileNotFoundError, match="Dive profile CSV not found"):
        utils.add_dive_profile_to_dataset(
            _mvbs_ds(), "memory://line/nope.csv", "dive"
        )


# ---------------------------------------------------------------------------
# _execution_storage_options tolerance
# ---------------------------------------------------------------------------


def test_execution_storage_options_from_context():
    from aa_recipe_manager.executor.runtime_context import execution_context

    with execution_context(mode="direct", storage_options={"token": "x"}):
        assert utils._execution_storage_options() == {"token": "x"}


def test_execution_storage_options_none_standalone():
    assert utils._execution_storage_options() is None
