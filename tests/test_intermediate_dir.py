# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for where read_raw_files_to_stores puts its intermediate stores."""

from pathlib import Path

from aa_si_utils import utils


def test_the_environment_variable_names_the_folder_outside_the_executor(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(utils, "_execution_temp_dir", lambda: None)
    monkeypatch.setenv("AA_SI_UTILS_TEMP_DIR", str(tmp_path / "scratch"))

    data_dir, options = utils._resolve_intermediate_dir()

    assert Path(data_dir) == tmp_path / "scratch" / "data"
    assert options is None


def test_the_executor_folder_wins_over_the_environment_variable(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(utils, "_execution_temp_dir", lambda: tmp_path / "exe_temp")
    monkeypatch.setenv("AA_SI_UTILS_TEMP_DIR", str(tmp_path / "scratch"))

    data_dir, _ = utils._resolve_intermediate_dir()

    assert Path(data_dir) == tmp_path / "exe_temp" / "data"


def test_without_either_it_uses_the_os_temp_folder(monkeypatch):
    monkeypatch.setattr(utils, "_execution_temp_dir", lambda: None)
    monkeypatch.delenv("AA_SI_UTILS_TEMP_DIR", raising=False)

    data_dir, _ = utils._resolve_intermediate_dir()

    assert Path(data_dir).parts[-2:] == ("aa_si_utils_temp", "data")
