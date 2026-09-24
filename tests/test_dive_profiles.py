# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for planning dive windows out of the NEFSC dive configs."""

import json
from pathlib import Path

import pytest

from aa_si_utils import dive_profiles

from test_seafloor_line_evl import _write_evl


REAL_AUX = (
    Path(__file__).resolve().parents[2]
    / "NEFSC_UC1" / "full_data" / "Auxiliary"
)


def _line_entry(name):
    return {"path": "/home/mjech/elsewhere", "filenames": [name],
            "linecolor": "magenta", "linewidth": 1}


def _write_config(tmp_path, label, nc_stems, line_names, seabed="seabed.evl",
                  range_bin="2m", ping_bin="10S", freqs=("18kHz", "38kHz")):
    """Write a config shaped like the real SWD_*.json files."""
    ev_lines = {key: _line_entry(name) for key, name in line_names.items()}
    if seabed:
        ev_lines["ev_bottom"] = _line_entry(seabed)
    config = {
        "path_config": {
            "EK_data_path": "/home/mjech/netCDF4_Files",
            "EK_data_filenames": [f"{stem}.nc" for stem in nc_stems],
            "save_path": "/tmp/generated",
            "fig_path": "/tmp/figures",
        },
        "analysis": {"frequency_list": list(freqs), "eqval": 3.0},
        "sonar_model": "EK60",
        "data_reduction": {
            "range_var": "depth",
            "range_meter_bin": range_bin,
            "ping_time_bin": ping_bin,
        },
        "sub_selection": {"align_line_time": True, "EV_lines": ev_lines},
    }
    json_dir = tmp_path / "JSON_Files"
    json_dir.mkdir(exist_ok=True)
    (json_dir / f"{label}.json").write_text(json.dumps(config), encoding="utf-8")
    return json_dir


def _write_dive_lines(tmp_path, stem="dive", date="20160707",
                      first="1948561960", last="2006123490"):
    """Write a fit/U99/L99 triplet plus a seabed line under an lines/ folder."""
    lines_dir = tmp_path / "lines"
    lines_dir.mkdir(exist_ok=True)
    names = {}
    for key, suffix, depth in (
        ("SWdiveprofile", "depth", 500.0),
        ("U99CI", "U99CI", 480.0),
        ("L99CI", "L99CI", 520.0),
    ):
        name = f"{stem}_{suffix}.evl"
        _write_evl(
            lines_dir,
            [(first, depth, 3), (last, depth + 20.0, 3)],
            name=name,
            date=date,
        )
        names[key] = name
    _write_evl(lines_dir, [(first, 2900.0, 3), (last, 2950.0, 3)],
               name="seabed.evl", date=date)
    return names


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------


def test_window_span_comes_from_the_dive_line_not_the_file_list(tmp_path):
    names = _write_dive_lines(tmp_path)
    json_dir = _write_config(
        tmp_path, "SWD_20160707-OE20",
        ["HB1603_L1-D20160707-T192446", "HB1603_L1-D20160707-T201039"],
        names,
    )

    out = dive_profiles.plan_dive_datasets(json_dir, evl_root=tmp_path)

    (window,) = out["windows"]
    # The line runs 19:48:56.196 to 20:06:12.349, well inside the file span,
    # which starts at 19:24:46.
    assert window["start"] == "2016-07-07T19:48:56.196000"
    assert window["end"] == "2016-07-07T20:06:12.349000"
    assert window["duration_minutes"] == pytest.approx(17.27, abs=0.01)


def test_pad_minutes_widens_both_sides(tmp_path):
    names = _write_dive_lines(tmp_path)
    json_dir = _write_config(tmp_path, "SWD_a", ["D20160707-T192446"], names)

    out = dive_profiles.plan_dive_datasets(
        json_dir, evl_root=tmp_path, pad_minutes=2.0
    )

    (window,) = out["windows"]
    assert window["start"] == "2016-07-07T19:46:56.196000"
    assert window["end"] == "2016-07-07T20:08:12.349000"


def test_raw_files_take_the_stem_and_the_raw_suffix(tmp_path):
    names = _write_dive_lines(tmp_path)
    json_dir = _write_config(
        tmp_path, "SWD_a",
        ["HB1603_L1-D20160707-T192446", "D20160707-T201039"],
        names,
    )

    out = dive_profiles.plan_dive_datasets(json_dir, evl_root=tmp_path)

    assert out["windows"][0]["raw_files"] == [
        "D20160707-T201039.raw",
        "HB1603_L1-D20160707-T192446.raw",
    ]
    assert out["raw_files"] == out["windows"][0]["raw_files"]


def test_both_file_name_spellings_are_accepted(tmp_path):
    """July files carry an HB1603_L1- prefix; later ones are bare."""
    names = _write_dive_lines(tmp_path)
    json_dir = _write_config(
        tmp_path, "SWD_a", ["HB1603_L1-D20160703-T183957", "D20160825-T114557"],
        names,
    )

    out = dive_profiles.plan_dive_datasets(json_dir, evl_root=tmp_path)

    assert len(out["windows"][0]["raw_files"]) == 2


def test_binning_and_frequencies_come_from_the_config(tmp_path):
    names = _write_dive_lines(tmp_path)
    json_dir = _write_config(
        tmp_path, "SWD_a", ["D20160707-T192446"], names,
        range_bin="2m", ping_bin="10S", freqs=("18kHz", "38kHz"),
    )

    (window,) = dive_profiles.plan_dive_datasets(
        json_dir, evl_root=tmp_path
    )["windows"]

    assert window["frequencies_khz"] == [18.0, 38.0]
    assert window["range_bin"] == "2m"
    # "10S" is lowercased: pandas renamed the second alias and warns on "S".
    assert window["ping_time_bin"] == "10s"
    assert window["range_var"] == "depth"


def test_line_paths_resolve_by_basename_under_the_search_root(tmp_path):
    """Configs carry absolute paths from another machine, so only the name is usable."""
    names = _write_dive_lines(tmp_path)
    json_dir = _write_config(tmp_path, "SWD_a", ["D20160707-T192446"], names)

    (window,) = dive_profiles.plan_dive_datasets(
        json_dir, evl_root=tmp_path
    )["windows"]

    for key in ("dive_fit_evl", "dive_u99_evl", "dive_l99_evl", "seabed_evl"):
        assert Path(window[key]).exists()
    assert window["dive_fit_evl"].endswith("dive_depth.evl")
    assert window["seabed_evl"].endswith("seabed.evl")


def test_windows_come_back_in_time_order(tmp_path):
    early = _write_dive_lines(tmp_path, stem="early", date="20160703")
    _write_config(tmp_path, "SWD_late", ["D20160814-T130425"],
                  _write_dive_lines(tmp_path, stem="late", date="20160814"))
    json_dir = _write_config(tmp_path, "SWD_early", ["D20160703-T183957"], early)

    out = dive_profiles.plan_dive_datasets(json_dir, evl_root=tmp_path)

    starts = [w["start"] for w in out["windows"]]
    assert starts == sorted(starts)
    assert out["windows"][0]["label"] == "SWD_early"


# ---------------------------------------------------------------------------
# Multi-dive configs and failure modes
# ---------------------------------------------------------------------------


def test_multi_dive_config_is_skipped_by_default(tmp_path):
    names = _write_dive_lines(tmp_path)
    _write_config(tmp_path, "SWD_20160707-OE20", ["D20160707-T192446"], names)
    # A combined config: several dive lines, no U99CI/L99CI triplet.
    json_dir = _write_config(
        tmp_path, "SWD_20160707-OE20-OE29-OE32", ["D20160707-T192446"],
        {"SWD_OE20": names["SWdiveprofile"], "SWD_OE29": names["U99CI"]},
    )

    out = dive_profiles.plan_dive_datasets(json_dir, evl_root=tmp_path)

    assert [w["label"] for w in out["windows"]] == ["SWD_20160707-OE20"]


def test_multi_dive_config_can_be_opted_into(tmp_path):
    names = _write_dive_lines(tmp_path)
    json_dir = _write_config(
        tmp_path, "SWD_combined", ["D20160707-T192446"],
        {"SWD_OE20": names["SWdiveprofile"], "SWD_OE29": names["U99CI"]},
    )

    out = dive_profiles.plan_dive_datasets(
        json_dir, evl_root=tmp_path, include_multi_dive=True
    )

    (window,) = out["windows"]
    assert window["label"] == "SWD_combined"
    # No triplet, so no CI line keys are attached.
    assert "dive_u99_evl" not in window


def test_missing_line_file_names_the_config(tmp_path):
    names = _write_dive_lines(tmp_path)
    names["SWdiveprofile"] = "not_written.evl"
    json_dir = _write_config(tmp_path, "SWD_a", ["D20160707-T192446"], names)

    with pytest.raises(FileNotFoundError, match="SWD_a: line file 'not_written.evl'"):
        dive_profiles.plan_dive_datasets(json_dir, evl_root=tmp_path)


def test_ambiguous_line_file_raises(tmp_path):
    names = _write_dive_lines(tmp_path)
    duplicate = tmp_path / "lines_copy"
    duplicate.mkdir()
    (duplicate / names["SWdiveprofile"]).write_text(
        (tmp_path / "lines" / names["SWdiveprofile"]).read_text(encoding="utf-8-sig"),
        encoding="utf-8-sig",
    )
    json_dir = _write_config(tmp_path, "SWD_a", ["D20160707-T192446"], names)

    with pytest.raises(ValueError, match="is ambiguous"):
        dive_profiles.plan_dive_datasets(json_dir, evl_root=tmp_path)


def test_empty_config_folder_raises(tmp_path):
    empty = tmp_path / "JSON_Files"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="No .json configs"):
        dive_profiles.plan_dive_datasets(empty, evl_root=tmp_path)


# ---------------------------------------------------------------------------
# Against the real NEFSC config set
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_AUX.exists(), reason="NEFSC_UC1 data not present")
def test_real_configs_give_fourteen_dives_and_twentyeight_files():
    out = dive_profiles.plan_dive_datasets(
        REAL_AUX / "JSON_Files", evl_root=REAL_AUX
    )

    # 15 configs, one of which is the combined July 7 view that is skipped.
    assert len(out["windows"]) == 14
    assert len(out["raw_files"]) == 28
    assert all(w["frequencies_khz"] == [18.0, 38.0] for w in out["windows"])
    assert all(w["range_bin"] == "2m" for w in out["windows"])
    assert all(w["ping_time_bin"] == "10s" for w in out["windows"])
    assert all(Path(w["dive_u99_evl"]).exists() for w in out["windows"])
    assert all(Path(w["seabed_evl"]).exists() for w in out["windows"])
    # Every dive is a localized section, tens of minutes not hours.
    assert all(1.0 < w["duration_minutes"] < 120.0 for w in out["windows"])


# ---------------------------------------------------------------------------
# include_labels: running the workflow over a slice of the survey
# ---------------------------------------------------------------------------


def test_include_labels_keeps_only_the_named_dive(tmp_path):
    names = _write_dive_lines(tmp_path, stem="a", date="20160703")
    _write_config(tmp_path, "SWD_keep", ["D20160703-T183957"], names)
    json_dir = _write_config(
        tmp_path, "SWD_drop", ["D20160814-T130425"],
        _write_dive_lines(tmp_path, stem="b", date="20160814"),
    )

    out = dive_profiles.plan_dive_datasets(
        json_dir, evl_root=tmp_path, include_labels=["SWD_keep"]
    )

    assert [w["label"] for w in out["windows"]] == ["SWD_keep"]
    # raw_files narrows with it, so a sliced survey does not claim files it
    # never processed.
    assert out["raw_files"] == ["D20160703-T183957.raw"]


def test_include_labels_none_keeps_everything(tmp_path):
    names = _write_dive_lines(tmp_path, stem="a", date="20160703")
    _write_config(tmp_path, "SWD_one", ["D20160703-T183957"], names)
    json_dir = _write_config(
        tmp_path, "SWD_two", ["D20160814-T130425"],
        _write_dive_lines(tmp_path, stem="b", date="20160814"),
    )

    out = dive_profiles.plan_dive_datasets(json_dir, evl_root=tmp_path)

    assert len(out["windows"]) == 2


def test_include_labels_rejects_an_unknown_dive(tmp_path):
    names = _write_dive_lines(tmp_path)
    json_dir = _write_config(tmp_path, "SWD_a", ["D20160707-T192446"], names)

    with pytest.raises(ValueError, match="no config"):
        dive_profiles.plan_dive_datasets(
            json_dir, evl_root=tmp_path, include_labels=["SWD_typo"]
        )


@pytest.mark.skipif(not REAL_AUX.exists(), reason="NEFSC_UC1 data not present")
def test_include_labels_selects_the_smoke_test_dive():
    out = dive_profiles.plan_dive_datasets(
        REAL_AUX / "JSON_Files", evl_root=REAL_AUX,
        include_labels=["SWD_20160707-OE20"],
    )

    (window,) = out["windows"]
    assert window["label"] == "SWD_20160707-OE20"
    # The three files the smoke-test survey slice covers.
    assert len(out["raw_files"]) == 3
    assert window["start"].startswith("2016-07-07T19:48")


# ---------------------------------------------------------------------------
# _bin_seconds: the grid's ping-time bin in the spelling compute_mvbs takes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, seconds",
    [("10s", 10.0), ("10S", 10.0), ("5s", 5.0), ("1min", 60.0), (10, 10.0), (2.5, 2.5)],
)
def test_bin_seconds_reads_offset_strings_and_numbers(value, seconds):
    assert dive_profiles._bin_seconds(value) == seconds


@pytest.mark.parametrize("value", ["ten seconds", "0s", -3, True, ""])
def test_bin_seconds_rejects_what_is_not_a_positive_duration(value):
    with pytest.raises(ValueError):
        dive_profiles._bin_seconds(value)
