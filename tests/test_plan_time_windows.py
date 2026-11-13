# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for plan_time_windows.

The expected windows are the slices echodataflow's build_MVBS_ledger plans for
the same files: clock aligned, the file starting just before a window counted
as one of its files, and windows after a long recording gap left out.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aa_si_utils.utils import plan_time_windows, select_ping_time_range


def _raw(stamp):
    return f"/data/SH2407-D{stamp}.raw"


def test_windows_are_clock_aligned_and_cover_the_last_file():
    files = [_raw("20240801-T120700"), _raw("20240801-T121100"), _raw("20240801-T122300")]

    windows = plan_time_windows(files, window_mins=10)["windows"]

    assert [w["label"] for w in windows] == [
        "20240801T120000", "20240801T121000", "20240801T122000",
    ]
    assert windows[0]["start"] == "2024-08-01T12:00:00"
    assert windows[0]["end"] == "2024-08-01T12:09:59.999999999"


def test_the_file_starting_before_a_window_is_one_of_its_files():
    files = [_raw("20240801-T120700"), _raw("20240801-T121100"), _raw("20240801-T122300")]

    windows = plan_time_windows(files, window_mins=10)["windows"]

    assert windows[0]["raw_filenames"] == ["SH2407-D20240801-T120700.raw"]
    assert windows[1]["raw_filenames"] == [
        "SH2407-D20240801-T120700.raw", "SH2407-D20240801-T121100.raw",
    ]
    assert windows[2]["raw_filenames"] == [
        "SH2407-D20240801-T121100.raw", "SH2407-D20240801-T122300.raw",
    ]


def test_windows_after_a_long_gap_are_left_out():
    files = [_raw("20240801-T120000"), _raw("20240801-T180000")]

    windows = plan_time_windows(files, window_mins=60, no_data_gap_hours=3.0)["windows"]

    assert [w["label"][-6:-2] for w in windows] == ["1200", "1300", "1400", "1500", "1800"]


def test_unordered_input_gives_the_same_windows():
    files = [_raw("20240801-T122300"), _raw("20240801-T120700"), _raw("20240801-T121100")]

    assert plan_time_windows(files, window_mins=10) == plan_time_windows(
        sorted(files), window_mins=10
    )


def test_a_name_without_a_start_time_is_rejected():
    with pytest.raises(ValueError, match="No start time"):
        plan_time_windows(["/data/notes.raw"])


@pytest.mark.parametrize("kwargs", [{"window_mins": 0}, {"no_data_gap_hours": 0}])
def test_non_positive_settings_are_rejected(kwargs):
    with pytest.raises(ValueError, match="greater than zero"):
        plan_time_windows([_raw("20240801-T120000")], **kwargs)


def test_windows_select_half_open_ranges():
    """A ping on a window edge belongs to the later window only."""
    ping_time = pd.to_datetime([
        "2024-08-01T12:09:59.5", "2024-08-01T12:10:00.0", "2024-08-01T12:10:00.5",
    ], format="ISO8601").values
    ds = xr.Dataset(
        {"Sv": ("ping_time", np.arange(3.0))}, coords={"ping_time": ping_time}
    )
    windows = plan_time_windows(
        [_raw("20240801-T120000"), _raw("20240801-T121000")], window_mins=10
    )["windows"]

    first = select_ping_time_range(ds, window=windows[0])
    second = select_ping_time_range(ds, window=windows[1])

    assert first.sizes["ping_time"] == 1
    assert second.sizes["ping_time"] == 2
