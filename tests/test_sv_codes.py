# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for SVCode generation and the depth-interval duration tables."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aa_si_utils import dive_profiles


PING_BIN_S = 10.0


def _ds(n_pings=6, depths=(100.0, 200.0, 300.0, 400.0, 500.0),
        labels=None, fit=300.0, half_band=120.0, freqs=(18000.0, 38000.0),
        with_lines=True, start="2016-07-07T19:49:00"):
    """MVBS-shaped dataset with a gridded label field and three dive lines."""
    ping_time = pd.date_range(start, periods=n_pings, freq="10s")
    depth = np.array(depths, dtype=float)
    shape = (len(ping_time), len(depth))

    if labels is None:
        labels = np.zeros(shape)
    labels = np.asarray(labels, dtype=float)

    sv = np.stack([
        np.full(shape, -70.0) + i * 5.0 for i in range(len(freqs))
    ])
    fitted = np.full(len(ping_time), fit, dtype=float)

    data = {
        "Sv": (("channel", "ping_time", "depth"), sv),
        "frequency_nominal": (("channel",), np.array(freqs)),
        "ml_dataset_hdbscan_results_grid": (("ping_time", "depth"), labels),
    }
    if with_lines:
        data["dive_fit"] = (("ping_time",), fitted)
        data["dive_u99"] = (("ping_time",), fitted - half_band)
        data["dive_l99"] = (("ping_time",), fitted + half_band)
    return xr.Dataset(
        data,
        coords={
            "channel": np.array([f"ch{i}" for i in range(len(freqs))]),
            "ping_time": ping_time,
            "depth": depth,
        },
    )


def _window(ds, label="SWD_test"):
    return {
        "label": label,
        "start": pd.Timestamp(ds["ping_time"].values[0]).isoformat(),
        "end": pd.Timestamp(ds["ping_time"].values[-1]).isoformat(),
    }


# ---------------------------------------------------------------------------
# Per-cell CSV
# ---------------------------------------------------------------------------


def test_csv_matches_the_reference_column_layout(tmp_path):
    ds = _ds()
    out = dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path)

    frame = pd.read_csv(out["code_csv_paths"][0])
    assert list(frame.columns) == [
        "DateTime", "depth_m", "18kHz_Sv", "38kHz_Sv", "code"
    ]


def test_timestamps_use_the_reference_spelling(tmp_path):
    ds = _ds()
    out = dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path)

    frame = pd.read_csv(out["code_csv_paths"][0])
    assert frame["DateTime"].iloc[0] == "2016-07-07_19:49:00.00"


def test_only_cells_between_the_confidence_bounds_are_coded(tmp_path):
    # Band is 300 +/- 120, so 200/300/400 are in and 100/500 are out.
    ds = _ds()
    out = dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path)

    frame = pd.read_csv(out["code_csv_paths"][0])
    assert sorted(frame["depth_m"].unique()) == [200.0, 300.0, 400.0]


def test_a_narrow_band_selects_fewer_cells(tmp_path):
    ds = _ds(half_band=10.0)
    out = dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path)

    frame = pd.read_csv(out["code_csv_paths"][0])
    assert sorted(frame["depth_m"].unique()) == [300.0]


def test_cluster_labels_become_the_code_column(tmp_path):
    labels = np.tile([9.0, 1.0, 2.0, 3.0, 9.0], (6, 1))
    ds = _ds(labels=labels)
    out = dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path)

    frame = pd.read_csv(out["code_csv_paths"][0])
    in_band = frame[frame["depth_m"] == 200.0]
    assert set(in_band["code"]) == {1}


def test_noise_is_kept_by_default_and_can_be_dropped(tmp_path):
    labels = np.tile([0.0, -1.0, 5.0, -1.0, 0.0], (6, 1))
    ds = _ds(labels=labels)

    kept = dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path / "a")
    dropped = dive_profiles.generate_sv_codes(
        ds, [_window(ds)], tmp_path / "b", include_noise=False
    )

    assert -1 in pd.read_csv(kept["code_csv_paths"][0])["code"].values
    assert -1 not in pd.read_csv(dropped["code_csv_paths"][0])["code"].values


def test_nan_labels_are_never_coded(tmp_path):
    labels = np.tile([0.0, np.nan, 5.0, 5.0, 0.0], (6, 1))
    ds = _ds(labels=labels)
    out = dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path)

    frame = pd.read_csv(out["code_csv_paths"][0])
    assert 200.0 not in frame["depth_m"].values
    assert sorted(frame["depth_m"].unique()) == [300.0, 400.0]


def test_one_csv_per_dive(tmp_path):
    ds = _ds(n_pings=6)
    early = {"label": "dive_a", "start": "2016-07-07T19:49:00",
             "end": "2016-07-07T19:49:20"}
    late = {"label": "dive_b", "start": "2016-07-07T19:49:30",
            "end": "2016-07-07T19:49:50"}

    out = dive_profiles.generate_sv_codes(ds, [early, late], tmp_path)

    assert out["dive_labels"] == ["dive_a", "dive_b"]
    assert {p.rsplit("/", 1)[-1] for p in out["code_csv_paths"]} == {
        "dive_a.csv", "dive_b.csv"
    }


def test_missing_label_grid_raises(tmp_path):
    ds = _ds().drop_vars("ml_dataset_hdbscan_results_grid")

    with pytest.raises(KeyError, match="embed_clustering_results"):
        dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path)


def test_missing_dive_line_raises(tmp_path):
    ds = _ds().drop_vars("dive_u99")

    with pytest.raises(KeyError, match="add_line_overlay"):
        dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path)


# ---------------------------------------------------------------------------
# Per-ping dominant code
# ---------------------------------------------------------------------------


def test_dominant_code_is_the_most_common_in_the_band(tmp_path):
    # In band (200/300/400): two 7s and one 4, so 7 dominates.
    labels = np.tile([0.0, 7.0, 7.0, 4.0, 0.0], (6, 1))
    ds = _ds(labels=labels)
    out = dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path)

    summary = pd.read_csv(out["summary_csv_path"])
    assert set(summary["code"]) == {7}
    assert set(summary["n_cells"]) == {3}
    assert set(summary["code_cells"]) == {2}


def test_a_tie_goes_to_the_cell_nearest_the_fitted_depth(tmp_path):
    # In band: 200 -> 3, 300 -> 8, 400 -> 3 gives 3 twice and 8 once, so 3 wins
    # on count. Drop to a two-cell band so the tie is real.
    labels = np.tile([0.0, 0.0, 8.0, 3.0, 0.0], (6, 1))
    ds = _ds(labels=labels, fit=350.0, half_band=55.0)  # band covers 300 and 400
    out = dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path)

    summary = pd.read_csv(out["summary_csv_path"])
    # 300 is 50 m from the fit, 400 is 50 m away too, so counts tie 1-1 and the
    # nearest cell decides; both are equidistant, so the answer must at least be
    # one of the two present labels rather than an invention.
    assert set(summary["code"]) <= {8, 3}


def test_tie_break_prefers_the_closer_cell(tmp_path):
    labels = np.tile([0.0, 0.0, 8.0, 3.0, 0.0], (6, 1))
    # Fit at 310 puts 300 (10 m away) closer than 400 (90 m away).
    ds = _ds(labels=labels, fit=310.0, half_band=95.0)
    out = dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path)

    summary = pd.read_csv(out["summary_csv_path"])
    assert set(summary["code"]) == {8}


def test_each_ping_contributes_one_bin_of_duration(tmp_path):
    ds = _ds(n_pings=6)
    out = dive_profiles.generate_sv_codes(
        ds, [_window(ds)], tmp_path, ping_time_bin_s=PING_BIN_S
    )

    summary = pd.read_csv(out["summary_csv_path"])
    assert len(summary) == 6
    assert summary["duration_s"].sum() == 60.0


# ---------------------------------------------------------------------------
# Depth-interval tables
# ---------------------------------------------------------------------------


def _summary_csv(tmp_path, rows):
    path = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_depth_table_percentages_sum_to_one_hundred_per_dive(tmp_path):
    rows = [
        {"label": "d1", "ping_time": f"t{i}", "code": i % 2,
         "n_cells": 3, "code_cells": 2,
         "dive_depth_m": 100.0 + 150.0 * i, "duration_s": 10.0}
        for i in range(8)
    ]
    paths = dive_profiles.sv_code_depth_table(
        _summary_csv(tmp_path, rows), tmp_path / "out"
    )

    per_dive = pd.read_csv(paths["per_dive_csv_path"])
    assert per_dive["percent_of_dive"].sum() == pytest.approx(100.0)


def test_depth_intervals_are_two_hundred_metres_by_default(tmp_path):
    rows = [
        {"label": "d1", "ping_time": "t0", "code": 1, "n_cells": 1,
         "code_cells": 1, "dive_depth_m": 50.0, "duration_s": 10.0},
        {"label": "d1", "ping_time": "t1", "code": 1, "n_cells": 1,
         "code_cells": 1, "dive_depth_m": 250.0, "duration_s": 10.0},
        {"label": "d1", "ping_time": "t2", "code": 1, "n_cells": 1,
         "code_cells": 1, "dive_depth_m": 650.0, "duration_s": 10.0},
    ]
    paths = dive_profiles.sv_code_depth_table(
        _summary_csv(tmp_path, rows), tmp_path / "out"
    )

    per_dive = pd.read_csv(paths["per_dive_csv_path"])
    assert set(per_dive["depth_interval_m"]) == {"0-200", "200-400", "600-800"}


def test_interval_width_is_configurable(tmp_path):
    rows = [
        {"label": "d1", "ping_time": "t0", "code": 1, "n_cells": 1,
         "code_cells": 1, "dive_depth_m": 150.0, "duration_s": 10.0},
    ]
    paths = dive_profiles.sv_code_depth_table(
        _summary_csv(tmp_path, rows), tmp_path / "out", depth_interval_m=100.0
    )

    per_dive = pd.read_csv(paths["per_dive_csv_path"])
    assert per_dive["depth_interval_m"].iloc[0] == "100-200"


def test_durations_are_reported_in_minutes(tmp_path):
    rows = [
        {"label": "d1", "ping_time": f"t{i}", "code": 1, "n_cells": 1,
         "code_cells": 1, "dive_depth_m": 100.0, "duration_s": 10.0}
        for i in range(6)
    ]
    paths = dive_profiles.sv_code_depth_table(
        _summary_csv(tmp_path, rows), tmp_path / "out"
    )

    totals = pd.read_csv(paths["totals_csv_path"])
    assert totals["duration_min"].iloc[0] == pytest.approx(1.0)
    assert totals["percent_of_dive"].iloc[0] == pytest.approx(100.0)


def test_multiple_dives_are_kept_separate(tmp_path):
    rows = [
        {"label": "d1", "ping_time": "t0", "code": 1, "n_cells": 1,
         "code_cells": 1, "dive_depth_m": 100.0, "duration_s": 30.0},
        {"label": "d2", "ping_time": "t1", "code": 2, "n_cells": 1,
         "code_cells": 1, "dive_depth_m": 100.0, "duration_s": 10.0},
    ]
    paths = dive_profiles.sv_code_depth_table(
        _summary_csv(tmp_path, rows), tmp_path / "out"
    )

    totals = pd.read_csv(paths["totals_csv_path"])
    assert set(totals["label"]) == {"d1", "d2"}
    # Each dive is 100% of itself regardless of how long the other one was.
    assert totals["percent_of_dive"].tolist() == pytest.approx([100.0, 100.0])

    combined = pd.read_csv(paths["combined_csv_path"])
    assert combined["percent_of_total"].sum() == pytest.approx(100.0)


def test_empty_summary_raises(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame(
        columns=["label", "ping_time", "code", "dive_depth_m", "duration_s"]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="No ping rows"):
        dive_profiles.sv_code_depth_table(path, tmp_path / "out")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_codes_flow_through_to_the_depth_table(tmp_path):
    labels = np.tile([0.0, 1.0, 1.0, 2.0, 0.0], (6, 1))
    ds = _ds(labels=labels)

    codes = dive_profiles.generate_sv_codes(
        ds, [_window(ds)], tmp_path / "codes", ping_time_bin_s=PING_BIN_S
    )
    tables = dive_profiles.sv_code_depth_table(
        codes["summary_csv_path"], tmp_path / "tables"
    )

    totals = pd.read_csv(tables["totals_csv_path"])
    # Six pings of 10 s, all dominated by code 1, all at the 300 m fitted depth.
    assert totals["code"].tolist() == [1]
    assert totals["duration_min"].iloc[0] == pytest.approx(1.0)
    per_dive = pd.read_csv(tables["per_dive_csv_path"])
    assert per_dive["depth_interval_m"].tolist() == ["200-400"]


# ---------------------------------------------------------------------------
# Regressions found in self-review
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deepest", [200.0, 400.0, 600.0, 1800.0])
def test_a_depth_on_a_bin_edge_is_not_dropped(tmp_path, deepest):
    """Left-closed bins need a bin ABOVE an exact multiple, or the ping vanishes.

    The original built edges with ceil, so a dive bottoming at exactly 400 m on
    200 m intervals got edges [0, 200, 400] and pd.cut put 400 outside every
    bin. The ping disappeared and the percentages summed to less than 100.
    """
    rows = [
        {"label": "d1", "ping_time": "t0", "code": 1, "n_cells": 1,
         "code_cells": 1, "dive_depth_m": deepest, "duration_s": 10.0},
        {"label": "d1", "ping_time": "t1", "code": 1, "n_cells": 1,
         "code_cells": 1, "dive_depth_m": 50.0, "duration_s": 10.0},
    ]
    paths = dive_profiles.sv_code_depth_table(
        _summary_csv(tmp_path, rows), tmp_path / "out"
    )

    per_dive = pd.read_csv(paths["per_dive_csv_path"])
    assert per_dive["duration_s"].sum() == 20.0
    assert per_dive["percent_of_dive"].sum() == pytest.approx(100.0)


def test_a_depth_beyond_max_depth_m_is_an_error_not_a_silent_drop(tmp_path):
    rows = [
        {"label": "d1", "ping_time": "t0", "code": 1, "n_cells": 1,
         "code_cells": 1, "dive_depth_m": 900.0, "duration_s": 10.0},
    ]

    with pytest.raises(ValueError, match="fall outside the depth bins"):
        dive_profiles.sv_code_depth_table(
            _summary_csv(tmp_path, rows), tmp_path / "out", max_depth_m=400.0
        )


def _masked_ds(n_pings=4):
    """MVBS as the recipe really produces it: five channels, three masked away.

    create_frequency_mask sets the unused channels to NaN and leaves them in
    place, so a fixture with only the two analysis channels cannot catch a
    column-selection bug.
    """
    freqs = [18000.0, 38000.0, 70000.0, 120000.0, 200000.0]
    ping = pd.date_range("2016-07-07T19:49:00", periods=n_pings, freq="10s")
    depth = np.array([200.0, 300.0, 400.0])
    sv = np.stack([np.full((n_pings, 3), -70.0 + i) for i in range(5)])
    sv[2:] = np.nan
    return xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "depth"), sv),
            "frequency_nominal": (("channel",), np.array(freqs)),
            "ml_dataset_hdbscan_results_grid": (
                ("ping_time", "depth"), np.ones((n_pings, 3))),
            "dive_fit": (("ping_time",), np.full(n_pings, 300.0)),
            "dive_u99": (("ping_time",), np.full(n_pings, 180.0)),
            "dive_l99": (("ping_time",), np.full(n_pings, 420.0)),
        },
        coords={"channel": [f"ch{i}" for i in range(5)],
                "ping_time": ping, "depth": depth},
    )


def test_masked_channels_do_not_become_all_nan_columns(tmp_path):
    """The reference layout is two Sv columns, not five with three empty."""
    ds = _masked_ds()
    out = dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path)

    frame = pd.read_csv(out["code_csv_paths"][0])
    assert list(frame.columns) == [
        "DateTime", "depth_m", "18kHz_Sv", "38kHz_Sv", "code"
    ]


def test_the_window_frequency_list_selects_the_columns(tmp_path):
    """plan_dive_datasets already carries the analysis frequencies; use them."""
    ds = _masked_ds()
    window = dict(_window(ds), frequencies_khz=[38.0, 18.0])

    out = dive_profiles.generate_sv_codes(ds, [window], tmp_path)

    frame = pd.read_csv(out["code_csv_paths"][0])
    # Column order follows the requested order, not the channel order.
    assert list(frame.columns) == [
        "DateTime", "depth_m", "38kHz_Sv", "18kHz_Sv", "code"
    ]


def test_an_absent_frequency_is_an_error(tmp_path):
    ds = _masked_ds()
    window = dict(_window(ds), frequencies_khz=[333.0])

    with pytest.raises(ValueError, match="no channel at 333.0 kHz"):
        dive_profiles.generate_sv_codes(ds, [window], tmp_path)


def test_an_explicit_override_beats_the_window(tmp_path):
    ds = _masked_ds()
    window = dict(_window(ds), frequencies_khz=[18.0, 38.0])

    out = dive_profiles.generate_sv_codes(
        ds, [window], tmp_path, frequencies_khz=[18.0]
    )

    frame = pd.read_csv(out["code_csv_paths"][0])
    assert list(frame.columns) == ["DateTime", "depth_m", "18kHz_Sv", "code"]


# ---------------------------------------------------------------------------
# Pings the dive spent somewhere but that produced no code
# ---------------------------------------------------------------------------


def _half_coded_ds(n_pings=6):
    """A dive whose second half produces no coded cell at all."""
    labels = np.zeros((n_pings, 5))
    labels[n_pings // 2:] = np.nan
    return _ds(n_pings=n_pings, labels=labels)


def test_a_ping_with_no_coded_cells_still_gets_a_row(tmp_path):
    ds = _half_coded_ds()
    out = dive_profiles.generate_sv_codes(
        ds, [_window(ds)], tmp_path, ping_time_bin_s=PING_BIN_S
    )

    summary = pd.read_csv(out["summary_csv_path"])
    assert len(summary) == 6
    assert summary["duration_s"].sum() == 60.0
    uncoded = summary[summary["code"] == dive_profiles.UNCODED]
    assert len(uncoded) == 3
    assert (uncoded["n_cells"] == 0).all()
    assert (uncoded["code_cells"] == 0).all()
    # The whale was still somewhere, so the depth has to survive for the table.
    assert uncoded["dive_depth_m"].notna().all()


def test_the_denominator_is_the_dive_not_the_pings_that_survived(tmp_path):
    """Regression: half a dive coded must not report as 100% of the dive.

    The summary used to hold only the pings a coded cell fell in, so the
    percentages were a fraction of those rather than of the dive. On the first
    live run that turned 20 of 104 ping bins into "100% of the dive".
    """
    ds = _half_coded_ds()
    out = dive_profiles.generate_sv_codes(
        ds, [_window(ds)], tmp_path, ping_time_bin_s=PING_BIN_S
    )
    tables = dive_profiles.sv_code_depth_table(out["summary_csv_path"], tmp_path)

    totals = pd.read_csv(tables["totals_csv_path"])
    share = dict(zip(totals["code"], totals["percent_of_dive"]))
    assert share[0] == pytest.approx(50.0)
    assert share[dive_profiles.UNCODED] == pytest.approx(50.0)
    assert totals["percent_of_dive"].sum() == pytest.approx(100.0)
    assert totals["duration_s"].sum() == pytest.approx(60.0)


def test_uncoded_pings_land_in_their_own_depth_interval(tmp_path):
    """They carry the fitted depth, so they bin like any other ping."""
    ds = _half_coded_ds()
    out = dive_profiles.generate_sv_codes(
        ds, [_window(ds)], tmp_path, ping_time_bin_s=PING_BIN_S
    )
    tables = dive_profiles.sv_code_depth_table(out["summary_csv_path"], tmp_path)

    per_dive = pd.read_csv(tables["per_dive_csv_path"])
    uncoded = per_dive[per_dive["code"] == dive_profiles.UNCODED]
    assert list(uncoded["depth_interval_m"]) == ["200-400"]
    assert uncoded["percent_of_dive"].iloc[0] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Dive lines read per dive, from the window's own files
# ---------------------------------------------------------------------------


def _write_flat_evl(path, times, depth):
    """A minimal Echoview .evl holding *depth* at every one of *times*."""
    lines = ["EVBD 3 15.1.65.0", str(len(times))]
    for when in times:
        stamp = pd.Timestamp(when)
        lines.append(
            f"{stamp:%Y%m%d} {stamp:%H%M%S}{stamp.microsecond // 100:04d}  "
            f"{depth} 3"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def _window_with_lines(tmp_path, ds, label, start, end, fit, half_band):
    """A window naming three flat line files spanning [start, end]."""
    times = ds["ping_time"].sel(ping_time=slice(start, end)).values
    return {
        "label": label,
        "start": pd.Timestamp(start).isoformat(),
        "end": pd.Timestamp(end).isoformat(),
        "dive_fit_evl": _write_flat_evl(tmp_path / f"{label}_fit.evl", times, fit).as_posix(),
        "dive_u99_evl": _write_flat_evl(tmp_path / f"{label}_u99.evl", times, fit - half_band).as_posix(),
        "dive_l99_evl": _write_flat_evl(tmp_path / f"{label}_l99.evl", times, fit + half_band).as_posix(),
    }


def test_lines_are_read_from_the_window_when_the_dataset_has_none(tmp_path):
    """Same answer whether the lines ride in on ds or come from the files."""
    with_lines = _ds()
    without = _ds(with_lines=False)
    t0 = pd.Timestamp(without["ping_time"].values[0])
    t1 = pd.Timestamp(without["ping_time"].values[-1])
    window = _window_with_lines(tmp_path, without, "SWD_files", t0, t1, 300.0, 120.0)

    a = dive_profiles.generate_sv_codes(with_lines, [_window(with_lines, "SWD_files")], tmp_path / "a")
    b = dive_profiles.generate_sv_codes(without, [window], tmp_path / "b")

    pd.testing.assert_frame_equal(
        pd.read_csv(a["code_csv_paths"][0]), pd.read_csv(b["code_csv_paths"][0])
    )


def test_overlapping_dives_each_get_their_own_lines(tmp_path):
    """Two dives sharing minutes are coded over their full windows, separately.

    This is HB1603: four of the 13 dives overlap another in time. The combined
    dataset holds each cell once and carries no dive line, and each dive reads
    its own lines onto its own slice. The two bands differ, so the coded depths
    show which dive's line was used.
    """
    ds = _ds(n_pings=12, with_lines=False, start="2016-07-25T21:20:00")
    t = ds["ping_time"].values
    dive_a = _window_with_lines(tmp_path, ds, "SWD_A", t[0], t[6], 300.0, 120.0)
    dive_b = _window_with_lines(tmp_path, ds, "SWD_B", t[3], t[11], 200.0, 120.0)

    out = dive_profiles.generate_sv_codes(ds, [dive_a, dive_b], tmp_path / "codes")

    assert out["dive_labels"] == ["SWD_A", "SWD_B"]
    frame_a = pd.read_csv(out["code_csv_paths"][0])
    frame_b = pd.read_csv(out["code_csv_paths"][1])
    assert sorted(frame_a["depth_m"].unique()) == [200.0, 300.0, 400.0]
    assert sorted(frame_b["depth_m"].unique()) == [100.0, 200.0, 300.0]
    # Each dive covers its whole window, overlap included, in the summary.
    summary = pd.read_csv(out["summary_csv_path"])
    assert (summary["label"] == "SWD_A").sum() == 7
    assert (summary["label"] == "SWD_B").sum() == 9


def test_a_window_without_a_line_file_is_still_an_error(tmp_path):
    ds = _ds(with_lines=False)
    with pytest.raises(KeyError, match="names no file under 'dive_fit_evl'"):
        dive_profiles.generate_sv_codes(ds, [_window(ds)], tmp_path)
