# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Unit tests for reading an Echoview .evl seafloor line onto a ping grid."""

import numpy as np
import pytest
import xarray as xr

from aa_si_utils import utils


def _write_evl(tmp_path, points, name="line.evl", declared=None, header=None):
    """Write an Echoview .evl file from (time_of_day, depth, status) triples.

    Times are 'HHMMSSssss' strings on 2024-01-01; ``declared`` overrides the
    point count in the header so truncation can be simulated.
    """
    header = header or "EVBD 3 15.1.65.0"
    count = len(points) if declared is None else declared
    lines = [header, str(count)]
    lines += [f"20240101 {when}  {depth} {status}" for when, depth, status in points]
    path = tmp_path / name
    # Echoview writes a BOM.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def _make_ds_sv(seconds=(0, 1, 2), with_depth=True, transducer_depth_m=5.0):
    """Sv dataset on 2024-01-01T00:00:<seconds>, optionally depth-enriched."""
    channels = np.array(["38000", "120000"])
    ping_time = np.array(
        [f"2024-01-01T00:00:{second:02d}" for second in seconds],
        dtype="datetime64[ns]",
    )
    range_sample = np.arange(4)
    echo_range_values = np.array([0.0, 10.0, 20.0, 30.0])
    echo_range = np.broadcast_to(
        echo_range_values, (len(channels), len(ping_time), len(range_sample))
    )

    data_vars = {
        "Sv": (("channel", "ping_time", "range_sample"), np.zeros_like(echo_range)),
        "echo_range": (("channel", "ping_time", "range_sample"), echo_range),
    }
    if with_depth:
        # What add_depth produces: depth = transducer_depth + echo_range.
        data_vars["depth"] = (
            ("channel", "ping_time", "range_sample"),
            echo_range + transducer_depth_m,
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "channel": channels,
            "ping_time": ping_time,
            "range_sample": range_sample,
        },
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_evl_reads_time_depth_and_status(tmp_path):
    path = _write_evl(
        tmp_path,
        [("0000000000", "20.0", "3"), ("0000010000", "21.5", "1")],
    )

    points = utils._parse_evl(path)

    assert list(points.columns) == ["time", "depth", "status"]
    assert points["time"].tolist() == [
        np.datetime64("2024-01-01T00:00:00"),
        np.datetime64("2024-01-01T00:00:01"),
    ]
    np.testing.assert_allclose(points["depth"].to_numpy(), [20.0, 21.5])
    np.testing.assert_array_equal(points["status"].to_numpy(), [3, 1])


def test_parse_evl_reads_tenth_millisecond_fraction(tmp_path):
    # Echoview's time field is HHMMSS plus four digits of 1/10000 s.
    path = _write_evl(tmp_path, [("0051442810", "68.5", "3")])

    points = utils._parse_evl(path)

    assert points["time"].iloc[0] == np.datetime64("2024-01-01T00:51:44.281000")


def test_parse_evl_replaces_no_bottom_sentinel_with_nan(tmp_path):
    path = _write_evl(
        tmp_path,
        [("0000000000", "20.0", "3"), ("0000010000", "-10000.99", "0")],
    )

    points = utils._parse_evl(path)

    assert points["depth"].iloc[0] == 20.0
    assert np.isnan(points["depth"].iloc[1])


def test_parse_evl_rejects_truncated_file(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", "20.0", "3")], declared=5)

    with pytest.raises(ValueError, match="declares 5 points but contains 1"):
        utils._parse_evl(path)


def test_parse_evl_rejects_malformed_point_line(tmp_path):
    path = tmp_path / "bad.evl"
    path.write_text("EVBD 3 15.1.65.0\n1\n20240101 0000000000 20.0\n", encoding="utf-8-sig")

    with pytest.raises(ValueError, match="4 fields"):
        utils._parse_evl(path)


def test_parse_evl_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="EVL line file not found"):
        utils._parse_evl(tmp_path / "absent.evl")


# ---------------------------------------------------------------------------
# Alignment to ping_time
# ---------------------------------------------------------------------------


def test_read_seafloor_line_matches_ping_time_exactly(tmp_path):
    ds_Sv = _make_ds_sv()
    path = _write_evl(
        tmp_path,
        [
            ("0000000000", "20.0", "3"),
            ("0000010000", "22.0", "3"),
            ("0000020000", "24.0", "3"),
        ],
    )

    line = utils.read_seafloor_line_evl(ds_Sv, path)

    assert line.name == "seafloor_depth"
    assert line.dims == ("ping_time",)
    xr.testing.assert_identical(line["ping_time"], ds_Sv["ping_time"])
    np.testing.assert_allclose(line.values, [20.0, 22.0, 24.0])
    assert line.attrs["units"] == "m"
    assert line.attrs["vertical_reference"] == "surface"
    assert line.attrs["ping_coverage"] == 1.0


def test_read_seafloor_line_interpolates_between_points(tmp_path):
    ds_Sv = _make_ds_sv()
    # Line points straddle the pings: interpolation must be linear in time.
    path = _write_evl(
        tmp_path, [("0000000000", "20.0", "3"), ("0000040000", "28.0", "3")]
    )

    line = utils.read_seafloor_line_evl(ds_Sv, path)

    np.testing.assert_allclose(line.values, [20.0, 22.0, 24.0])


def test_no_bottom_sentinel_nans_only_the_pings_it_brackets(tmp_path):
    ds_Sv = _make_ds_sv(seconds=(0, 1, 2, 3, 4))
    path = _write_evl(
        tmp_path,
        [
            ("0000000000", "20.0", "3"),
            ("0000020000", "-10000.99", "0"),
            ("0000040000", "24.0", "3"),
        ],
    )

    line = utils.read_seafloor_line_evl(ds_Sv, path)

    # Every ping is bracketed by the NaN point except the two endpoints, which
    # np.interp resolves conservatively -- what matters is that the finite
    # values survive at the far ends and the middle is unknown.
    assert np.isnan(line.values[1:4]).all()
    assert line.values[0] == 20.0
    assert line.values[4] == 24.0


# ---------------------------------------------------------------------------
# edge_extend_s / max_gap_s
# ---------------------------------------------------------------------------


def test_pings_outside_the_line_span_are_nan_by_default(tmp_path):
    ds_Sv = _make_ds_sv(seconds=(0, 1, 2))
    # Line covers only the middle ping.
    path = _write_evl(tmp_path, [("0000010000", "22.0", "3")])

    line = utils.read_seafloor_line_evl(ds_Sv, path)

    assert np.isnan(line.values[0])
    assert line.values[1] == 22.0
    assert np.isnan(line.values[2])
    assert line.attrs["ping_coverage"] == pytest.approx(1 / 3)


def test_edge_extend_s_holds_the_end_depth_within_tolerance(tmp_path):
    ds_Sv = _make_ds_sv(seconds=(0, 1, 2, 5))
    path = _write_evl(
        tmp_path, [("0000010000", "22.0", "3"), ("0000020000", "24.0", "3")]
    )

    line = utils.read_seafloor_line_evl(ds_Sv, path, edge_extend_s=2.0)

    np.testing.assert_allclose(line.values[:3], [22.0, 22.0, 24.0])
    # 5 s is more than 2 s past the last point at 2 s.
    assert np.isnan(line.values[3])


def test_edge_extend_none_holds_the_end_depth_indefinitely(tmp_path):
    ds_Sv = _make_ds_sv(seconds=(0, 1, 59))
    path = _write_evl(tmp_path, [("0000010000", "22.0", "3")])

    line = utils.read_seafloor_line_evl(ds_Sv, path, edge_extend_s=None)

    np.testing.assert_allclose(line.values, [22.0, 22.0, 22.0])


def test_max_gap_s_nans_pings_inside_an_over_long_hole(tmp_path):
    ds_Sv = _make_ds_sv(seconds=(0, 5, 10))
    # A 10 s hole between the two line points.
    path = _write_evl(
        tmp_path, [("0000000000", "20.0", "3"), ("0000100000", "30.0", "3")]
    )

    line = utils.read_seafloor_line_evl(ds_Sv, path, max_gap_s=4.0)

    # The endpoints sit exactly on line points; only the interior is unknown.
    assert np.isnan(line.values[1])
    np.testing.assert_allclose([line.values[0], line.values[2]], [20.0, 30.0])


def test_max_gap_s_leaves_short_gaps_interpolated(tmp_path):
    ds_Sv = _make_ds_sv(seconds=(0, 1, 2))
    path = _write_evl(
        tmp_path, [("0000000000", "20.0", "3"), ("0000020000", "24.0", "3")]
    )

    line = utils.read_seafloor_line_evl(ds_Sv, path, max_gap_s=5.0)

    np.testing.assert_allclose(line.values, [20.0, 22.0, 24.0])


# ---------------------------------------------------------------------------
# Filtering, vertical reference, offsets, coverage guard
# ---------------------------------------------------------------------------


def test_min_status_drops_unverified_points(tmp_path):
    ds_Sv = _make_ds_sv(seconds=(0, 1, 2))
    path = _write_evl(
        tmp_path,
        [
            ("0000000000", "20.0", "3"),
            ("0000010000", "99.0", "1"),  # unverified: should be ignored
            ("0000020000", "24.0", "3"),
        ],
    )

    kept = utils.read_seafloor_line_evl(ds_Sv, path)
    filtered = utils.read_seafloor_line_evl(ds_Sv, path, min_status=3)

    assert kept.values[1] == 99.0
    # With the bad point gone, the middle ping interpolates across it.
    np.testing.assert_allclose(filtered.values, [20.0, 22.0, 24.0])


def test_min_status_filtering_everything_raises(tmp_path):
    ds_Sv = _make_ds_sv()
    path = _write_evl(tmp_path, [("0000000000", "20.0", "1")])

    with pytest.raises(ValueError, match="no line points with status >= 3"):
        utils.read_seafloor_line_evl(ds_Sv, path, min_status=3)


def test_transducer_referenced_line_is_promoted_to_depth(tmp_path):
    ds_Sv = _make_ds_sv(transducer_depth_m=5.0)
    path = _write_evl(
        tmp_path,
        [
            ("0000000000", "20.0", "3"),
            ("0000010000", "20.0", "3"),
            ("0000020000", "20.0", "3"),
        ],
    )

    line = utils.read_seafloor_line_evl(
        ds_Sv, path, vertical_reference="transducer"
    )

    # ds_Sv carries depth, so a beam-referenced line gains the transducer depth.
    np.testing.assert_allclose(line.values, [25.0, 25.0, 25.0])
    assert line.attrs["vertical_reference"] == "surface"


def test_transducer_referenced_line_stays_as_is_without_depth(tmp_path):
    ds_Sv = _make_ds_sv(with_depth=False)
    path = _write_evl(
        tmp_path,
        [
            ("0000000000", "20.0", "3"),
            ("0000010000", "20.0", "3"),
            ("0000020000", "20.0", "3"),
        ],
    )

    line = utils.read_seafloor_line_evl(
        ds_Sv, path, vertical_reference="transducer"
    )

    np.testing.assert_allclose(line.values, [20.0, 20.0, 20.0])
    assert line.attrs["range_var"] == "echo_range"


def test_surface_reference_without_depth_variable_raises(tmp_path):
    ds_Sv = _make_ds_sv(with_depth=False)
    path = _write_evl(tmp_path, [("0000000000", "20.0", "3")])

    with pytest.raises(ValueError, match="ep_add_depth"):
        utils.read_seafloor_line_evl(ds_Sv, path)


def test_unknown_vertical_reference_raises(tmp_path):
    ds_Sv = _make_ds_sv()
    path = _write_evl(tmp_path, [("0000000000", "20.0", "3")])

    with pytest.raises(ValueError, match="must be 'surface' or 'transducer'"):
        utils.read_seafloor_line_evl(ds_Sv, path, vertical_reference="seabed")


def test_depth_offset_shifts_the_line(tmp_path):
    ds_Sv = _make_ds_sv()
    path = _write_evl(
        tmp_path,
        [
            ("0000000000", "20.0", "3"),
            ("0000010000", "20.0", "3"),
            ("0000020000", "20.0", "3"),
        ],
    )

    line = utils.read_seafloor_line_evl(ds_Sv, path, depth_offset_m=1.5)

    np.testing.assert_allclose(line.values, [21.5, 21.5, 21.5])


def test_min_coverage_raises_when_the_line_falls_short(tmp_path):
    ds_Sv = _make_ds_sv(seconds=(0, 1, 2))
    path = _write_evl(tmp_path, [("0000010000", "22.0", "3")])

    with pytest.raises(ValueError, match="covers 33.3% of pings"):
        utils.read_seafloor_line_evl(ds_Sv, path, min_coverage=0.9)


# ---------------------------------------------------------------------------
# Contract with create_seafloor_mask
# ---------------------------------------------------------------------------


def test_line_is_a_drop_in_for_create_seafloor_mask(tmp_path):
    ds_Sv = _make_ds_sv(transducer_depth_m=5.0)
    path = _write_evl(
        tmp_path,
        [
            ("0000000000", "25.0", "3"),
            ("0000010000", "25.0", "3"),
            ("0000020000", "25.0", "3"),
        ],
    )

    line = utils.read_seafloor_line_evl(ds_Sv, path)
    mask = utils.create_seafloor_mask(ds_Sv, line)

    assert mask.dims == ds_Sv["Sv"].dims
    assert mask.dtype == bool
    # depth is 5, 15, 25, 35; a 25 m seafloor keeps the first three samples.
    np.testing.assert_array_equal(
        mask.isel(channel=0, ping_time=0).values,
        np.array([True, True, True, False]),
    )


def test_uncovered_pings_mask_out_entirely(tmp_path):
    ds_Sv = _make_ds_sv(seconds=(0, 1, 2), transducer_depth_m=5.0)
    # The line covers only the middle ping.
    path = _write_evl(tmp_path, [("0000010000", "25.0", "3")])

    line = utils.read_seafloor_line_evl(ds_Sv, path)
    mask = utils.create_seafloor_mask(ds_Sv, line)

    # This is the documented hazard: a NaN seafloor rejects the whole ping.
    assert not mask.isel(channel=0, ping_time=0).values.any()
    assert mask.isel(channel=0, ping_time=1).values.any()
    assert not mask.isel(channel=0, ping_time=2).values.any()
