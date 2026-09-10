# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Unit tests for generic .evl line reading and dive-profile line overlays."""

import numpy as np
import pytest
import xarray as xr

from aa_si_utils import utils

from test_seafloor_line_evl import _make_ds_sv, _write_evl


# ---------------------------------------------------------------------------
# read_line_evl: the generalized core
# ---------------------------------------------------------------------------


def test_read_line_evl_names_the_array_and_its_long_name(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3), ("0000020000", 60.0, 3)])
    ds_Sv = _make_ds_sv()

    line = utils.read_line_evl(ds_Sv, path, line_name="dive_fit")

    assert line.name == "dive_fit"
    assert line.attrs["long_name"] == "dive_fit from Echoview line file"
    assert line.dims == ("ping_time",)
    np.testing.assert_array_equal(line["ping_time"].values, ds_Sv["ping_time"].values)


def test_read_line_evl_long_name_override(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3), ("0000020000", 60.0, 3)])

    line = utils.read_line_evl(
        _make_ds_sv(), path, line_name="dive_fit", long_name="Fitted dive depth"
    )

    assert line.attrs["long_name"] == "Fitted dive depth"


def test_read_line_evl_interpolates_like_the_seafloor_reader(tmp_path):
    """The seafloor reader is now a wrapper, so the two must agree exactly."""
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3), ("0000020000", 60.0, 3)])
    ds_Sv = _make_ds_sv()

    generic = utils.read_line_evl(ds_Sv, path, line_name="seafloor_depth")
    seafloor = utils.read_seafloor_line_evl(ds_Sv, path)

    np.testing.assert_allclose(generic.values, seafloor.values)
    assert generic.attrs["ping_coverage"] == seafloor.attrs["ping_coverage"]
    assert generic.attrs["range_var"] == seafloor.attrs["range_var"]


def test_read_line_evl_min_coverage_names_the_line(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3)])

    with pytest.raises(ValueError, match="Line 'dive_fit' covers"):
        utils.read_line_evl(
            _make_ds_sv(), path, line_name="dive_fit", min_coverage=0.9
        )


def test_seafloor_wrapper_still_warns_about_masking(tmp_path, capsys):
    """The create_seafloor_mask warning belongs to the seafloor wrapper only."""
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3)])
    ds_Sv = _make_ds_sv()

    utils.read_line_evl(ds_Sv, path, line_name="dive_fit")
    assert "create_seafloor_mask" not in capsys.readouterr().out

    utils.read_seafloor_line_evl(ds_Sv, path)
    assert "create_seafloor_mask" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# add_line_from_evl
# ---------------------------------------------------------------------------


def test_add_line_from_evl_adds_a_ping_time_variable(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3), ("0000020000", 60.0, 3)])
    ds_Sv = _make_ds_sv()

    out = utils.add_line_from_evl(ds_Sv, path, "dive_fit")

    assert "dive_fit" in out
    assert out["dive_fit"].dims == ("ping_time",)
    np.testing.assert_allclose(out["dive_fit"].values, [40.0, 50.0, 60.0])


def test_add_line_from_evl_does_not_mutate_the_input(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3), ("0000020000", 60.0, 3)])
    ds_Sv = _make_ds_sv()

    utils.add_line_from_evl(ds_Sv, path, "dive_fit")

    assert "dive_fit" not in ds_Sv


def test_dive_profile_triplet_attaches_as_three_variables(tmp_path):
    """The real shape of a dive profile: fit plus two confidence bounds."""
    fit = _write_evl(
        tmp_path, [("0000000000", 50.0, 3), ("0000020000", 70.0, 3)], name="fit.evl"
    )
    upper = _write_evl(
        tmp_path, [("0000000000", 40.0, 3), ("0000020000", 60.0, 3)], name="u99.evl"
    )
    lower = _write_evl(
        tmp_path, [("0000000000", 60.0, 3), ("0000020000", 80.0, 3)], name="l99.evl"
    )
    ds = _make_ds_sv()

    for path, name in ((fit, "dive_fit"), (upper, "dive_u99"), (lower, "dive_l99")):
        ds = utils.add_line_from_evl(ds, path, name)

    assert {"dive_fit", "dive_u99", "dive_l99"} <= set(ds.data_vars)
    # The upper bound is shallower than the fit, which is shallower than the lower.
    assert np.all(ds["dive_u99"].values < ds["dive_fit"].values)
    assert np.all(ds["dive_fit"].values < ds["dive_l99"].values)


# ---------------------------------------------------------------------------
# select_ping_time_range's window form
# ---------------------------------------------------------------------------


def _ds_over_seconds(n=10):
    ping_time = np.array(
        [f"2024-01-01T00:00:{second:02d}" for second in range(n)],
        dtype="datetime64[ns]",
    )
    return xr.Dataset(
        {"Sv": (("ping_time",), np.arange(n, dtype=float))},
        coords={"ping_time": ping_time},
    )


def test_window_form_matches_the_range_form():
    ds = _ds_over_seconds()
    window = {"start": "2024-01-01T00:00:02", "end": "2024-01-01T00:00:05"}

    by_window = utils.select_ping_time_range(ds, window=window)
    by_range = utils.select_ping_time_range(ds, start=window["start"], end=window["end"])

    xr.testing.assert_identical(by_window, by_range)
    assert by_window.sizes["ping_time"] == 4


def test_window_form_ignores_extra_keys():
    """A window dict carries a whole instance's context, not just the bounds."""
    ds = _ds_over_seconds()
    window = {
        "label": "SWD_20160707-OE20",
        "start": "2024-01-01T00:00:02",
        "end": "2024-01-01T00:00:03",
        "raw_files": ["a.raw", "b.raw"],
        "dive_profile_evl": "fit.evl",
    }

    assert utils.select_ping_time_range(ds, window=window).sizes["ping_time"] == 2


def test_window_form_prints_the_label(capsys):
    ds = _ds_over_seconds()
    window = {"label": "OE20", "start": None, "end": "2024-01-01T00:00:03"}

    utils.select_ping_time_range(ds, window=window)

    assert "OE20" in capsys.readouterr().out


def test_window_form_allows_an_open_side():
    ds = _ds_over_seconds()

    assert utils.select_ping_time_range(
        ds, window={"start": "2024-01-01T00:00:07"}
    ).sizes["ping_time"] == 3
    assert utils.select_ping_time_range(
        ds, window={"end": "2024-01-01T00:00:02"}
    ).sizes["ping_time"] == 3


def test_window_form_rejects_a_non_mapping():
    with pytest.raises(TypeError, match="window must be a mapping"):
        utils.select_ping_time_range(_ds_over_seconds(), window=["start", "end"])


def test_window_form_rejects_a_dict_without_bounds():
    with pytest.raises(ValueError, match="neither a 'start' nor an 'end'"):
        utils.select_ping_time_range(_ds_over_seconds(), window={"label": "OE20"})


def test_window_form_propagates_an_empty_window():
    ds = _ds_over_seconds()
    window = {"start": "2024-01-02T00:00:00", "end": "2024-01-02T00:00:05"}

    with pytest.raises(ValueError, match="selects no pings"):
        utils.select_ping_time_range(ds, window=window)


def test_window_and_explicit_bounds_together_are_rejected():
    """Silently preferring one over the other would hide a wiring mistake."""
    ds = _ds_over_seconds()

    with pytest.raises(ValueError, match="not both"):
        utils.select_ping_time_range(
            ds, start="2024-01-01T00:00:02", window={"start": "2024-01-01T00:00:05"}
        )


# ---------------------------------------------------------------------------
# add_line_overlay: one op over both line formats
# ---------------------------------------------------------------------------


def _write_click_csv(tmp_path, name="clicks.csv"):
    """The R-written click file, which carries a whole profile in one file."""
    path = tmp_path / name
    path.write_text(
        '"","Date_UTC","ClickTime_UTC","clickDepth_m","fit_line","lwr_CI_99","upr_CI_99"\n'
        '"1","2024-01-01","2024-01-01_00:00:00",40.0,41.0,39.0,43.0\n'
        '"2","2024-01-01","2024-01-01_00:00:01",50.0,51.0,49.0,53.0\n'
        '"3","2024-01-01","2024-01-01_00:00:02",60.0,61.0,59.0,63.0\n',
        encoding="utf-8",
    )
    return path


def test_add_line_overlay_reads_an_evl_as_one_variable(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3), ("0000020000", 60.0, 3)])

    out = utils.add_line_overlay(_make_ds_sv(), path, line_name="dive_fit")

    assert "dive_fit" in out
    np.testing.assert_allclose(out["dive_fit"].values, [40.0, 50.0, 60.0])


def test_add_line_overlay_reads_a_csv_as_four_variables(tmp_path):
    """A click CSV holds the profile and both bounds, so line_name is a prefix."""
    path = _write_click_csv(tmp_path)

    out = utils.add_line_overlay(_make_ds_sv(), path, line_name="sw_dive")

    for suffix in ("depth", "fit", "lower_ci", "upper_ci"):
        assert f"sw_dive_{suffix}" in out, suffix


def test_add_line_overlay_matches_calling_the_readers_directly(tmp_path):
    """The merged op must be a pass-through, not a reimplementation."""
    evl = _write_evl(tmp_path, [("0000000000", 40.0, 3), ("0000020000", 60.0, 3)])
    csv = _write_click_csv(tmp_path)
    ds = _make_ds_sv()

    xr.testing.assert_identical(
        utils.add_line_overlay(ds, evl, line_name="l"),
        utils.add_line_from_evl(ds, evl, "l"),
    )
    xr.testing.assert_identical(
        utils.add_line_overlay(ds, csv, line_name="p"),
        utils.add_dive_profile_to_dataset(ds, csv, dive_profile_name="p"),
    )


def test_add_line_overlay_format_can_be_forced(tmp_path):
    """An .evl named oddly still reads as EVL when told to."""
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3), ("0000020000", 60.0, 3)],
                      name="line.dat")

    out = utils.add_line_overlay(
        _make_ds_sv(), path, line_name="dive_fit", file_format="evl"
    )

    assert "dive_fit" in out


def test_add_line_overlay_rejects_an_unknown_format(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3)])

    with pytest.raises(ValueError, match="file_format must be"):
        utils.add_line_overlay(
            _make_ds_sv(), path, line_name="l", file_format="netcdf"
        )


def test_add_line_overlay_auto_detects_by_extension(tmp_path):
    assert utils._resolve_line_format("a/b/line.evl", "auto") == "evl"
    assert utils._resolve_line_format("a/b/clicks.csv", "auto") == "csv"
    assert utils._resolve_line_format("a/b/CLICKS.CSV", "auto") == "csv"
    # A folder or list is only ever the EVL reader's shape.
    assert utils._resolve_line_format("a/b/seabed_lines", "auto") == "evl"
    assert utils._resolve_line_format(["a/x.evl", "a/y.evl"], "auto") == "evl"


# ---------------------------------------------------------------------------
# read_seafloor_line_evl's window form
# ---------------------------------------------------------------------------


def test_seafloor_window_form_matches_the_path_form(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3), ("0000020000", 60.0, 3)])
    ds_Sv = _make_ds_sv()

    by_path = utils.read_seafloor_line_evl(ds_Sv, path)
    by_window = utils.read_seafloor_line_evl(
        ds_Sv, window={"label": "OE20", "seabed_evl": str(path)}
    )

    np.testing.assert_allclose(by_window.values, by_path.values)
    assert by_window.name == "seafloor_depth"


def test_seafloor_window_key_is_configurable(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3), ("0000020000", 60.0, 3)])

    line = utils.read_seafloor_line_evl(
        _make_ds_sv(), window={"bottom": str(path)}, window_key="bottom"
    )

    assert np.isfinite(line.values).all()


def test_seafloor_requires_exactly_one_of_path_or_window(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3)])
    ds_Sv = _make_ds_sv()

    with pytest.raises(ValueError, match="exactly one"):
        utils.read_seafloor_line_evl(ds_Sv)
    with pytest.raises(ValueError, match="exactly one"):
        utils.read_seafloor_line_evl(ds_Sv, path, window={"seabed_evl": str(path)})


def test_seafloor_window_without_the_key_names_the_dive(tmp_path):
    with pytest.raises(ValueError, match="OE20"):
        utils.read_seafloor_line_evl(
            _make_ds_sv(), window={"label": "OE20", "start": "2024-01-01"}
        )


def test_seafloor_window_rejects_a_non_mapping():
    with pytest.raises(TypeError, match="window must be a mapping"):
        utils.read_seafloor_line_evl(_make_ds_sv(), window=["a.evl"])


def test_seafloor_window_supplies_the_folder_selection_bounds(tmp_path):
    """A window narrows a folder of part-day exports without repeating bounds."""
    folder = tmp_path / "seabed"
    folder.mkdir()
    _write_evl(folder, [("0000000000", 40.0, 3), ("0000020000", 60.0, 3)],
               name="d20240101_t000000-t235959_seabed.evl", date="20240101")
    _write_evl(folder, [("0000000000", 900.0, 3), ("0000020000", 900.0, 3)],
               name="d20240202_t000000-t235959_seabed.evl", date="20240202")

    line = utils.read_seafloor_line_evl(
        _make_ds_sv(),
        window={
            "seabed_evl": str(folder),
            "start": "2024-01-01T00:00:00",
            "end": "2024-01-01T00:00:02",
        },
    )

    # Only the January file covers the window, so the February depths are absent.
    assert np.nanmax(line.values) < 100.0


# ---------------------------------------------------------------------------
# add_line_overlay's window form
# ---------------------------------------------------------------------------


def test_overlay_window_form_matches_the_path_form(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3), ("0000020000", 60.0, 3)])
    ds = _make_ds_sv()

    by_path = utils.add_line_overlay(ds, path, line_name="dive_fit")
    by_window = utils.add_line_overlay(
        ds, line_name="dive_fit",
        window={"label": "OE20", "dive_fit_evl": str(path)},
        window_key="dive_fit_evl",
    )

    xr.testing.assert_identical(by_window, by_path)


def test_overlay_window_attaches_a_dive_triplet_in_three_calls(tmp_path):
    """The real shape: three .evl exports chained onto one dataset."""
    window = {"label": "OE20"}
    for key, depth in (("dive_fit_evl", 50.0), ("dive_u99_evl", 40.0),
                       ("dive_l99_evl", 60.0)):
        path = _write_evl(tmp_path, [("0000000000", depth, 3),
                                     ("0000020000", depth + 20.0, 3)],
                          name=f"{key}.evl")
        window[key] = str(path)

    ds = _make_ds_sv()
    for key, name in (("dive_fit_evl", "dive_fit"), ("dive_u99_evl", "dive_u99"),
                      ("dive_l99_evl", "dive_l99")):
        ds = utils.add_line_overlay(ds, line_name=name, window=window,
                                    window_key=key)

    assert {"dive_fit", "dive_u99", "dive_l99"} <= set(ds.data_vars)
    assert np.all(ds["dive_u99"].values < ds["dive_fit"].values)
    assert np.all(ds["dive_fit"].values < ds["dive_l99"].values)


def test_overlay_requires_exactly_one_of_path_or_window(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3)])
    ds = _make_ds_sv()

    with pytest.raises(ValueError, match="exactly one"):
        utils.add_line_overlay(ds, line_name="l")
    with pytest.raises(ValueError, match="exactly one"):
        utils.add_line_overlay(ds, path, line_name="l",
                               window={"a": str(path)}, window_key="a")


def test_overlay_window_requires_a_window_key(tmp_path):
    path = _write_evl(tmp_path, [("0000000000", 40.0, 3)])

    with pytest.raises(ValueError, match="window_key is required"):
        utils.add_line_overlay(
            _make_ds_sv(), line_name="l", window={"dive_fit_evl": str(path)}
        )


def test_overlay_window_missing_the_key_names_the_dive():
    with pytest.raises(ValueError, match="OE20"):
        utils.add_line_overlay(
            _make_ds_sv(), line_name="l",
            window={"label": "OE20", "dive_fit_evl": "x.evl"},
            window_key="dive_l99_evl",
        )


def test_overlay_window_rejects_a_non_mapping():
    with pytest.raises(TypeError, match="window must be a mapping"):
        utils.add_line_overlay(
            _make_ds_sv(), line_name="l", window=["a.evl"], window_key="a"
        )
