# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Fanning in per-segment BINNED products, where labels collide at boundaries.

compute_MVBS and compute_per_cell_statistics anchor their time bins to a global
origin, so a segment boundary falling inside a bin makes the segment on each
side emit that bin, each holding part of its pings. Concatenating leaves two
entries under one label, and every downstream reindex rejects that - it is what
raised InvalidIndexError in add_auxiliary_features 5h42m into an HB1603 run.

A mean of decibels is not the mean of what they measure, so dB variables are
averaged in the linear domain.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from aa_si_utils.utils import concat_datasets

T0 = np.datetime64("2016-06-27T00:00:00")


def _segment(bins, sv_db, cv=None):
    """One segment's binned product, labelled at 10 s bin edges."""
    t = T0 + np.array(bins) * np.timedelta64(10, "s")
    data = {"Sv": (("channel", "ping_time"), np.array([sv_db], dtype=float),
                   {"units": "dB", "long_name": "MVBS"})}
    if cv is not None:
        data["cell_cv"] = (("channel", "ping_time"), np.array([cv], dtype=float))
    return xr.Dataset(data, coords={"channel": ["ch0"], "ping_time": t})


def _at(ds, var, bin_index):
    return float(ds[var].sel(ping_time=T0 + bin_index * np.timedelta64(10, "s"))
                 .isel(channel=0))


def test_without_the_option_duplicates_survive():
    """The default must not change: segments that cannot overlap keep both."""
    merged = concat_datasets([_segment([4, 5], [-80.0, -70.0]),
                              _segment([5, 6], [-60.0, -50.0])], dim="ping_time")

    assert not merged.indexes["ping_time"].is_unique


def test_mean_collapses_the_shared_label():
    merged = concat_datasets([_segment([4, 5], [-80.0, -70.0]),
                              _segment([5, 6], [-60.0, -50.0])],
                             dim="ping_time", on_duplicate="mean")

    assert merged.indexes["ping_time"].is_unique
    assert merged.sizes["ping_time"] == 3


def test_db_is_averaged_in_the_linear_domain():
    merged = concat_datasets([_segment([4, 5], [-80.0, -70.0]),
                              _segment([5, 6], [-60.0, -50.0])],
                             dim="ping_time", on_duplicate="mean")

    expected = 10 * np.log10((10 ** (-70 / 10) + 10 ** (-60 / 10)) / 2)
    assert _at(merged, "Sv", 5) == pytest.approx(expected)
    # The naive decibel mean would be -65.0, which is not the same quantity.
    assert _at(merged, "Sv", 5) != pytest.approx(-65.0)


def test_a_variable_without_db_units_is_averaged_arithmetically():
    merged = concat_datasets([_segment([4, 5], [-80.0, -70.0], cv=[0.1, 0.2]),
                              _segment([5, 6], [-60.0, -50.0], cv=[0.4, 0.6])],
                             dim="ping_time", on_duplicate="mean")

    assert _at(merged, "cell_cv", 5) == pytest.approx(0.3)


def test_labels_that_were_not_duplicated_are_untouched():
    merged = concat_datasets([_segment([4, 5], [-80.0, -70.0]),
                              _segment([5, 6], [-60.0, -50.0])],
                             dim="ping_time", on_duplicate="mean")

    assert _at(merged, "Sv", 4) == -80.0
    assert _at(merged, "Sv", 6) == -50.0


def test_attributes_survive_the_round_trip():
    merged = concat_datasets([_segment([4, 5], [-80.0, -70.0]),
                              _segment([5, 6], [-60.0, -50.0])],
                             dim="ping_time", on_duplicate="mean")

    assert merged["Sv"].attrs["units"] == "dB"
    assert merged["Sv"].attrs["long_name"] == "MVBS"


def test_no_duplicates_is_a_cheap_no_op():
    a, b = _segment([4, 5], [-80.0, -70.0]), _segment([6, 7], [-60.0, -50.0])

    merged = concat_datasets([a, b], dim="ping_time", on_duplicate="mean")

    assert merged.sizes["ping_time"] == 4
    assert _at(merged, "Sv", 4) == -80.0


def test_a_lazy_input_stays_lazy():
    import dask.array as da
    t = T0 + np.arange(4) * np.timedelta64(10, "s")
    t[3] = t[2]
    ds = xr.Dataset(
        {"Sv": (("channel", "ping_time"),
                da.from_array(np.full((1, 4), -70.0), chunks=(1, 2)),
                {"units": "dB"})},
        coords={"channel": ["ch0"], "ping_time": t})

    merged = concat_datasets([ds], dim="ping_time", on_duplicate="mean")

    # A single dataset passes through unchanged (single-item transparency).
    assert merged is ds


def test_an_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="on_duplicate"):
        concat_datasets([_segment([4], [-80.0]), _segment([5], [-70.0])],
                        dim="ping_time", on_duplicate="median")


# ---------------------------------------------------------------------------
# on_duplicate="first": overlapping slices of one store
# ---------------------------------------------------------------------------


def _overlapping_dives():
    """Two dive windows cut from one store, overlapping on two bins.

    Concatenated A then B, so the raw index is both duplicated and
    non-monotonic. Sv is identical on the overlap, as slices of one store are;
    ``dive_fit`` is a per-dive line and genuinely differs.
    """
    import pandas as pd

    t = pd.date_range("2016-07-25T21:20:00", periods=6, freq="10s")
    a = xr.Dataset(
        {"Sv": ("ping_time", [-60.0, -61.0, -62.0, -63.0]),
         "dive_fit": ("ping_time", [100.0, 110.0, 120.0, 130.0])},
        coords={"ping_time": t[:4]},
    )
    b = xr.Dataset(
        {"Sv": ("ping_time", [-62.0, -63.0, -64.0, -65.0]),
         "dive_fit": ("ping_time", [500.0, 510.0, 520.0, 530.0])},
        coords={"ping_time": t[2:]},
    )
    a["Sv"].attrs["units"] = "dB"
    b["Sv"].attrs["units"] = "dB"
    return a, b, t


def test_first_makes_the_index_unique_and_monotonic():
    a, b, t = _overlapping_dives()
    out = concat_datasets([a, b], dim="ping_time", on_duplicate="first")

    index = out.indexes["ping_time"]
    assert index.is_unique
    assert index.is_monotonic_increasing
    assert list(index) == list(t)


def test_first_keeps_identical_values_bit_for_bit():
    """No arithmetic, so no dB round trip: slices of one store come back exact."""
    a, b, _ = _overlapping_dives()
    out = concat_datasets([a, b], dim="ping_time", on_duplicate="first")

    np.testing.assert_array_equal(out["Sv"].values, [-60, -61, -62, -63, -64, -65])
    assert out["Sv"].attrs["units"] == "dB"


def test_first_does_not_average_a_per_segment_variable():
    """The overlap keeps the FIRST dive's line, not a blend of the two.

    That is the right behaviour for a dedup and the wrong place for a
    per-dive variable, which is why the recipe attaches dive lines after the
    fan-in rather than before it. Pinned so the choice is explicit.
    """
    a, b, _ = _overlapping_dives()
    out = concat_datasets([a, b], dim="ping_time", on_duplicate="first")

    np.testing.assert_array_equal(
        out["dive_fit"].values, [100, 110, 120, 130, 520, 530]
    )


def test_first_on_the_same_input_as_mean_gives_the_same_sv():
    a, b, _ = _overlapping_dives()
    first = concat_datasets([a, b], dim="ping_time", on_duplicate="first")
    mean = concat_datasets([a, b], dim="ping_time", on_duplicate="mean")

    np.testing.assert_allclose(first["Sv"].values, mean["Sv"].values, rtol=1e-12)


def test_first_is_a_no_op_on_a_clean_index():
    a, _, _ = _overlapping_dives()
    out = concat_datasets([a, a.assign_coords(ping_time=a.ping_time + np.timedelta64(40, "s"))],
                          dim="ping_time", on_duplicate="first")
    assert out.sizes["ping_time"] == 8


def test_first_stays_lazy():
    a, b, _ = _overlapping_dives()
    out = concat_datasets([a.chunk(), b.chunk()], dim="ping_time", on_duplicate="first")
    assert out["Sv"].chunks is not None
