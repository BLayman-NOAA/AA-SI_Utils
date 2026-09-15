# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Applying a mask to a gridded product as well as to fine-resolution Sv.

echopype's apply_mask checks the mask against ping_time and range_sample
specifically, so it rejects an MVBS binned on depth, which carries a depth
dimension and no range_sample at all. A recipe that grids in the survey tier and
masks per dive below it hits that immediately, so the wrapper takes the gridded
case itself rather than failing.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from aa_si_utils.utils import apply_mask_to_sv, create_seafloor_mask


def _gridded(n_ch=2, n_pt=6, n_depth=106, bin_m=2.0):
    """The shape compute_MVBS returns with range_var='depth'."""
    return xr.Dataset(
        {"Sv": (("channel", "ping_time", "depth"),
                np.full((n_ch, n_pt, n_depth), -70.0)),
         "frequency_nominal": (("channel",), np.arange(n_ch, dtype=float))},
        coords={
            "channel": [f"ch{i}" for i in range(n_ch)],
            "ping_time": (np.datetime64("2016-06-27T00:00:00")
                          + np.arange(n_pt) * np.timedelta64(10, "s")),
            "depth": np.arange(n_depth) * bin_m,
        },
    )


def _fine(n_ch=2, n_pt=20, n_range=40):
    rng = np.linspace(0, 200, n_range)
    shape = (n_ch, n_pt, n_range)
    return xr.Dataset(
        {"Sv": (("channel", "ping_time", "range_sample"), np.full(shape, -70.0)),
         "echo_range": (("channel", "ping_time", "range_sample"),
                        np.broadcast_to(rng, shape).copy()),
         "depth": (("channel", "ping_time", "range_sample"),
                   np.broadcast_to(rng, shape).copy()),
         "frequency_nominal": (("channel",), np.arange(n_ch, dtype=float))},
        coords={"channel": [f"ch{i}" for i in range(n_ch)],
                "ping_time": (np.datetime64("2016-06-27T00:00:00")
                              + np.arange(n_pt) * np.timedelta64(1, "s")),
                "range_sample": np.arange(n_range)},
    )


def _seafloor(ds, depth_m=150.0):
    return xr.DataArray(np.full(ds.sizes["ping_time"], depth_m), dims="ping_time",
                        coords={"ping_time": ds["ping_time"]})


def test_a_gridded_product_can_be_masked():
    ds = _gridded()
    mask = create_seafloor_mask(ds, seafloor_depth=_seafloor(ds),
                                seafloor_buffer_m=10.0)

    out = apply_mask_to_sv(ds, mask)

    assert int(np.isfinite(out["Sv"]).sum()) == int(mask.sum())
    assert out["Sv"].dims == ("channel", "ping_time", "depth")


def test_the_gridded_cut_lands_where_the_buffer_says():
    ds = _gridded()
    mask = create_seafloor_mask(ds, seafloor_depth=_seafloor(ds, 150.0),
                                seafloor_buffer_m=10.0)

    out = apply_mask_to_sv(ds, mask)
    kept = out["depth"].where(
        np.isfinite(out["Sv"].isel(channel=0, ping_time=0))
    ).max()

    assert float(kept) == 140.0


def test_fine_resolution_still_goes_through_echopype():
    """The default path must be untouched, provenance and all."""
    ds = _fine()
    mask = create_seafloor_mask(ds, seafloor_depth=_seafloor(ds, 100.0),
                                seafloor_buffer_m=10.0)

    out = apply_mask_to_sv(ds, mask)

    assert "range_sample" in out.dims
    assert 0 < int(np.isfinite(out["Sv"]).sum()) < out["Sv"].size


def test_masking_does_not_mutate_the_input():
    ds = _gridded()
    mask = create_seafloor_mask(ds, seafloor_depth=_seafloor(ds),
                                seafloor_buffer_m=10.0)

    apply_mask_to_sv(ds, mask)

    assert int(np.isfinite(ds["Sv"]).sum()) == ds["Sv"].size
