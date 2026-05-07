# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Unit tests for the pure mask-building helpers."""

import numpy as np
import xarray as xr

from aa_si_utils import utils


def _make_ds_sv():
    channels = np.array(["38000", "120000"])
    ping_time = np.array(
        [
            "2024-01-01T00:00:00",
            "2024-01-01T00:00:01",
            "2024-01-01T00:00:02",
        ],
        dtype="datetime64[ns]",
    )
    range_sample = np.arange(4)
    echo_range_values = np.array([5.0, 15.0, 25.0, 35.0])
    echo_range = np.broadcast_to(
        echo_range_values,
        (len(channels), len(ping_time), len(range_sample)),
    )
    sv = np.zeros_like(echo_range, dtype=float)

    return xr.Dataset(
        data_vars={
            "Sv": (("channel", "ping_time", "range_sample"), sv),
            "echo_range": (("channel", "ping_time", "range_sample"), echo_range),
            "frequency_nominal": (("channel",), np.array([38000.0, 120000.0])),
        },
        coords={
            "channel": channels,
            "ping_time": ping_time,
            "range_sample": range_sample,
        },
    )


def _make_echodata(ds_Sv):
    seafloor_depth = xr.DataArray(
        np.array(
            [
                [30.0, 31.0, 29.0],
                [5.0, 5.0, 5.0],
            ]
        ),
        coords={
            "channel": ds_Sv["channel"],
            "ping_time": ds_Sv["ping_time"],
        },
        dims=["channel", "ping_time"],
    )
    return {"Vendor_specific": {"detected_seafloor_depth": seafloor_depth}}


def test_detect_seafloor_returns_best_channel_depth():
    ds_Sv = _make_ds_sv()
    echodata = _make_echodata(ds_Sv)

    seafloor_depth = utils.detect_seafloor(ds_Sv=ds_Sv, echodata=echodata)

    assert seafloor_depth.dims == ("ping_time",)
    np.testing.assert_allclose(seafloor_depth.values, np.array([30.0, 31.0, 29.0]))


def test_detect_seafloor_supports_explicit_channel_selection():
    ds_Sv = _make_ds_sv()
    echodata = _make_echodata(ds_Sv)

    seafloor_depth = utils.detect_seafloor(
        ds_Sv=ds_Sv,
        echodata=echodata,
        channel="120000",
    )

    assert seafloor_depth.dims == ("ping_time",)
    np.testing.assert_allclose(seafloor_depth.values, np.array([5.0, 5.0, 5.0]))


def test_find_best_seafloor_detection_preserves_legacy_frequency_return():
    ds_Sv = _make_ds_sv()
    echodata = _make_echodata(ds_Sv)

    best_channel_idx, best_freq_khz, best_seafloor = utils.find_best_seafloor_detection(
        echodata,
        ds_Sv,
    )

    assert best_channel_idx == 0
    assert best_freq_khz == 38
    assert best_seafloor.dims == ("ping_time",)
    np.testing.assert_allclose(best_seafloor.values, np.array([30.0, 31.0, 29.0]))


def test_detect_seafloor_requires_echodata():
    ds_Sv = _make_ds_sv()

    try:
        utils.detect_seafloor(ds_Sv=ds_Sv, echodata=None)
    except ValueError as exc:
        assert "echodata is required for seafloor detection" in str(exc)
    else:
        raise AssertionError("detect_seafloor should require echodata")


def test_create_seafloor_mask_broadcasts_depth_across_channels():
    ds_Sv = _make_ds_sv()
    seafloor_depth = xr.DataArray(
        np.array([30.0, 20.0, 15.0]),
        coords={"ping_time": ds_Sv["ping_time"]},
        dims=["ping_time"],
    )

    mask = utils.create_seafloor_mask(ds_Sv, seafloor_depth, seafloor_buffer_m=5.0)

    assert mask.dims == ds_Sv["Sv"].dims
    expected_ping_0 = np.array([True, True, True, False])
    expected_ping_1 = np.array([True, True, False, False])
    expected_ping_2 = np.array([True, False, False, False])
    np.testing.assert_array_equal(mask.isel(channel=0, ping_time=0).values, expected_ping_0)
    np.testing.assert_array_equal(mask.isel(channel=1, ping_time=1).values, expected_ping_1)
    np.testing.assert_array_equal(mask.isel(channel=0, ping_time=2).values, expected_ping_2)


def test_create_frequency_mask_masks_selected_channels():
    ds_Sv = _make_ds_sv()

    mask = utils.create_frequency_mask(ds_Sv, frequencies_to_mask=[120])

    assert mask.dtype == bool
    assert mask.isel(channel=0).all().item() is True
    assert mask.isel(channel=1).any().item() is False


def test_combine_masks_validates_boolean_inputs_and_combines_masks():
    ds_Sv = _make_ds_sv()
    surface_mask = utils.create_surface_mask(ds_Sv, surface_depth_m=10.0)
    frequency_mask = utils.create_frequency_mask(ds_Sv, frequencies_to_mask=[120])

    combined_mask = utils.combine_masks([surface_mask, frequency_mask], mode="and")

    assert combined_mask.dtype == bool
    assert combined_mask.isel(channel=0, ping_time=0).values.tolist() == [False, True, True, True]
    assert combined_mask.isel(channel=1).any().item() is False

    invalid_mask = ds_Sv["Sv"]
    try:
        utils.combine_masks([surface_mask, invalid_mask], mode="and")
    except TypeError as exc:
        assert "boolean dtype" in str(exc)
    else:
        raise AssertionError("combine_masks should reject non-boolean inputs")


def test_create_data_mask_matches_pure_mask_composition():
    ds_Sv = _make_ds_sv()
    echodata = _make_echodata(ds_Sv)

    composed_mask = utils.combine_masks(
        [
            utils.create_seafloor_mask(
                ds_Sv,
                utils.detect_seafloor(ds_Sv=ds_Sv, echodata=echodata),
                seafloor_buffer_m=5.0,
            ),
            utils.create_surface_mask(ds_Sv, surface_depth_m=10.0),
            utils.create_frequency_mask(ds_Sv, frequencies_to_mask=[120]),
        ],
        mode="and",
    )

    wrapper_mask = utils.create_data_mask(
        echodata,
        ds_Sv,
        seafloor_buffer_m=5.0,
        surface_depth_m=10.0,
        frequencies_to_mask=[120],
    )

    xr.testing.assert_equal(wrapper_mask, composed_mask)