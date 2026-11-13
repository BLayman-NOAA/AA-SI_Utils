# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for add_location, the wrapper around echopype's add_location."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aa_si_utils import utils

RAW_FILE = (
    Path(__file__).resolve().parents[2]
    / "AA-SI_calibration" / "notebooks" / "example_data" / "HB2407_raw_cal"
    / "D20241112-T052842.raw"
)


def _ds_sv(n_pings=4):
    return xr.Dataset(
        {"Sv": ("ping_time", np.arange(float(n_pings)))},
        coords={"ping_time": np.datetime64("2024-11-12T05:30:00")
                + np.arange(n_pings) * np.timedelta64(1, "s")},
    )


class _FakeEchoData(dict):
    """Stands in for EchoData: groups by name, as add_location reads them."""


def _platform(times, latitudes):
    return xr.Dataset(
        {"latitude": ("time1", latitudes), "longitude": ("time1", latitudes)},
        coords={"time1": np.array(times, dtype="datetime64[ns]")},
    )


def test_a_file_without_fixes_gets_nan_position():
    echodata = _FakeEchoData(Platform=_platform(
        ["2024-11-12T05:30:00", "2024-11-12T05:30:02"], [np.nan, np.nan]
    ))

    out = utils.add_location(_ds_sv(), echodata)

    assert out["latitude"].dims == ("ping_time",)
    assert np.isnan(out["latitude"].values).all()
    assert np.isnan(out["longitude"].values).all()
    assert out["latitude"].attrs["units"] == "degrees_north"


def test_a_file_without_fixes_raises_when_asked():
    echodata = _FakeEchoData(Platform=xr.Dataset())

    with pytest.raises(ValueError, match="no position fixes"):
        utils.add_location(_ds_sv(), echodata, missing="error")


def test_repeated_fix_times_are_dropped_before_echopype(monkeypatch):
    seen = {}

    def fake_add_location(ds, echodata, datagram_type=None, nmea_sentence=None):
        seen["times"] = echodata["Platform"]["time1"].values
        seen["nmea_sentence"] = nmea_sentence
        return ds

    monkeypatch.setattr(utils.ep.consolidate, "add_location", fake_add_location)
    echodata = _FakeEchoData(Platform=_platform(
        ["2024-11-12T05:30:00", "2024-11-12T05:30:00", "2024-11-12T05:30:02"],
        [41.0, 41.5, 42.0],
    ))

    utils.add_location(_ds_sv(), echodata, nmea_sentence="GGA")

    assert len(seen["times"]) == 2
    assert float(echodata["Platform"]["latitude"].values[0]) == 41.0
    assert seen["nmea_sentence"] == "GGA"


def test_an_unknown_missing_policy_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        utils.add_location(_ds_sv(), _FakeEchoData(), missing="skip")


@pytest.mark.skipif(not RAW_FILE.exists(), reason="HB2407 example raw file not present")
def test_a_real_ek80_file_gets_its_position():
    import echopype as ep

    echodata = ep.open_raw(RAW_FILE, sonar_model="EK80")
    ds_Sv = ep.calibrate.compute_Sv(echodata, waveform_mode="CW", encode_mode="power")

    out = utils.add_location(ds_Sv, echodata)

    assert out["latitude"].sizes == {"ping_time": ds_Sv.sizes["ping_time"]}
    assert np.isfinite(out["latitude"].values).any()
