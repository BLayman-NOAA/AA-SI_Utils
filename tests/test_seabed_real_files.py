# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Real-file checks of the phase-aware detector against reference lines.

Slow: each case opens and calibrates a full raw file. The references (.bot
picks, Echoview lines) are themselves imperfect, so the accuracy bounds here
are loose floors that catch a broken detector, not a tuned target.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from aa_si_utils.seabed.validation import run_validation

EXAMPLES = Path(__file__).resolve().parents[2] / "AA-SI_recipe_manager" / "examples"
EK60_RAW = EXAMPLES / "HB1603" / "raw_file_inputs" / "HB1603_2016-07-25_2058-2345" / "D20160725-T205832.raw"
EK80_RAW = EXAMPLES / "HB2407" / "raw" / "D20241012-T064618.raw"
EK80_EVL = EXAMPLES / "HB2407" / "seabed_lines"

pytestmark = pytest.mark.slow


def _check_contract(line, metrics):
    assert isinstance(line, xr.DataArray)
    assert line.dims == ("ping_time",)
    assert line.attrs["units"] == "m"
    assert line.attrs["vertical_reference"] == "surface"
    assert metrics["line_coverage"] >= 0.9
    assert metrics["alias_false_accept_rate"] == 0.0


@pytest.mark.skipif(not EK60_RAW.exists() or not EK60_RAW.with_suffix(".bot").exists(), reason="HB1603 raw/bot files absent")
def test_ek60_against_bot_line(tmp_path):
    """HB1603 EK60 file with .bot reference."""
    metrics, diag, line, ref = run_validation(EK60_RAW, "EK60", "bot", out_dir=tmp_path, channel=38)

    # Measured 2026-09-24: the .bot line runs about 8 m below the Sv rise
    # on this file, so the detector's leading edge sits a steady 8 m above
    # it (median deviation 8.3 m, bias -3.3 m, gross-error rate 0.10 on
    # the steep step at pings 85 to 120).
    _check_contract(line, metrics)
    assert metrics["has_angles"]
    assert metrics["median_abs_deviation_m"] <= 12.0
    assert metrics["gross_error_rate"] <= 0.15


@pytest.mark.skipif(not EK80_RAW.exists() or not EK80_EVL.exists(), reason="HB2407 raw/evl files absent")
def test_ek80_against_evl_line(tmp_path):
    """HB2407 EK80 file with Echoview line reference."""
    metrics, diag, line, ref = run_validation(EK80_RAW, "EK80", "evl", evl_path=EK80_EVL, out_dir=tmp_path)

    # Measured 2026-09-24: the detector sits a steady 2.2 m below the
    # Echoview line on 18 kHz (median deviation 2.2 m, no gross errors).
    _check_contract(line, metrics)
    assert metrics["median_abs_deviation_m"] <= 4.0
    assert metrics["gross_error_rate"] <= 0.05
