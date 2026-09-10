# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Unit tests for deriving transducer depth from the EK Platform group."""

import numpy as np
import pytest
import xarray as xr

from aa_si_utils import utils


def _times(seconds):
    """datetime64 stamps on 2024-01-01T00:00:<seconds>."""
    return np.array(
        [f"2024-01-01T00:00:{second:02d}" for second in seconds],
        dtype="datetime64[ns]",
    )


def _make_ds_sv(seconds=(0, 1, 2)):
    """Minimal Sv dataset supplying only the ping_time coordinate."""
    ping_time = _times(seconds)
    return xr.Dataset(
        data_vars={"Sv": (("ping_time",), np.zeros(len(ping_time)))},
        coords={"ping_time": ping_time},
    )


def _make_platform(
    drop_keel_offset=7.5, heave=None, heave_seconds=(0, 2), water_level=None
):
    """Platform group with a scalar drop keel offset and optional time2 heave.

    ``drop_keel_offset`` may be a scalar or a sequence (the time3-dimensioned
    shape older converted files carry). ``water_level`` is the EK60 shape,
    where there is no drop keel offset at all.
    """
    data_vars = {}
    coords = {}
    if drop_keel_offset is not None:
        if np.ndim(drop_keel_offset) == 0:
            data_vars["drop_keel_offset"] = ((), drop_keel_offset)
        else:
            data_vars["drop_keel_offset"] = (("time3",), np.asarray(drop_keel_offset))
            coords["time3"] = _times(range(len(drop_keel_offset)))
    if water_level is not None:
        data_vars["water_level"] = ((), water_level)
    if heave is not None:
        data_vars["vertical_offset"] = (("time2",), np.asarray(heave, dtype=float))
        coords["time2"] = _times(heave_seconds)
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _echodata(platform):
    """Stand-in for EchoData: the function only indexes the Platform group."""
    return {"Platform": platform}


# ---------------------------------------------------------------------------
# Static keel depth
# ---------------------------------------------------------------------------


def test_static_depth_is_the_drop_keel_offset():
    depth = utils.compute_transducer_depth(
        _echodata(_make_platform(drop_keel_offset=7.5)),
        _make_ds_sv(),
        use_heave=False,
    )
    assert depth.name == "transducer_depth"
    assert depth.dims == ("ping_time",)
    np.testing.assert_allclose(depth.values, [7.5, 7.5, 7.5])


def test_datum_correction_shifts_the_depth():
    depth = utils.compute_transducer_depth(
        _echodata(_make_platform(drop_keel_offset=7.5)),
        _make_ds_sv(),
        datum_correction_m=-1.25,
        use_heave=False,
    )
    np.testing.assert_allclose(depth.values, [6.25, 6.25, 6.25])


def test_missing_heave_falls_back_to_static_depth():
    depth = utils.compute_transducer_depth(
        _echodata(_make_platform(heave=None)),
        _make_ds_sv(),
        use_heave=True,
    )
    np.testing.assert_allclose(depth.values, [7.5, 7.5, 7.5])
    assert depth.attrs["heave_pings"] == 0


def test_all_nan_heave_falls_back_to_static_depth():
    platform = _make_platform(heave=[np.nan, np.nan])
    depth = utils.compute_transducer_depth(
        _echodata(platform), _make_ds_sv(), use_heave=True
    )
    np.testing.assert_allclose(depth.values, [7.5, 7.5, 7.5])
    assert depth.attrs["heave_pings"] == 0


# ---------------------------------------------------------------------------
# Per-ping heave
# ---------------------------------------------------------------------------


def test_heave_is_linearly_interpolated_onto_ping_time():
    # Heave 1.0 m at t=0 and 3.0 m at t=2 gives 2.0 m at the t=1 ping.
    platform = _make_platform(heave=[1.0, 3.0], heave_seconds=(0, 2))
    depth = utils.compute_transducer_depth(
        _echodata(platform), _make_ds_sv(), heave_sign=-1.0
    )
    np.testing.assert_allclose(depth.values, [6.5, 5.5, 4.5])
    assert depth.attrs["heave_pings"] == 3


def test_heave_sign_flips_the_correction():
    platform = _make_platform(heave=[1.0, 3.0], heave_seconds=(0, 2))
    depth = utils.compute_transducer_depth(
        _echodata(platform), _make_ds_sv(), heave_sign=1.0
    )
    np.testing.assert_allclose(depth.values, [8.5, 9.5, 10.5])


def test_pings_outside_the_heave_record_hold_the_static_depth():
    # Heave covers t=1..2 only, so the t=0 ping gets no correction rather
    # than a NaN depth that would void the whole ping downstream.
    platform = _make_platform(heave=[1.0, 2.0], heave_seconds=(1, 2))
    depth = utils.compute_transducer_depth(
        _echodata(platform), _make_ds_sv(), heave_sign=-1.0
    )
    np.testing.assert_allclose(depth.values, [7.5, 6.5, 5.5])
    assert np.isfinite(depth.values).all()
    assert depth.attrs["heave_pings"] == 2


def test_single_heave_sample_is_held_flat():
    platform = _make_platform(heave=[2.0], heave_seconds=(1,))
    depth = utils.compute_transducer_depth(
        _echodata(platform), _make_ds_sv(), heave_sign=-1.0
    )
    np.testing.assert_allclose(depth.values, [5.5, 5.5, 5.5])


def test_output_carries_no_stray_source_time_coord():
    platform = _make_platform(heave=[1.0, 3.0], heave_seconds=(0, 2))
    depth = utils.compute_transducer_depth(_echodata(platform), _make_ds_sv())
    assert "time2" not in depth.coords
    assert set(depth.coords) == {"ping_time"}


def test_ping_time_matches_the_sv_dataset_exactly():
    ds_Sv = _make_ds_sv(seconds=(0, 1, 2))
    platform = _make_platform(heave=[1.0, 3.0], heave_seconds=(0, 2))
    depth = utils.compute_transducer_depth(_echodata(platform), ds_Sv)
    assert depth["ping_time"].equals(ds_Sv["ping_time"])


# ---------------------------------------------------------------------------
# Provenance and failure modes
# ---------------------------------------------------------------------------


def test_attrs_record_the_derivation():
    platform = _make_platform(drop_keel_offset=7.5, heave=[1.0, 3.0])
    depth = utils.compute_transducer_depth(
        _echodata(platform),
        _make_ds_sv(),
        datum_correction_m=-1.0,
        heave_sign=-1.0,
    )
    assert depth.attrs["static_offset_m"] == 7.5
    assert depth.attrs["static_offset_field"] == "drop_keel_offset"
    assert depth.attrs["datum_correction_m"] == -1.0
    assert depth.attrs["heave_sign"] == -1.0
    assert depth.attrs["units"] == "m"
    assert depth.attrs["interp_method"] == "linear"


def test_no_usable_offset_raises_and_says_what_it_tried():
    platform = _make_platform(drop_keel_offset=None)
    with pytest.raises(KeyError, match="no usable transducer depth") as excinfo:
        utils.compute_transducer_depth(_echodata(platform), _make_ds_sv())
    assert "'drop_keel_offset' (absent)" in str(excinfo.value)
    assert "'water_level' (absent)" in str(excinfo.value)


def test_nan_drop_keel_offset_with_nothing_to_fall_back_on_raises():
    platform = _make_platform(drop_keel_offset=np.nan)
    with pytest.raises(KeyError, match="all NaN"):
        utils.compute_transducer_depth(_echodata(platform), _make_ds_sv())


# ---------------------------------------------------------------------------
# The EK60 path: water_level instead of a drop keel offset
# ---------------------------------------------------------------------------


def test_water_level_is_used_when_there_is_no_drop_keel_offset():
    """EK60. echopype maps the ER60 transducer depth into water_level.

    Its own use_platform_vertical_offsets computes transducer_offset_z -
    (water_level + heave), and the ER60 leaves transducer_offset_z at zero, so
    that path puts the transducer above the surface and every depth comes out
    short by twice the draft. Reading water_level directly gets the sign right.
    """
    platform = _make_platform(drop_keel_offset=None, water_level=9.0)
    depth = utils.compute_transducer_depth(
        _echodata(platform), _make_ds_sv(), use_heave=False
    )
    np.testing.assert_allclose(depth.values, [9.0, 9.0, 9.0])
    assert depth.attrs["static_offset_field"] == "water_level"
    assert depth.attrs["static_offset_m"] == 9.0


def test_a_drop_keel_offset_wins_over_water_level():
    """EK80 files carry both, and the keel position is the real one."""
    platform = _make_platform(drop_keel_offset=7.5, water_level=0.0)
    depth = utils.compute_transducer_depth(
        _echodata(platform), _make_ds_sv(), use_heave=False
    )
    np.testing.assert_allclose(depth.values, [7.5, 7.5, 7.5])
    assert depth.attrs["static_offset_field"] == "drop_keel_offset"


def test_water_level_still_takes_heave():
    platform = _make_platform(
        drop_keel_offset=None, water_level=9.0, heave=[0.5, -0.5]
    )
    depth = utils.compute_transducer_depth(
        _echodata(platform), _make_ds_sv(), heave_sign=1.0
    )
    np.testing.assert_allclose(depth.values, [9.5, 9.0, 8.5])


def test_conflicting_drop_keel_offsets_raise():
    # The shape a combine across two keel positions leaves behind.
    platform = _make_platform(drop_keel_offset=[7.0, 7.5])
    with pytest.raises(ValueError, match="more than one value"):
        utils.compute_transducer_depth(_echodata(platform), _make_ds_sv())


def test_repeated_identical_drop_keel_offsets_are_accepted():
    platform = _make_platform(drop_keel_offset=[7.5, 7.5])
    depth = utils.compute_transducer_depth(
        _echodata(platform), _make_ds_sv(), use_heave=False
    )
    np.testing.assert_allclose(depth.values, [7.5, 7.5, 7.5])
