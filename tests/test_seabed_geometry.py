# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Stage 0 geometry: physical-unit windows and search bounds."""

import numpy as np
import pytest
import xarray as xr

from aa_si_utils.seabed.geometry import build_geometry, crop_block
from seabed_synthetic import TRANSDUCER_DEPTH_M, make_synthetic_sv


def test_geometry_reads_file_metadata():
    ds, truth = make_synthetic_sv(n_ping=20, dr=0.5, pulse_length_m=2.0)

    geom = build_geometry(ds, "38000")

    assert geom.range_var == "depth"
    assert geom.dr == pytest.approx(0.5)
    assert geom.pulse_length_m == pytest.approx(2.0)
    assert geom.pulse_samples == 4
    assert geom.beamwidth_deg == pytest.approx(7.0)
    assert geom.has_angles
    np.testing.assert_allclose(geom.range0, TRANSDUCER_DEPTH_M)
    np.testing.assert_allclose(geom.dt_ping, 1.0)
    np.testing.assert_allclose(geom.dx_ping, 5.0, rtol=0.01)


def test_window_conversions_round_up():
    ds, _ = make_synthetic_sv(n_ping=20, dr=0.5, pulse_length_m=2.0)
    geom = build_geometry(ds, "38000")

    assert geom.metres_to_samples(5.0) == 10
    assert geom.metres_to_samples(0.1) == 1
    assert geom.pulse_lengths_to_samples(2) == 8
    assert geom.seconds_to_pings(30) == 30
    assert geom.seconds_to_pings(0.1) == 1


def test_fixed_window_sets_crop():
    ds, _ = make_synthetic_sv(n_ping=20, dr=0.5)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)

    np.testing.assert_allclose(geom.r_min, 50.0)
    np.testing.assert_allclose(geom.r_max, 150.0)
    assert geom.i_lo == 90
    assert geom.i_hi == 291
    assert geom.window_mask().all()
    axis = geom.crop_range_axis()
    assert axis[0, 0] == pytest.approx(50.0)
    assert axis.shape == (20, geom.n_crop)


def test_default_window_skips_near_field_and_reaches_end():
    ds, _ = make_synthetic_sv(n_ping=10, n_sample=100, dr=0.5)
    geom = build_geometry(ds, "38000")

    np.testing.assert_allclose(geom.r_min, TRANSDUCER_DEPTH_M + 10.0)
    assert geom.i_hi == 100


def test_prior_line_sets_per_ping_window():
    ds, _ = make_synthetic_sv(n_ping=10, dr=0.5)
    z = xr.DataArray(np.linspace(80.0, 120.0, 10), coords={"ping_time": ds["ping_time"]}, dims=["ping_time"])
    geom = build_geometry(ds, "38000", z_prior=z, sigma_prior_m=5.0)

    np.testing.assert_allclose(geom.r_min, z.values - 15.0)
    np.testing.assert_allclose(geom.r_max, z.values + 15.0)
    assert geom.i_min[0] < geom.i_min[-1]
    mask = geom.window_mask()
    assert mask.any(axis=1).all()
    assert not mask.all()


def test_scalar_prior_uses_two_percent_sigma():
    ds, _ = make_synthetic_sv(n_ping=5, dr=0.5)
    geom = build_geometry(ds, "38000", z_prior=100.0)

    np.testing.assert_allclose(geom.r_min, 94.0)
    np.testing.assert_allclose(geom.r_max, 106.0)


def test_empty_window_raises():
    ds, _ = make_synthetic_sv(n_ping=5)
    with pytest.raises(ValueError, match="search window is empty"):
        build_geometry(ds, "38000", r_min=150.0, r_max=100.0)


def test_channel_by_khz_and_hz():
    ds, _ = make_synthetic_sv(n_ping=5, frequencies_hz=(18000, 38000))

    assert build_geometry(ds, 38).channel == "38000"
    assert build_geometry(ds, 18000).channel_index == 0
    with pytest.raises(ValueError):
        build_geometry(ds, 120)


def test_without_angles_or_gps_or_depth():
    ds, _ = make_synthetic_sv(n_ping=5, angles=False, gps=False, depth=False)
    geom = build_geometry(ds, "38000", vessel_speed_m_s=4.0)

    assert not geom.has_angles
    assert geom.range_var == "echo_range"
    np.testing.assert_allclose(geom.range0, 0.0)
    np.testing.assert_allclose(geom.dx_ping, 4.0)

    geom = build_geometry(ds, "38000")
    assert np.isnan(geom.dx_ping).all()


def test_missing_pulse_length_needs_override():
    ds, _ = make_synthetic_sv(n_ping=5)
    ds = ds.drop_vars("tau_effective")
    with pytest.raises(KeyError, match="tau_effective"):
        build_geometry(ds, "38000")
    geom = build_geometry(ds, "38000", pulse_length_s=0.001)
    assert geom.pulse_length_m == pytest.approx(0.75)


def test_crop_block_and_range_lookup():
    ds, truth = make_synthetic_sv(n_ping=8, dr=0.5)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)

    sv = crop_block(ds, geom, "Sv")
    assert sv.shape == (8, geom.n_crop)
    np.testing.assert_allclose(sv, ds["Sv"].isel(channel=0, range_sample=slice(geom.i_lo, geom.i_hi)).values)

    edge_m = geom.sample_to_range(truth["edge_idx"].astype(float))
    np.testing.assert_allclose(edge_m, truth["edge_m"])
    np.testing.assert_allclose(geom.range_to_sample(edge_m), truth["edge_idx"])


def test_lazy_dataset_is_computed():
    ds, _ = make_synthetic_sv(n_ping=6)
    geom = build_geometry(ds.chunk({"ping_time": 2}), "38000", r_min=50.0, r_max=150.0)
    sv = crop_block(ds.chunk({"ping_time": 2}), geom, "Sv")
    assert isinstance(sv, np.ndarray)
    assert geom.has_angles
