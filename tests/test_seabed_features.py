# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Stage 1 background estimate and Stage 3 point features."""

import numpy as np
import pytest

from aa_si_utils.seabed.features import (
    background_estimate,
    echo_asymmetry,
    energy_gradient,
    point_features,
    shape_features,
)
from aa_si_utils.seabed.geometry import build_geometry, crop_block
from aa_si_utils.seabed.phase import seed_mask
from seabed_synthetic import WATER_SV_DB, make_synthetic_sv


def _setup(**kwargs):
    ds, truth = make_synthetic_sv(n_ping=60, seabed_depth_m=100.0, **kwargs)
    geom = build_geometry(ds, "38000", r_min=50.0, r_max=150.0)
    sv = crop_block(ds, geom, "Sv")
    theta = crop_block(ds, geom, "angle_alongship") if kwargs.get("angles", True) else None
    phi = crop_block(ds, geom, "angle_athwartship") if kwargs.get("angles", True) else None
    return ds, truth, geom, sv, theta, phi


def test_background_tracks_water_column_and_ignores_seabed():
    ds, truth, geom, sv, theta, phi = _setup()
    seeds = seed_mask(sv, theta, phi, geom).seeds

    bg = background_estimate(sv, geom, seeds=seeds)

    assert bg.shape == sv.shape
    edge = truth["edge_idx"][0] - geom.i_lo
    water = bg[:, :edge]
    assert WATER_SV_DB - 10.0 < np.median(water) < WATER_SV_DB + 2.0
    # The seabed plateau is 60 dB louder than the background estimate.
    assert np.median(sv[:, edge + 2 : edge + 6] - bg[:, edge + 2 : edge + 6]) > 40.0


def test_background_without_masks_still_finite():
    _, _, geom, sv, _, _ = _setup(angles=False)
    bg = background_estimate(sv, geom)
    assert np.isfinite(bg).all()


def test_asymmetry_is_positive_at_leading_edge_and_negative_at_tail():
    _, truth, geom, sv, _, _ = _setup()
    f3 = echo_asymmetry(sv, geom.pulse_samples)
    edge = truth["edge_idx"] - geom.i_lo
    rows = np.arange(sv.shape[0])
    assert np.median(f3[rows, edge]) > 0.5
    tail = edge + 6 * geom.pulse_samples
    assert np.median(f3[rows, tail]) < 0.0
    assert f3.min() >= -1.0 and f3.max() <= 1.0


def test_energy_gradient_peaks_within_a_pulse_of_the_edge():
    _, truth, geom, sv, _, _ = _setup()
    f6 = energy_gradient(sv, geom.window_mask(), geom.pulse_samples)
    edge = truth["edge_idx"] - geom.i_lo
    peak = np.nanargmax(f6, axis=1)
    assert np.abs(peak - edge).max() <= geom.pulse_samples
    assert np.isfinite(f6).all()


def test_point_features_full_set_and_reference_shapes():
    _, truth, geom, sv, theta, phi = _setup()
    result = seed_mask(sv, theta, phi, geom)
    bg = background_estimate(sv, geom, seeds=result.seeds)

    features, attrs = point_features(sv, geom, bg, theta, phi, activity=result.phase_activity)

    assert set(features) == {"f1", "f2", "f3", "f4", "f5", "f6"}
    assert set(attrs) == set(features)
    for arr in features.values():
        assert arr.shape == sv.shape
    edge = truth["edge_idx"] - geom.i_lo
    rows = np.arange(sv.shape[0])
    plateau = (rows, edge + 2)
    water = (rows, edge - 4 * geom.pulse_samples)
    assert np.median(features["f2"][plateau]) < np.median(features["f2"][water])
    assert np.median(features["f5"][plateau]) < np.median(features["f5"][water])
    assert np.median(features["f4"][plateau]) > 40.0
    assert attrs["f2"]["window_pulse_lengths"] == 1.0


def test_point_features_without_angles_drop_phase_features():
    _, _, geom, sv, _, _ = _setup(angles=False)
    bg = background_estimate(sv, geom)
    features, _ = point_features(sv, geom, bg)
    assert set(features) == {"f1", "f3", "f4", "f6"}


def test_alias_pixels_are_nan_in_every_feature():
    _, _, geom, sv, theta, phi = _setup()
    bg = background_estimate(sv, geom)
    alias = np.zeros(sv.shape, dtype=bool)
    alias[3:6, 10:20] = True
    features, _ = point_features(sv, geom, bg, theta, phi, alias=alias)
    for arr in features.values():
        assert np.isnan(arr[alias]).all()
        assert np.isfinite(arr[~alias]).any()


def test_shape_features_mark_the_edge_and_the_plateau():
    _, truth, geom, sv, theta, phi = _setup()
    activity = seed_mask(sv, theta, phi, geom).phase_activity

    features, attrs = shape_features(sv, geom, 5.0, activity=activity)

    assert set(features) == {"f8", "f9", "f10"}
    assert attrs["f8"]["window_samples"] >= 2 * geom.pulse_samples
    edge = truth["edge_idx"] - geom.i_lo
    rows = np.arange(sv.shape[0])
    # Contrast peaks within a pulse of the edge and collapses inside the plateau.
    peak = np.nanargmax(np.where(geom.window_mask(), features["f8"], -np.inf), axis=1)
    assert np.abs(peak - edge).max() <= geom.pulse_samples
    assert np.median(features["f8"][rows, edge]) > 30.0
    assert np.median(features["f8"][rows, edge + 4 * geom.pulse_samples]) < 5.0
    # The plateau is stationary and phase-stable below the edge, water is not.
    assert np.median(features["f9"][rows, edge]) < 3.0
    water = edge - 6 * geom.pulse_samples
    assert np.median(features["f9"][rows, water]) > 2.0
    assert np.median(features["f10"][rows, edge]) < np.median(features["f10"][rows, water])


def test_shape_features_without_angles_and_at_edges():
    _, _, geom, sv, _, _ = _setup(angles=False)
    features, _ = shape_features(sv, geom, 5.0)
    assert set(features) == {"f8", "f9"}
    # An empty window is NaN at the very ends; a partly clipped one is finite.
    n_w = geom.metres_to_samples(5.0)
    assert np.isnan(features["f8"][:, 0]).all()
    assert np.isnan(features["f9"][:, -1]).all()
    assert np.isfinite(features["f8"][:, n_w // 2]).all()
    assert np.isfinite(features["f9"][:, -(geom.pulse_samples + 2)]).all()
