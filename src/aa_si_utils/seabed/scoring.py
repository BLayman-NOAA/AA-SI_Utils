# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Stage 4, Mode 1: fixed physics-grounded scoring with regime presets."""

import numpy as np

PRESET_VERSION = "v4.0"

# Depth-dependent references. Calibration values, not learned parameters.
PRESETS = {
    "shelf": {
        "version": PRESET_VERSION,
        "max_depth_m": 200.0,
        "snr_ref_db": 30.0,
        "seed_sv_floor_db": -45.0,
        "shape_window_m": 5.0,
        "contrast_ref_db": 20.0,
    },
    "slope": {
        "version": PRESET_VERSION,
        "max_depth_m": 1000.0,
        "snr_ref_db": 20.0,
        "seed_sv_floor_db": -50.0,
        "shape_window_m": 10.0,
        "contrast_ref_db": 20.0,
    },
    "deep": {
        "version": PRESET_VERSION,
        "max_depth_m": np.inf,
        "snr_ref_db": 12.0,
        "seed_sv_floor_db": -55.0,
        "shape_window_m": 10.0,
        "contrast_ref_db": 15.0,
    },
}


def select_preset(regime, expected_depth_m):
    """Pick a regime preset by name or from the expected seabed depth.

    Args:
        regime: ``"shelf"``, ``"slope"``, ``"deep"`` or ``"auto"``.
        expected_depth_m: Depth used when ``regime`` is ``"auto"``.

    Returns:
        tuple: ``(name, preset)``.
    """
    if regime != "auto":
        if regime not in PRESETS:
            raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(PRESETS)} or 'auto'")
        return regime, PRESETS[regime]
    for name in ("shelf", "slope", "deep"):
        if expected_depth_m < PRESETS[name]["max_depth_m"]:
            return name, PRESETS[name]
    return "deep", PRESETS["deep"]


def reference_table(preset, seed_sv_db, beamwidth_deg, seed_f2_deg=0.0, seed_f5_deg=0.0, mode=1):
    """Reference mean, scale and sign per feature for Mode 1 or Mode 2.

    The amplitude and phase references come from the survey's own seeds,
    as the spec does for f1: real angle noise on a deep seabed is several
    degrees, not the sub-degree figures of the idealised references, so a
    fixed zero mean would saturate f2 and f5 on seabed and water alike.

    Args:
        preset: A ``PRESETS`` entry.
        seed_sv_db: Seed-median Sv, the f1 reference; when there were no
            seeds pass the preset seed floor.
        beamwidth_deg: Nominal beamwidth setting the phase scales.
        seed_f2_deg: Seed-median local phase activity, the f2 reference.
        seed_f5_deg: Seed-median phase stability, the f5 reference.
        mode: 1 for the point features f1 to f6, 2 to add the shape
            features f8 to f10.

    Returns:
        dict: ``{feature: (mu, sigma, sign)}``.
    """
    if mode not in (1, 2):
        raise ValueError(f"mode must be 1 or 2, got {mode!r}")
    phase_scale = beamwidth_deg / 2.0
    refs = {
        "f1": (float(seed_sv_db), 10.0, 1.0),
        "f2": (float(seed_f2_deg), phase_scale, -1.0),
        "f3": (0.5, 0.3, 1.0),
        "f4": (float(preset["snr_ref_db"]), 10.0, 1.0),
        "f5": (float(seed_f5_deg), phase_scale, -1.0),
        "f6": (10.0, 5.0, 1.0),
    }
    if mode == 2:
        refs["f8"] = (float(preset["contrast_ref_db"]), 10.0, 1.0)
        refs["f9"] = (5.0, 3.0, -1.0)
        refs["f10"] = (phase_scale, phase_scale, -1.0)
    return refs


def score_mode1(features, refs, weights=None):
    """Bounded per-feature scores summed into the bottom likelihood Lambda.

    Args:
        features: ``{name: (P, S) array}``; only names present in ``refs``
            contribute.
        refs: Output of :func:`reference_table`.
        weights: Optional ``{name: weight}``; default 1 for every feature.

    Returns:
        tuple: ``(Lambda, used)`` with ``used`` the list of feature names
        that contributed.
    """
    weights = weights or {}
    total = None
    used = []
    for name, (mu, sigma, sign) in refs.items():
        if name not in features:
            continue
        w = float(weights.get(name, 1.0))
        s = w * sign * np.tanh((features[name] - mu) / sigma)
        total = s if total is None else total + s
        used.append(name)
    if total is None:
        raise ValueError("no features available to score")
    return total, used
