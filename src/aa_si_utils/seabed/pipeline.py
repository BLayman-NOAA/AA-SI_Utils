# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Phase-aware DP seabed detection: orchestration and the recipe entry point.

``detect_seabed`` runs Stages 0 to 8 (Mode 1) on one channel and returns
every intermediate as an xarray Dataset. ``detect_seafloor_phase`` is the
recipe op: it returns the leading-edge line on the shared ``seafloor_depth``
contract, a 1-D ``(ping_time,)`` DataArray in metres.
"""

import warnings

import numpy as np
import xarray as xr

from aa_si_utils.seabed.dp import run_dp, select_candidates, transition_params
from aa_si_utils.seabed.features import background_estimate, point_features, shape_features
from aa_si_utils.seabed.geometry import build_geometry, crop_block
from aa_si_utils.seabed.phase import alias_candidates, seed_mask
from aa_si_utils.seabed.prior import estimate_prior
from aa_si_utils.seabed.refine import backstep, leading_edge
from aa_si_utils.seabed.scoring import reference_table, score_mode1, select_preset

FLAG_SKIPPED = 1
FLAG_WALKBACK_BOUND = 2
FLAG_LOW_CONFIDENCE = 4
FLAG_LOW_MARGIN = 8
FLAG_INTERPOLATED = 16

_COVERAGE_WARNING = (
    "  WARNING: create_seafloor_mask drops a ping entirely where the "
    "seafloor is NaN. Widen max_gap_s, lower min_score, or check the search "
    "window against the seabed depth."
)


def _channels_by_frequency(ds_Sv):
    labels = [str(c) for c in ds_Sv["channel"].values]
    if "frequency_nominal" in ds_Sv:
        freqs = np.asarray(ds_Sv["frequency_nominal"].values, dtype=float)
        order = np.argsort(freqs, kind="stable")
        return [labels[i] for i in order]
    return labels


def _blocks(ds_Sv, geom):
    sv = crop_block(ds_Sv, geom, "Sv")
    theta = phi = None
    if geom.has_angles:
        theta = crop_block(ds_Sv, geom, "angle_alongship")
        phi = crop_block(ds_Sv, geom, "angle_athwartship")
    return sv, theta, phi


def _stage_2(ds_Sv, channel, regime, geometry_kwargs, detect_aliases, phase_threshold_deg2):
    """Geometry, preset, alias mask and seeds for one channel."""
    geom = build_geometry(ds_Sv, channel, **geometry_kwargs)
    expected_depth = float(np.median((geom.r_min + geom.r_max) / 2.0))
    preset_name, preset = select_preset(regime, expected_depth)
    sv, theta, phi = _blocks(ds_Sv, geom)
    if detect_aliases:
        alias = alias_candidates(sv, theta, phi, geom)
    else:
        alias = np.zeros(sv.shape, dtype=bool)
    seeds = seed_mask(
        sv,
        theta,
        phi,
        geom,
        sv_floor_db=preset["seed_sv_floor_db"],
        alias=alias,
        phase_threshold_deg2=phase_threshold_deg2,
    )
    return geom, preset_name, preset, sv, theta, phi, alias, seeds


def _seeds_usable(seeds, min_ping_fraction=0.2):
    """Seeds count for channel selection only if they span enough pings."""
    return seeds.n_components > 0 and seeds.seeds.any(axis=1).mean() >= min_ping_fraction


def _seed_median(feature, seeds):
    values = feature[seeds.seeds]
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else 0.0


def detect_seabed(
    ds_Sv,
    channel=None,
    r_min=None,
    r_max=None,
    z_prior=None,
    sigma_prior_m=None,
    regime="auto",
    n_candidates=5,
    max_slope_deg=30.0,
    alpha=1.0,
    beta_fraction=0.1,
    min_score=0.0,
    walkback_db=10.0,
    effective_beamwidth_deg=None,
    vessel_speed_m_s=None,
    min_confidence=0.8,
    phase_weight=1.0,
    transition="huber",
    pulse_length_s=None,
    beamwidth_deg=None,
    detect_aliases=False,
    phase_threshold_deg2=None,
    confidence_tolerance_pulses=2.0,
    mode=1,
    prior="auto",
    prior_budget=64,
):
    """Run the Mode 1 or Mode 2 pipeline on one channel and keep every intermediate.

    Args:
        ds_Sv: Calibrated Sv dataset. With ``depth`` the line is surface
            referenced, otherwise transducer referenced. Split-beam angles
            (``angle_alongship``, ``angle_athwartship``) enable the phase
            features; without them the detector runs amplitude-only.
        channel: Primary channel label, Hz or kHz. None picks the lowest
            frequency channel that produces seabed seeds.
        r_min: Fixed lower search bound in metres.
        r_max: Fixed upper search bound in metres.
        z_prior: Expected seabed depth, scalar or ``(ping_time,)`` line.
        sigma_prior_m: Prior uncertainty; default 2 % of z_prior.
        regime: ``"shelf"``, ``"slope"``, ``"deep"`` or ``"auto"``.
        n_candidates: Score maxima kept per ping.
        max_slope_deg: Maximum-slope prior for the transition break-point.
        alpha: Transition strength multiplier.
        beta_fraction: Shallow-first prior as a fraction of IQR(Lambda).
        min_score: Candidates below this score are dropped.
        walkback_db: Leading-edge walk-back threshold below the peak.
        effective_beamwidth_deg: Effective beamwidth for the backstep.
        vessel_speed_m_s: Speed when the dataset carries no position.
        min_confidence: Pings below this posterior are flagged.
        phase_weight: Weight of the phase features f2 and f5.
        transition: ``"huber"`` or ``"truncated"``.
        pulse_length_s: Pulse duration override.
        beamwidth_deg: Nominal beamwidth override.
        detect_aliases: Run the Blackwell alias detector (Stage 2a) and
            exclude its mask. Off by default; see
            :func:`aa_si_utils.seabed.phase.alias_candidates`.
        phase_threshold_deg2: Seed phase-stability threshold; default
            (beamwidth / 2)^2.
        confidence_tolerance_pulses: Candidates within this many pulse
            lengths (plus the beam-geometry spread) of the pick, or inside
            the same run of samples louder than the seed median minus 10 dB,
            count as the same pick for confidence and margin.
        mode: 1 scores the point features f1 to f6; 2 adds the local shape
            features f8 to f10 (below-above contrast, below stationarity,
            below phase activity) with the preset's shape window.
        prior: ``"auto"`` runs the Mode 0 slab estimator when no z_prior is
            given and uses its line and uncertainty as the search window,
            inside r_min / r_max if those are set; ``"none"`` searches the
            fixed window only.
        prior_budget: Mode 0b slab budget.

    Returns:
        xr.Dataset: Masks, features, score, candidates, the raw, refined
        and integration lines, confidence, margin and per-ping flags on the
        cropped ``(ping_time, range_sample)`` grid, with run attributes.
    """
    prior_attrs = {"prior": "none" if z_prior is None else "given"}
    if z_prior is None and prior == "auto":
        prior_channel = channel if channel is not None else _channels_by_frequency(ds_Sv)[0]
        estimate = estimate_prior(
            ds_Sv,
            prior_channel,
            r_min=r_min,
            r_max=r_max,
            budget=prior_budget,
            max_slope_deg=max_slope_deg,
            vessel_speed_m_s=vessel_speed_m_s,
            pulse_length_s=pulse_length_s,
            beamwidth_deg=beamwidth_deg,
        )
        if estimate is None:
            warnings.warn("Mode 0 prior found no seabed; searching the fixed window")
        else:
            z_prior = estimate["z_prior"]
            sigma_prior_m = estimate["sigma_prior"]
            prior_attrs = {
                "prior": "mode0",
                "prior_slabs": estimate.attrs["n_slabs"],
                "prior_inconsistent_gaps": estimate.attrs["n_inconsistent_gaps"],
                "prior_median_m": float(np.nanmedian(estimate["z_prior"].values)),
                "prior_median_sigma_m": float(np.nanmedian(estimate["sigma_prior"].values)),
            }
    elif prior not in ("auto", "none"):
        raise ValueError(f"prior must be 'auto' or 'none', got {prior!r}")
    geometry_kwargs = dict(
        r_min=r_min,
        r_max=r_max,
        z_prior=z_prior,
        sigma_prior_m=sigma_prior_m,
        vessel_speed_m_s=vessel_speed_m_s,
        pulse_length_s=pulse_length_s,
        beamwidth_deg=beamwidth_deg,
    )
    stage2_args = (regime, geometry_kwargs, detect_aliases, phase_threshold_deg2)
    if channel is None:
        stage2 = None
        for label in _channels_by_frequency(ds_Sv):
            stage2 = _stage_2(ds_Sv, label, *stage2_args)
            if _seeds_usable(stage2[7]):
                break
        else:
            warnings.warn("no channel produced seabed seeds; using the lowest frequency")
            stage2 = _stage_2(ds_Sv, _channels_by_frequency(ds_Sv)[0], *stage2_args)
    else:
        stage2 = _stage_2(ds_Sv, channel, *stage2_args)
    geom, preset_name, preset, sv, theta, phi, alias, seeds = stage2

    background = background_estimate(sv, geom, alias=alias, seeds=seeds.seeds)
    features, feature_attrs = point_features(
        sv, geom, background, theta, phi, activity=seeds.phase_activity, alias=alias
    )
    if mode == 2:
        shape, shape_attrs = shape_features(sv, geom, preset["shape_window_m"], activity=seeds.phase_activity)
        if alias.any():
            for arr in shape.values():
                arr[alias] = np.nan
        features.update(shape)
        feature_attrs.update(shape_attrs)
    refs = reference_table(
        preset,
        seeds.sv_threshold_db,
        geom.beamwidth_deg,
        seed_f2_deg=_seed_median(features["f2"], seeds) if "f2" in features else 0.0,
        seed_f5_deg=_seed_median(features["f5"], seeds) if "f5" in features else 0.0,
        mode=mode,
    )
    weights = {"f2": phase_weight, "f5": phase_weight}
    score, used = score_mode1(features, refs, weights)

    cand = select_candidates(score, geom, n_candidates=n_candidates, min_score=min_score)
    params = transition_params(
        cand, geom, max_slope_deg=max_slope_deg, alpha=alpha, beta_fraction=beta_fraction, kind=transition
    )
    loud = np.isfinite(sv) & (sv > seeds.sv_threshold_db - 10.0)
    dp = run_dp(cand, geom, params, tolerance_pulses=confidence_tolerance_pulses, loud=loud)

    le_idx, bound_hit = leading_edge(sv, dp.r_star_idx, geom, walkback_db=walkback_db)
    r_star_m = geom.sample_to_range(dp.r_star_idx + geom.i_lo)
    r_le_m = geom.sample_to_range(le_idx + geom.i_lo)
    r_int_m = backstep(r_le_m, geom, effective_beamwidth_deg=effective_beamwidth_deg)

    flags = np.zeros(geom.n_ping, dtype=np.int16)
    flags[~cand.valid] |= FLAG_SKIPPED
    flags[bound_hit] |= FLAG_WALKBACK_BOUND
    flags[np.isfinite(dp.confidence) & (dp.confidence < min_confidence)] |= FLAG_LOW_CONFIDENCE
    flags[np.isfinite(dp.margin) & (dp.margin < params.lam * params.delta)] |= FLAG_LOW_MARGIN

    ping = ds_Sv["ping_time"]
    sample = np.arange(geom.i_lo, geom.i_hi)
    dims2 = ("ping_time", "range_sample")
    data = {
        "alias": (dims2, alias),
        "seeds": (dims2, seeds.seeds),
        "uncertain": (dims2, seeds.uncertain),
        "background": (dims2, background, {"units": "dB"}),
        "score": (dims2, score, {"features": " ".join(used)}),
        "range_m": (dims2, geom.crop_range_axis(), {"units": "m", "reference": geom.range_var}),
        "candidate_range_m": (("ping_time", "candidate"), cand.range_m, {"units": "m"}),
        "candidate_score": (("ping_time", "candidate"), cand.score),
        "r_star": (("ping_time",), r_star_m, {"units": "m", "long_name": "raw DP pick"}),
        "r_le": (("ping_time",), r_le_m, {"units": "m", "long_name": "leading edge"}),
        "r_int": (("ping_time",), r_int_m, {"units": "m", "long_name": "integration line"}),
        "confidence": (("ping_time",), dp.confidence),
        "margin": (("ping_time",), dp.margin),
        "delta": (("ping_time",), params.delta, {"units": "m"}),
        "range_end_m": (("ping_time",), geom.range0 + (geom.n_valid - 1) * geom.dr, {"units": "m", "long_name": "last recorded range"}),
        "flags": (("ping_time",), flags, {"bits": "1 skipped, 2 walkback bound, 4 low confidence, 8 low margin, 16 interpolated"}),
    }
    for name, arr in features.items():
        data[name] = (dims2, arr, feature_attrs[name])
    attrs = {
        "primary_channel": geom.channel,
        "range_var": geom.range_var,
        "mode": int(mode),
        **prior_attrs,
        "window_median_r_min_m": float(np.median(geom.r_min)),
        "window_median_r_max_m": float(np.median(geom.r_max)),
        "regime": preset_name,
        "preset_version": preset["version"],
        "has_angles": int(geom.has_angles),
        "seed_sv_db": float(seeds.sv_threshold_db),
        "detection_range_m": float(seeds.detection_range_m),
        "n_seed_components": int(seeds.n_components),
        "pulse_length_m": geom.pulse_length_m,
        "beamwidth_deg": geom.beamwidth_deg,
        "lambda": float(params.lam),
        "beta": float(params.beta),
        "temperature": float(params.temperature),
        "total_cost": float(dp.total_cost),
        "features": " ".join(used),
    }
    return xr.Dataset(
        data,
        coords={"ping_time": ping, "range_sample": sample, "candidate": np.arange(n_candidates)},
        attrs=attrs,
    )


def _fill_gaps(values, ping_time, max_gap_s):
    """Linearly interpolate interior NaN runs no longer than ``max_gap_s``."""
    out = np.asarray(values, dtype=float).copy()
    filled = np.zeros(out.size, dtype=bool)
    if max_gap_s is None:
        return out, filled
    valid = np.nonzero(np.isfinite(out))[0]
    if valid.size < 2:
        return out, filled
    t = np.asarray(ping_time, dtype="datetime64[ns]").astype("int64") / 1e9
    for a, b in zip(valid[:-1], valid[1:]):
        if b - a > 1 and (t[b] - t[a]) <= max_gap_s:
            out[a + 1 : b] = np.interp(t[a + 1 : b], [t[a], t[b]], [out[a], out[b]])
            filled[a + 1 : b] = True
    return out, filled


def detect_seafloor_phase(
    ds_Sv,
    echodata=None,
    channel=None,
    r_min=None,
    r_max=None,
    z_prior=None,
    sigma_prior_m=None,
    regime="auto",
    n_candidates=5,
    max_slope_deg=30.0,
    alpha=1.0,
    beta_fraction=0.1,
    min_score=0.0,
    walkback_db=10.0,
    effective_beamwidth_deg=None,
    vessel_speed_m_s=None,
    min_confidence=0.8,
    phase_weight=1.0,
    max_gap_s=30.0,
    transition="huber",
    pulse_length_s=None,
    beamwidth_deg=None,
    detect_aliases=False,
    phase_threshold_deg2=None,
    confidence_tolerance_pulses=2.0,
    mode=1,
    line="leading_edge",
    prior="auto",
    prior_budget=64,
    missing="nan",
    diagnostics_path=None,
):
    """Seafloor line from the phase-aware DP detector.

    Drop-in swap with the other seafloor detection ops: the return is a 1-D
    ``(ping_time,)`` DataArray in metres on ``ds_Sv``'s exact ping grid,
    surface referenced when ``ds_Sv`` carries ``depth``. The value is the
    leading edge of the seabed echo; the backstepped integration line and
    the raw pick are in the diagnostics.

    Args:
        ds_Sv: See :func:`detect_seabed`.
        echodata: Accepted for interface parity with the other detectors;
            unused.
        max_gap_s: Interior runs of skipped pings up to this long are
            interpolated; longer runs and the ends stay NaN.
        mode: 1 (point features) or 2 (adds the shape features); see
            :func:`detect_seabed`.
        prior: ``"auto"`` (default) runs the Mode 0 slab estimator when no
            z_prior is given, so the op is self-contained; ``"none"``
            searches only the fixed window.
        prior_budget: Mode 0b slab budget.
        missing: What a ping with no seabed found gets after gap filling.
            ``"nan"`` leaves it NaN, which ``create_seafloor_mask`` masks
            entirely (right for an analysis that must not integrate an
            unknown bottom). ``"max_range"`` sets it to the last recorded
            range so the whole ping is kept (right for a picture of a
            survey whose seabed is sometimes below the recording).
        line: ``"leading_edge"`` returns the seabed echo's leading edge;
            ``"integration"`` returns the Ona-Mitson backstepped line with
            the slope correction, which on steep walls clears the up-slope
            echo that a fixed buffer above the leading edge leaves behind.
        diagnostics_path: Optional zarr path for the full diagnostics
            Dataset from :func:`detect_seabed`.

    Other arguments are passed to :func:`detect_seabed`.

    Returns:
        xr.DataArray: ``seafloor_depth`` in metres, dims ``("ping_time",)``.
    """
    diag = detect_seabed(
        ds_Sv,
        channel=channel,
        r_min=r_min,
        r_max=r_max,
        z_prior=z_prior,
        sigma_prior_m=sigma_prior_m,
        regime=regime,
        n_candidates=n_candidates,
        max_slope_deg=max_slope_deg,
        alpha=alpha,
        beta_fraction=beta_fraction,
        min_score=min_score,
        walkback_db=walkback_db,
        effective_beamwidth_deg=effective_beamwidth_deg,
        vessel_speed_m_s=vessel_speed_m_s,
        min_confidence=min_confidence,
        phase_weight=phase_weight,
        transition=transition,
        pulse_length_s=pulse_length_s,
        beamwidth_deg=beamwidth_deg,
        detect_aliases=detect_aliases,
        phase_threshold_deg2=phase_threshold_deg2,
        confidence_tolerance_pulses=confidence_tolerance_pulses,
        mode=mode,
        prior=prior,
        prior_budget=prior_budget,
    )
    if line not in ("leading_edge", "integration"):
        raise ValueError(f"line must be 'leading_edge' or 'integration', got {line!r}")
    source = "r_le" if line == "leading_edge" else "r_int"
    values, filled = _fill_gaps(diag[source].values, ds_Sv["ping_time"].values, max_gap_s)
    flags = diag["flags"].values.copy()
    flags[filled] |= FLAG_INTERPOLATED
    n_missing = int(np.isnan(values).sum())
    if missing == "max_range":
        values = np.where(np.isnan(values), diag["range_end_m"].values, values)
    elif missing != "nan":
        raise ValueError(f"missing must be 'nan' or 'max_range', got {missing!r}")
    diag["flags"].values[:] = flags
    if diagnostics_path is not None:
        diag.to_zarr(diagnostics_path, mode="w")

    n_ping = values.size
    coverage = float((n_ping - n_missing) / n_ping) if n_ping else 0.0
    n_flagged = int((flags & (FLAG_LOW_CONFIDENCE | FLAG_LOW_MARGIN | FLAG_WALKBACK_BOUND)).astype(bool).sum())
    reference = "surface" if diag.attrs["range_var"] == "depth" else "transducer"
    print(
        f"phase_detect_seafloor: channel {diag.attrs['primary_channel']}, mode "
        f"{diag.attrs['mode']}, regime {diag.attrs['regime']}, {'phase' if diag.attrs['has_angles'] else 'amplitude-only'}, "
        f"prior {diag.attrs['prior']}, window {diag.attrs['window_median_r_min_m']:.0f} to "
        f"{diag.attrs['window_median_r_max_m']:.0f} m, "
        f"{n_ping} pings, coverage {coverage:.1%}, {int(filled.sum())} interpolated, "
        f"{n_flagged} flagged, {diag.attrs['n_seed_components']} seed components"
    )
    if coverage < 1.0 and missing == "nan":
        print(_COVERAGE_WARNING)
    elif coverage < 1.0:
        print(f"  {n_missing} pings without a seabed were set to the last recorded range and are kept whole")

    return xr.DataArray(
        values,
        coords={"ping_time": ds_Sv["ping_time"]},
        dims=["ping_time"],
        name="seafloor_depth",
        attrs={
            "long_name": f"Seafloor depth from the phase-aware DP detector ({line.replace('_', ' ')})",
            "units": "m",
            "line": line,
            "vertical_reference": reference,
            "range_var": diag.attrs["range_var"],
            "primary_channel": diag.attrs["primary_channel"],
            "mode": diag.attrs["mode"],
            "prior": diag.attrs["prior"],
            "regime": diag.attrs["regime"],
            "preset_version": diag.attrs["preset_version"],
            "has_angles": diag.attrs["has_angles"],
            "ping_coverage": coverage,
            "missing": missing,
            "n_flagged": n_flagged,
            "n_interpolated": int(filled.sum()),
            "n_seed_components": diag.attrs["n_seed_components"],
        },
    )
