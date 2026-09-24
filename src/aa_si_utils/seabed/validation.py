# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Run the detector on a raw file and compare it with a reference line.

The reference is the echosounder's own .bot pick or an Echoview .evl line.
Neither is ground truth: .bot picks are unreliable in places and analyst
lines carry about a pulse length of uncertainty, so the numbers here measure
agreement with the reference, not accuracy. The overlay plot is the check
that decides which of the two is wrong where they disagree.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from aa_si_utils import utils
from aa_si_utils.seabed.pipeline import (
    FLAG_LOW_CONFIDENCE,
    FLAG_LOW_MARGIN,
    FLAG_WALKBACK_BOUND,
    detect_seafloor_phase,
)

REVIEW_FLAGS = FLAG_LOW_CONFIDENCE | FLAG_LOW_MARGIN | FLAG_WALKBACK_BOUND


def prepare_sv(raw_path, sonar_model, include_bot=False, waveform_mode="CW", encode_mode="power", use_heave=True):
    """Open a raw file and build the Sv dataset the detector expects.

    Adds surface-referenced depth from the Platform group, split-beam
    angles when the file's calibration allows it, and position when the
    file carries NMEA data.

    Args:
        raw_path: Path to the .raw file.
        sonar_model: ``"EK60"`` or ``"EK80"``.
        include_bot: Read the matching .bot file for a reference line.
        waveform_mode: EK80 waveform mode.
        encode_mode: EK80 encode mode.
        use_heave: Add per-ping heave to the transducer depth. False gives
            a depth grid that is identical on every ping, which echopype's
            own seafloor detectors require.

    Returns:
        tuple: ``(echodata, ds_Sv, notes)`` where ``notes`` lists the
        optional steps that failed.
    """
    import echopype as ep

    notes = []
    echodata = ep.open_raw(str(raw_path), sonar_model=sonar_model, include_bot=include_bot)
    if sonar_model.upper() == "EK60":
        ds_Sv = ep.calibrate.compute_Sv(echodata)
    else:
        ds_Sv = ep.calibrate.compute_Sv(echodata, waveform_mode=waveform_mode, encode_mode=encode_mode)
    transducer_depth = utils.compute_transducer_depth(echodata, ds_Sv, use_heave=use_heave)
    ds_Sv = ep.consolidate.add_depth(ds_Sv, depth_offset=transducer_depth)
    try:
        ds_Sv = ep.consolidate.add_splitbeam_angle(
            ds_Sv, echodata, waveform_mode=waveform_mode, encode_mode=encode_mode, to_disk=False
        )
    except Exception as exc:  # noqa: BLE001 - any failure means amplitude-only
        notes.append(f"no split-beam angles: {type(exc).__name__}: {exc}")
    try:
        ds_Sv = ep.consolidate.add_location(ds_Sv, echodata)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"no position: {type(exc).__name__}: {exc}")
    return echodata, ds_Sv, notes


def reference_line(ds_Sv, echodata, reference, evl_path=None):
    """Reference seafloor line on ``ds_Sv``'s ping grid.

    Args:
        ds_Sv: Depth-carrying Sv dataset.
        echodata: EchoData with .bot data when ``reference`` is ``"bot"``.
        reference: ``"bot"`` or ``"evl"``.
        evl_path: A .evl file or a folder of them for ``"evl"``.

    Returns:
        xr.DataArray: Surface-referenced line in metres, ``(ping_time,)``.
    """
    if reference == "bot":
        return utils.detect_seafloor(ds_Sv=ds_Sv, echodata=echodata)
    if reference == "evl":
        if evl_path is None:
            raise ValueError("evl_path is required for an evl reference")
        times = ds_Sv["ping_time"].values
        return utils.read_seafloor_line_evl(
            ds_Sv,
            evl_path=str(evl_path),
            file_time_start=str(np.datetime64(times.min(), "s")),
            file_time_end=str(np.datetime64(times.max(), "s")),
        )
    raise ValueError(f"reference must be 'bot' or 'evl', got {reference!r}")


def line_metrics(line, ref, diag):
    """Agreement between the detector's line and a reference.

    Args:
        line: Detector line, ``(ping_time,)`` DataArray.
        ref: Reference line on the same pings.
        diag: Diagnostics Dataset from the run.

    Returns:
        dict: The spec's validation metrics. Positive bias means the
        detector picked deeper than the reference.
    """
    pulse = float(diag.attrs["pulse_length_m"])
    a = np.asarray(line.values, dtype=float)
    b = np.asarray(ref.values, dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    err = a[both] - b[both]
    abs_err = np.abs(err)
    gross_threshold = np.maximum(5.0, 0.01 * b[both])
    gross = abs_err > gross_threshold

    flags = diag["flags"].values
    flagged = (flags & REVIEW_FLAGS).astype(bool)
    flagged_compared = flagged[both]

    alias = diag["alias"].values
    range_axis = diag["range_m"].values
    inside_alias = []
    for p in np.nonzero(np.isfinite(a))[0]:
        col = int(np.clip(np.round((a[p] - range_axis[p, 0]) / (range_axis[p, 1] - range_axis[p, 0])), 0, alias.shape[1] - 1))
        inside_alias.append(bool(alias[p, col]))

    return {
        "n_ping": int(a.size),
        "n_compared": int(both.sum()),
        "line_coverage": float(np.isfinite(a).mean()),
        "reference_coverage": float(np.isfinite(b).mean()),
        "pulse_length_m": pulse,
        "median_abs_deviation_m": float(np.median(abs_err)) if err.size else np.nan,
        "within_pulse_agreement": float(np.mean(abs_err <= pulse)) if err.size else np.nan,
        "gross_error_rate": float(np.mean(gross)) if err.size else np.nan,
        "deep_pick_bias_m": float(np.mean(err)) if err.size else np.nan,
        "alias_false_accept_rate": float(np.mean(inside_alias)) if inside_alias else np.nan,
        "flagged_fraction": float(flagged.mean()),
        "flag_precision": float(np.mean(gross[flagged_compared])) if flagged_compared.any() else np.nan,
        "n_seed_components": int(diag.attrs["n_seed_components"]),
        "primary_channel": str(diag.attrs["primary_channel"]),
        "regime": str(diag.attrs["regime"]),
        "has_angles": bool(diag.attrs["has_angles"]),
    }


def plot_overlay(diag, line, ref, path, title=""):
    """Echogram of the primary channel with the reference and detector lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sv = diag["f1"].values
    y = diag["range_m"].values.mean(axis=0)
    x = np.arange(sv.shape[0])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.pcolormesh(x, y, sv.T, cmap="viridis", vmin=-90, vmax=-30, shading="auto")
    ax.plot(x, np.asarray(ref.values, dtype=float), color="white", lw=1.0, ls="--", label="reference")
    ax.plot(x, diag["r_star"].values, color="orange", lw=0.6, alpha=0.7, label="raw DP pick")
    ax.plot(x, np.asarray(line.values, dtype=float), color="red", lw=1.0, label="leading edge")
    ax.plot(x, diag["r_int"].values, color="cyan", lw=0.8, label="integration line")
    flagged = (diag["flags"].values & REVIEW_FLAGS).astype(bool)
    if flagged.any():
        ax.plot(x[flagged], np.asarray(line.values, dtype=float)[flagged], "m^", ms=4, label="flagged")
    ax.set_ylim(y.max(), y.min())
    ax.set_xlabel("ping")
    ax.set_ylabel(f"{diag.attrs['range_var']} (m)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_validation(
    raw_path,
    sonar_model,
    reference,
    evl_path=None,
    out_dir=None,
    r_min=None,
    r_max=None,
    window_margin=0.5,
    **detector_kwargs,
):
    """Detect, compare and save metrics, an overlay and diagnostics.

    Args:
        raw_path: Raw file.
        sonar_model: ``"EK60"`` or ``"EK80"``.
        reference: ``"bot"`` or ``"evl"``.
        evl_path: Line file or folder for ``"evl"``.
        out_dir: Folder for ``metrics.json``, ``overlay.png`` and
            ``diagnostics.zarr``; None writes nothing.
        r_min: Search window lower bound; default from the reference.
        r_max: Search window upper bound; default from the reference.
        window_margin: Fraction of the reference span added on both sides
            when the window is taken from the reference.
        **detector_kwargs: Passed to :func:`detect_seafloor_phase`.

    Returns:
        tuple: ``(metrics, diag, line, ref)``.
    """
    raw_path = Path(raw_path)
    started = time.perf_counter()
    echodata, ds_Sv, notes = prepare_sv(raw_path, sonar_model, include_bot=(reference == "bot"))
    ref = reference_line(ds_Sv, echodata, reference, evl_path)
    ref_values = np.asarray(ref.values, dtype=float)
    finite = ref_values[np.isfinite(ref_values)]
    if finite.size == 0:
        raise ValueError("the reference line has no finite values on this file")
    if r_min is None:
        r_min = max(10.0, float(finite.min()) * (1.0 - window_margin))
    if r_max is None:
        r_max = float(finite.max()) * (1.0 + window_margin)
    prep_seconds = time.perf_counter() - started

    out_dir = Path(out_dir) if out_dir is not None else None
    diagnostics_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path = str(out_dir / "diagnostics.zarr")
    started = time.perf_counter()
    line = detect_seafloor_phase(ds_Sv, r_min=r_min, r_max=r_max, diagnostics_path=diagnostics_path, **detector_kwargs)
    detect_seconds = time.perf_counter() - started
    if diagnostics_path is None:
        from aa_si_utils.seabed.pipeline import detect_seabed

        diag = detect_seabed(ds_Sv, r_min=r_min, r_max=r_max, **_detect_kwargs(detector_kwargs))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diag = xr.open_zarr(diagnostics_path).load()

    metrics = line_metrics(line, ref, diag)
    metrics.update(
        {
            "raw_file": raw_path.name,
            "sonar_model": sonar_model,
            "reference": reference,
            "r_min": float(r_min),
            "r_max": float(r_max),
            "prepare_seconds": round(prep_seconds, 1),
            "detect_seconds": round(detect_seconds, 1),
            "notes": notes,
        }
    )
    if out_dir is not None:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
        plot_overlay(diag, line, ref, out_dir / "overlay.png", title=f"{raw_path.name} vs {reference}")
    return metrics, diag, line, ref


def _detect_kwargs(detector_kwargs):
    return {k: v for k, v in detector_kwargs.items() if k not in ("max_gap_s", "diagnostics_path")}


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)
