# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Inter-method comparison of seafloor detection techniques.

Every method runs on the same prepared Sv dataset and returns a line on the
same ping grid. Agreement between methods is reported alongside
reference-free sanity metrics (echo intensity at the line, contrast across
it, continuity), because none of the lines, including the .bot pick and an
analyst's Echoview line, is ground truth.
"""

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from aa_si_utils import utils
from aa_si_utils.seabed.geometry import build_geometry, crop_block, resolve_channel
from aa_si_utils.seabed.phase import nan_mean_filter
from aa_si_utils.seabed.pipeline import detect_seafloor_phase
from aa_si_utils.seabed.validation import prepare_sv, reference_line

# Blackwell (2020) angle thresholds as published, in electrical degrees
# squared; converted per channel with the transducer's angle sensitivity.
BLACKWELL_ELECTRICAL = (702.0, 282.0)


@dataclass
class Dataset:
    """A contiguous set of raw files treated as one comparison dataset."""

    name: str
    sonar_model: str
    raw_paths: list
    r_min: float
    r_max: float
    primary_khz: float = 38.0
    secondary_khz: float = 18.0
    evl_path: str = None
    regime: str = "auto"
    hdbscan_bin_m: float = 5.0
    notes: str = ""


@dataclass
class MethodResult:
    name: str
    lines: list = field(default_factory=list)
    seconds: float = 0.0
    n_ping: int = 0
    error: str = None
    experimental: bool = False

    @property
    def line(self):
        return xr.concat(self.lines, dim="ping_time") if self.lines else None


def _channel_label(ds_Sv, khz):
    return str(ds_Sv["channel"].values[resolve_channel(ds_Sv, khz)])


def _bin_skip(ds_Sv, khz, r_min):
    geom = build_geometry(ds_Sv, khz, r_min=r_min)
    return int(geom.i_lo)


def max_sv_line(ds_Sv, khz, r_min, r_max, walkback_db=10.0):
    """Baseline: range-smoothed Sv maximum per ping, backstepped to -10 dB.

    Echoview's Maximum Sv pick: the walk back from the maximum to the
    threshold crossing is unbounded, unlike the detector's one pulse
    length refinement, because the maximum can sit anywhere in the echo.
    """
    geom = build_geometry(ds_Sv, khz, r_min=r_min, r_max=r_max)
    sv = crop_block(ds_Sv, geom, "Sv")
    smooth = nan_mean_filter(sv, (1, geom.pulse_samples))
    masked = np.where(geom.window_mask() & np.isfinite(smooth), smooth, -np.inf)
    idx = np.argmax(masked, axis=1)
    le = np.full(sv.shape[0], np.nan)
    for p in range(sv.shape[0]):
        if not np.isfinite(masked[p]).any():
            continue
        peak = smooth[p, idx[p]]
        j = idx[p]
        while j > geom.i_min[p] - geom.i_lo and sv[p, j] >= peak - walkback_db:
            j -= 1
        le[p] = j
    return xr.DataArray(
        geom.sample_to_range(le + geom.i_lo),
        coords={"ping_time": ds_Sv["ping_time"]},
        dims=["ping_time"],
        name="seafloor_depth",
        attrs={"units": "m"},
    )


def hdbscan_line(ds_Sv, r_min, r_max, bin_m, primary_khz, max_points=150_000):
    """HDBSCAN detector on a range-coarsened copy of the window (experimental).

    The range bin is widened beyond ``bin_m`` when needed to keep the point
    count under ``max_points``; the detector's cost grows with the product
    of pings and bins and a 300k-point file exhausted memory.
    """
    from seabed_detection import detect_seafloor_hdbscan

    geom = build_geometry(ds_Sv, primary_khz, r_min=r_min, r_max=r_max)
    n = max(1, int(round(bin_m / geom.dr)))
    n = max(n, -(-geom.n_ping * geom.n_crop // max_points))
    sub = ds_Sv[["Sv", "depth", "frequency_nominal"]].isel(range_sample=slice(geom.i_lo, geom.i_hi))
    sub = sub.compute() if hasattr(sub, "compute") else sub
    linear = 10.0 ** (sub["Sv"] / 10.0)
    coarse = xr.Dataset(
        {
            "Sv": 10.0 * np.log10(linear.coarsen(range_sample=n, boundary="trim").mean()),
            "depth": sub["depth"].coarsen(range_sample=n, boundary="trim").mean(),
            "frequency_nominal": sub["frequency_nominal"],
        }
    )
    coarse = coarse.assign_coords(range_sample=np.arange(coarse.sizes["range_sample"]))
    n_points = coarse.sizes["ping_time"] * coarse.sizes["range_sample"]
    min_cluster = max(300, n_points // 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return detect_seafloor_hdbscan(
            coarse,
            min_cluster_size=min_cluster,
            min_samples=max(50, min_cluster // 4),
            num_feature_channels=2,
            offset_m=0.0,
            core_dist_n_jobs=1,
        )


def build_methods(dataset, ds_Sv, echodata, has_angles):
    """Callables ``() -> line`` for every method that can run on this file."""
    import echopype as ep

    primary = _channel_label(ds_Sv, dataset.primary_khz)
    r_min, r_max = dataset.r_min, dataset.r_max
    methods = {}
    if dataset.sonar_model.upper() == "EK60":
        methods["bot"] = lambda: reference_line(ds_Sv, echodata, "bot")
    if dataset.evl_path:
        methods["evl"] = lambda: reference_line(ds_Sv, echodata, "evl", dataset.evl_path)
    methods["max_sv"] = lambda: max_sv_line(ds_Sv, dataset.primary_khz, r_min, r_max)
    skip = _bin_skip(ds_Sv, dataset.primary_khz, r_min)
    methods["ep_basic"] = lambda: ep.mask.api.detect_seafloor(
        ds_Sv,
        "basic",
        {"var_name": "Sv", "channel": primary, "threshold": (-50.0, 20.0), "offset_m": 0.0, "bin_skip_from_surface": skip},
    )
    if has_angles and "angle_sensitivity_alongship" in ds_Sv:
        sens_a = float(ds_Sv["angle_sensitivity_alongship"].sel(channel=primary).values)
        sens_b = float(ds_Sv["angle_sensitivity_athwartship"].sel(channel=primary).values)
        thresholds = (-75.0, BLACKWELL_ELECTRICAL[0] / sens_a**2, BLACKWELL_ELECTRICAL[1] / sens_b**2)
        methods["ep_blackwell"] = lambda: ep.mask.api.detect_seafloor(
            ds_Sv,
            "blackwell",
            {"var_name": "Sv", "channel": primary, "threshold": thresholds, "offset": 0.0, "r0": r_min, "r1": r_max},
        )
    common = dict(r_min=r_min, r_max=r_max, regime=dataset.regime, max_gap_s=None, prior="none")
    methods["phase_m1"] = lambda: detect_seafloor_phase(ds_Sv, channel=dataset.primary_khz, **common)
    # Self-contained: no window, no prior handed in; Mode 0 sets the window.
    methods["phase_auto"] = lambda: detect_seafloor_phase(
        ds_Sv, channel=dataset.primary_khz, regime=dataset.regime, max_gap_s=None
    )
    methods["phase_m2"] = lambda: detect_seafloor_phase(ds_Sv, channel=dataset.primary_khz, mode=2, **common)
    methods["phase_m1_int"] = lambda: detect_seafloor_phase(ds_Sv, channel=dataset.primary_khz, line="integration", **common)
    if has_angles:
        stripped = ds_Sv.drop_vars(["angle_alongship", "angle_athwartship"])
        methods["phase_amp"] = lambda: detect_seafloor_phase(stripped, channel=dataset.primary_khz, **common)
    try:
        resolve_channel(ds_Sv, dataset.secondary_khz)
        methods["phase_m1_2nd"] = lambda: detect_seafloor_phase(ds_Sv, channel=dataset.secondary_khz, **common)
    except ValueError:
        pass
    methods["hdbscan"] = lambda: hdbscan_line(ds_Sv, r_min, r_max, dataset.hdbscan_bin_m, dataset.primary_khz)
    return methods


EXPERIMENTAL = {"hdbscan"}

# Variants of the phase detector; the consensus uses one line per family so
# that five near-identical phase lines cannot outvote the other methods.
VARIANTS = {"phase_m2", "phase_amp", "phase_m1_2nd", "phase_m1_int", "phase_auto"}


def sanity_metrics(line, sv, geom):
    """Reference-free checks of one line against the primary channel's Sv.

    Args:
        line: Line values in metres on ``geom.range_var``, shape (P,).
        sv: Cropped Sv block, shape (P, n_crop).
        geom: ``PingGeometry`` of the crop.

    Returns:
        dict: contrast across the line (below minus above, dB), echo
        intensity just below the line (dB), the fraction of pings whose
        below band is quieter than -60 dB, the fraction of ping-to-ping
        jumps over 10 m, the median ping-to-ping change, and coverage.
    """
    n_ping, n_crop = sv.shape
    idx = np.round(geom.range_to_sample(line)).astype(float) - geom.i_lo
    ok = np.isfinite(idx) & (idx >= 0) & (idx < n_crop)
    n_k = geom.pulse_samples
    above_lo, above_hi = geom.metres_to_samples(10.0), geom.metres_to_samples(5.0)
    lin = 10.0 ** (sv / 10.0)
    below_db = np.full(n_ping, np.nan)
    contrast = np.full(n_ping, np.nan)
    for p in np.nonzero(ok)[0]:
        i = int(idx[p])
        b = lin[p, i : min(n_crop, i + 2 * n_k + 1)]
        a = lin[p, max(0, i - above_lo) : max(0, i - above_hi)]
        if np.isfinite(b).any():
            below_db[p] = 10.0 * np.log10(np.nanmean(b))
        if np.isfinite(b).any() and np.isfinite(a).any() and a.size:
            contrast[p] = below_db[p] - 10.0 * np.log10(np.nanmean(a))
    finite = np.isfinite(line)
    steps = np.abs(np.diff(line))
    steps = steps[np.isfinite(steps)]
    return {
        "coverage": float(finite.mean()),
        "contrast_db": float(np.nanmedian(contrast)) if np.isfinite(contrast).any() else np.nan,
        "below_sv_db": float(np.nanmedian(below_db)) if np.isfinite(below_db).any() else np.nan,
        "quiet_below_fraction": float(np.mean(below_db[ok] < -60.0)) if ok.any() else np.nan,
        "jump_fraction": float(np.mean(steps > 10.0)) if steps.size else np.nan,
        "step_median_m": float(np.median(steps)) if steps.size else np.nan,
    }


def agreement(lines):
    """Pairwise agreement and deviation from the per-ping consensus.

    Args:
        lines: ``{method: (P,) array}`` on one ping grid.

    Returns:
        tuple: ``(pairwise, consensus)``; ``pairwise`` is a DataFrame of
        median absolute differences in metres (lower triangle) and the
        fraction within 5 m (upper triangle); ``consensus`` maps each method
        to its median absolute deviation from the per-ping median of one
        line per method family (``VARIANTS`` excluded) and the fraction
        within 5 m of it.
    """
    names = list(lines)
    core = [n for n in names if n not in VARIANTS] or names
    stack = np.vstack([lines[n] for n in core])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
    pairwise = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            d = np.abs(lines[a] - lines[b])
            d = d[np.isfinite(d)]
            if d.size == 0:
                continue
            if i > j:
                pairwise.loc[a, b] = float(np.median(d))
            elif i < j:
                pairwise.loc[a, b] = float(np.mean(d <= 5.0))
    consensus = {}
    for n in names:
        d = np.abs(lines[n] - median)
        d = d[np.isfinite(d)]
        consensus[n] = {
            "consensus_mad_m": float(np.median(d)) if d.size else np.nan,
            "consensus_within_5m": float(np.mean(d <= 5.0)) if d.size else np.nan,
        }
    return pairwise, consensus


def run_dataset(dataset, out_dir, methods_filter=None, skip_experimental=False):
    """Run every method on every file of a dataset and write the results.

    Args:
        dataset: ``Dataset`` definition.
        out_dir: Folder for ``metrics.csv``, ``pairwise.csv``, ``overlay.png``
            and ``lines.nc``.
        methods_filter: Optional set of method names to run.
        skip_experimental: Skip methods in ``EXPERIMENTAL``.

    Returns:
        pd.DataFrame: One row per method with speed, sanity and agreement.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    blocks, geoms, n_pings = [], [], []
    prep_seconds = 0.0
    for raw in dataset.raw_paths:
        started = time.perf_counter()
        echodata, ds_Sv, notes = prepare_sv(
            raw, dataset.sonar_model, include_bot=(dataset.sonar_model.upper() == "EK60"), use_heave=False
        )
        ds_Sv = ds_Sv.compute() if hasattr(ds_Sv, "compute") else ds_Sv
        prep_seconds += time.perf_counter() - started
        has_angles = "angle_alongship" in ds_Sv
        geom = build_geometry(ds_Sv, dataset.primary_khz, r_min=dataset.r_min, r_max=dataset.r_max)
        blocks.append(crop_block(ds_Sv, geom, "Sv"))
        geoms.append(geom)
        n_pings.append(ds_Sv.sizes["ping_time"])
        print(f"[{dataset.name}] {Path(raw).name}: {ds_Sv.sizes['ping_time']} pings, angles={has_angles}, notes={notes}")
        for name, fn in build_methods(dataset, ds_Sv, echodata, has_angles).items():
            if methods_filter and name not in methods_filter:
                continue
            if skip_experimental and name in EXPERIMENTAL:
                continue
            result = results.setdefault(name, MethodResult(name, experimental=name in EXPERIMENTAL))
            if result.error:
                continue
            started = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    line = fn()
                line = line.reindex(ping_time=ds_Sv["ping_time"])
                result.lines.append(line.astype(float))
            except Exception as exc:  # noqa: BLE001 - a failing method is a result
                result.error = f"{type(exc).__name__}: {str(exc)[:200]}"
                result.lines = []
                print(f"  {name}: FAILED {result.error}")
            result.seconds += time.perf_counter() - started
            result.n_ping += ds_Sv.sizes["ping_time"]
            if not result.error:
                print(f"  {name}: {result.seconds:.1f} s so far")

    lines = {n: r.line.values for n, r in results.items() if r.line is not None}
    rows = []
    for name, r in results.items():
        row = {"method": name, "experimental": r.experimental, "seconds": round(r.seconds, 1), "error": r.error}
        if r.line is not None:
            row["pings_per_s"] = round(r.n_ping / r.seconds, 1) if r.seconds > 0 else np.nan
            per_file = []
            offset = 0
            for sv, geom, n in zip(blocks, geoms, n_pings):
                segment = r.line.values[offset : offset + n]
                offset += n
                if segment.size != n:
                    continue
                per_file.append((n, sanity_metrics(segment, sv, geom)))
            total = sum(n for n, _ in per_file)
            for key in per_file[0][1] if per_file else ():
                row[key] = float(np.nansum([n * m[key] for n, m in per_file]) / total)
        rows.append(row)
    pairwise, consensus = agreement(lines) if len(lines) > 1 else (pd.DataFrame(), {})
    for row in rows:
        row.update(consensus.get(row["method"], {}))
    table = pd.DataFrame(rows).set_index("method")
    table.insert(0, "dataset", dataset.name)
    table.to_csv(out_dir / "metrics.csv")
    pairwise.to_csv(out_dir / "pairwise.csv")
    if lines:
        ping_time = np.concatenate([r.line["ping_time"].values for r in results.values() if r.line is not None][:1])
        xr.Dataset({n: ("ping_time", v) for n, v in lines.items()}, coords={"ping_time": ping_time}).to_netcdf(out_dir / "lines.nc")
        plot_overlay(dataset, blocks, geoms, lines, out_dir / "overlay.png")
    (out_dir / "dataset.json").write_text(
        json.dumps(
            {
                "name": dataset.name,
                "sonar_model": dataset.sonar_model,
                "files": [Path(p).name for p in dataset.raw_paths],
                "n_ping": int(sum(n_pings)),
                "r_min": dataset.r_min,
                "r_max": dataset.r_max,
                "primary_khz": dataset.primary_khz,
                "prepare_seconds": round(prep_seconds, 1),
                "notes": dataset.notes,
            },
            indent=2,
        )
    )
    return table


def plot_overlay(dataset, blocks, geoms, lines, path):
    """Primary-channel echogram of the window with every method's line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = min(b.shape[1] for b in blocks)
    sv = np.concatenate([b[:, :width] for b in blocks], axis=0)
    y = geoms[0].crop_range_axis()[0, :width]
    x = np.arange(sv.shape[0])
    step = max(1, sv.shape[1] // 1500)
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.pcolormesh(x, y[::step], sv[:, ::step].T, cmap="viridis", vmin=-90, vmax=-30, shading="auto")
    styles = {
        "bot": ("white", "--", 1.0),
        "evl": ("white", "-", 1.2),
        "max_sv": ("orange", "-", 0.6),
        "ep_basic": ("yellow", "-", 0.6),
        "ep_blackwell": ("magenta", "-", 0.6),
        "hdbscan": ("grey", "-", 0.6),
        "phase_m1": ("red", "-", 1.0),
        "phase_m2": ("darkred", "--", 0.8),
        "phase_m1_int": ("cyan", "-", 0.6),
        "phase_amp": ("lime", "--", 0.7),
        "phase_m1_2nd": ("pink", ":", 0.8),
        "phase_auto": ("blue", "-", 1.0),
    }
    for name, values in lines.items():
        color, ls, lw = styles.get(name, ("black", "-", 0.6))
        ax.plot(x, values, color=color, ls=ls, lw=lw, label=name)
    ax.set_ylim(y.max(), y.min())
    ax.set_xlabel("ping")
    ax.set_ylabel(f"{geoms[0].range_var} (m)")
    ax.set_title(f"{dataset.name}: {dataset.sonar_model}, {int(dataset.primary_khz)} kHz, {sv.shape[0]} pings")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
