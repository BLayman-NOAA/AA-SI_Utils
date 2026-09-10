# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Planning helpers for the NEFSC sperm whale dive-profile analysis.

The NEFSC use case describes each localized dive with a JSON config naming the
netCDF files it was analysed from and the Echoview lines that bound it. These
helpers turn a folder of those configs into the window dicts a recipe fans a
mapped chain out over, one instance per dive.

They are deliberately not registry ops. The JSON layout is specific to this one
use case, so the recipe wires them as inline ``op: custom`` steps and only the
generally useful pieces they lean on (``select_ping_time_range``,
``add_line_overlay``) are registered ops.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import _parse_evl

logger = logging.getLogger(__name__)


FILE_STAMP = re.compile(r"D(\d{8})-T(\d{6})")
"""The ``D{YYYYMMDD}-T{HHMMSS}`` stamp every HB1603 data file name carries.

Matched anywhere in the name so that both spellings in the configs parse: the
July files are listed as ``HB1603_L1-D20160703-T183957.nc`` and the later ones
as the bare ``D20160725-T201239.nc``.
"""

DIVE_LINE_KEYS = {
    "SWdiveprofile": "dive_fit_evl",
    "U99CI": "dive_u99_evl",
    "L99CI": "dive_l99_evl",
}
"""EV_lines keys holding a single dive's fitted depth and 99% bounds."""

SEABED_LINE_KEY = "ev_bottom"


CSV_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S.%f"

#: Code for a ping of the dive that no coded cell fell in. Distinct from
#: HDBSCAN's -1, which marks a cell the clusterer saw and called noise; this
#: marks a ping the clusterer had nothing to say about at all. Both are real
#: outcomes and both count toward the dive's duration.
UNCODED = -2
"""Timestamp spelling used by the reference Sv_codes CSVs, to hundredths."""


def generate_sv_codes(
    ds,
    windows,
    output_dir,
    label_grid_var=None,
    dataset_name="ml_dataset",
    ml_result_name="hdbscan_results",
    range_var="depth",
    data_var="Sv",
    dive_fit_var="dive_fit",
    upper_var="dive_u99",
    lower_var="dive_l99",
    ping_time_bin_s=10.0,
    include_noise=True,
    frequencies_khz=None,
):
    """Write per-dive SVCode CSVs from an embedded clustering result.

    The HDBSCAN replacement for ``EK_CodeSv.py``'s pairwise frequency code. For
    each dive it takes the MVBS cells lying between the 99% confidence bounds of
    the dive profile, records each cell's cluster label as its code, and also
    reduces each ping to the single code that dominates it.

    Two products, because the analysis needs both. The per-cell CSV matches the
    layout of the reference ``Sv_codes`` files, so a run can be diffed against
    them cell for cell. The per-ping summary is what the duration tables are
    built from: a ping contributes its whole bin to one code, which is what makes
    the per-code minutes add up to the dive length.

    Args:
        ds (xr.Dataset): Embedded clustering result, carrying the gridded cluster
            labels, the Sv the codes describe, and the three dive lines.
        windows (list): Window dicts from :func:`plan_dive_datasets`. Each one's
            ping_time span selects its dive out of the combined dataset.
        output_dir (str | Path): Folder the CSVs are written into.
        label_grid_var (str or None): Gridded cluster-label variable. Defaults to
            ``"{dataset_name}_{ml_result_name}_grid"``, which is what
            ``embed_clustering_results`` writes.
        dataset_name (str): ML dataset name, used to build the default label
            variable name.
        ml_result_name (str): Clustering result name, likewise.
        range_var (str): Vertical coordinate of the grid. Defaults to "depth",
            since the dive lines are metres below the surface.
        data_var (str): Sv variable written into the CSV columns.
        dive_fit_var (str): Fitted dive-depth variable.
        upper_var (str): Upper (shallower) confidence bound variable.
        lower_var (str): Lower (deeper) confidence bound variable.
        ping_time_bin_s (float): Seconds one ping bin represents, used to turn a
            ping count into a duration. Should match compute_mvbs's
            ping_time_bin.
        include_noise (bool): Keep cells HDBSCAN labelled noise (-1) as their own
            code. True by default, so the per-code durations account for the
            whole dive rather than silently dropping the unclustered part.
        frequencies_khz (list[float] | None): Frequencies in kHz to write as Sv
            columns, overriding each window's own ``frequencies_khz``. None uses
            the window's list, and falls back to every channel carrying data.

    Returns:
        dict: With keys ``code_csv_paths`` (one path per dive),
        ``summary_csv_path`` (the per-ping summary feeding
        :func:`sv_code_depth_table`), and ``dive_labels``.

    Raises:
        KeyError: If the label grid or a dive line is missing from *ds*.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_grid_var = label_grid_var or f"{dataset_name}_{ml_result_name}_grid"
    if label_grid_var not in ds:
        raise KeyError(
            f"Gridded cluster labels {label_grid_var!r} not in the dataset. "
            f"Run embed_clustering_results first, or pass label_grid_var. "
            f"Available: {sorted(ds.data_vars)}"
        )
    for name in (dive_fit_var, upper_var, lower_var):
        if name not in ds:
            raise KeyError(
                f"Dive line {name!r} not in the dataset; attach it with "
                f"the add_line_overlay op before generating codes"
            )

    csv_paths = []
    summaries = []
    labels_written = []
    for window in windows:
        label = window["label"]
        subset = ds.sel(ping_time=slice(window["start"], window["end"]))
        if subset.sizes.get("ping_time", 0) == 0:
            logger.warning("%s: no pings in the dataset for this window", label)
            continue

        cells, per_ping = _codes_for_one_dive(
            subset, label, label_grid_var, range_var, data_var,
            dive_fit_var, upper_var, lower_var, ping_time_bin_s, include_noise,
            frequencies_khz or window.get("frequencies_khz"),
        )
        if cells.empty:
            logger.warning("%s: no cells fell between the confidence bounds", label)
            continue

        path = output_dir / f"{label}.csv"
        cells.to_csv(path, index=False)
        csv_paths.append(path.as_posix())
        summaries.append(per_ping)
        labels_written.append(label)
        logger.info(
            "%s: %d cell(s) over %d ping(s), codes %s",
            label, len(cells), len(per_ping), sorted(cells["code"].unique())
        )

    if not summaries:
        raise ValueError(
            "No dive produced any coded cells. Check that the dive lines and the "
            "clustering grid share a ping_time range."
        )

    summary = pd.concat(summaries, ignore_index=True)
    summary_path = output_dir / "sv_code_ping_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(
        "Wrote %d dive CSV(s) and a %d-ping summary to %s",
        len(csv_paths), len(summary), output_dir
    )

    return {
        "code_csv_paths": csv_paths,
        "summary_csv_path": summary_path.as_posix(),
        "dive_labels": labels_written,
    }


def _codes_for_one_dive(
    ds, label, label_grid_var, range_var, data_var,
    dive_fit_var, upper_var, lower_var, ping_time_bin_s, include_noise,
    frequencies_khz=None,
):
    """Per-cell rows and per-ping dominant codes for a single dive.

    Args:
        ds (xr.Dataset): This dive's slice of the combined dataset.
        label (str): Dive label, carried into both frames.
        label_grid_var (str): Gridded cluster-label variable.
        range_var (str): Vertical coordinate name.
        data_var (str): Sv variable.
        dive_fit_var (str): Fitted dive-depth variable.
        upper_var (str): Shallower confidence bound.
        lower_var (str): Deeper confidence bound.
        ping_time_bin_s (float): Seconds per ping bin.
        include_noise (bool): Keep label -1 cells.
        frequencies_khz (list[float] | None): Frequencies to write as Sv
            columns. None keeps every channel carrying data.

    Returns:
        tuple: (cells DataFrame, per-ping DataFrame).
    """
    labels = ds[label_grid_var]
    depths = ds[range_var]
    upper = ds[upper_var]
    lower = ds[lower_var]

    # The band the whale's localized position could have occupied. Both bounds
    # are metres below the surface, so "upper" is the smaller number.
    in_band = (depths >= upper) & (depths <= lower)
    valid = in_band & labels.notnull()
    if not include_noise:
        valid = valid & (labels >= 0)

    ping_idx, range_idx = np.where(valid.transpose("ping_time", range_var).values)
    if ping_idx.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    ping_times = ds["ping_time"].values[ping_idx]
    cell_depths = depths.values[range_idx]
    codes = (
        labels.transpose("ping_time", range_var)
        .values[ping_idx, range_idx]
        .astype(int)
    )

    frame = {
        "DateTime": [
            pd.Timestamp(t).strftime(CSV_TIME_FORMAT)[:-4] for t in ping_times
        ],
        "depth_m": cell_depths,
    }
    for column, values in _sv_columns(ds, data_var, frequencies_khz).items():
        frame[column] = np.round(
            values.transpose("ping_time", range_var).values[ping_idx, range_idx], 2
        )
    frame["code"] = codes
    cells = pd.DataFrame(frame)

    per_ping = _dominant_code_per_ping(
        ds, label, ping_idx, codes, cell_depths,
        dive_fit_var, ping_time_bin_s,
    )
    return cells, per_ping


def _sv_columns(ds, data_var, frequencies_khz=None):
    """Sv DataArrays keyed by the CSV column name for each channel kept.

    Names follow the reference files (``18kHz_Sv``), taken from
    ``frequency_nominal`` when the dataset carries it and from the channel
    coordinate otherwise.

    Which channels are kept matters. ``create_frequency_mask`` masks the
    channels an analysis does not use to NaN but leaves them in the dataset, so
    writing one column per channel puts three all-NaN columns beside the two
    real ones and the file no longer matches the reference layout. Passing the
    analysis frequencies keeps only those; without them, any channel that is
    entirely NaN over this dive is dropped, which recovers the same answer for a
    masked dataset without having to be told.

    Args:
        ds (xr.Dataset): Dataset holding *data_var*.
        data_var (str): Sv variable name.
        frequencies_khz (list[float] | None): Frequencies in kHz to write, in
            the order the columns should appear. None keeps every channel that
            carries data.

    Returns:
        dict: Column name to the 2-D DataArray for that channel.

    Raises:
        ValueError: If a requested frequency is not among the dataset's
            channels.
    """
    if data_var not in ds:
        return {}
    data = ds[data_var]
    if "channel" not in data.dims:
        return {f"{data_var}": data}

    has_freq = "frequency_nominal" in ds
    if has_freq:
        khz = [
            float(hz) / 1000.0
            for hz in np.atleast_1d(ds["frequency_nominal"].values)
        ]
        names = [f"{int(round(value))}kHz_Sv" for value in khz]
    else:
        khz = [None] * data.sizes["channel"]
        names = [f"{str(ch)}_Sv" for ch in np.atleast_1d(ds["channel"].values)]

    if frequencies_khz:
        if not has_freq:
            raise ValueError(
                "frequencies_khz was given but the dataset has no "
                "'frequency_nominal' to match it against"
            )
        keep = []
        for wanted in frequencies_khz:
            matches = [i for i, value in enumerate(khz)
                       if abs(value - float(wanted)) < 0.5]
            if not matches:
                raise ValueError(
                    f"no channel at {wanted} kHz; the dataset carries "
                    f"{[round(v, 1) for v in khz]}"
                )
            keep.append(matches[0])
    else:
        keep = [
            index for index in range(data.sizes["channel"])
            if bool(np.isfinite(data.isel(channel=index).values).any())
        ]

    return {names[index]: data.isel(channel=index) for index in keep}


def _dominant_code_per_ping(
    ds, label, ping_idx, codes, cell_depths, dive_fit_var, ping_time_bin_s,
):
    """Reduce each ping's cells to the one code that dominates it.

    Most cells wins. Ties go to the code on the cell nearest the fitted dive
    depth, which is the best single estimate of where the whale actually was,
    rather than to whichever label sorts first.

    Every ping of the dive gets a row, including those no coded cell fell in;
    those carry :data:`UNCODED`. The dive's duration is the sum of its rows, so
    dropping the empty ones would shrink the denominator to the covered pings
    and report them as the whole dive.

    Args:
        ds (xr.Dataset): This dive's slice.
        label (str): Dive label.
        ping_idx (np.ndarray): Ping index per coded cell.
        codes (np.ndarray): Cluster label per coded cell.
        cell_depths (np.ndarray): Depth per coded cell.
        dive_fit_var (str): Fitted dive-depth variable.
        ping_time_bin_s (float): Seconds per ping bin.

    Returns:
        pd.DataFrame: One row per ping with the dominant code.
    """
    ping_times = ds["ping_time"].values
    fitted = ds[dive_fit_var].values
    coded = set(np.unique(ping_idx).tolist())

    rows = []
    for position in range(len(ping_times)):
        if position not in coded:
            # A ping the whale was present for but that produced no coded
            # cell. It still has to appear, or the duration it represents
            # leaves the denominator and the percentages below report a
            # fraction of the pings that happened to survive rather than a
            # fraction of the dive.
            rows.append({
                "label": label,
                "ping_time": pd.Timestamp(ping_times[position]).isoformat(),
                "code": UNCODED,
                "n_cells": 0,
                "code_cells": 0,
                "dive_depth_m": float(fitted[position]),
                "duration_s": float(ping_time_bin_s),
            })
            continue
        mask = ping_idx == position
        ping_codes = codes[mask]
        depth_offset = np.abs(cell_depths[mask] - fitted[position])

        values, counts = np.unique(ping_codes, return_counts=True)
        best = values[counts == counts.max()]
        if len(best) == 1:
            dominant = int(best[0])
        else:
            nearest = np.argmin(
                [depth_offset[ping_codes == value].min() for value in best]
            )
            dominant = int(best[nearest])

        rows.append({
            "label": label,
            "ping_time": pd.Timestamp(ping_times[position]).isoformat(),
            "code": dominant,
            "n_cells": int(mask.sum()),
            "code_cells": int((ping_codes == dominant).sum()),
            "dive_depth_m": float(fitted[position]),
            "duration_s": float(ping_time_bin_s),
        })
    return pd.DataFrame(rows)


def sv_code_depth_table(
    summary_csv_path,
    output_dir,
    depth_interval_m=200.0,
    max_depth_m=None,
):
    """Percent of dive duration spent at each depth interval, per code.

    Builds the table behind the report's "Percent of total dive duration the
    whale spent at 200 m depth intervals per code". Each ping contributes its
    whole bin to one interval and one code, so a dive's percentages sum to 100.

    "Total dive duration" is every ping of the dive, not only the ones that
    produced a coded cell: :func:`generate_sv_codes` writes a row per ping and
    marks the empty ones :data:`UNCODED`, so they hold their share of the
    denominator and show up as their own code in the table. A dive whose cells
    are mostly masked away therefore reports a large UNCODED share rather than
    reporting its handful of surviving pings as the whole dive.

    Depth intervals come from the fitted dive profile, not from the cells: the
    question is how long the whale spent at a depth, and the code says what was
    around it there.

    Args:
        summary_csv_path (str | Path): Per-ping summary from
            :func:`generate_sv_codes`.
        output_dir (str | Path): Folder the tables are written into.
        depth_interval_m (float): Interval width in metres. Defaults to 200.
        max_depth_m (float or None): Highest interval edge. Defaults to the
            deepest fitted depth, rounded up to the next interval.

    Returns:
        dict: With keys ``per_dive_csv_path``, ``combined_csv_path`` and
        ``totals_csv_path``.

    Raises:
        ValueError: If the summary holds no rows.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_csv_path)
    if summary.empty:
        raise ValueError(f"No ping rows in {summary_csv_path}")

    deepest = max_depth_m if max_depth_m is not None else summary["dive_depth_m"].max()
    # floor + 1, not ceil. Bins are left-closed, so a depth landing exactly on
    # an edge belongs to the interval starting there and needs a bin above it:
    # with ceil, a dive bottoming at exactly 400 m on 200 m intervals gets
    # edges [0, 200, 400], and pd.cut puts 400 outside every bin. That ping
    # then vanishes from the table and the percentages sum to less than 100.
    n_bins = int(np.floor(deepest / depth_interval_m)) + 1
    edges = np.arange(n_bins + 1) * depth_interval_m
    labels = [f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(n_bins)]
    summary["depth_interval_m"] = pd.cut(
        summary["dive_depth_m"], bins=edges, labels=labels,
        include_lowest=True, right=False,
    )

    # Nothing may fall outside: an unbinned row is silently dropped by the
    # grouping below, which is exactly how the bug above hid itself.
    unbinned = summary["depth_interval_m"].isna()
    if unbinned.any():
        depths = summary.loc[unbinned, "dive_depth_m"]
        if depths.isna().any():
            raise ValueError(
                f"{int(depths.isna().sum())} ping(s) have no fitted dive depth, "
                f"so they cannot be placed in a depth interval. The dive line "
                f"does not cover the whole window."
            )
        raise ValueError(
            f"{int(unbinned.sum())} ping(s) fall outside the depth bins "
            f"0 to {int(edges[-1])} m: depths "
            f"{sorted(depths.unique())[:5]}. Raise max_depth_m."
        )

    per_dive = (
        summary.groupby(["label", "depth_interval_m", "code"], observed=True)
        ["duration_s"].sum().reset_index()
    )
    dive_totals = summary.groupby("label", observed=True)["duration_s"].sum()
    per_dive["duration_min"] = per_dive["duration_s"] / 60.0
    per_dive["percent_of_dive"] = (
        100.0 * per_dive["duration_s"] / per_dive["label"].map(dive_totals)
    )

    combined = (
        summary.groupby(["depth_interval_m", "code"], observed=True)
        ["duration_s"].sum().reset_index()
    )
    combined["duration_min"] = combined["duration_s"] / 60.0
    combined["percent_of_total"] = (
        100.0 * combined["duration_s"] / summary["duration_s"].sum()
    )

    totals = (
        summary.groupby(["label", "code"], observed=True)["duration_s"]
        .sum().reset_index()
    )
    totals["duration_min"] = totals["duration_s"] / 60.0
    totals["percent_of_dive"] = (
        100.0 * totals["duration_s"] / totals["label"].map(dive_totals)
    )

    paths = {}
    for name, frame in (
        ("per_dive_csv_path", per_dive),
        ("combined_csv_path", combined),
        ("totals_csv_path", totals),
    ):
        path = output_dir / f"sv_code_{name.replace('_csv_path', '')}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path.as_posix()

    logger.info(
        "Depth-interval tables over %d dive(s), %d interval(s), codes %s",
        summary["label"].nunique(), len(labels), sorted(summary["code"].unique())
    )
    for _, row in totals.iterrows():
        logger.info(
            "  %-28s code %3d: %5.1f min (%4.1f%%)",
            row["label"], int(row["code"]), row["duration_min"],
            row["percent_of_dive"],
        )
    return paths


def plan_dive_datasets(
    json_dir,
    evl_root=None,
    pad_minutes=0.0,
    include_multi_dive=False,
    raw_suffix=".raw",
    include_labels=None,
):
    """Turn a folder of NEFSC dive configs into recipe window dicts.

    Each returned window describes one localized dive: the ping_time span to cut
    out of the survey Sv, the Echoview lines that bound it, and the binning the
    original analysis used. A recipe maps a chain over ``windows`` and passes
    ``${_item}`` to ``select_ping_time_range`` and to the line-attaching steps.

    The span comes from the dive profile line, not from the netCDF file list.
    The files are much wider than the dive they contain: for SWD_20160707-OE20
    the three files span about 68 minutes while the localized dive is 17.3, so
    taking the file span would carry four times the pings through clustering for
    nothing. ``raw_files`` still records the file stems, since those are what the
    survey tier had to process for the window to exist at all.

    Configs describing several dives at once are skipped by default. In the
    HB1603 set that is ``SWD_20160707-OE20-OE29-OE32.json``, whose three dives
    each have their own single-dive config, and which carries no 99% confidence
    lines, so it cannot drive the SVCode step that needs them.

    Args:
        json_dir (str | Path): Folder of ``SWD_*.json`` configs.
        evl_root (str | Path | None): Folder searched recursively for the line
            files a config names. Only the basename is matched, since the paths
            in the configs point at the machine the original analysis ran on.
            Defaults to *json_dir*'s parent.
        pad_minutes (float): Minutes added to each side of the dive span.
            Defaults to 0.0.
        include_multi_dive (bool): Keep configs describing several dives.
            Defaults to False.
        raw_suffix (str): Suffix put on the file stems in ``raw_files``.
            Defaults to ".raw", since the configs name the netCDF conversions
            rather than the raw files the recipe reads.
        include_labels (list[str] | None): Keep only these dive labels, which
            are the config stems such as "SWD_20160707-OE20". None (default)
            keeps every dive.

            For running the workflow over a slice of the survey rather than the
            whole of it. A window whose pings are not in the Sv store makes
            ``select_ping_time_range`` fail rather than return nothing, which
            is the right behaviour for a full run and the wrong one for a
            smoke test, so a sliced run names the dives its slice contains.

    Returns:
        dict: With keys ``windows`` (list of window dicts, ordered by start
        time) and ``raw_files`` (sorted, deduplicated file names across every
        window).

    Raises:
        FileNotFoundError: If *json_dir* holds no configs, or a line file a
            config names cannot be found under *evl_root*.
        ValueError: If a line file is named more than once under *evl_root*, or
            if *include_labels* names a dive no config provides.
    """
    json_dir = Path(json_dir)
    evl_root = Path(evl_root) if evl_root is not None else json_dir.parent

    config_paths = sorted(json_dir.glob("*.json"))
    if not config_paths:
        raise FileNotFoundError(f"No .json configs found in {json_dir}")

    wanted = set(include_labels) if include_labels else None
    if wanted is not None:
        available = {path.stem for path in config_paths}
        unknown = sorted(wanted - available)
        if unknown:
            raise ValueError(
                f"include_labels names dives with no config in {json_dir}: "
                f"{unknown}. Available: {sorted(available)}"
            )

    index = _index_evl_files(evl_root)
    windows = []
    skipped = []
    for config_path in config_paths:
        if wanted is not None and config_path.stem not in wanted:
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        ev_lines = config["sub_selection"]["EV_lines"]
        if not all(key in ev_lines for key in DIVE_LINE_KEYS):
            skipped.append(config_path.name)
            if not include_multi_dive:
                continue
        windows.append(
            _window_from_config(config, config_path, index, pad_minutes, raw_suffix)
        )

    windows.sort(key=lambda window: window["start"])
    raw_files = sorted({name for w in windows for name in w["raw_files"]})

    logger.info(
        "Planned %d dive window(s) from %d config(s); %d raw file(s) involved",
        len(windows), len(config_paths), len(raw_files)
    )
    if skipped:
        verb = "Included" if include_multi_dive else "Skipped"
        logger.info(
            "%s %d config(s) without a single-dive line triplet: %s",
            verb, len(skipped), ", ".join(skipped)
        )
    for window in windows:
        logger.info(
            "  %-28s %s .. %s  (%.1f min, %d file(s))",
            window["label"], window["start"], window["end"],
            window["duration_minutes"], len(window["raw_files"])
        )

    return {"windows": windows, "raw_files": raw_files}


def _window_from_config(config, config_path, index, pad_minutes, raw_suffix):
    """Build one window dict from a parsed config."""
    ev_lines = config["sub_selection"]["EV_lines"]
    label = config_path.stem

    lines = {}
    for source_key, window_key in DIVE_LINE_KEYS.items():
        if source_key in ev_lines:
            lines[window_key] = _resolve_line(ev_lines[source_key], index, label)
    if SEABED_LINE_KEY in ev_lines:
        lines["seabed_evl"] = _resolve_line(ev_lines[SEABED_LINE_KEY], index, label)

    span_source = lines.get("dive_fit_evl")
    if span_source is None:
        # A multi-dive config: fall back to the union of its dive profile lines.
        span_source = [
            _resolve_line(entry, index, label)
            for key, entry in ev_lines.items()
            if key != SEABED_LINE_KEY
        ]
    start, end = _line_time_span(span_source, pad_minutes)

    reduction = config.get("data_reduction", {})
    return {
        "label": label,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "duration_minutes": round((end - start).total_seconds() / 60.0, 2),
        "raw_files": _raw_file_names(config, raw_suffix),
        "frequencies_khz": _frequencies_khz(config),
        "range_bin": _normalize_bin(reduction.get("range_meter_bin", "2m")),
        "ping_time_bin": _normalize_bin(reduction.get("ping_time_bin", "10S")),
        "range_var": reduction.get("range_var", "depth"),
        "source_json": config_path.name,
        **lines,
    }


def _index_evl_files(evl_root):
    """Map every ``.evl`` basename under *evl_root* to its path.

    Args:
        evl_root (Path): Folder searched recursively.

    Returns:
        dict: Basename to list of matching paths.
    """
    index = {}
    for path in evl_root.rglob("*.evl"):
        index.setdefault(path.name, []).append(path)
    return index


def _resolve_line(entry, index, label):
    """Locate the line file an EV_lines entry names, by basename.

    The configs carry absolute paths from the machine the original analysis ran
    on (``/home/mjech/NOAA_Gdrive/...``), so only the file name is usable.

    Args:
        entry (dict): One EV_lines value, with a ``filenames`` list.
        index (dict): Basename to paths, from :func:`_index_evl_files`.
        label (str): Config name, for error messages.

    Returns:
        str: Path to the line file, as a forward-slash string.

    Raises:
        FileNotFoundError: If the name is not present under the search root.
        ValueError: If the name is ambiguous.
    """
    name = Path(entry["filenames"][0]).name
    matches = index.get(name, [])
    if not matches:
        raise FileNotFoundError(
            f"{label}: line file {name!r} is named in the config but was not "
            f"found under the evl_root search path"
        )
    if len(matches) > 1:
        raise ValueError(
            f"{label}: line file {name!r} is ambiguous, found at "
            f"{', '.join(str(m) for m in matches)}"
        )
    return matches[0].as_posix()


def _line_time_span(paths, pad_minutes):
    """First and last point time across one or more line files.

    Args:
        paths (str | list): One line path, or several to take the union of.
        pad_minutes (float): Minutes added to each side.

    Returns:
        tuple: (start, end) as pandas Timestamps.

    Raises:
        ValueError: If the lines hold no points with a usable time.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    times = pd.concat([_parse_evl(path)["time"] for path in paths])
    times = times.dropna()
    if times.empty:
        raise ValueError(f"No timed line points in {', '.join(str(p) for p in paths)}")
    pad = pd.Timedelta(minutes=pad_minutes)
    return times.min() - pad, times.max() + pad


def _raw_file_names(config, raw_suffix):
    """File names for the data files a config lists, with *raw_suffix*.

    The configs name netCDF conversions; the recipe reads the raw files they
    came from, which share the stem.

    Args:
        config (dict): Parsed config.
        raw_suffix (str): Suffix to put on each stem.

    Returns:
        list: Sorted, deduplicated file names.
    """
    names = set()
    for entry in config["path_config"]["EK_data_filenames"]:
        stem = Path(entry).stem
        if FILE_STAMP.search(stem):
            names.add(stem + raw_suffix)
    return sorted(names)


def _frequencies_khz(config):
    """Analysis frequencies in kHz, from the config's ``frequency_list``.

    Args:
        config (dict): Parsed config, whose list looks like ``["18kHz", "38kHz"]``.

    Returns:
        list: Frequencies as floats, in the order listed.
    """
    values = []
    for entry in config.get("analysis", {}).get("frequency_list", []):
        match = re.match(r"([\d.]+)\s*([kKmM]?)[hH][zZ]", str(entry))
        if not match:
            continue
        number, prefix = float(match.group(1)), match.group(2).lower()
        values.append(number * (1000.0 if prefix == "m" else 1.0))
    return values


def _normalize_bin(value):
    """Lowercase a pandas offset alias so "10S" does not trip the 's' rename.

    Args:
        value (str): Bin size such as "2m" or "10S".

    Returns:
        str: The same size with a lowercase unit.
    """
    text = str(value).strip()
    match = re.fullmatch(r"([\d.]+)\s*([A-Za-z]+)", text)
    return f"{match.group(1)}{match.group(2).lower()}" if match else text
