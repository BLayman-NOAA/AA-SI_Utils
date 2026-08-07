# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Utility functions for active acoustics data processing.

Provides helpers for depth indexing, masking (seafloor, surface, frequency),
haversine distance calculation, dive-profile alignment, and raw-file I/O
using the echopype library.
"""

import colorsys
import math
import os
import shutil
import stat
import sys
import time
from pathlib import Path

import echopype as ep
import numpy as np
import pandas as pd
import xarray as xr

from aa_si_utils import _storage
from aa_si_utils.data_retrieval import (
    filter_evl_paths_by_file_time,
    filter_paths_by_file_time,
    parse_evl_span_from_filename,
)


# Target number of pings per chunk along ``ping_time`` when writing the combined
# EchoData checkpoint.  Uniform chunks (equal size with a smaller remainder last)
# satisfy the zarr v2 rule that every chunk but the last be uniform and the last
# be <= the others, while keeping multiple chunks along time so bucket-backed
# (remote) reads stay lazy instead of pulling whole variables at once.  Tune down
# for smaller per-chunk transfers, up for fewer/larger reads.
DEFAULT_PING_TIME_CHUNK = 1000

# Depth Echoview writes into a line file for a ping where no bottom was found.
# Treated as a hole in the line rather than a 10 km deep seafloor.
ECHOVIEW_NAN_DEPTH_VALUE = -10000.99


def _computed_values(data):
    """Return NumPy values, explicitly computing lazy xarray/dask inputs."""
    if hasattr(data, "compute"):
        data = data.compute()
    if hasattr(data, "values"):
        return data.values
    return np.asarray(data)


def _computed_item(data):
    """Return a Python scalar, explicitly computing lazy xarray/dask inputs."""
    if hasattr(data, "compute"):
        data = data.compute()
    if hasattr(data, "item"):
        return data.item()
    return np.asarray(data).item()


def get_closest_index_for_depth(sv_data, target_depth):
    """Find the range_sample index closest to a target depth.
    
    Uses the echo_range coordinate to find the range_sample index that corresponds
    to the depth closest to the target depth. Useful for depth-based data selection.
    
    Args:
        sv_data (xr.Dataset): Sv dataset containing an ``echo_range`` coordinate.
        target_depth (float): Target depth in meters.
        
    Returns:
        int: Range sample index closest to the target depth.
    """
    # echo_range is gridded, so it is populated to the deepest depth for every ping
    echo_range_1d = sv_data['echo_range'].isel(ping_time=0, channel=0)

    # Calculate the absolute difference
    depth_diff = _computed_values(np.abs(echo_range_1d - target_depth))

    # echo_range is NaN-padded at depth when channels have differing sample
    # counts (multi-channel Sv combined onto a shared range_sample axis).  Plain
    # np.argmin would return the index of the first NaN rather than the closest
    # valid depth, so ignore NaNs here.
    if np.all(np.isnan(depth_diff)):
        raise ValueError(
            "echo_range at ping_time=0, channel=0 is entirely NaN; "
            "cannot resolve a range_sample index for the target depth"
        )
    range_sample_index = int(np.nanargmin(depth_diff))

    # Get the actual depth at that index
    actual_depth = _computed_item(echo_range_1d.isel(range_sample=range_sample_index))

    print(f"Target depth: {target_depth} m")
    print(f"Closest range_sample index: {range_sample_index}")
    print(f"Actual depth at that index: {actual_depth:.2f} m")
    return range_sample_index


def find_data_depth_range(sv_data, ping_min=None, ping_max=None, channel=0):
    """Find the depth range where actual Sv data exists within a ping window.
    
    Analyzes the Sv data to find where valid (non-NaN, non-inf) data starts and ends
    in the depth dimension, accounting for missing data near surface and below seafloor.
    
    Args:
        sv_data: Sv dataset containing Sv values and echo_range coordinate
        ping_min (int, optional): Starting ping index (default: 0)
        ping_max (int, optional): Ending ping index (default: last ping)
        channel (int): Channel index to analyze (default: 0)
        
    Returns:
        tuple: (min_depth, max_depth) in meters where data exists
    """
    
    # Set default ping range if not provided
    if ping_min is None:
        ping_min = 0
    if ping_max is None:
        ping_max = len(sv_data['ping_time']) - 1
    
    # Get Sv data for the specified ping and channel range
    sv_subset = sv_data['Sv'].isel(
        channel=channel, 
        ping_time=slice(ping_min, ping_max + 1)
    )
    
    # Find valid data (not NaN or infinite)
    valid_data = np.isfinite(sv_subset)
    
    # Find range samples that have any valid data across the ping window
    has_data_by_range = valid_data.any(dim='ping_time')
    
    # Find first and last range samples with data
    range_indices_with_data = np.where(_computed_values(has_data_by_range))[0]
    
    if len(range_indices_with_data) == 0:
        print("Warning: No valid data found in specified ping range")
        return 0, 100  # Return default fallback
    
    min_range_idx = range_indices_with_data[0]
    max_range_idx = range_indices_with_data[-1]
    
    # Get the corresponding depths from echo_range
    echo_range_data = sv_data['echo_range'].isel(channel=channel, ping_time=ping_min)
    min_depth = _computed_item(echo_range_data.isel(range_sample=min_range_idx))
    max_depth = _computed_item(echo_range_data.isel(range_sample=max_range_idx))
    
    return min_depth, max_depth


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth.
    
    Uses the haversine formula to calculate the distance between two points
    on Earth given their latitude and longitude coordinates.
    
    Args:
        lat1 (float): Latitude of first point in decimal degrees
        lon1 (float): Longitude of first point in decimal degrees
        lat2 (float): Latitude of second point in decimal degrees
        lon2 (float): Longitude of second point in decimal degrees
        
    Returns:
        float: Distance between the two points in meters
    """
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Calculate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Apply haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Calculate distance
    distance = R * c
    
    return distance * 1000


def mask_sparse_bins(ds_Sv: xr.Dataset,
                      range_bin: str = "2m",
                      ping_time_bin: str = "10s",
                      nan_threshold: float = 0.8,
                      range_var: str = "echo_range",
                      block_pings: int = 2000) -> xr.Dataset:
    """Mask Sv values in bins where NaN fraction meets or exceeds a threshold.

    Partitions the data into (time, range) bins and sets entire bins to NaN
    when the proportion of missing values within that bin is at or above
    *nan_threshold*.

    Sv is counted one ping block at a time and masked through
    :meth:`xarray.DataArray.where`, so a dask-backed input is never fully
    materialized and the returned Sv stays lazy.  Variables other than Sv are
    shared with the input rather than copied.

    Args:
        ds_Sv: Sv dataset to process.
        range_bin: Range bin size as a string (e.g. ``"2m"``).
        ping_time_bin: Time bin size as a pandas-compatible offset string
            (e.g. ``"10s"``).
        nan_threshold: Fraction of NaN values (0-1) at or above which a bin
            is masked.
        range_var: Name of the range coordinate variable.
        block_pings: Pings read per block while counting NaNs.  Caps peak
            memory and does not affect the result.

    Returns:
        xr.Dataset: Copy of *ds_Sv* with sparse bins set to NaN.
    """
    sv = ds_Sv["Sv"]

    # Parse range_bin and create bin edges
    range_bin_val = float(range_bin.rstrip('m'))
    range_max = float(ds_Sv[range_var].max(skipna=True).values)
    range_edges = np.arange(0, range_max + range_bin_val, range_bin_val)
    n_range_edges = len(range_edges)

    # Assign each point to a range bin
    range_bin_indices = np.digitize(
        ds_Sv[range_var].isel(ping_time=0, channel=0).values,
        range_edges
    ).astype(np.int32) - 1

    # Assign each ping to a time bin.  Flooring the ping times to the bin
    # width and factorizing gives the same bin ordinals as resampling, since
    # both anchor fixed-frequency bins on the epoch.
    time_bin_map = pd.DatetimeIndex(
        ds_Sv['ping_time'].values
    ).floor(ping_time_bin).factorize()[0].astype(np.int32)

    n_pings = sv.sizes["ping_time"]
    n_channels = sv.sizes["channel"]
    n_bins = (int(time_bin_map.max()) + 1) * n_range_edges

    def block_bin_id(lo, hi):
        """Unique bin ID for each (time_bin, range_bin) pair in a ping block."""
        return (time_bin_map[lo:hi, np.newaxis] * n_range_edges
                + range_bin_indices[np.newaxis, :]).ravel()

    # Count samples and NaNs per bin, reading one ping block at a time
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    nan_counts = np.zeros((n_channels, n_bins), dtype=np.int64)
    for lo in range(0, n_pings, block_pings):
        hi = min(lo + block_pings, n_pings)
        bin_id = block_bin_id(lo, hi)
        bin_counts += np.bincount(bin_id, minlength=n_bins)
        block = sv.isel(ping_time=slice(lo, hi)).values
        for ch in range(n_channels):
            nan_counts[ch] += np.bincount(
                bin_id[np.isnan(block[ch]).ravel()], minlength=n_bins
            )
        del block, bin_id

    # Build the keep mask from the counts alone; Sv is not read again
    keep = np.empty(sv.shape, dtype=bool)
    for ch in range(n_channels):
        nan_fractions = np.divide(nan_counts[ch], bin_counts,
                                  out=np.zeros(n_bins, dtype=float),
                                  where=bin_counts > 0)
        bins_to_mask = nan_fractions >= nan_threshold
        for lo in range(0, n_pings, block_pings):
            hi = min(lo + block_pings, n_pings)
            keep[ch, lo:hi] = ~bins_to_mask[block_bin_id(lo, hi)].reshape(hi - lo, -1)

    keep_mask = xr.DataArray(keep, dims=sv.dims, coords=sv.coords)
    return ds_Sv.assign(Sv=sv.where(keep_mask))


def select_ping_time_range(ds_Sv: xr.Dataset,
                           start: str | None = None,
                           end: str | None = None) -> xr.Dataset:
    """Narrow a Dataset to a ping_time window.

    The user-facing entry point into a survey-wide Sv store.  A survey-level
    recipe computes and checkpoints Sv once for the whole cruise; a user
    recipe reuses that checkpoint and calls this to work on its own segment.

    Cheap on a zarr-backed Dataset: ``sel`` narrows the dask graph to the
    chunks the window touches, so only those are read.  Keep the window step
    downstream of every step whose checkpoint you want to share -- step hashes
    are a Merkle chain, so a window wired into an upstream step's params
    changes that step's hash and every hash below it, and no two windows would
    ever share a cached Sv.

    Args:
        ds_Sv: Dataset carrying a ``ping_time`` coordinate.
        start: Inclusive ISO datetime lower bound (e.g. "2024-10-15T13:38").
            None leaves the start open.
        end: Inclusive ISO datetime upper bound.  None leaves the end open.

    Returns:
        xr.Dataset: The ``ping_time`` slice of *ds_Sv*.  Returned lazily when
        the input is dask-backed.

    Raises:
        ValueError: If the window selects no pings, or if *start* is after
            *end*.
    """
    if start is None and end is None:
        return ds_Sv
    if start is not None and end is not None:
        if pd.Timestamp(start) > pd.Timestamp(end):
            raise ValueError(
                f"start {start!r} is after end {end!r}; the window is empty"
            )

    n_before = ds_Sv.sizes.get("ping_time", 0)
    windowed = ds_Sv.sel(ping_time=slice(start, end))
    n_after = windowed.sizes.get("ping_time", 0)
    if n_after == 0:
        available = ds_Sv["ping_time"].values
        raise ValueError(
            f"ping_time window {start!r} to {end!r} selects no pings; the "
            f"dataset spans {available[0]} to {available[-1]}"
        )
    print(
        f"select_ping_time_range: {n_before} -> {n_after} pings "
        f"({100 * n_after / n_before:.1f}%), "
        f"{windowed['ping_time'].values[0]} to {windowed['ping_time'].values[-1]}"
    )
    return windowed


def rechunk_dataset(ds_Sv: xr.Dataset, ping_time_chunk: int) -> xr.Dataset:
    """Rechunk a Dataset along ping_time.

    A standalone step rather than a parameter on any particular masking or
    reduction op, so a recipe can insert (or remove) chunking wherever a
    downstream op should run dask-parallel, independent of which step
    happens to run immediately before it.

    Args:
        ds_Sv: Dataset to rechunk.
        ping_time_chunk: Target chunk size along ping_time.

    Returns:
        xr.Dataset: *ds_Sv* rechunked along ping_time.
    """
    return ds_Sv.chunk({"ping_time": ping_time_chunk})


def _deepest_finite_sample_per_ping(ds_Sv: xr.Dataset,
                                    data_var: str) -> tuple[np.ndarray, np.ndarray]:
    """Deepest range-sample index holding finite *data_var*, per ping.

    Returns ``(deepest, has_any)``.  ``deepest`` is 0 for a ping with no
    finite sample at all, which ``has_any`` distinguishes from a ping whose
    only finite sample really is index 0.  Both reductions stay lazy on a
    dask-backed input, so only the per-ping results are materialized.
    """
    finite = np.isfinite(ds_Sv[data_var])
    extra = [d for d in finite.dims if d not in ("ping_time", "range_sample")]
    if extra:
        finite = finite.any(dim=extra)
    index = xr.DataArray(
        np.arange(ds_Sv.sizes["range_sample"]), dims="range_sample"
    )
    deepest = (finite * index).max(dim="range_sample")
    has_any = finite.any(dim="range_sample")
    return np.asarray(deepest.values), np.asarray(has_any.values)


def crop_range_samples(ds_Sv: xr.Dataset,
                       max_range_m: float | None = None,
                       outlier_sigma: float | None = None,
                       outlier_margin: float = 1.5,
                       drop_trailing_all_nan: bool = True,
                       data_var: str = "Sv",
                       range_var: str = "echo_range") -> xr.Dataset:
    """Trim trailing range samples that carry no data.

    A raw file records to the sonar's configured range, so after seafloor
    masking the deep end of ``range_sample`` is entirely NaN.  Those samples
    still cost memory and time in every downstream step, and they widen the
    binned cell grid that :func:`compute_per_cell_statistics` and
    ``compute_MVBS`` build, so trimming them once is worth a step of its own.

    Run this survey-wide, after the per-file datasets have been merged.  The
    trim point is a property of the whole survey, and cropping per file would
    leave the segments with different ``range_sample`` lengths, which the
    ping_time concatenation resolves by padding the short ones back out with
    NaN.

    Three criteria are available and compose: whichever are active are each
    turned into a candidate cut and the shallowest wins.

    * *outlier_sigma* rejects anomalous pings, then crops to the deepest
      **retained** ping.  A single glitch ping reaching far past the seafloor
      otherwise holds the whole range axis open, which is what defeats
      *drop_trailing_all_nan*.  The threshold is deliberately generous, so
      uncommon-but-real depths survive and only wild outliers are cut.
    * *max_range_m* is a hard ceiling, applied whether or not samples above it
      hold data.
    * *drop_trailing_all_nan* is the lossless fallback when no outlier test is
      requested.

    Args:
        ds_Sv: Dataset to crop.
        max_range_m: Hard ceiling in metres.  Range samples whose shallowest
            *range_var* across pings exceeds this are dropped, whether or not
            they hold data, so a ceiling below the deepest echo of interest
            discards it.  Combines with the other criteria; None disables it.
        outlier_sigma: Standard deviations above the mean, over the per-ping
            deepest finite sample, at which a ping becomes a candidate
            outlier.  None disables outlier rejection.
        outlier_margin: Multiplier applied to ``mean + outlier_sigma * sd`` to
            get the reject threshold.  Raise it to flag fewer pings, lower it
            to flag more.  Only used when *outlier_sigma* is set.
        drop_trailing_all_nan: Crop to one past the deepest range sample
            holding a finite *data_var* value anywhere in the dataset.
            Lossless.  Ignored when *outlier_sigma* is set.  With this False
            and both other criteria unset the dataset is returned unchanged.
        data_var: Variable inspected for finite values.
        range_var: Variable supplying each sample's range in metres.

    Returns:
        xr.Dataset: *ds_Sv* sliced along ``range_sample``.

    Raises:
        ValueError: If *data_var* holds no finite values, if *max_range_m*
            excludes every range sample, or if outlier rejection would flag
            every ping.
    """
    if "range_sample" not in ds_Sv.dims:
        return ds_Sv
    n_before = ds_Sv.sizes["range_sample"]
    n_keep = n_before
    reasons: list[str] = []

    if outlier_sigma is not None:
        deepest, has_any = _deepest_finite_sample_per_ping(ds_Sv, data_var)
        sample = deepest[has_any]
        if sample.size == 0:
            raise ValueError(
                f"'{data_var}' holds no finite values; nothing to crop to"
            )
        threshold = outlier_margin * (sample.mean() + outlier_sigma * sample.std())
        retained = sample <= threshold
        n_flagged = int((~retained).sum())
        if not retained.any():
            raise ValueError(
                f"outlier rejection at {outlier_margin} x (mean + "
                f"{outlier_sigma} sd) = {threshold:.0f} flags every ping"
            )
        n_keep = int(sample[retained].max()) + 1
        reasons.append(
            f"outlier reject {outlier_margin} x (mean + {outlier_sigma} sd)"
        )
        print(
            f"crop_range_samples: flagged {n_flagged} of {len(sample)} pings "
            f"deeper than range sample {threshold:.0f} "
            f"({outlier_margin} x (mean + {outlier_sigma} sd))"
        )
    elif drop_trailing_all_nan:
        finite = np.isfinite(ds_Sv[data_var])
        reduce_dims = [d for d in finite.dims if d != "range_sample"]
        has_data = np.asarray(finite.any(dim=reduce_dims).values)
        if not has_data.any():
            raise ValueError(
                f"'{data_var}' holds no finite values; nothing to crop to"
            )
        n_keep = int(np.flatnonzero(has_data)[-1]) + 1
        reasons.append("trailing all-NaN")

    if max_range_m is not None:
        sample_range = ds_Sv[range_var]
        reduce_dims = [d for d in sample_range.dims if d != "range_sample"]
        shallowest = sample_range.min(dim=reduce_dims, skipna=True).values
        within = np.flatnonzero(np.nan_to_num(shallowest, nan=np.inf) <= max_range_m)
        if within.size == 0:
            raise ValueError(
                f"max_range_m={max_range_m} excludes every range sample; the "
                f"shallowest is {np.nanmin(shallowest):.1f} m"
            )
        capped = int(within[-1]) + 1
        reasons.append(f"max_range_m={max_range_m}")
        n_keep = min(n_keep, capped)

    if not reasons:
        return ds_Sv
    reason = " + ".join(reasons)

    if n_keep >= n_before:
        print(
            f"crop_range_samples: keeping all {n_before} range samples "
            f"({reason})"
        )
        return ds_Sv

    dropped = int(
        np.isfinite(ds_Sv[data_var].isel(range_sample=slice(n_keep, None)))
        .sum()
        .values
    )
    cropped = ds_Sv.isel(range_sample=slice(0, n_keep))
    deepest_m = float(np.nanmax(cropped[range_var].values[..., -1]))
    print(
        f"crop_range_samples: {n_before} -> {n_keep} range samples "
        f"({100 * n_keep / n_before:.1f}%), deepest kept {deepest_m:.1f} m; "
        f"dropped {dropped} finite sample(s) ({reason})"
    )
    return cropped


def generate_colors(hue_offset, num_additional_colors):
    """Generate a list of visually distinct hex colours using the golden ratio.

    Args:
        hue_offset (float): Starting hue value in the range [0, 1).
        num_additional_colors (int): Number of colours to generate.

    Returns:
        list[str]: Hex colour strings (e.g. ``'#ab12cd'``).
    """
    additional_colors = []
    golden_ratio = 0.618033988749895

    for i in range(num_additional_colors):
        hue = (hue_offset + i * golden_ratio) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.8)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
        )
        additional_colors.append(hex_color)
    return additional_colors


def add_dive_profile_to_dataset(ds_MVBS, csv_filepath, dive_profile_name='dive_profile'):
    """Add dive profile line data to an MVBS dataset.

    Aligns dive profile times to MVBS ping_time using nearest-neighbor matching.
    Only dive times within the MVBS time range will have depth values; other
    pings will be NaN.

    Args:
        ds_MVBS (xr.Dataset): MVBS dataset with ``ping_time`` coordinate.
        csv_filepath (str | Path): Path to dive profile CSV with columns:
            ``ClickTime_UTC``, ``clickDepth_m``, ``fit_line``,
            ``lwr_CI_99``, ``upr_CI_99``. May be a remote fsspec URL
            (``gs://...``), which pandas reads in place — these files are small,
            so no local copy is made.
        dive_profile_name (str): Base name for the added variables
            (default: ``'dive_profile'``).

    Returns:
        xr.Dataset: Dataset with added dive-profile variables sharing the
        existing ``ping_time`` dimension. Values are NaN where no dive
        data exists.
    """
    # Validate file exists
    if _storage.is_remote(csv_filepath):
        storage_options = _execution_storage_options()
        fs = _storage.get_fs(csv_filepath, storage_options)
        if not fs.exists(str(csv_filepath)):
            raise FileNotFoundError(f"Dive profile CSV not found: {csv_filepath}")
        csv_path = str(csv_filepath)
        read_csv_kwargs = {"storage_options": storage_options or None}
    else:
        csv_path = Path(csv_filepath)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dive profile CSV not found: {csv_filepath}")
        read_csv_kwargs = {}

    csv_name = _storage.basename(csv_path)
    print(f"Reading dive profile: {csv_name}")

    # Read CSV
    dive_df = pd.read_csv(csv_path, **read_csv_kwargs)
    dive_df['ClickTime_UTC'] = pd.to_datetime(dive_df['ClickTime_UTC'].str.replace('_', ' '))
    
    # Rename to 'time' for merge_asof
    dive_df = dive_df.rename(columns={'ClickTime_UTC': 'time'})
    
    # Get time ranges
    mvbs_times = pd.to_datetime(ds_MVBS['ping_time'].values)
    dive_start = pd.Timestamp(dive_df['time'].min())
    dive_end = pd.Timestamp(dive_df['time'].max())
    
    # Check for overlap
    has_overlap = not (dive_end < mvbs_times[0] or dive_start > mvbs_times[-1])
    if not has_overlap:
        print("WARNING: No time overlap between dive profile and MVBS data")
        print(f"  MVBS: {mvbs_times[0]} to {mvbs_times[-1]}")
        print(f"  Dive: {dive_start} to {dive_end}")
        return ds_MVBS
    
    print(f"  Dive points: {len(dive_df)}")
    print(f"  MVBS pings: {len(mvbs_times)}")
    print(f"  Overlap: {max(dive_start, mvbs_times[0])} to {min(dive_end, mvbs_times[-1])}")
    
    # Create DataFrame with all MVBS times 
    mvbs_time_df = pd.DataFrame({'time': mvbs_times})
    
    # Ensure both time columns have the same datetime resolution for merge_asof
    mvbs_time_df['time'] = mvbs_time_df['time'].astype('datetime64[ns]')
    dive_df['time'] = dive_df['time'].astype('datetime64[ns]')
    
    # Merge dive data to MVBS times using nearest-neighbor 
    # This creates a FULL array (length = MVBS pings) with NaN where no dive occurs
    aligned_data = pd.merge_asof(
        mvbs_time_df,
        dive_df[['time', 'clickDepth_m', 'fit_line', 'lwr_CI_99', 'upr_CI_99']],
        on='time',
        direction='nearest',
        tolerance=pd.Timedelta('30s')  # NaN if no match within 30 seconds
    )
    
    # Now we have arrays of length = len(mvbs_times), can use ping_time dimension
    ds_MVBS[f'{dive_profile_name}_depth'] = (
        ('ping_time',), 
        aligned_data['clickDepth_m'].values
    )
    ds_MVBS[f'{dive_profile_name}_fit'] = (
        ('ping_time',), 
        aligned_data['fit_line'].values
    )
    ds_MVBS[f'{dive_profile_name}_lower_ci'] = (
        ('ping_time',), 
        aligned_data['lwr_CI_99'].values
    )
    ds_MVBS[f'{dive_profile_name}_upper_ci'] = (
        ('ping_time',), 
        aligned_data['upr_CI_99'].values
    )
    
    # Add metadata
    ds_MVBS[f'{dive_profile_name}_fit'].attrs.update({
        'long_name': 'Sperm whale dive depth (fitted)',
        'units': 'meters',
        'source_file': csv_name
    })
    
    ds_MVBS[f'{dive_profile_name}_depth'].attrs.update({
        'long_name': 'Sperm whale dive depth (measured)',
        'units': 'meters',
        'source_file': csv_name
    })
    
    ds_MVBS[f'{dive_profile_name}_lower_ci'].attrs.update({
        'long_name': 'Dive depth lower 99% CI',
        'units': 'meters'
    })
    
    ds_MVBS[f'{dive_profile_name}_upper_ci'].attrs.update({
        'long_name': 'Dive depth upper 99% CI',
        'units': 'meters'
    })
    
    # Report results
    n_valid = (~np.isnan(aligned_data['fit_line'])).sum()
    print(f"Added dive profile variables (length={len(aligned_data)})")
    print(f"  Valid dive points: {n_valid} ({n_valid/len(aligned_data)*100:.1f}%)")
    print(f"  NaN outside dive: {len(aligned_data) - n_valid}")
    
    return ds_MVBS


def createSvMask(ds_Sv):
    """Create an all-True boolean mask matching the shape of the Sv variable.

    Args:
        ds_Sv (xr.Dataset): Dataset containing an ``Sv`` variable.

    Returns:
        xr.DataArray: Boolean mask with the same shape as ``ds_Sv['Sv']``.
    """
    return xr.ones_like(ds_Sv['Sv'], dtype=bool)


def _get_detected_seafloor_depth(echodata):
    if echodata is None:
        raise ValueError(
            "echodata is required for seafloor detection: no .bot data source was provided"
        )

    try:
        return echodata["Vendor_specific"]["detected_seafloor_depth"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "echodata does not contain Vendor_specific.detected_seafloor_depth"
        ) from exc


def _channel_display_label(channel_value, freq_hz=None):
    label = str(channel_value)
    if freq_hz is None:
        return label

    freq_khz = int(round(float(freq_hz) / 1000))
    if label == str(freq_khz):
        return f"{label} kHz"
    return f"{label} ({freq_khz} kHz)"


def _resolve_channel_index(requested_channel, channel_values, ds_Sv=None):
    requested_str = str(requested_channel)

    for idx, channel_value in enumerate(channel_values):
        if str(channel_value) == requested_str:
            return idx

    if ds_Sv is not None and "frequency_nominal" in ds_Sv:
        for idx, freq_hz in enumerate(ds_Sv["frequency_nominal"].values):
            freq_hz_int = int(round(float(freq_hz)))
            freq_khz_int = int(round(float(freq_hz) / 1000))
            if requested_str in {str(freq_hz_int), str(freq_khz_int)}:
                return idx

    available_channels = ", ".join(str(value) for value in channel_values)
    raise ValueError(
        f"Channel {requested_channel!r} was not found in detected seafloor data. "
        f"Available channels: {available_channels}"
    )


def _normalize_seafloor_depth(seafloor_depth, ds_Sv):
    if not isinstance(seafloor_depth, xr.DataArray):
        raise TypeError("seafloor_depth must be an xarray DataArray")

    if "ping_time" not in seafloor_depth.dims:
        raise ValueError("seafloor_depth must include a ping_time dimension")

    if "channel" in seafloor_depth.dims:
        if seafloor_depth.sizes["channel"] != 1:
            raise ValueError(
                "seafloor_depth must be 1-D over ping_time; detected multiple channels"
            )
        seafloor_depth = seafloor_depth.squeeze("channel", drop=True)

    if tuple(seafloor_depth.dims) != ("ping_time",):
        raise ValueError(
            "seafloor_depth must only have the ping_time dimension after squeezing"
        )

    ping_template = xr.DataArray(
        np.zeros(ds_Sv.sizes["ping_time"]),
        coords={"ping_time": ds_Sv["ping_time"]},
        dims=["ping_time"],
    )

    try:
        aligned_depth, _ = xr.align(seafloor_depth, ping_template, join="exact")
    except ValueError as exc:
        raise ValueError(
            "seafloor_depth ping_time coordinate must match ds_Sv ping_time exactly"
        ) from exc

    return aligned_depth


def _validate_boolean_mask(mask, mask_name):
    if not isinstance(mask, xr.DataArray):
        raise TypeError(f"{mask_name} must be an xarray DataArray")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError(f"{mask_name} must have boolean dtype")


def _drop_keel_offset(platform):
    """Single drop keel offset in metres from a Platform group."""
    if "drop_keel_offset" not in platform:
        raise KeyError(
            "Platform group has no 'drop_keel_offset'. Only EK80-family files "
            "record one; pass a constant depth_offset to add_depth instead."
        )

    values = np.atleast_1d(_computed_values(platform["drop_keel_offset"])).astype(float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(
            "Platform 'drop_keel_offset' is NaN, so transducer depth cannot be "
            "derived from it. The raw file's Environment datagram carried no "
            "DropKeelOffset."
        )
    if not np.allclose(finite, finite[0]):
        raise ValueError(
            "Platform 'drop_keel_offset' holds more than one value "
            f"({sorted(set(finite.tolist()))} m), so no single keel position "
            "describes this data. Combine only files recorded with the keel at "
            "one position, or compute depth per file before combining."
        )
    return float(finite[0])


def _heave_on_ping_time(platform, ping_time, interp_method):
    """Platform heave in metres on ping_time, or None when it carries no signal."""
    if "vertical_offset" not in platform:
        return None

    heave = platform["vertical_offset"]
    if len(heave.dims) != 1:
        return None

    time_dim = heave.dims[0]
    if time_dim in heave.coords:
        heave = heave.where(heave[time_dim].notnull(), drop=True)
    heave = heave.dropna(dim=time_dim)
    if heave.sizes[time_dim] == 0:
        return None
    if heave.sizes[time_dim] == 1:
        return xr.full_like(ping_time, _computed_item(heave[0]), dtype=float)

    aligned = heave.interp({time_dim: ping_time}, method=interp_method)
    # interp leaves the source time as a stray non-dimension coordinate, which
    # would otherwise ride along into ds_Sv['depth'].
    return aligned.drop_vars(time_dim, errors="ignore")


def compute_transducer_depth(
    echodata,
    ds_Sv,
    datum_correction_m=0.0,
    use_heave=True,
    heave_sign=1.0,
    interp_method="linear",
):
    """Per-ping transducer depth (m below surface) from the EK Platform group.

    Built from ``drop_keel_offset`` rather than from echopype's own platform
    vertical offsets. On a drop keel vessel the transducers ride on a
    retractable keel and ``drop_keel_offset`` records how far it was lowered,
    while the ``transducer_offset_z`` and ``water_level`` fields that
    ``add_depth(use_platform_vertical_offsets=True)`` reads are often left at
    zero. echopype carries ``drop_keel_offset`` for provenance and never reads
    it, so pass this result to ``add_depth`` as ``depth_offset``.

    The keel position is per file and heave is per ping, giving
    ``drop_keel_offset + datum_correction_m + heave_sign * vertical_offset``.
    Heave is interpolated onto ``ds_Sv``'s ping_time here so that add_depth
    passes the result straight through instead of resampling it again with
    nearest neighbour.

    Args:
        echodata (EchoData): Source EchoData supplying the Platform group.
        ds_Sv (xr.Dataset): Sv dataset supplying the target ``ping_time``.
        datum_correction_m (float): Metres added to ``drop_keel_offset`` to
            reach depth below the water surface, for when the recorded offset
            is measured from a hull datum rather than the waterline. Positive
            pushes the transducer deeper.
        use_heave (bool): Add the per-ping ``vertical_offset`` (heave) term.
            False holds the static keel depth on every ping.
        heave_sign (float): Sign applied to ``vertical_offset``. The default
            ``1.0`` was determined empirically on HB2407 (EK80, Henry B.
            Bigelow) by correlating raw heave against the hand-verified EVL
            seabed line: combining EVL depth with a heave_sign=+1 transducer
            depth left a residual std of 0.14 m against 0.75 m for -1.0, a
            5x difference (see AA-SI_recipe_manager/example_recipes/HB2407).
            echopype's own platform-vertical-offsets formula uses -1.0, but
            that formula is referenced from transducer_offset_z/water_level
            (a hull datum), not drop_keel_offset, so there is no reason to
            expect the same sign to carry over — re-verify on new
            vessels/instrument setups the same way rather than assuming.
        interp_method (str): Interpolation used to put heave on ``ping_time``.

    Returns:
        xr.DataArray: Transducer depth in metres, dims ``("ping_time",)``,
        named ``transducer_depth``.
    """
    platform = echodata["Platform"]
    keel_offset = _drop_keel_offset(platform)
    static_depth = keel_offset + datum_correction_m
    ping_time = ds_Sv["ping_time"]

    heave = None
    if use_heave:
        heave = _heave_on_ping_time(platform, ping_time, interp_method)

    if heave is None:
        depth = xr.full_like(ping_time, static_depth, dtype=float)
        heave_pings = 0
        if use_heave:
            print(
                "[transducer_depth] Platform group carries no usable heave; "
                "holding the static keel depth on every ping"
            )
    else:
        heave_pings = int(np.isfinite(_computed_values(heave)).sum())
        # Pings outside the heave record fall back to the static depth rather
        # than to NaN, which would void every sample of those pings downstream.
        depth = static_depth + heave_sign * heave.fillna(0.0)

    depth = depth.rename("transducer_depth").assign_coords(ping_time=ping_time)
    depth.attrs = {
        "long_name": "Transducer depth below water surface",
        "units": "m",
        "drop_keel_offset": keel_offset,
        "datum_correction_m": float(datum_correction_m),
        "heave_sign": float(heave_sign),
        "heave_pings": heave_pings,
        "interp_method": interp_method,
    }

    values = _computed_values(depth)
    print(
        f"[transducer_depth] drop_keel_offset {keel_offset} m "
        f"{datum_correction_m:+g} m correction, heave on "
        f"{heave_pings}/{ping_time.size} pings, "
        f"depth {np.nanmin(values):.2f} to {np.nanmax(values):.2f} m"
    )
    return depth


def get_transducer_depth(ds_Sv):
    """Per-ping transducer depth (m below surface) from a depth-enriched Sv ds.

    Reads the implicit transducer offset baked into ``ds_Sv['depth']`` by
    ``echopype.consolidate.add_depth``. Because depth = transducer_depth +
    echo_range * cos(tilt) and echo_range == 0 at ``range_sample=0``, the
    sample at ``range_sample=0`` equals the transducer depth regardless of
    which add_depth technique was used (constant offset, platform offsets,
    beam-angle-derived, etc.).

    Args:
        ds_Sv (xr.Dataset): Sv dataset that has been passed through
            ``add_depth`` (must contain a ``depth`` variable).

    Returns:
        xr.DataArray: Per-ping transducer depth in metres, dim ``ping_time``.
    """
    if "depth" not in ds_Sv:
        raise KeyError("ds_Sv has no 'depth' variable; run add_depth first")
    td = ds_Sv["depth"].isel(range_sample=0)
    if "channel" in td.dims:
        td = td.isel(channel=0, drop=True)
    return td


def detect_seafloor(ds_Sv=None, echodata=None, channel=None, min_valid_depth_m=10.0):
    """Detect seafloor depth and return a 1-D ``(ping_time,)`` DataArray.

    The returned line is in metres. Its vertical reference depends on whether
    ``ds_Sv`` has been passed through ``echopype.consolidate.add_depth``:

    * If ``ds_Sv`` contains a ``depth`` variable, the line is **surface
      referenced** (metres below surface) so it can be compared directly
      against ``ds_Sv['depth']`` in :func:`create_seafloor_mask`.
    * Otherwise the line is **transducer referenced** (metres along the beam
      from the transducer face), matching ``ds_Sv['echo_range']``.
    """
    seafloor_depth = _get_detected_seafloor_depth(echodata)

    if channel is None:
        _, _, best_seafloor = _find_best_seafloor_detection(
            echodata, ds_Sv, min_valid_depth_m=min_valid_depth_m
        )
        seafloor_line = best_seafloor.squeeze(drop=True)
    else:
        channel_values = seafloor_depth["channel"].values
        channel_index = _resolve_channel_index(channel, channel_values, ds_Sv=ds_Sv)
        seafloor_line = seafloor_depth.isel(channel=channel_index).squeeze(drop=True)

    # Promote to surface-referenced depth when ds_Sv carries a depth variable
    # (i.e., add_depth was run upstream). Keeps the output consistent with
    # echopype's detect_seafloor and lets create_seafloor_mask compare
    # against ds_Sv['depth'] directly.
    if ds_Sv is not None and "depth" in ds_Sv:
        seafloor_line = seafloor_line + get_transducer_depth(ds_Sv)

    return seafloor_line


def _read_evl_text(evl_path):
    """Return the text of a local or remote ``.evl`` file.

    Decoded as utf-8-sig because Echoview writes a BOM. Line files are small
    (tens of thousands of points, a couple of MB), so a remote one is read in
    place rather than copied to local scratch.
    """
    if _storage.is_remote(evl_path):
        storage_options = _execution_storage_options()
        fs = _storage.get_fs(evl_path, storage_options)
        if not fs.exists(str(evl_path)):
            raise FileNotFoundError(f"EVL line file not found: {evl_path}")
        with fs.open(str(evl_path), "rt", encoding="utf-8-sig") as handle:
            return handle.read()

    path = Path(evl_path)
    if not path.exists():
        raise FileNotFoundError(f"EVL line file not found: {evl_path}")
    return path.read_text(encoding="utf-8-sig")


def _parse_evl(evl_path):
    """Parse an Echoview ``.evl`` line file into a time/depth/status DataFrame.

    The format is two header lines — ``EVBD <format_version> <echoview_version>``
    and the point count — followed by one point per line:

        ``CCYYMMDD HHmmSSssss <depth_m> <status>``

    The time-of-day field carries four fractional digits (1/10000 s), which
    ``%f`` reads as microseconds after right-padding. The declared point count
    is checked against the rows actually present, so a truncated export fails
    here rather than silently producing a short line.

    Args:
        evl_path (str | Path): Path or fsspec URL of the ``.evl`` file.

    Returns:
        pd.DataFrame: Columns ``time`` (datetime64), ``depth`` (float, metres,
        with the ``-10000.99`` "no bottom" sentinel replaced by NaN), and
        ``status`` (numeric: 0 none, 1 unverified, 2 bad, 3 good).
    """
    file_lines = _read_evl_text(evl_path).splitlines()
    if len(file_lines) < 2:
        raise ValueError(
            f"EVL file is missing its two header lines: {evl_path}"
        )

    try:
        declared_points = int(file_lines[1].strip())
    except ValueError as exc:
        raise ValueError(
            f"EVL file's second line should be the point count, got "
            f"{file_lines[1].strip()!r}: {evl_path}"
        ) from exc

    rows = [line.split() for line in file_lines[2:] if line.strip()]
    if len(rows) != declared_points:
        raise ValueError(
            f"EVL file declares {declared_points} points but contains "
            f"{len(rows)}: {evl_path}"
        )
    if not rows:
        raise ValueError(f"EVL file contains no line points: {evl_path}")

    widths = {len(row) for row in rows}
    if widths != {4}:
        raise ValueError(
            f"EVL point lines should have 4 fields (date, time, depth, status); "
            f"found rows with {sorted(widths)} fields: {evl_path}"
        )

    fields = pd.DataFrame(rows, columns=["date", "time_of_day", "depth", "status"])
    depth = fields["depth"].astype("float64")
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                fields["date"] + " " + fields["time_of_day"],
                format="%Y%m%d %H%M%S%f",
            ),
            "depth": depth.mask(depth == ECHOVIEW_NAN_DEPTH_VALUE),
            "status": pd.to_numeric(fields["status"], errors="coerce"),
        }
    )


def _resolve_evl_paths(evl_path, file_time_start=None, file_time_end=None):
    """Expand a file, folder, or sequence into the ``.evl`` files to read.

    A folder (local path or remote prefix) is listed for ``*.evl`` and narrowed
    to the files whose name-encoded span overlaps the window. A single file is
    returned as-is and never filtered -- naming one file is an override, not a
    candidate set.

    Args:
        evl_path: Path, URL, folder, or sequence of either.
        file_time_start: Optional inclusive lower bound (ISO string or datetime).
        file_time_end: Optional inclusive upper bound.

    Returns:
        list: The ``.evl`` paths to parse, in chronological name order.
    """
    if isinstance(evl_path, (list, tuple, set)):
        candidates = sorted(evl_path, key=str)
        from_folder = True
    elif _storage.is_remote(evl_path):
        fs = _storage.get_fs(evl_path, _execution_storage_options())
        from_folder = fs.isdir(str(evl_path))
        if from_folder:
            candidates = _storage.glob_url(
                evl_path, "*.evl", _execution_storage_options()
            )
        else:
            candidates = [evl_path]
    else:
        path = Path(evl_path)
        from_folder = path.is_dir()
        candidates = sorted(path.glob("*.evl")) if from_folder else [evl_path]

    if not from_folder:
        return candidates

    if not candidates:
        raise FileNotFoundError(f"No .evl line files found in folder: {evl_path}")

    selected = filter_evl_paths_by_file_time(
        candidates, file_time_start, file_time_end
    )
    if not selected:
        spans = [
            parse_evl_span_from_filename(_storage.basename(c)) for c in candidates
        ]
        starts = [start for start, _ in spans if start is not None]
        ends = [end for _, end in spans if end is not None]
        covered = (
            f"{min(starts)} to {max(ends)}" if starts else "no parseable file names"
        )
        raise FileNotFoundError(
            f"No .evl line files in {evl_path} cover "
            f"{file_time_start} to {file_time_end}; the folder's "
            f"{len(candidates)} files span {covered}"
        )
    return selected


def _tolerance_ns(seconds):
    """Nanoseconds for a tolerance in seconds; ``None`` means "no limit"."""
    if seconds is None:
        return np.iinfo(np.int64).max
    return np.int64(round(float(seconds) * 1_000_000_000))


def _interp_line_to_ping_time(points, ping_time, max_gap_s, edge_extend_s):
    """Interpolate line points onto a ping_time coordinate, linearly in time.

    Args:
        points (pd.DataFrame): Line points sorted by ``time``, with unique
            timestamps and a ``depth`` column in metres.
        ping_time (xr.DataArray): Target ping times.
        max_gap_s (float | None): Widest hole in the line to interpolate across.
        edge_extend_s (float | None): How far past the line's first/last point
            to hold that point's depth.

    Returns:
        np.ndarray: Depths shaped like ``ping_time``, NaN wherever the line
        does not support a value.
    """
    line_ns = points["time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    line_depth = points["depth"].to_numpy(dtype="float64")
    ping_ns = np.asarray(ping_time).astype("datetime64[ns]").astype("int64")

    # Linear between the bracketing points, clamped to the end values outside
    # the line's span. A NaN line depth propagates only to the pings it
    # brackets, which is the wanted "bottom unknown here".
    values = np.interp(ping_ns, line_ns, line_depth)

    # Nanoseconds a ping sits beyond the line's span; <= 0 while inside it.
    outside_ns = np.maximum(line_ns[0] - ping_ns, ping_ns - line_ns[-1])
    values[outside_ns > _tolerance_ns(edge_extend_s)] = np.nan

    if max_gap_s is not None and line_ns.size > 1:
        # Interior pings: drop those whose two bracketing line points are
        # further apart in time than the limit. A ping landing exactly on a
        # line point keeps that point's depth -- it is measured, not guessed,
        # so the gap on either side of it is irrelevant.
        left = np.clip(
            np.searchsorted(line_ns, ping_ns, side="right") - 1,
            0,
            line_ns.size - 2,
        )
        bracket_ns = line_ns[left + 1] - line_ns[left]
        on_line_point = (line_ns[left] == ping_ns) | (line_ns[left + 1] == ping_ns)
        interpolated = (outside_ns <= 0) & ~on_line_point
        values[interpolated & (bracket_ns > _tolerance_ns(max_gap_s))] = np.nan

    return values


def _log_seafloor_line_summary(points, ping_time, values, evl_paths):
    """Print line/ping alignment stats; return the covered fraction of pings."""
    n_pings = values.size
    n_filled = int(np.isfinite(values).sum())
    coverage = n_filled / n_pings if n_pings else 0.0
    ping_values = np.asarray(ping_time)

    names = [_storage.basename(path) for path in evl_paths]
    if len(names) == 1:
        print(f"Seafloor line: {names[0]}")
    elif len(names) <= 5:
        print(f"Seafloor line: {len(names)} files")
        for name in names:
            print(f"    {name}")
    else:
        print(f"Seafloor line: {len(names)} files, {names[0]} .. {names[-1]}")
    print(
        f"  Line points: {len(points)} spanning "
        f"{points['time'].iloc[0]} to {points['time'].iloc[-1]}"
    )
    if n_pings:
        print(
            f"  Pings: {n_pings} spanning {ping_values.min()} to {ping_values.max()}"
        )
    print(
        f"  Coverage: {n_filled}/{n_pings} pings ({coverage:.1%}) have a seafloor depth"
    )
    if n_filled:
        print(
            f"  Depth: min={np.nanmin(values):.1f} m, "
            f"median={np.nanmedian(values):.1f} m, max={np.nanmax(values):.1f} m"
        )
    if coverage < 1.0:
        print(
            "  WARNING: create_seafloor_mask drops a ping entirely where the "
            "seafloor is NaN (the comparison is False for every sample). Widen "
            "edge_extend_s / max_gap_s, or use a line covering more of the survey."
        )
    return coverage


def read_seafloor_line_evl(
    ds_Sv,
    evl_path,
    vertical_reference="surface",
    min_status=None,
    max_gap_s=None,
    edge_extend_s=0.0,
    depth_offset_m=0.0,
    min_coverage=0.0,
    file_time_start=None,
    file_time_end=None,
):
    """Read an Echoview ``.evl`` seafloor line onto ``ds_Sv``'s ping grid.

    Drop-in replacement for :func:`detect_seafloor`: returns the same 1-D
    ``(ping_time,)`` line in metres, on ``ds_Sv``'s exact ``ping_time``
    coordinate, so :func:`create_seafloor_mask` consumes it unchanged. Use it
    when a hand-verified Echoview bottom line is more trustworthy than running
    detection.

    Echoview exports a survey as a series of part-day line files, so ``evl_path``
    may be a folder: the files whose names span ``file_time_start`` to
    ``file_time_end`` are selected and concatenated into one line before
    interpolation. Passing the same window that selected the raw files keeps the
    line and the pings in step.

    Pings the line does not cover are NaN, and :func:`create_seafloor_mask`
    rejects **every sample** of a ping whose seafloor is NaN (the metres-vs-metres
    comparison is False against NaN). Coverage is therefore printed on every
    call, and ``min_coverage`` turns a shortfall into an error.

    Args:
        ds_Sv (xr.Dataset): Sv dataset supplying the target ``ping_time``.
        evl_path (str | Path | list): A single ``.evl`` file, or a folder of
            them to select from with ``file_time_start`` / ``file_time_end``.
            May be a remote fsspec URL (``gs://...``), which is read in place —
            these files are small, so no local copy is made.
        vertical_reference (str): What the EVL depths are measured from.
            ``"surface"`` (default) is Echoview's usual seabed-line reference,
            metres below the water surface; ``"transducer"`` is metres along the
            beam from the transducer face. The line is converted to whichever
            reference ``ds_Sv`` carries — ``depth`` when ``add_depth`` has been
            run upstream, otherwise ``echo_range``.
        min_status (int | None): Drop line points whose Echoview status is below
            this (0 none, 1 unverified, 2 bad, 3 good). ``None`` (default) keeps
            every point. Dropped points leave a gap, subject to ``max_gap_s``.
        max_gap_s (float | None): Widest hole in the line, in seconds, to
            interpolate across; pings inside a wider hole are NaN. A ping that
            lands exactly on a line point always keeps that point's depth.
            ``None`` (default) interpolates across holes of any width.
        edge_extend_s (float | None): How many seconds past the line's first and
            last point to hold that point's depth. Default ``0.0`` — no
            extrapolation, so pings outside the line's span are NaN. ``None``
            holds the end depths indefinitely.
        depth_offset_m (float): Constant metres added to the line, for when the
            transducer draft configured in Echoview differs from the one baked
            into ``ds_Sv['depth']``. Positive pushes the seafloor deeper.
        min_coverage (float): Fraction of pings (0-1) that must end up with a
            finite depth; below it a ValueError is raised. Default 0.0 never
            raises.
        file_time_start (str | datetime | None): Inclusive lower bound used to
            pick line files out of an ``evl_path`` folder, by the
            ``d{YYYYMMDD}_t{HHMMSS}-t{HHMMSS}`` span in their names. Pass the
            same window that selected the raw files. Ignored when ``evl_path``
            names a single file.
        file_time_end (str | datetime | None): Inclusive upper bound; see
            ``file_time_start``.

    Returns:
        xr.DataArray: Seafloor depth in metres, dims ``("ping_time",)``, named
        ``seafloor_depth``, on ``ds_Sv``'s ``ping_time``.
    """
    evl_paths = _resolve_evl_paths(evl_path, file_time_start, file_time_end)
    points = pd.concat(
        [_parse_evl(path) for path in evl_paths], ignore_index=True
    )

    if min_status is not None:
        points = points[points["status"] >= min_status]

    points = (
        points.dropna(subset=["time"])
        .sort_values("time")
        .drop_duplicates(subset="time", keep="first")
    )
    if points.empty:
        detail = "" if min_status is None else f" with status >= {min_status}"
        raise ValueError(
            f"EVL input has no line points{detail}: "
            f"{', '.join(str(path) for path in evl_paths)}"
        )

    values = _interp_line_to_ping_time(
        points, ds_Sv["ping_time"], max_gap_s, edge_extend_s
    )

    # Both sides of the create_seafloor_mask comparison must use the same
    # vertical reference, so convert the line to whichever one ds_Sv provides.
    range_var = "depth" if "depth" in ds_Sv else "echo_range"
    if vertical_reference == "surface":
        if range_var != "depth":
            raise ValueError(
                "A surface-referenced seafloor line has nothing to compare "
                "against: ds_Sv has no 'depth' variable. Run add_depth (recipe "
                "op 'ep_add_depth') upstream, or pass "
                "vertical_reference='transducer' if the EVL depths are measured "
                "from the transducer face."
            )
    elif vertical_reference == "transducer":
        if range_var == "depth":
            values = values + _computed_values(get_transducer_depth(ds_Sv))
    else:
        raise ValueError(
            "vertical_reference must be 'surface' or 'transducer', got "
            f"{vertical_reference!r}"
        )

    values = values + depth_offset_m

    coverage = _log_seafloor_line_summary(
        points, ds_Sv["ping_time"], values, evl_paths
    )
    if coverage < min_coverage:
        raise ValueError(
            f"Seafloor line covers {coverage:.1%} of pings, below the required "
            f"min_coverage of {min_coverage:.1%}: {evl_path}"
        )

    return xr.DataArray(
        values,
        coords={"ping_time": ds_Sv["ping_time"]},
        dims=["ping_time"],
        name="seafloor_depth",
        attrs={
            "long_name": "Seafloor depth from Echoview line file",
            "units": "m",
            "source_file": ", ".join(
                _storage.basename(path) for path in evl_paths
            ),
            "source_file_count": len(evl_paths),
            "vertical_reference": (
                "surface" if range_var == "depth" else "transducer"
            ),
            "range_var": range_var,
            "ping_coverage": float(coverage),
        },
    )


def create_seafloor_mask(ds_Sv, seafloor_depth, seafloor_buffer_m=0.0, range_var=None):
    """Create a boolean mask that keeps samples above the detected seafloor.

    The comparison is metres-vs-metres at every ``(channel, ping_time,
    range_sample)`` cell. Both sides of the comparison must use the same
    vertical reference. ``range_var`` selects which meter-valued variable on
    ``ds_Sv`` to compare against:

    * ``"depth"`` (metres below surface) — pair with a surface-referenced
      ``seafloor_depth`` line.
    * ``"echo_range"`` (metres along beam from transducer) — pair with a
      transducer-referenced ``seafloor_depth`` line.

    When ``range_var`` is ``None`` (default), it auto-selects ``"depth"`` if
    that variable is present on ``ds_Sv`` (i.e., ``add_depth`` was run),
    otherwise falls back to ``"echo_range"``.
    """
    if range_var is None:
        range_var = "depth" if "depth" in ds_Sv else "echo_range"
    if range_var not in ds_Sv:
        raise KeyError(
            f"range_var '{range_var}' not found in ds_Sv. "
            "Run add_depth before create_seafloor_mask if using 'depth'."
        )
    normalized_depth = _normalize_seafloor_depth(seafloor_depth, ds_Sv)
    expanded_depth = normalized_depth.expand_dims(channel=ds_Sv["channel"])
    expanded_depth = expanded_depth.transpose("channel", "ping_time")
    return (ds_Sv[range_var] <= (expanded_depth - seafloor_buffer_m)).astype(bool)


def create_surface_mask(ds_Sv, surface_depth_m=0.0):
    """Create a boolean mask that excludes surface interference."""
    return (ds_Sv["echo_range"] > surface_depth_m).astype(bool)


def create_frequency_mask(ds_Sv, frequencies_to_mask=None):
    """Create a boolean mask that excludes selected frequency channels."""
    if not frequencies_to_mask:
        return xr.ones_like(ds_Sv["Sv"], dtype=bool)

    target_frequencies = {int(float(freq)) for freq in frequencies_to_mask}
    channel_keep = []
    for freq_hz in ds_Sv["frequency_nominal"].values:
        freq_khz = int(round(float(freq_hz) / 1000))
        channel_keep.append(freq_khz not in target_frequencies)

    channel_mask = xr.DataArray(
        np.asarray(channel_keep, dtype=bool),
        coords={"channel": ds_Sv["channel"]},
        dims=["channel"],
    )
    return channel_mask.broadcast_like(ds_Sv["Sv"])


def combine_masks(masks, mode="and"):
    """Combine a list of boolean masks using logical AND or OR."""
    if not masks:
        raise ValueError("masks must contain at least one DataArray")

    for index, mask in enumerate(masks):
        _validate_boolean_mask(mask, f"masks[{index}]")

    try:
        broadcast_masks = xr.broadcast(*masks)
    except ValueError as exc:
        raise ValueError("Mask inputs are not broadcast-compatible") from exc

    if mode not in {"and", "or"}:
        raise ValueError("mode must be 'and' or 'or'")

    combined_mask = broadcast_masks[0]
    for mask in broadcast_masks[1:]:
        if mode == "and":
            combined_mask = combined_mask & mask
        else:
            combined_mask = combined_mask | mask

    return combined_mask.astype(bool)


def _find_best_seafloor_detection(echodata, ds_Sv=None, min_valid_depth_m=10.0):
    seafloor_depth = _get_detected_seafloor_depth(echodata)

    best_channel_idx = None
    best_score = -1
    best_freq_khz = None
    best_channel_label = None

    print("\n Analyzing seafloor detection quality across channels...")
    print(f"   Minimum valid depth threshold: {min_valid_depth_m}m")

    channel_values = seafloor_depth["channel"].values
    for ch_idx in range(seafloor_depth.sizes["channel"]):
        freq_hz = None
        if ds_Sv is not None and "frequency_nominal" in ds_Sv:
            freq_hz = ds_Sv["frequency_nominal"].values[ch_idx]
        freq_khz = None if freq_hz is None else int(round(float(freq_hz) / 1000))

        channel_label = _channel_display_label(channel_values[ch_idx], freq_hz)
        seafloor_ch = seafloor_depth.isel(channel=ch_idx)
        all_values = seafloor_ch.values
        valid_values = all_values[~np.isnan(all_values)]

        n_valid = len(valid_values)
        n_zeros = np.sum(valid_values == 0.0)
        n_deep_enough = np.sum(valid_values >= min_valid_depth_m)

        if n_valid == 0:
            print(f"   {channel_label}: no data")
            continue
        if n_zeros == n_valid:
            print(f"   {channel_label}: all zeros")
            continue
        if n_deep_enough == 0:
            mean_depth = valid_values.mean()
            print(f"   {channel_label}: too shallow (mean={mean_depth:.1f}m)")
            continue

        valid_deep_values = valid_values[valid_values >= min_valid_depth_m]
        mean_depth = valid_deep_values.mean()
        std_depth = valid_deep_values.std()
        score = n_deep_enough * (1.0 - min(std_depth / mean_depth, 1.0))

        print(
            f"   {channel_label}: {n_deep_enough}/{n_valid} valid, "
            f"mean={mean_depth:.1f}m, std={std_depth:.1f}m, score={score:.0f}"
        )

        if score > best_score:
            best_score = score
            best_channel_idx = ch_idx
            best_freq_khz = freq_khz
            best_channel_label = channel_label

    if best_channel_idx is None:
        raise ValueError("No valid seafloor detection found in any channel")

    print(f"\n Best seafloor detection: {best_channel_label}")

    return best_channel_idx, best_freq_khz, seafloor_depth.isel(channel=best_channel_idx)


def log_seafloor_detection_stats(echodata, ds_Sv=None, min_valid_depth_m=10.0):
    """Print summary statistics for seafloor detections across channels."""
    seafloor_depth = _get_detected_seafloor_depth(echodata)
    channel_values = seafloor_depth["channel"].values

    print("Seafloor detection statistics:")
    for ch_idx in range(seafloor_depth.sizes["channel"]):
        freq_hz = None
        if ds_Sv is not None and "frequency_nominal" in ds_Sv:
            freq_hz = ds_Sv["frequency_nominal"].values[ch_idx]

        channel_label = _channel_display_label(channel_values[ch_idx], freq_hz)
        seafloor_ch = seafloor_depth.isel(channel=ch_idx)
        valid_values = seafloor_ch.values[np.isfinite(seafloor_ch.values)]
        valid_deep_values = valid_values[valid_values >= min_valid_depth_m]

        if len(valid_deep_values) == 0:
            print(f"  {channel_label}: no valid detections >= {min_valid_depth_m}m")
            continue

        quantiles = np.percentile(valid_deep_values, [10, 50, 90])
        print(
            f"  {channel_label}: min={valid_deep_values.min():.1f}m, "
            f"max={valid_deep_values.max():.1f}m, valid_pings={len(valid_deep_values)}, "
            f"p10={quantiles[0]:.1f}m, p50={quantiles[1]:.1f}m, p90={quantiles[2]:.1f}m"
        )

def find_best_seafloor_detection(ed_raw, ds_Sv, min_valid_depth_m=10.0):
    """Find the best seafloor detection across all channels.
    
    Args:
        ed_raw: Original EchoData object with seafloor detection
        ds_Sv: Calibrated Sv dataset from ep.calibrate.compute_Sv()
        min_valid_depth_m: Minimum depth to consider as valid seafloor (default 10m)
        
    Returns:
        tuple: (best_channel_idx, best_freq_khz, best_seafloor_data)
    """
    return _find_best_seafloor_detection(
        echodata=ed_raw,
        ds_Sv=ds_Sv,
        min_valid_depth_m=min_valid_depth_m,
    )


def remove_seafloor_from_mask(ed_raw, ds_Sv, mask, buffer_m=1.0, use_best_detection=True, min_valid_depth_m=10.0):
    """Update a mask to exclude data at and below the detected seafloor.

    Args:
        ed_raw: Original EchoData object with seafloor detection.
        ds_Sv (xr.Dataset): Calibrated Sv dataset from ``ep.calibrate.compute_Sv()``.
        mask (xr.DataArray): Boolean mask to update (True = keep).
        buffer_m (float): Buffer in meters above seafloor to keep (default 1.0).
        use_best_detection (bool): If True, use the best single-channel
            seafloor detection for all channels (default True).
        min_valid_depth_m (float): Minimum depth to consider as valid
            seafloor (default 10.0).

    Returns:
        xr.DataArray: Updated mask with seafloor data excluded.
    """
    
    # Get bottom detection data from raw EchoData object
    seafloor_depth = ed_raw['Vendor_specific']['detected_seafloor_depth']
    
    # Get the echo_range from the calibrated dataset
    echo_range = ds_Sv['echo_range']
    
    if use_best_detection:
        best_seafloor = detect_seafloor(
            ds_Sv=ds_Sv,
            echodata=ed_raw,
            min_valid_depth_m=min_valid_depth_m,
        )
        seafloor_mask = create_seafloor_mask(
            ds_Sv,
            best_seafloor,
            seafloor_buffer_m=buffer_m,
        )
        print(f" Applied seafloor mask to all channels")
        return combine_masks([mask, seafloor_mask], mode="and")
        
    else:
        # Original behavior: use per-channel detection
        print(f"Removing seafloor and below from mask with {buffer_m}m buffer (per-channel detection)...")
        
        for ch_idx, channel in enumerate(ds_Sv.channel):
            freq_hz = ds_Sv["frequency_nominal"].values[ch_idx]
            freq_khz = int(freq_hz / 1000)
            
            seafloor_ch = seafloor_depth.isel(channel=ch_idx)
            valid_seafloor = seafloor_ch.values[~np.isnan(seafloor_ch.values)]
            
            if len(valid_seafloor) == 0:
                print(f"WARNING: {freq_khz} kHz has NO seafloor detection data!")
            elif np.all(valid_seafloor == 0.0):
                print(f"WARNING: {freq_khz} kHz has ALL ZEROs for seafloor detection!")
            elif valid_seafloor.mean() < min_valid_depth_m:
                print(f"WARNING: {freq_khz} kHz has suspiciously shallow seafloor detection (mean={valid_seafloor.mean():.1f}m)")
                print(f"   This may indicate failed detection. Verify results carefully.")

            for ping_idx, ping_time in enumerate(ds_Sv.ping_time):
                try:
                    seafloor_at_ping = seafloor_depth.isel(channel=ch_idx, ping_time=ping_idx)
                    ping_ranges = echo_range.isel(channel=ch_idx, ping_time=ping_idx)
                    range_mask = ping_ranges <= (seafloor_at_ping - buffer_m)
                    mask[ch_idx, ping_idx, :] = range_mask.values & mask[ch_idx, ping_idx, :]
                    
                except (KeyError, IndexError, ValueError) as e:
                    if ping_idx % 100 == 0:
                        print(f"No seafloor data for channel {ch_idx}, ping {ping_idx}, keeping all data")
                    continue
                    
    return mask


def mask_frequency_channels(ds_Sv, mask, frequencies_to_mask_khz):
    """Mask out specific frequency channels entirely.
    
    Useful for excluding high-frequency channels that don't work well at deeper depths
    or have known issues with the data.
    
    Args:
        ds_Sv: Calibrated Sv dataset from ep.calibrate.compute_Sv()
        mask: Boolean array mask to be updated
        frequencies_to_mask_khz: List of frequencies in kHz to completely mask out
    
    Returns:
        mask: Updated boolean array with specified frequencies masked
    """
    if not frequencies_to_mask_khz:
        return mask

    print(f"\n Masking out frequency channels: {frequencies_to_mask_khz} kHz...")
    frequency_mask = create_frequency_mask(ds_Sv, frequencies_to_mask_khz)
    return combine_masks([mask, frequency_mask], mode="and")


def remove_surface_from_mask(ds_Sv, mask, depth_threshold_m):
    """Exclude data from the surface down to a specified depth threshold.
    
    Args:
        ds_Sv (xr.Dataset): Calibrated Sv dataset containing ``echo_range``.
        mask (xr.DataArray): Boolean mask to update (True = keep).
        depth_threshold_m (float): Depth in meters below the surface to mask.
    
    Returns:
        xr.DataArray: Updated mask with surface data excluded.
    """

    print(f"Removing surface from mask for depths <= {depth_threshold_m}m...")
    surface_mask = create_surface_mask(ds_Sv, surface_depth_m=depth_threshold_m)
    return combine_masks([mask, surface_mask], mode="and")


def apply_mask_to_sv(ds_Sv, mask, fill_value=np.nan):
    """Apply a boolean mask to the Sv variable in a dataset.
    
    Args:
        ds_Sv (xr.Dataset): Sv dataset to mask.
        mask (xr.DataArray): Boolean mask (True = keep, False = exclude).
        fill_value: Value to assign where mask is False (default: ``np.nan``).
    
    Returns:
        xr.Dataset: Masked copy of *ds_Sv*.
    """
    ed_masked = ep.mask.apply_mask(
        source_ds=ds_Sv,
        mask=mask,
        var_name="Sv",
        fill_value=fill_value
    )
    return ed_masked


def create_data_mask(echodata, ds_Sv, seafloor_buffer_m=10.0, surface_depth_m=10.0, frequencies_to_mask=None):
    """Create a data mask combining seafloor, surface, and frequency exclusions.

    Args:
        echodata: Echopype EchoData object.
        ds_Sv: Sv dataset to create the mask for.
        seafloor_buffer_m: Buffer in meters below seafloor to mask (default 10.0).
        surface_depth_m: Depth in meters above which to mask surface echoes (default 10.0).
        frequencies_to_mask: List of frequencies in kHz to mask entirely 
    Returns:
        xarray.DataArray: Boolean mask (True = keep, False = exclude).
    """
        
    seafloor_depth = detect_seafloor(ds_Sv=ds_Sv, echodata=echodata)
    seafloor_mask = create_seafloor_mask(
        ds_Sv,
        seafloor_depth,
        seafloor_buffer_m=seafloor_buffer_m,
    )
    surface_mask = create_surface_mask(ds_Sv, surface_depth_m=surface_depth_m)
    frequency_mask = create_frequency_mask(ds_Sv, frequencies_to_mask)
    mask = combine_masks([seafloor_mask, surface_mask, frequency_mask], mode="and")
    log_mask_stats(mask)

    return mask


def apply_mask_to_sv_datasets(sv_datasets, mask):
    """Apply a mask to multiple Sv datasets.

    Args:
        sv_datasets: Dict of Sv datasets (e.g., from compute_calibrated_sv_datasets).
        mask: Boolean mask to apply (True = keep, False = exclude).

    Returns:
        dict: Dictionary of masked Sv datasets with the same keys.
    """
    return {name: apply_mask_to_sv(ds, mask) for name, ds in sv_datasets.items()}


def apply_seafloor_and_surface_masks(ds_Sv, ed_raw, depth_threshold_m=10.0, buffer_m=1.0, 
                                      use_best_seafloor=True, min_valid_depth_m=10.0,
                                      exclude_frequencies_khz=None):
    """Apply seafloor, surface, and frequency exclusion masks.
    
    Args:
        ds_Sv: Calibrated Sv dataset
        ed_raw: Original EchoData object with seafloor detection
        depth_threshold_m: Surface exclusion depth threshold
        buffer_m: Buffer above seafloor to keep
        use_best_seafloor: If True, use best seafloor detection for all channels
        min_valid_depth_m: Minimum depth to consider valid seafloor
        exclude_frequencies_khz: List of frequencies (kHz) to completely mask out
    
    Returns:
        mask: Final combined mask
    """
    mask = createSvMask(ds_Sv)
    mask = remove_seafloor_from_mask(ed_raw, ds_Sv, mask, buffer_m, use_best_seafloor, min_valid_depth_m)
    mask = remove_surface_from_mask(ds_Sv, mask, depth_threshold_m)
    
    if exclude_frequencies_khz:
        mask = mask_frequency_channels(ds_Sv, mask, exclude_frequencies_khz)
    
    log_mask_stats(mask)
    return mask


def log_mask_stats(mask):
    """Print summary statistics for a boolean mask.

    Args:
        mask (xr.DataArray): Boolean mask (True = kept, False = masked).
    """
    total_points = mask.size
    kept_points = mask.sum().values
    masked_points = total_points - kept_points
    
    print(f"  - Total points: {total_points}")
    print(f"  - Points kept: {kept_points} ({100*(kept_points/total_points):.1f}%)")
    print(f"  - Points masked: {masked_points} ({100*(masked_points/total_points):.1f}%)")


def initial_setup_and_validation(raw_input_folder, calibration_outputs_string="calibration",
                                  raw_file_names=None, clear_previous_json_logs=True,
                                  file_time_start=None, file_time_end=None):
    """Set up output folders, resolve raw file paths, and optionally clear previous logs.

    Args:
        raw_input_folder: Path to the folder containing raw input files. May be a
            local path or a remote fsspec URL (``gs://bucket/survey/raw``); a
            remote folder is listed without downloading any data.
        calibration_outputs_string: Subdirectory name for calibration artifacts. When
            running inside the recipe executor this is resolved relative to the pipeline's
            user-facing outputs directory (``artifacts_dir`` from the execution context)
            so calibration files land alongside images and logs under ``outputs/``.
            Falls back to a CWD-relative path when called standalone (e.g. in a notebook).
            Calibration JSON logs are written to a ``logs/`` subdirectory within this
            folder (i.e. ``<calibration_outputs>/logs/``).
        raw_file_names: Optional list of specific raw file name strings to process.
            If empty or None, all .raw files in raw_input_folder are used.
        clear_previous_json_logs: If True, delete existing calibration_flags.json (default: True).
        file_time_start: Optional inclusive lower bound (ISO string or datetime)
            on each raw file's recording span. The span is inferred from the
            ``D{YYYYMMDD}-T{HHMMSS}`` name stamps: a file's own stamp is its
            start and the next file's stamp is its end, so a file that starts
            before this bound but records into the window is kept. Filtering is
            name-based except for the single file straddling this bound, whose
            real end is read from its datagram headers so a gap between survey
            legs is not mistaken for one long recording.
        file_time_end: Optional inclusive upper bound; see *file_time_start*.

    Returns:
        dict with keys:
            raw_file_paths: list[str] of absolute paths (or gs:// URLs) to raw
                acoustic files.
            calibration_output_dir: str resolved full path to the calibration outputs folder.
    """
    # Coerce only local values: Path() mangles a URL ("gs://b/x" -> "gs:/b/x").
    raw_input_remote = _storage.is_remote(raw_input_folder)
    if not raw_input_remote:
        raw_input_folder = Path(raw_input_folder)

    # Resolve the calibration output directory relative to the pipeline outputs dir
    # when running inside the executor; fall back to CWD-relative when standalone.
    try:
        from aa_recipe_manager.executor.runtime_context import get_execution_context  # noqa: PLC0415
        ctx = get_execution_context()
        if ctx.artifacts_dir is not None:
            calibration_output_dir = Path(ctx.artifacts_dir) / calibration_outputs_string
        else:
            calibration_output_dir = Path(calibration_outputs_string)
    except ImportError:
        calibration_output_dir = Path(calibration_outputs_string)

    # Logs always live at <calibration_outputs>/logs — not user-configurable.
    calibration_logs_dir = calibration_output_dir / "logs"

    # Get list of raw files to process
    if raw_file_names is not None and len(raw_file_names) > 0:
        raw_file_paths = [
            _storage.join(raw_input_folder, filename) for filename in raw_file_names
        ]
    elif raw_input_remote:
        raw_file_paths = _storage.glob_url(
            raw_input_folder, "*.raw", _execution_storage_options()
        )
    else:
        raw_file_paths = sorted(raw_input_folder.glob("*.raw"))

    if not raw_file_paths:
        raise FileNotFoundError("No raw files found to process")

    if file_time_start is not None or file_time_end is not None:
        before = len(raw_file_paths)
        raw_file_paths = filter_paths_by_file_time(
            raw_file_paths,
            file_time_start,
            file_time_end,
            storage_options=_execution_storage_options(),
        )
        print(
            f"  Filename-time filter: {before} -> {len(raw_file_paths)} raw file(s) "
            f"({file_time_start} to {file_time_end})"
        )
        if not raw_file_paths:
            raise FileNotFoundError(
                "No raw files found to process within the filename-time window "
                f"({file_time_start} to {file_time_end})"
            )

    print(f"Found {len(raw_file_paths)} raw file(s) to process:")
    for path in raw_file_paths:
        print(f"  - {_storage.basename(path)}")

    # Ensure calibration log folder exists (parents=True also creates calibration_output_dir)
    if not calibration_logs_dir.exists():
        calibration_logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created missing output folder: {calibration_logs_dir}")

    if clear_previous_json_logs:
        flags_file = calibration_logs_dir / "calibration_flags.json"
        if flags_file.exists():
            flags_file.unlink()

    return {
        "raw_file_paths": [str(p) for p in raw_file_paths],
        "calibration_output_dir": str(calibration_output_dir),
    }


def _execution_temp_dir():
    """Return ``ExecutionContext.temp_dir`` (Path/StorageLocation) or None."""
    try:
        from aa_recipe_manager.executor.runtime_context import get_execution_context  # noqa: PLC0415
        return get_execution_context().temp_dir
    except ImportError:
        return None


def _execution_storage_options():
    """Return the pipeline's fsspec options for remote *input* paths, or None.

    ``getattr`` keeps this working against recipe-manager versions predating the
    ``storage_options`` context field; ``None`` then means "use ambient
    credentials" (Application Default Credentials on GCP), which is the intended
    default anyway.
    """
    try:
        from aa_recipe_manager.executor.runtime_context import get_execution_context  # noqa: PLC0415
        options = getattr(get_execution_context(), "storage_options", None)
        return dict(options) if options else None
    except ImportError:
        return None


def _resolve_intermediate_dir() -> tuple[Path | str, dict | None]:
    """Resolve the per-file intermediate store directory and its storage options.

    When running inside the recipe executor the directory is read from
    ``ExecutionContext.temp_dir`` (set to ``exe_temp/``) and the stores are
    written under ``<temp_dir>/data/``.  Falls back to a sub-directory of the
    OS temp folder when called standalone (e.g. in a plain script or notebook).

    Returns a ``(data_dir, storage_options)`` pair where ``data_dir`` is a
    local ``Path`` or, for a ``gs://`` scratch dir, a URL string.
    """
    import tempfile
    temp_dir = _execution_temp_dir()
    if temp_dir is not None:
        return _storage.join(temp_dir, "data"), _storage.storage_options_of(temp_dir)
    return Path(tempfile.gettempdir()) / "aa_si_utils_temp" / "data", None


def _grant_access(path):
    """Add the owner permissions needed to remove ``path``.

    Read and write are always granted; execute is added for directories only,
    so a data file is never made executable.  Bits are OR-ed onto the current
    mode rather than replacing it: assigning ``stat.S_IWRITE`` outright is
    ``0o200``, which on POSIX strips read and execute from a directory and
    leaves it permanently unlistable -- turning a transient rmtree error into a
    permanent "Permission denied" on that path.
    """
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return
    wanted = mode | stat.S_IWRITE | stat.S_IREAD
    if stat.S_ISDIR(mode):
        wanted |= stat.S_IEXEC
    if wanted != mode:
        try:
            os.chmod(path, wanted)
        except OSError:
            pass


def _rmtree_onerror(func, fpath, _exc):
    """``shutil.rmtree`` error handler that fixes up permissions and retries.

    Removing an entry needs write and execute on its *parent*; listing a
    directory needs read and execute on the directory *itself*.  Both are
    granted, since which one is missing depends on whether rmtree failed while
    scanning or while unlinking.
    """
    _grant_access(os.path.dirname(fpath) or ".")
    _grant_access(fpath)
    func(fpath)


def _remove_existing_store(path):
    """Remove a store dir/file, fixing up restrictive permissions if needed."""
    path = Path(path)
    if not path.exists():
        return
    if path.is_dir():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_rmtree_onerror)
        else:
            shutil.rmtree(path, onerror=_rmtree_onerror)
    else:
        path.unlink()


def _is_transient_windows_lock(exc):
    """Return True for a Windows file-lock/rename PermissionError.

    A POSIX ``EACCES`` carries no ``winerror`` and is not transient: the
    permissions will be exactly the same on the next attempt, so retrying only
    repeats the failed removal and delays the real error.
    """
    return getattr(exc, "winerror", None) is not None


def _store_write_context(store_path):
    """Describe a failed store write: path, existing remains, free disk."""
    parts = [f"store: {store_path}"]
    path = Path(store_path)
    if path.exists():
        parts.append("a partial store is present")
    try:
        usage = shutil.disk_usage(path.parent if path.parent.exists() else ".")
        parts.append(f"free disk: {usage.free / 1024 ** 3:.1f} GiB")
    except OSError:
        pass
    return "; ".join(parts)


def _write_store_with_retry(write_fn, store_path, max_retries=3, base_delay=1.0):
    """Write a zarr store, retrying a transient Windows PermissionError.

    zarr-python v3 writes each metadata file atomically: bytes go to a
    ``.partial`` temp file which is then renamed into place.  On Windows,
    antivirus or the file-indexing service can hold an exclusive lock on the
    temp file between the write and rename steps, failing the rename with
    PermissionError [WinError 5].  Mirrors the recipe manager's checkpoint
    writer mitigation: clean up any partially-written store and retry after
    an increasing delay (1 s, 2 s).

    Only that Windows race is retried.  Any other ``PermissionError`` is raised
    on the first attempt with the store path, remains, and free disk attached.
    """
    last_exc = None
    for attempt in range(max_retries):
        _remove_existing_store(store_path)
        try:
            write_fn()
            return
        except PermissionError as exc:
            last_exc = exc
            if not _is_transient_windows_lock(exc):
                raise PermissionError(
                    f"{exc} ({_store_write_context(store_path)})"
                ) from exc
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc


def _fmt_gib(num_bytes):
    """Format a byte count as GiB."""
    return f"{num_bytes / 1024 ** 3:.2f} GiB"


def _free_disk_note(target):
    """Free space on the filesystem holding ``target``, or '' if remote."""
    if target is None or _storage.is_remote(target):
        return ""
    path = Path(os.fspath(target))
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        return f", free disk {_fmt_gib(shutil.disk_usage(path).free)}"
    except OSError:
        return ""


def _memory_note():
    """Available RAM, plus echopype's swap threshold, or '' if unavailable."""
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        return ""
    mem = psutil.virtual_memory()
    return (
        f", RAM {_fmt_gib(mem.available)} free of {_fmt_gib(mem.total)}"
        f" (swap above {_fmt_gib(mem.total * 0.4)})"
    )


def _swap_note(ed):
    """Report whether echopype backed backscatter_r with dask (swap mode).

    ``use_swap`` is decided per file from live memory pressure, so it can flip
    partway through a survey. Reading it off the object is the only way to know
    which path a given file actually took.
    """
    try:
        beam = ed["Sonar/Beam_group1"]
        chunks = beam["backscatter_r"].chunks
    except Exception:
        return ""
    return ", backscatter_r dask-backed (swap)" if chunks else ""


def _store_size_note(store_path):
    """On-disk size of a written store, or '' when it cannot be measured."""
    try:
        if _storage.is_remote(store_path):
            return ""
        total = sum(
            f.stat().st_size for f in Path(store_path).rglob("*") if f.is_file()
        )
    except OSError:
        return ""
    return f", store {_fmt_gib(total)}"


def _open_and_store(local_raw_path, sonar_model, include_bot, use_swap,
                     intermediate_format, store_dir, store_options, remote):
    """Convert one *local* raw file and return its EchoData or store path.

    Deliberately a module-level function taking explicit arguments (rather than
    a closure over the caller's loop variables): this is the per-file unit that
    a future ``map_over`` step will dispatch, and it must stay picklable for
    the Dask/Prefect executors.

    Prints one line before parsing and one after writing. This is the record of
    how far a conversion got when a later file fails, and of whether echopype
    switched to swap mode partway through, so it is not gated behind a flag.
    """
    label = Path(local_raw_path).name
    try:
        raw_size = f", raw {_fmt_gib(Path(local_raw_path).stat().st_size)}"
    except OSError:
        raw_size = ""
    print(f"  open_raw {label}{raw_size}{_memory_note()}", flush=True)

    parse_start = time.perf_counter()
    ed = ep.open_raw(local_raw_path, sonar_model=sonar_model,
                     include_bot=include_bot, use_swap=use_swap)
    print(
        f"  parsed {label} in {time.perf_counter() - parse_start:.1f}s"
        f"{_swap_note(ed)}",
        flush=True,
    )

    if intermediate_format == "none":
        return ed
    if intermediate_format == "netcdf":
        store_path = store_dir / f"{local_raw_path.stem}.nc"
        ed.to_netcdf(save_path=store_path)
        return str(store_path)

    store_path = _storage.join(store_dir, f"{local_raw_path.stem}.zarr")
    print(
        f"  writing {store_path}{_free_disk_note(store_dir)}",
        flush=True,
    )
    write_start = time.perf_counter()
    # Match the checkpoint writer exactly: zarr_format=2 and compress=False.
    # Writing v3 encodes echopype's fixed-length UTF-32 string metadata
    # with a v3 serializer the v2 checkpoint write cannot express
    # ("Zarr format 2 arrays do not support serializer").  Leaving
    # compression on makes echopype attach a zarr-v3 BloscCodec that a
    # v2 write also rejects ("Invalid compressor ... Got BloscCodec");
    # compress=False sidesteps it, consistent with the checkpoint store.
    if remote:
        # echopype 0.11.1 cannot write a zarr to a remote store: its save_file()
        # hands the protocol-stripped fsspec mapper root (e.g. "bucket/key.zarr")
        # to xarray.to_zarr with no filesystem, so a gs:// save_path silently
        # becomes a LOCAL relative write and never reaches the bucket.
        #
        # A per-file store is one raw file's worth of data, so stage it on local
        # scratch (fast, and echopype writes consolidated metadata so it reopens
        # quickly) and bulk-upload with fs.put, which parallelizes the many small
        # zarr objects — ~10x faster than a sequential per-chunk remote write.
        # Local disk holds only this one store, then it is deleted.
        import tempfile
        scratch = Path(tempfile.mkdtemp(prefix="aa_si_zarr_"))
        try:
            local_store = scratch / f"{local_raw_path.stem}.zarr"
            _write_store_with_retry(
                lambda: ed.to_zarr(save_path=local_store, zarr_format=2,
                                   compress=False, overwrite=True),
                local_store,
            )
            _storage.remove_store(store_path, store_options)  # clear stale store
            _storage.get_fs(store_path, store_options).put(
                str(local_store), str(store_path), recursive=True)
        finally:
            _storage._rmtree_local(scratch)
    else:
        # overwrite=True matches the remote branch above. Without it, echopype's
        # to_file() logs "already exists, will not overwrite" and returns
        # WITHOUT writing whenever the store still exists, so a store this
        # function only partly removed would be reported as a success and fail
        # much later, in combine_raw, as a corrupt store.
        _write_store_with_retry(
            lambda: ed.to_zarr(save_path=store_path, zarr_format=2,
                               compress=False, overwrite=True),
            store_path,
        )
    print(
        f"  wrote {label} in {time.perf_counter() - write_start:.1f}s"
        f"{_store_size_note(store_path)}{_free_disk_note(store_dir)}",
        flush=True,
    )
    return str(store_path)


def read_raw_files_to_stores(raw_file_paths, sonar_model="EK60", include_bot=True,
                              intermediate_format="netcdf", use_swap="auto"):
    """Open raw sonar files and write each as an intermediate store.

    Args:
        raw_file_paths: List of string paths or Path objects pointing to raw files.
            Entries may also be remote fsspec URLs (``gs://bucket/.../f.raw``).
            Each remote file is downloaded to a private local scratch directory,
            converted, and its local copy deleted before the next file is
            fetched — local disk therefore holds at most one raw file at a time
            (per concurrent instance), and bucket objects are never modified.
        sonar_model: Sonar model string for echopype (default: "EK60").
        include_bot: Whether to include .bot files (default: True).
        intermediate_format: ``"netcdf"`` (default), ``"zarr"``, or ``"none"``.
            ``"none"`` returns in-memory EchoData objects without writing any
            files.

            Pick by where the intermediates must live:

            ``"netcdf"`` is the default and the simplest choice for a *local*
            temp dir: each raw file is written to a temporary ``.nc`` file, then
            reopened and combined by ``combine_raw_stores``.  After combining,
            all groups are rechunked to a single uniform chunk so the downstream
            zarr v2 checkpoint write (``combine_raw`` step with
            ``checkpoint: always``) always succeeds.  HDF5 needs seekable
            writes, so netcdf **cannot** target object storage — a remote temp
            dir raises.

            ``"none"`` keeps every ``EchoData`` object in memory without
            touching disk.  Suitable when the full dataset fits comfortably in
            RAM.

            ``"zarr"`` is the option for surveys too large to stage locally: it
            is the only format that can write intermediates to a bucket, so a
            ``gs://`` temp dir requires it.  Stores are written with
            ``zarr_format=2`` and ``compress=False`` to match the checkpoint
            writer exactly; that is what avoids the v3-serializer and
            v3-BloscCodec errors a default ``to_zarr`` would trigger on the
            downstream v2 checkpoint write (see ``_open_and_store``).  Remote
            stores are staged to local scratch and bulk-uploaded, so local disk
            still holds only one store at a time (per concurrent instance).
        use_swap: Passed to ``ep.open_raw`` as ``use_swap`` (default: ``"auto"``).

    Returns:
        list: Store path strings (zarr/netcdf) for file-backed formats, or a
        list of in-memory EchoData objects for ``intermediate_format="none"``.
    """
    if not raw_file_paths:
        raise ValueError("No raw files provided")

    store_dir = None
    store_options = None
    remote = False
    if intermediate_format != "none":
        store_dir, store_options = _resolve_intermediate_dir()
        remote = _storage.is_remote(store_dir)
        if remote and intermediate_format == "netcdf":
            raise ValueError(
                "netcdf intermediates require a local temp dir (HDF5 needs "
                "seekable writes and cannot be written to object storage). "
                "Use intermediate_format='zarr' or point --temp-dir at a local "
                f"directory instead of {store_dir!r}."
            )
        _storage.makedirs(store_dir, store_options)

    input_options = _execution_storage_options()
    # echopype locates a .bot companion by swapping the raw file's extension,
    # so it must be downloaded beside the raw file rather than passed in.
    companions = (".bot",) if include_bot else ()

    results = []
    total = len(raw_file_paths)
    if store_dir is not None:
        print(f"Intermediate stores: {store_dir}{_free_disk_note(store_dir)}")
    for index, raw_path in enumerate(raw_file_paths, start=1):
        print(f"[{index}/{total}] {Path(str(raw_path)).name}", flush=True)
        if _storage.is_remote(raw_path):
            with _storage.localized_file(
                str(raw_path),
                storage_options=input_options,
                companion_suffixes=companions,
            ) as local_raw:
                # open_raw fully parses the file before returning, so nothing
                # references the local copy once the context block exits.
                results.append(_open_and_store(
                    local_raw, sonar_model, include_bot, use_swap,
                    intermediate_format, store_dir, store_options, remote,
                ))
        else:
            results.append(_open_and_store(
                Path(raw_path), sonar_model, include_bot, use_swap,
                intermediate_format, store_dir, store_options, remote,
            ))

    print(f"Read {len(raw_file_paths)} raw file(s) to '{intermediate_format}' format")
    return results


def combine_raw_stores(raw_stores, ping_time_chunk=DEFAULT_PING_TIME_CHUNK):
    """Combine per-file stores or in-memory EchoData objects into a single EchoData.

    Args:
        raw_stores: List of store path strings (zarr/netcdf) produced by
            ``read_raw_files_to_stores``, or a list of in-memory EchoData
            objects (for ``intermediate_format="none"`` mode).

            A **list of such lists** is also accepted and flattened one level.
            That is the shape a ``collect`` fan-in produces when the read step
            is parallelized with ``map_over``: each mapped instance reads one
            raw file and returns a one-element list, so the collected value is
            ``[[store_0], [store_1], ...]``. Accepting both shapes lets the same
            recipe step serve the whole-survey and per-file-parallel forms.
            Order is preserved, which matters because ``ep.combine_echodata``
            expects its inputs in time order.
        ping_time_chunk: Target chunk length along ``ping_time`` for the combined
            object (default: ``DEFAULT_PING_TIME_CHUNK``).  Controls the chunking
            of the downstream zarr checkpoint.

    Returns:
        EchoData: Combined EchoData object (lazily backed for file-based inputs).
    """
    if not raw_stores:
        raise ValueError("No raw stores provided")

    # Flatten one level of nesting (see the map_over/collect note above). Only
    # genuine sequences are unwrapped: str/Path and EchoData pass through.
    if any(isinstance(entry, (list, tuple)) for entry in raw_stores):
        flattened = []
        for entry in raw_stores:
            if isinstance(entry, (list, tuple)):
                flattened.extend(entry)
            else:
                flattened.append(entry)
        raw_stores = flattened
        if not raw_stores:
            raise ValueError("No raw stores provided")

    # Remote (bucket-backed) stores need storage options to open. The store
    # strings are opaque, so recover the options from the same exe_temp scratch
    # location the read step used (usually empty for gs:// under ADC / memory://).
    remote_options = None
    if any(
        isinstance(item, (str, Path)) and _storage.is_remote(item)
        for item in raw_stores
    ):
        remote_options = _storage.storage_options_of(_execution_temp_dir())

    # Phase timing: combine_raw was observed spending minutes here while the
    # downstream checkpoint upload of the finished store is only seconds, so the
    # cost is in this function. These prints break the wall clock into open /
    # combine / rechunk so a run log shows which phase dominates.
    _t0 = time.perf_counter()
    echodata_list = []
    for item in raw_stores:
        if isinstance(item, (str, Path)):
            if _storage.is_remote(item):
                echodata_list.append(
                    ep.open_converted(
                        str(item), chunks={}, storage_options=remote_options
                    )
                )
            else:
                echodata_list.append(ep.open_converted(str(item), chunks={}))
        else:
            # Already an in-memory EchoData object (none mode)
            echodata_list.append(item)
    print(f"[combine_raw] opened {len(echodata_list)} store(s) in "
          f"{time.perf_counter() - _t0:.1f}s")

    _t0 = time.perf_counter()
    if len(echodata_list) == 1:
        echodata = echodata_list[0]
    else:
        echodata = ep.combine_echodata(echodata_list)
        print(f"Combined {len(echodata_list)} stores into one EchoData object")
    print(f"[combine_raw] combine_echodata in {time.perf_counter() - _t0:.1f}s")

    # EK80 only: combining N files stacks N filter-coefficient sets onto
    # Vendor_specific.filter_time, which forces echopype's compute_Sv into its
    # per-(channel, filter_time) loop.  That loop slices the beam to one channel
    # at a time and then rejects user-supplied per-channel cal/env params whose
    # length no longer matches the 1-channel slice (echopype param2da raises
    # "lengths of 'p_val' and 'channel' should be identical").  The
    # WBT_coeffs/PC_coeffs indexed by filter_time are used ONLY for broadband
    # (complex_FM) pulse-compression calibration, so when the combined data has
    # no FM beam group they are dead weight and we can safely collapse to a
    # single filter_time -- restoring echopype's single-pass (all-channels)
    # branch that works with per-channel cal/env lists.  Broadband data is
    # detected via Sonar/waveform_encode_descr and left completely untouched, so
    # FM pipelines keep every filter_time/coefficient set they need.
    if getattr(echodata, "sonar_model", None) in ("EK80", "ES80", "EA640"):
        _vend = echodata["Vendor_specific"]
        if "filter_time" in _vend.dims and _vend.sizes["filter_time"] > 1:
            _sonar = echodata["Sonar"]
            _descr = (
                _sonar["waveform_encode_descr"].values
                if "waveform_encode_descr" in _sonar
                else []
            )
            _has_fm = "complex_FM" in np.asarray(_descr).ravel().tolist()
            if not _has_fm:
                _n = _vend.sizes["filter_time"]
                # list index keeps filter_time as a size-1 dim so echopype's
                # `len(...) == 1` single-pass gate trips
                echodata["Vendor_specific"] = _vend.isel(filter_time=[0])
                print(
                    "[combine_raw] no FM beam group; collapsed Vendor_specific "
                    f"filter_time {_n} -> 1 (WBT/PC coeffs unused in CW)"
                )

    _t0 = time.perf_counter()

    # After combine_echodata the per-file dask chunks are simply concatenated,
    # producing uneven chunks along ``ping_time`` whenever files have different
    # ping counts (e.g. (3896, 3921)).  Zarr v2 requires every chunk but the last
    # to be uniform and the last to be <= the others, so the naive concatenated
    # chunking fails to write.  Rechunk the concatenation dimension to a uniform
    # target size: this satisfies the v2 constraint while keeping multiple chunks
    # along time so remote (bucket-backed) reads stay lazy instead of pulling
    # whole variables at once.  Groups without a ``ping_time`` dimension are small
    # metadata tables and stay single-chunk.
    #
    # Every other dimension collapses to a single chunk.  combine_echodata aligns
    # its inputs with ``join="outer"``, so a dimension whose length differs
    # between files -- ``range_sample`` when the files record different numbers of
    # samples -- comes back reindexed into ragged blocks such as
    # (2783, 12, 4, 7), which zarr v2 rejects just like the ping_time case.  Each
    # source store already held that dimension in one chunk, so restoring one
    # chunk here does not grow the per-chunk footprint.
    for group_path in echodata.group_paths:
        ds = echodata[group_path]
        if ds is None:
            continue
        if "ping_time" in ds.sizes:
            target = min(ping_time_chunk, ds.sizes["ping_time"])
            chunks = {dim: -1 for dim in ds.sizes if dim != "ping_time"}
            chunks["ping_time"] = target
            ds = ds.chunk(chunks)
        else:
            ds = ds.chunk(-1)
        for v in ds.variables:
            # Clear stale chunk *and* codec encoding carried over from the source
            # stores.  For zarr intermediates, open_converted records the store's
            # compressor/filters as zarr-v3 codec objects (e.g. BloscCodec); left
            # in place they crash the zarr-v2 checkpoint write ("Invalid
            # compressor ... Got BloscCodec").  Dropping them lets the checkpoint
            # writer re-derive v2-compatible encoding from its own compress flag.
            for key in (
                "chunks",
                "preferred_chunks",
                "compressor",
                "filters",
                "codecs",
                "serializer",
            ):
                ds[v].encoding.pop(key, None)
        echodata[group_path] = ds
    print(f"[combine_raw] rechunk in {time.perf_counter() - _t0:.1f}s")

    _t0 = time.perf_counter()
    check_for_seafloor_depth_data(echodata)
    print(f"[combine_raw] seafloor check in {time.perf_counter() - _t0:.1f}s")
    print("EchoData ready for processing")
    return echodata


def open_and_combine_raw_files(raw_file_paths, netcdf_output_folder, sonar_model="EK60", include_bot=True):
    """Open raw sonar files, convert to netCDF, and combine into a single echodata object.

    Args:
        raw_file_paths: List of string paths or Path objects pointing to raw files.
        netcdf_output_folder: String or Path to folder for intermediate netCDF files.
        sonar_model: Sonar model string for echopype (default: "EK60").
        include_bot: Whether to include .bot files (default: True).

    Returns:
        echodata: Combined EchoData object (or single if only one file).
    """
    if not raw_file_paths:
        raise ValueError("No raw files provided")

    netcdf_output_folder = Path(netcdf_output_folder)
    netcdf_output_folder.mkdir(parents=True, exist_ok=True)

    echodata_nc_list = []
    echodata_raw_list = []
    netcdf_paths = []

    print(f"Processing {len(raw_file_paths)} raw file(s)...")

    for raw_path in raw_file_paths:
        raw_path = Path(raw_path)
        netcdf_path = netcdf_output_folder / f"{raw_path.stem}.nc"
        netcdf_paths.append(netcdf_path)

        ed_elem = ep.open_raw(raw_path, sonar_model=sonar_model, include_bot=include_bot)
        echodata_raw_list.append(ed_elem)

        ed_elem.to_netcdf(save_path=netcdf_path)

        echodata_single = ep.open_converted(netcdf_path)
        echodata_nc_list.append(echodata_single)

    if len(echodata_nc_list) > 1:
        echodata = ep.combine_echodata(echodata_nc_list)
        print(f"Combined {len(echodata_nc_list)} files into one echodata object")
    else:
        echodata = echodata_raw_list[0]
        print(f"Using single file: {netcdf_paths[0]}")

    check_for_seafloor_depth_data(echodata)
    print("Echodata ready for processing")

    return echodata


def check_for_seafloor_depth_data(ed):
    """Check for the presence of bottom detection data and print summary statistics.
    
    Args:
        ed: EchoData object from echopype containing sonar data
        
    Raises:
        ValueError: If no seafloor depth data is found in the Vendor_specific group
    """
    if 'detected_seafloor_depth' in ed['Vendor_specific']:
        seafloor_depth = ed['Vendor_specific']['detected_seafloor_depth']
        seafloor_depth_values = _computed_values(seafloor_depth)
        print("Bottom detection data available!")
        print(f"Min depth: {np.nanmin(seafloor_depth_values):.1f} m")
        print(f"Max depth: {np.nanmax(seafloor_depth_values):.1f} m")
        print(f"Mean depth: {np.nanmean(seafloor_depth_values):.1f} m")
        print(f"Median depth: {np.nanmedian(seafloor_depth_values):.1f} m")
        print(f"Std deviation: {np.nanstd(seafloor_depth_values):.1f} m")
    else:
        print("Warning: No bottom detection data found in the raw file. Sv effects not valid!")
        # raise ValueError("Error: No seafloor depth data found in the raw file. Sv effects not valid!")


#: Variables recording which source files a dataset came from. Under
#: ``data_vars="minimal"`` these would be taken from the first segment alone,
#: understating the provenance of a merged survey, so they are concatenated
#: along their own dimension instead.
_PROVENANCE_CONCAT_VARS = ("source_filenames",)


def _restore_provenance_vars(merged, items, dim):
    """Re-merge provenance variables so every segment's sources are listed.

    Args:
        merged: Dataset produced by the ``data_vars="minimal"`` concat.
        items: The source segments, in order.
        dim: The dimension the segments were concatenated along.

    Returns:
        ``merged`` with each provenance variable replaced by the
        concatenation of that variable across all segments. Variables that
        are absent, scalar, or already carry ``dim`` are left untouched.
        A non-Dataset (e.g. a concatenated DataArray) is returned unchanged.
    """
    if not isinstance(merged, xr.Dataset):
        return merged
    for name in _PROVENANCE_CONCAT_VARS:
        present = [ds for ds in items if name in ds.variables]
        if len(present) < 2:
            continue
        var_dims = present[0][name].dims
        # A scalar has no dimension to grow, and one already spanning the
        # concat dim was handled by xr.concat itself.
        if not var_dims or dim in var_dims:
            continue
        var_dim = var_dims[0]
        # Concatenate the raw values rather than the DataArrays. The
        # provenance dimension usually carries its own index coordinate that
        # restarts at 0 in every segment (echopype writes filenames=[0] per
        # file), so an xr.concat would build a duplicate-valued index and the
        # assignment back into `merged` would fail to align. A fresh
        # positional index over the combined length is the honest result.
        values = np.concatenate(
            [np.atleast_1d(np.asarray(ds[name].values)) for ds in present]
        )
        combined = xr.DataArray(values, dims=(var_dim,))
        was_coord = name in merged.coords
        # Drop the stale index coordinate too, or the assign realigns the
        # new length-N variable against the old length-1 index.
        merged = merged.drop_vars([name, var_dim], errors="ignore")
        merged = (
            merged.assign_coords({name: combined})
            if was_coord
            else merged.assign({name: combined})
        )
        if var_dim in present[0].indexes:
            merged = merged.assign_coords({var_dim: np.arange(values.size)})
    return merged


def concat_datasets(datasets, dim="ping_time", **kwargs):
    """Concatenate a list of xarray Datasets along a dimension.

    Reconsolidation (fan-in) helper for the recipe system's ``collect``
    pattern: a mapped step that produces one Dataset per data segment (e.g.
    per-file MVBS) is gathered into a list, and this joins them back into one
    Dataset along ``dim`` (default ``ping_time``).

    A single Dataset (not a list) is returned unchanged so a ``map_over``
    source that resolves to one item still works (single-item transparency).

    Only variables that already have ``dim`` are concatenated. Variables
    without it (e.g. an Sv dataset's ``frequency_nominal`` on ``(channel,)``,
    or a scalar ``sound_speed``) are taken from the first segment and keep
    their original shape. ``xarray.concat``'s default (``data_vars="all"``)
    would instead broadcast every variable along ``dim``, turning
    ``frequency_nominal`` into ``(ping_time, channel)`` and scalars into
    ``(ping_time,)`` -- which silently corrupts the schema for downstream
    consumers that expect per-channel metadata to stay 1-D.

    Note that per-segment metadata differences are therefore resolved in
    favour of the first segment: if segments were calibrated with different
    environmental parameters, the merged dataset records the first segment's
    ``sound_speed``/``sound_absorption``. The data variables themselves are
    unaffected -- each segment's values were already computed with its own
    parameters.

    The provenance variables in :data:`_PROVENANCE_CONCAT_VARS` are exempt
    from that rule: taking only the first segment's ``source_filenames``
    would make a survey merged from many raw files claim a single source, so
    they are concatenated along their own dimension to list every
    contributing file.

    Args:
        datasets: A list of ``xarray.Dataset`` objects, or a single Dataset.
        dim: Dimension to concatenate along. Defaults to ``"ping_time"``.
        **kwargs: Forwarded to :func:`xarray.concat`, and override the
            defaults set here.

    Returns:
        A single concatenated ``xarray.Dataset``.

    Raises:
        ValueError: If ``datasets`` is an empty collection.
    """
    if isinstance(datasets, xr.Dataset):
        return datasets
    items = [ds for ds in datasets if ds is not None]
    if not items:
        raise ValueError("concat_datasets received no datasets to merge")
    if len(items) == 1:
        return items[0]
    if isinstance(items[0], xr.Dataset):
        concat_kwargs = {
            "data_vars": "minimal",
            "coords": "minimal",
            "compat": "override",
            **kwargs,
        }
    else:
        # xr.concat rejects data_vars/coords on DataArray input ("data_vars is
        # not a valid argument when concatenating DataArray objects"). A
        # DataArray has no non-concat-dim variables to protect anyway, so the
        # plain concat is already the correct behaviour. Collecting per-segment
        # DataArrays (e.g. detect_seafloor's per-file seafloor_depth) is a real
        # fan-in shape, so it must keep working.
        concat_kwargs = dict(kwargs)
    merged = xr.concat(items, dim=dim, **concat_kwargs)
    return _restore_provenance_vars(merged, items, dim)

