# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Utility functions for active acoustics data processing.

Provides helpers for depth indexing, masking (seafloor, surface, frequency),
haversine distance calculation, dive-profile alignment, and raw-file I/O
using the echopype library.
"""

import colorsys
import math
from pathlib import Path

import echopype as ep
import numpy as np
import pandas as pd
import xarray as xr


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
    depth_diff = np.abs(echo_range_1d - target_depth)

    # Find the index of the minimum difference
    range_sample_index = int(np.argmin(_computed_values(depth_diff)))

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
                      range_var: str = "echo_range") -> xr.Dataset:
    """Mask Sv values in bins where NaN fraction meets or exceeds a threshold.

    Partitions the data into (time, range) bins and sets entire bins to NaN
    when the proportion of missing values within that bin is at or above
    *nan_threshold*.

    Args:
        ds_Sv: Sv dataset to process.
        range_bin: Range bin size as a string (e.g. ``"2m"``).
        ping_time_bin: Time bin size as a pandas-compatible offset string
            (e.g. ``"10s"``).
        nan_threshold: Fraction of NaN values (0–1) at or above which a bin
            is masked.
        range_var: Name of the range coordinate variable.

    Returns:
        xr.Dataset: Copy of *ds_Sv* with sparse bins set to NaN.
    """
    
    # Parse range_bin and create bin edges
    range_bin_val = float(range_bin.rstrip('m'))
    range_max = float(ds_Sv[range_var].max(skipna=True).values)
    range_edges = np.arange(0, range_max + range_bin_val, range_bin_val)
    
    # Assign each point to a range bin
    range_bin_indices = np.digitize(
        ds_Sv[range_var].isel(ping_time=0, channel=0).values, 
        range_edges
    ) - 1
    
    # Assign each point to a time bin
    time_bin_map = np.zeros(len(ds_Sv['ping_time']), dtype=int)
    for bin_idx, (_, time_group) in enumerate(ds_Sv.resample(ping_time=ping_time_bin)):
        time_mask = np.isin(ds_Sv['ping_time'].values, time_group.indexes['ping_time'].values)
        time_bin_map[time_mask] = bin_idx
    
    # Create unique bin ID for each (time_bin, range_bin) combination
    time_grid = np.repeat(time_bin_map[:, np.newaxis], len(range_bin_indices), axis=1)
    range_grid = np.repeat(range_bin_indices[np.newaxis, :], len(time_bin_map), axis=0)
    combined_bin_id = time_grid * len(range_edges) + range_grid
    
    # Work with copy of Sv values
    sv_values = ds_Sv["Sv"].values.copy()
    
    # Process each channel
    for ch in range(sv_values.shape[0]):
        # Flatten arrays for vectorized operations
        sv_flat = sv_values[ch].ravel()
        bin_id_flat = combined_bin_id.ravel()
        
        # Calculate NaN fraction per bin using bincount
        is_nan = np.isnan(sv_flat).astype(int)
        bin_counts = np.bincount(bin_id_flat)
        nan_counts = np.bincount(bin_id_flat, weights=is_nan)
        nan_fractions = np.divide(nan_counts, bin_counts, 
                                 out=np.zeros_like(nan_counts, dtype=float), 
                                 where=bin_counts > 0)
        
        # Mask bins exceeding threshold
        bins_to_mask = nan_fractions >= nan_threshold
        mask = bins_to_mask[bin_id_flat].reshape(sv_values[ch].shape)
        sv_values[ch][mask] = np.nan
    
    # Return modified dataset
    ds_result = ds_Sv.copy(deep=True)
    ds_result["Sv"].values = sv_values
    return ds_result


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
            ``lwr_CI_99``, ``upr_CI_99``.
        dive_profile_name (str): Base name for the added variables
            (default: ``'dive_profile'``).

    Returns:
        xr.Dataset: Dataset with added dive-profile variables sharing the
        existing ``ping_time`` dimension. Values are NaN where no dive
        data exists.
    """
    # Validate file exists
    csv_path = Path(csv_filepath)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dive profile CSV not found: {csv_filepath}")
    
    print(f"Reading dive profile: {csv_path.name}")
    
    # Read CSV
    dive_df = pd.read_csv(csv_path)
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
        'source_file': csv_path.name
    })
    
    ds_MVBS[f'{dive_profile_name}_depth'].attrs.update({
        'long_name': 'Sperm whale dive depth (measured)',
        'units': 'meters',
        'source_file': csv_path.name
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


def initial_setup_and_validation(raw_input_folder, netcdf_output_folder_string, sv_output_folder_string,
                                  output_logs_folder_string, raw_file_names=None, clear_previous_json_logs=True):
    """Set up output folders, resolve raw file paths, and optionally clear previous logs.

    Args:
        raw_input_folder: Path to the folder containing raw input files.
        netcdf_output_folder_string: String path for netCDF output folder.
        sv_output_folder_string: String path for Sv output folder.
        output_logs_folder_string: String path for output logs folder.
        raw_file_names: Optional list of specific raw file name strings to process.
            If empty or None, all .raw files in raw_input_folder are used.
        clear_previous_json_logs: If True, delete existing calibration_flags.json (default: True).

    Returns:
        list[str]: List of string paths to raw files found in raw_input_folder.
    """
    raw_input_folder = Path(raw_input_folder)
    netcdf_output_folder = Path(netcdf_output_folder_string)
    sv_output_folder = Path(sv_output_folder_string)
    output_logs_folder = Path(output_logs_folder_string)

    # Get list of raw files to process
    if raw_file_names is not None and len(raw_file_names) > 0:
        raw_file_paths = [raw_input_folder / filename for filename in raw_file_names]
    else:
        raw_file_paths = sorted(raw_input_folder.glob("*.raw"))

    if not raw_file_paths:
        raise FileNotFoundError("No raw files found to process")

    print(f"Found {len(raw_file_paths)} raw file(s) to process:")
    for path in raw_file_paths:
        print(f"  - {path.name}")

    # Ensure output folders exist
    output_folders = [
        netcdf_output_folder,
        sv_output_folder,
        output_logs_folder,
    ]
    for folder in output_folders:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Created missing output folder: {folder}")

    if clear_previous_json_logs:
        flags_file = Path(output_logs_folder) / "calibration_flags.json"
        if flags_file.exists():
            flags_file.unlink()

    return [str(p) for p in raw_file_paths]


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
        print("Bottom detection data available!")
        print(f"Min depth: {seafloor_depth.min().values:.1f} m")
        print(f"Max depth: {seafloor_depth.max().values:.1f} m")
        print(f"Mean depth: {seafloor_depth.mean().values:.1f} m")
        print(f"Median depth: {seafloor_depth.median().values:.1f} m")
        print(f"Std deviation: {seafloor_depth.std().values:.1f} m")
    else:
        print("Error: No bottom detection data found in the raw file. Sv effects not valid!")
        raise ValueError("Error: No seafloor depth data found in the raw file. Sv effects not valid!")
    
