# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""aa_si_utils - Utility functions for NOAA Fisheries AA-SI active acoustics data processing."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aa-si-utils")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source without install)
    __version__ = "0.0.0.dev"

from .utils import (
    add_dive_profile_to_dataset,
    apply_mask_to_sv,
    apply_mask_to_sv_datasets,
    apply_seafloor_and_surface_masks,
    check_for_seafloor_depth_data,
    combine_masks,
    combine_raw_stores,
    create_data_mask,
    create_frequency_mask,
    create_seafloor_mask,
    create_surface_mask,
    createSvMask,
    detect_seafloor,
    find_best_seafloor_detection,
    find_data_depth_range,
    generate_colors,
    get_closest_index_for_depth,
    haversine_distance,
    initial_setup_and_validation,
    log_mask_stats,
    log_seafloor_detection_stats,
    mask_frequency_channels,
    mask_sparse_bins,
    read_raw_files_to_stores,
    read_seafloor_line_evl,
    remove_seafloor_from_mask,
    remove_surface_from_mask,
)

from .data_retrieval import (
    query_ncei_data,
    download_ncei_data,
)

__all__ = [
    "__version__",
    "add_dive_profile_to_dataset",
    "apply_mask_to_sv",
    "apply_mask_to_sv_datasets",
    "apply_seafloor_and_surface_masks",
    "check_for_seafloor_depth_data",
    "combine_masks",
    "create_data_mask",
    "create_frequency_mask",
    "create_seafloor_mask",
    "create_surface_mask",
    "createSvMask",
    "detect_seafloor",
    "download_ncei_data",
    "find_best_seafloor_detection",
    "find_data_depth_range",
    "generate_colors",
    "get_closest_index_for_depth",
    "haversine_distance",
    "initial_setup_and_validation",
    "log_mask_stats",
    "log_seafloor_detection_stats",
    "mask_frequency_channels",
    "mask_sparse_bins",
    "query_ncei_data",
    "read_seafloor_line_evl",
    "remove_seafloor_from_mask",
    "remove_surface_from_mask",
]
