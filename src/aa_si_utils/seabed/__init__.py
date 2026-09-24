# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Phase-aware dynamic-programming seabed detection.

``detect_seafloor_phase`` is the recipe op; ``detect_seabed`` returns the
full diagnostics Dataset it is built from.
"""

from .prior import estimate_prior
from .pipeline import (
    FLAG_INTERPOLATED,
    FLAG_LOW_CONFIDENCE,
    FLAG_LOW_MARGIN,
    FLAG_SKIPPED,
    FLAG_WALKBACK_BOUND,
    detect_seabed,
    detect_seafloor_phase,
)

__all__ = [
    "FLAG_INTERPOLATED",
    "FLAG_LOW_CONFIDENCE",
    "FLAG_LOW_MARGIN",
    "FLAG_SKIPPED",
    "FLAG_WALKBACK_BOUND",
    "detect_seabed",
    "detect_seafloor_phase",
    "estimate_prior",
]
