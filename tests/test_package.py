# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""
Sample test file demonstrating pytest structure.

Run tests with: pytest
Run with coverage: pytest --cov=aa_si_utils
"""

import aa_si_utils


def test_version_exists():
    """Test that the package has a version string."""
    assert hasattr(aa_si_utils, "__version__")
    assert isinstance(aa_si_utils.__version__, str)


def test_version_format():
    """Test that version follows semantic versioning format (X.Y.Z)."""
    version = aa_si_utils.__version__
    # Skip detailed check for dev versions (package not installed)
    if version == "0.0.0.dev":
        return
    parts = version.split(".")
    assert len(parts) >= 2, "Version should have at least major.minor"
    # Check that parts are numeric (allowing for pre-release suffixes)
    assert parts[0].isdigit(), "Major version should be numeric"
    assert parts[1].isdigit(), "Minor version should be numeric"


# =============================================================================
# TODO: Add your own tests below
# =============================================================================
#
# Example test structure:
#
# def test_my_function():
#     """Test description."""
#     from aa_si_utils.utils import my_function
#     result = my_function(input_value)
#     assert result == expected_value
#
# For more pytest features, see: https://docs.pytest.org/
# =============================================================================
