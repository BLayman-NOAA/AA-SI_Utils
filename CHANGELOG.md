# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Optional Google Cloud Storage backing for `exe_temp` intermediate stores:
  when the recipe executor's scratch dir is a `gs://` URL, `read_raw_files_to_stores`
  writes per-file zarr stores to the bucket and `combine_raw_stores` reads them
  back lazily. Requires the `gcs` extra; credentials via Application Default
  Credentials. New `aa_si_utils._storage` helpers (duck-typed, no hard
  dependency on the recipe manager).
- NetCDF intermediates now raise a clear error when the scratch dir is remote
  (HDF5 needs seekable writes and cannot be written to object storage — use
  `intermediate_format="zarr"` or a local `--temp-dir`).
- Initial project structure from NOAA Fisheries AA-SI Python template
- Utility functions for depth analysis, masking, distance calculations
- Dive profile integration with MVBS datasets
- Seafloor and surface mask tools for Sv data

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Nothing yet

### Security
- Nothing yet

## [0.1.0] - YYYY-MM-DD

### Added
- Initial release
- Basic package structure with src layout
- Development tooling (pytest, black, pylint, pre-commit)

<!--
=============================================================================
CHANGELOG GUIDELINES
=============================================================================

When adding entries, use the following categories:
- Added: for new features
- Changed: for changes in existing functionality
- Deprecated: for soon-to-be removed features
- Removed: for now removed features
- Fixed: for any bug fixes
- Security: in case of vulnerabilities

Each release should have a version number and date in the format:
## [X.Y.Z] - YYYY-MM-DD

Link definitions should be added at the bottom (optional)
