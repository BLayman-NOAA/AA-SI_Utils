# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `build_range_grid`, producing the uniformly spaced range grid that
  `echopype.commongrid.resample_to_geometry` takes as its `target_grid`. That
  function wants a DataArray rather than a number, which a recipe params block
  cannot express, so building the grid as its own step lets the resampling step
  receive it as a wired input. `spacing_m` sets the axis length and
  `max_range_m` caps how deep it reaches. On HB1603 a 5 m grid capped at 2100 m
  takes a ping from 15207 range samples to 421, which is what brings clustering
  over the full survey from 23.5 million points down to 652 thousand.
- `read_seafloor_line_evl` accepts a folder of Echoview exports (local or
  `gs://`) as well as a single `.evl` file, and takes `file_time_start` /
  `file_time_end` — the same window that selects the raw files. The line files
  spanning that window are read and concatenated into one line before
  interpolation, so a survey exported as part-day line files needs no manual
  picking. A single `.evl` path is read whole regardless of the window: naming
  one file is an override, not a candidate set. `source_file` lists every file
  used and the printed summary names them.
- `filter_evl_paths_by_file_time` and `parse_evl_span_from_filename` in
  `data_retrieval`, beside the raw-file equivalents they reuse. A line file's
  span runs from its own `d{YYYYMMDD}_t{HHMMSS}` start stamp to the **next**
  file's, not to the end stamp in its own name — Echoview's end stamp
  under-reports the line's real last point by about one raw file's duration, so
  trusting it would drop the export straddling the window start and leave those
  pings with a NaN seafloor, which `create_seafloor_mask` masks away entirely.
  The last file, which no later file can bound, falls back to its own end stamp.
  Selection is name-based, so no line data is read to filter.
- `read_raw_files_to_stores` reports progress per file: `[i/n]` with the raw
  file's name and size, available RAM against echopype's swap threshold, parse
  time, whether `backscatter_r` came back dask-backed (echopype decided to swap
  it to disk, a per-file decision driven by live memory pressure), the store
  being written, its size, and free disk before and after. Not gated behind a
  flag: this is the record of how far a conversion got, and it is only useful
  if it is already there when a run fails.

### Fixed
- Removing an existing intermediate store could fail permanently with
  `[Errno 13] Permission denied` naming a directory inside it. The `rmtree`
  error handler assigned `stat.S_IWRITE` to the failing path, which is `0o200`
  exactly: on POSIX that strips read and execute from a directory, so the retry
  it performs immediately afterwards fails with EACCES on that same path, and
  the original error is lost. `_remove_existing_store` now grants read and write
  (plus execute for directories only) by OR-ing onto the existing mode, and
  fixes up the parent as well, since unlinking an entry needs write and execute
  there rather than on the entry itself.
- `_write_store_with_retry` retried any `PermissionError` three times over three
  seconds. Only the Windows rename race it was written for is transient; a POSIX
  EACCES will be identical on the next attempt, so retrying only delayed the
  error and re-ran a removal that could not succeed. Non-Windows permission
  errors now raise on the first attempt, with the store path, whether a partial
  store is present, and free disk space attached to the message.
- The local zarr write did not pass `overwrite=True` while the remote one did.
  If a store was only partly removed, echopype's `to_file` logs "already exists,
  will not overwrite" and returns **without writing**, so a truncated store was
  reported as a successful conversion and failed later, in `combine_raw`, as a
  corrupt store. The two branches are now symmetric.
- Filename-time filtering pulled in a stale raw file from before a gap between
  survey legs. Inferring a file's end from the next file's start stamp assumes
  recording ran continuously, so the last file before a gap looked like it
  recorded for the whole gap and was kept as if it straddled the window start.
  `filter_paths_by_file_time` now reads the real last ping from the one file
  whose verdict depends on that inference, via the new `raw_file_times` module.
  This also settles the chronologically last file, whose end the names cannot
  bound at all, so a long final file recording into the window is no longer
  dropped. At most one file per call is opened and only its datagram headers
  are read; pass `verify_boundary=False` to keep the filter name-only.
  `query_ncei_data` is unaffected: it filters catalog metadata before anything
  is downloaded, so `initial_setup_and_validation` applies the check afterward,
  once the files are local.
- Filename-time filtering (`file_time_start` / `file_time_end`) missed raw
  files that start before the window but record into it, because only each
  file's own name stamp (its recording *start*) was compared against the
  window. `query_ncei_data` and `filter_paths_by_file_time` now use overlap
  semantics: a file's end time is inferred from the next file's start stamp
  (within the same dataset for NCEI queries) and the file is kept when the
  resulting span overlaps the window. The auto-derived server-side
  `collection_start` is widened by one day so a straddling file from the
  previous day is fetched as a candidate. The chronologically last file has no
  inferred end and still uses the own-stamp rule.
- Remote (`gs://`) zarr intermediates were silently written to a local relative
  directory instead of the bucket: echopype 0.11.1's `EchoData.to_zarr` passes
  the protocol-stripped fsspec mapper root to `xarray.to_zarr` with no
  filesystem, so a `gs://` save path became a local write. `read_raw_files_to_stores`
  now streams the EchoData's datatree straight to the bucket (no local copy — the
  raw file remains the only thing on local disk), with a local-write-then-upload
  fallback for EchoData stand-ins that lack a datatree.

### Added
- `read_seafloor_line_evl`: reads an Echoview `.evl` seabed line and returns it
  as a 1-D `(ping_time,)` DataArray in metres on `ds_Sv`'s exact `ping_time`
  coordinate — a drop-in replacement for `detect_seafloor` that feeds
  `create_seafloor_mask` unchanged, for when a hand-verified Echoview line beats
  running detection. Local paths and remote (`gs://`) URLs are both supported;
  line files are small, so a remote one is read in place with no local copy.
  - Alignment is linear in time, with two independent limits. `max_gap_s` caps
    the widest hole in the line that will be interpolated across (a ping landing
    exactly on a line point always keeps that point's depth); `edge_extend_s`
    caps how far past the line's first/last point its depth is held, and
    **defaults to `0.0`** — no extrapolation, so pings outside the line's span
    are NaN rather than silently inheriting a constant seafloor. Pass `None` to
    either for "no limit".
  - Because `create_seafloor_mask` rejects *every* sample of a ping whose
    seafloor is NaN, ping coverage is printed on every call and `min_coverage`
    turns a shortfall into an error.
  - `vertical_reference` converts a surface- or transducer-referenced line to
    whichever reference `ds_Sv` carries (`depth` after `add_depth`, else
    `echo_range`); `depth_offset_m` absorbs an Echoview transducer draft that
    differs from the one baked into `ds_Sv['depth']`.
  - The `.evl` parser is local (no new dependency). echoregions was evaluated
    first and rejected on two counts: its released 0.2.3 pins `zarr<3` and
    `scipy<1.15.2`, which conflicts with `echopype>=0.11` (`zarr>=3`), and its
    `parse_evl` fails outright under pandas 3.x by assigning datetimes into a
    string column — on `main` as well as in the release. `_parse_evl` is kept
    as a seam so parsing can be delegated upstream if a compatible release lands.
- Optional Google Cloud Storage backing for `exe_temp` intermediate stores:
  when the recipe executor's scratch dir is a `gs://` URL, `read_raw_files_to_stores`
  writes per-file zarr stores to the bucket and `combine_raw_stores` reads them
  back lazily. Requires the `gcs` extra; credentials via Application Default
  Credentials. New `aa_si_utils._storage` helpers (duck-typed, no hard
  dependency on the recipe manager).
- NetCDF intermediates now raise a clear error when the scratch dir is remote
  (HDF5 needs seekable writes and cannot be written to object storage — use
  `intermediate_format="zarr"` or a local `--temp-dir`).
- Remote (`gs://`) raw-file **inputs**: `initial_setup_and_validation` lists a
  remote `raw_input_folder` without downloading, and `read_raw_files_to_stores`
  downloads each remote `.raw` (plus its `.bot` companion) to a private local
  scratch dir, converts it, and deletes the local copy before the next file —
  local disk holds ~1 raw file at a time. `add_dive_profile_to_dataset` reads a
  `gs://` line CSV in place via pandas. New `_storage` helpers `basename`,
  `glob_url`, and the `localized_file` context manager; input storage options
  come from the execution context (`_execution_storage_options`).
- Optional filename-datetime filtering: `initial_setup_and_validation` gained
  `file_time_start` / `file_time_end` (inclusive ISO/`datetime` bounds matched
  against the `D{YYYYMMDD}-T{HHMMSS}` file-name stamp). The NCEI filter logic is
  now the shared, public `data_retrieval.filter_paths_by_file_time` /
  `parse_datetime_from_filename` (works on local paths and `gs://` URLs).
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
