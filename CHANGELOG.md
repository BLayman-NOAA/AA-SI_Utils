# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `aa_si_utils.seabed`, a phase-aware dynamic-programming seabed detector
  implementing the Mode 1 core path of the v4 design (Stages 0, 2b, 3, 4, 5,
  5c, 6 and 8). `detect_seafloor_phase` is the recipe entry point and returns
  the seabed echo's leading edge as the same 1-D `(ping_time,)` line in
  metres the other seafloor ops return, surface-referenced when `ds_Sv`
  carries `depth`; `detect_seabed` returns every intermediate (seed and alias
  masks, features f1 to f6, score, candidates, raw pick, leading edge,
  Ona-Mitson integration line, confidence, margin, flags) as a Dataset.
  Windows are given in metres, pulse lengths and seconds and converted from
  the file's own `tau_effective`, `echo_range`, ping times and position.
  Two departures from the design, both forced by HB1603 38 kHz data at
  1750 m: phase activity is the variance of the physical angles within the
  window rather than their mean square, because the seabed centroid sat
  2 degrees off axis and the mean square could not separate it from water;
  and the f2/f5 references are the seed medians, as f1's already is. The
  Blackwell alias detector is implemented but off by default, since on that
  file (8 s pinging, no aliasing possible) its mask covered the true seabed.
  Without split-beam angles the detector runs amplitude-only and warns.
  `aa_si_utils.seabed.prior.estimate_prior` implements Mode 0 (0a regular
  and 0b adaptive slab sampling): a few dozen slabs of pings, averaged and
  binned to two pulse lengths, scored with the reduced score and joined by
  the same line search, with bisection of gaps whose picks disagree by more
  than the break-point. Its line and uncertainty become the per-ping search
  window, so `detect_seafloor_phase` with no window, prior or line file is
  self-contained (`prior="auto"`, the default; `prior="none"` keeps the
  fixed window). Geometry now reads only the first sample of every ping and
  the first ping's row, so a whole-survey dataset costs no more than one
  file to set up.
  A pick whose mean Sv over the pulse length below it is under
  `min_seabed_sv_db` (default -50 dB) is rejected, in the prior's slabs and
  in the final line: the scores are relative to each ping, so on the first
  HB1603 survey run files whose seabed lies beyond the 1800 m crop were
  given a line on a -57 dB scattering layer at 16 to 23 m, a third of the
  files finished. Real seabed echoes there run -2 to -40 dB. Pings whose
  search window is empty (a prior outside the recorded range, or a ping
  with no valid depth, as on a file with 38 m heave spikes) now get no
  seabed instead of stopping the run; a file with no window anywhere
  returns no line.
  `line="integration"` returns the slope-corrected Ona-Mitson backstep
  instead of the leading edge; on the HB1603 canyon walls a 10 m buffer
  above the leading edge still left up-slope echo in the 18 kHz channel.
  `mode=2` adds the Stage 3b shape features (below-above Sv contrast,
  below-region Sv std, below-region phase activity) over the preset's shape
  window; on HB1603 it moved the edge 3 m further up the rise and cut the
  flagged pings from 31 % to 27 %, on HB2407 it changed nothing.
- `scripts/validate_seabed_detection.py` and `aa_si_utils.seabed.validation`:
  run the detector on one raw file against its .bot pick or an Echoview line
  and write the design's agreement metrics, an echogram overlay and the
  diagnostics. The references are not ground truth: on HB1603 the .bot line
  sits about 8 m below the Sv rise, on HB2407 the Echoview line 2 m above
  it, and the overlay is what decides which side is right.
- `aa_si_utils.seabed.compare` and `scripts/compare_seabed_methods.py`: an
  inter-method comparison of seafloor detectors (the .bot pick, Echoview
  lines, a max-Sv baseline, echopype basic and Blackwell, the experimental
  HDBSCAN detector, and the phase detector in Mode 1, Mode 2, amplitude-only,
  integration-line and secondary-channel variants) on six two-hour datasets
  from HB1603 and HB2407. Reports detector time, reference-free sanity checks
  (contrast across the line, echo intensity below it, lines left in water,
  ping-to-ping jumps) and pairwise plus consensus agreement, with one line
  per method family voting so the phase variants cannot outvote the rest.
  `scripts/report_seabed_comparison.py` turns the result folders into a
  markdown report with overlays.
- `slow` pytest marker for `tests/test_seabed_real_files.py`, which skips
  when the HB1603 and HB2407 example files are absent.
- `scipy` is now a dependency (`ndimage` filters and connected components).
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
