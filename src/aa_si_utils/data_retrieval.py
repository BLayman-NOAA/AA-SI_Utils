# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Functions for querying and downloading NOAA NCEI water column sonar data."""

import re
import tarfile
from datetime import datetime
from pathlib import Path

import requests


# Constants

_WCSD_BASE_URL = (
    "https://gis.ngdc.noaa.gov/arcgis/rest/services/wcsd_files/MapServer/1/query"
)

_S3_BUCKET_PREFIX = "s3://noaa-wcsd-pds/"
_HTTPS_BUCKET_PREFIX = "https://noaa-wcsd-pds.s3.amazonaws.com/"

# Maximum records per page returned by the ArcGIS MapServer
_PAGE_SIZE = 1000


# Private helpers

def _parse_datetime_from_filename(filename):
    """Extract a datetime from a WCSD filename.

    Matches the ``D{YYYYMMDD}-T{HHMMSS}`` pattern anywhere in the filename,
    e.g. ``D20160725-T205832.tar`` or ``DY2207_EK80-D20220604-T074711.raw``.

    Returns ``None`` when the pattern is not found.
    """
    match = re.search(r"D(\d{8})-T(\d{6})", filename)
    if not match:
        return None
    date_part, time_part = match.groups()
    return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")


def _validate_string_value(value, param_name):
    """Reject values that could break the ArcGIS REST SQL WHERE clause."""
    dangerous = {"'", ";", "--", "/*", "*/"}
    text = str(value)
    for ch in dangerous:
        if ch in text:
            raise ValueError(
                f"Parameter '{param_name}' contains forbidden character sequence "
                f"'{ch}'. Please remove it to avoid query injection."
            )


def _build_where_clause(
    dataset_name=None,
    instrument_type=None,
    frequencies=None,
    calibration_state=None,
    collection_start=None,
    collection_end=None,
    filters=None,
):
    """Build a SQL WHERE clause string from the supplied parameters."""
    clauses = ["CLOUD_PATH IS NOT NULL"]

    if dataset_name is not None:
        _validate_string_value(dataset_name, "dataset_name")
        clauses.append(f"DATASET_NAME = '{dataset_name}'")

    if instrument_type is not None:
        _validate_string_value(instrument_type, "instrument_type")
        clauses.append(
            f"UPPER(INSTRUMENT_NAME) LIKE '%{instrument_type.upper()}%'"
        )

    if frequencies:
        for freq in frequencies:
            _validate_string_value(freq, "frequencies")
        freq_conditions = [
            f"UPPER(FREQUENCY) LIKE '%{freq.upper()}%'" for freq in frequencies
        ]
        clauses.append(f"({' OR '.join(freq_conditions)})")

    if calibration_state is not None:
        if isinstance(calibration_state, (list, tuple)):
            cal_values = ", ".join(str(int(v)) for v in calibration_state)
        else:
            cal_values = str(int(calibration_state))
        clauses.append(f"CAL_STATE_VALUE IN ({cal_values})")

    if collection_start and collection_end:
        _validate_string_value(collection_start, "collection_start")
        _validate_string_value(collection_end, "collection_end")
        clauses.append(
            f"COLLECTION_DATE BETWEEN DATE '{collection_start}' "
            f"AND DATE '{collection_end}'"
        )
    elif collection_start:
        _validate_string_value(collection_start, "collection_start")
        clauses.append(f"COLLECTION_DATE >= DATE '{collection_start}'")
    elif collection_end:
        _validate_string_value(collection_end, "collection_end")
        clauses.append(f"COLLECTION_DATE <= DATE '{collection_end}'")

    if filters:
        for key, value in filters.items():
            _validate_string_value(key, f"filters key '{key}'")
            _validate_string_value(value, f"filters['{key}']")
            if isinstance(value, str):
                clauses.append(f"{key} = '{value}'")
            else:
                clauses.append(f"{key} = {value}")

    return " AND ".join(clauses)


def _add_spatial_params(params, bbox=None, center_point=None, radius_nm=None):
    """Mutate *params* dict to include ArcGIS spatial query parameters."""
    if bbox is not None:
        west, south, east, north = bbox
        params["geometry"] = f"{west},{south},{east},{north}"
        params["geometryType"] = "esriGeometryEnvelope"
        params["spatialRel"] = "esriSpatialRelIntersects"
        params["inSR"] = "4326"
        print(f"Spatial filter: bbox ({west}, {south}) to ({east}, {north})")
    elif center_point is not None and radius_nm is not None:
        lon, lat = center_point
        params["geometry"] = f"{lon},{lat}"
        params["geometryType"] = "esriGeometryPoint"
        params["spatialRel"] = "esriSpatialRelIntersects"
        params["distance"] = radius_nm
        params["units"] = "esriSRUnit_NauticalMile"
        params["inSR"] = "4326"
        print(f"Spatial filter: {radius_nm} nm radius around ({lon}, {lat})")


def _fetch_all_pages(params):
    """Paginate through the ArcGIS MapServer and return all attribute dicts."""
    all_items = []
    offset = 0

    while True:
        params["resultOffset"] = offset
        params["resultRecordCount"] = _PAGE_SIZE

        response = requests.get(_WCSD_BASE_URL, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            print("API Error:", data["error"])
            return []

        features = data.get("features", [])
        all_items.extend(f["attributes"] for f in features)

        if not data.get("exceededTransferLimit"):
            break

        offset += _PAGE_SIZE
        print(f"  Fetching next page (offset {offset})...")

    return all_items


def _s3_to_https(cloud_path):
    """Convert an ``s3://noaa-wcsd-pds/...`` path to a public HTTPS URL."""
    if cloud_path.startswith(_S3_BUCKET_PREFIX):
        return _HTTPS_BUCKET_PREFIX + cloud_path[len(_S3_BUCKET_PREFIX):]
    return cloud_path


def _extract_tar(tar_path, output_dir):
    """Safely extract a tar archive and return the list of extracted paths.

    Extracts into a subfolder named after the archive (minus extension).
    Deletes the ``.tar`` file after successful extraction.
    """
    tar_path = Path(tar_path)
    extract_dir = output_dir / tar_path.stem

    extracted_paths = []
    with tarfile.open(tar_path) as tf:
        # Security: reject paths that escape the target directory
        for member in tf.getmembers():
            member_path = (extract_dir / member.name).resolve()
            if not str(member_path).startswith(str(extract_dir.resolve())):
                raise RuntimeError(
                    f"Tar member '{member.name}' would extract outside the "
                    f"target directory. Aborting for safety."
                )
        # Use data filter when available (Python 3.12+) for additional safety
        try:
            tf.extractall(path=extract_dir, filter="data")
        except TypeError:
            tf.extractall(path=extract_dir)
        extracted_paths = [extract_dir / m.name for m in tf.getmembers() if m.isfile()]

    tar_path.unlink()
    print(f"  Extracted {len(extracted_paths)} file(s) to {extract_dir.name}/")
    return extracted_paths


def _probe_url_extension(base_url, extensions=(".raw", ".tar")):
    """Try HEAD requests to find which file extension exists on the server."""
    for ext in extensions:
        try:
            resp = requests.head(base_url + ext, timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                return ext
        except requests.exceptions.RequestException:
            continue
    return None


def _download_companions(primary_url, primary_dest, output_dir, extensions, overwrite):
    """Try to download companion files that share the same base name.

    For a primary URL like ``.../HB1603-D20160725-T205832.raw``, this will
    attempt to download ``.../HB1603-D20160725-T205832.bot``, etc.

    Returns a list of successfully downloaded ``Path`` objects.  Missing files
    (HTTP 404) are silently skipped.
    """
    downloaded = []
    stem = primary_dest.stem  # e.g. "HB1603-D20160725-T205832"
    base_url = primary_url.rsplit(".", 1)[0]  # strip the extension from URL

    for ext in extensions:
        ext = ext if ext.startswith(".") else f".{ext}"
        comp_url = base_url + ext
        comp_dest = output_dir / (stem + ext)

        if not overwrite and comp_dest.exists():
            downloaded.append(comp_dest)
            continue

        try:
            resp = requests.get(comp_url, stream=True, timeout=120)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()

            with open(comp_dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    fh.write(chunk)

            size_mb = comp_dest.stat().st_size / (1024 * 1024)
            print(f"           Companion ({size_mb:.1f} MB): {comp_dest.name}")
            downloaded.append(comp_dest)
        except requests.exceptions.RequestException:
            # Companion not available, not an error
            pass

    return downloaded


# Public API

def query_ncei_data(
    dataset_name=None,
    instrument_type=None,
    frequencies=None,
    calibration_state=None,
    collection_start=None,
    collection_end=None,
    file_time_start=None,
    file_time_end=None,
    max_files=None,
    bbox=None,
    center_point=None,
    radius_nm=None,
    filters=None,
):
    """Query the NOAA NCEI Water Column Sonar Data (WCSD) archive.

    Searches the WCSD ArcGIS MapServer for sonar data files matching the given
    criteria.  All parameters are optional; omit a parameter to leave that
    field unconstrained.

    Args:
        dataset_name (str | None): Dataset identifier, e.g. ``'DY2207_EK80'``.
        instrument_type (str | None): Instrument name substring, e.g. ``'EK80'``.
        frequencies (list[str] | None): Frequency strings to match,
            e.g. ``['18WKHZ', '38WKHZ']``.  Results must contain *any* of the
            listed frequencies (OR logic).
        calibration_state (int | list[int] | None): Calibration state value(s),
            e.g. ``4`` for "Calibrated w/ calibration data".
        collection_start (str | None): Inclusive start date as ``'YYYY-MM-DD'``.
            Filters the server-side ``COLLECTION_DATE`` field (day-level
            granularity).
        collection_end (str | None): Inclusive end date as ``'YYYY-MM-DD'``.
        file_time_start (str | datetime | None): Inclusive start datetime for
            fine-grained filtering.  Parsed from each filename's embedded
            timestamp (``-DYYYYMMDD-THHMMSS``).  Accepts an ISO-format string
            or a ``datetime`` object.
        file_time_end (str | datetime | None): Inclusive end datetime for
            fine-grained filtering (same format as *file_time_start*).
        max_files (int | None): Maximum number of results to return.  Applied
            after all other filters; results are sorted chronologically by
            filename timestamp.
        bbox (tuple | None): Bounding box ``(west, south, east, north)`` in
            WGS 84 decimal degrees.
        center_point (tuple | None): ``(longitude, latitude)`` for a
            radius-based spatial query.
        radius_nm (float | None): Search radius in nautical miles (requires
            *center_point*).
        filters (dict | None): Additional field-level filters as
            ``{FIELD_NAME: value}`` pairs.  Each becomes an exact-match clause
            in the query.  Use this for any WCSD attribute not covered by the
            explicit parameters above (e.g.
            ``{'CRUISE_NAME': 'DY2207', 'PLATFORM_NAME': 'Oscar Dyson (DY)'}``).

    Returns:
        list[dict]: A list of result dictionaries, one per file.  Each dict
        contains all WCSD attribute fields plus a ``FILE_DATETIME``
        (``datetime`` or ``None``) parsed from the filename.

    Example::

        results = query_ncei_data(
            dataset_name="DY2207_EK80",
            frequencies=["38WKHZ", "120WKHZ"],
            collection_start="2022-06-04",
            collection_end="2022-06-04",
            file_time_start="2022-06-04T08:00:00",
            file_time_end="2022-06-04T10:00:00",
        )
    """
    # Auto-derive collection date range from file_time if not given
    if file_time_start is not None and collection_start is None:
        ft_start = (
            datetime.fromisoformat(file_time_start)
            if isinstance(file_time_start, str)
            else file_time_start
        )
        collection_start = ft_start.strftime("%Y-%m-%d")
    if file_time_end is not None and collection_end is None:
        ft_end = (
            datetime.fromisoformat(file_time_end)
            if isinstance(file_time_end, str)
            else file_time_end
        )
        collection_end = ft_end.strftime("%Y-%m-%d")

    # Build WHERE clause
    where_clause = _build_where_clause(
        dataset_name=dataset_name,
        instrument_type=instrument_type,
        frequencies=frequencies,
        calibration_state=calibration_state,
        collection_start=collection_start,
        collection_end=collection_end,
        filters=filters,
    )

    params = {
        "f": "json",
        "where": where_clause,
        "outFields": "*",
        "returnGeometry": "false",
        "orderByFields": "FILE_NAME",
    }

    _add_spatial_params(params, bbox=bbox, center_point=center_point, radius_nm=radius_nm)

    print(f"Querying NCEI WCSD archive...")

    # Paginated fetch
    try:
        items = _fetch_all_pages(params)
    except requests.exceptions.RequestException as exc:
        print(f"Network error: {exc}")
        return []

    # Enrich with FILE_DATETIME
    for item in items:
        item["FILE_DATETIME"] = _parse_datetime_from_filename(
            item.get("FILE_NAME", "")
        )

    # Fine-grained filename-time filtering
    if file_time_start is not None or file_time_end is not None:
        if isinstance(file_time_start, str):
            file_time_start = datetime.fromisoformat(file_time_start)
        if isinstance(file_time_end, str):
            file_time_end = datetime.fromisoformat(file_time_end)

        def _in_window(item):
            dt = item.get("FILE_DATETIME")
            if dt is None:
                return False
            if file_time_start and dt < file_time_start:
                return False
            if file_time_end and dt > file_time_end:
                return False
            return True

        before = len(items)
        items = [i for i in items if _in_window(i)]
        print(
            f"  Filename-time filter: {before} -> {len(items)} results "
            f"({file_time_start} to {file_time_end})"
        )

    # Sort by filename time and limit
    items_with_time = [i for i in items if i.get("FILE_DATETIME")]
    items_without_time = [i for i in items if not i.get("FILE_DATETIME")]
    items_with_time.sort(key=lambda x: x["FILE_DATETIME"])
    items = items_with_time + items_without_time

    if max_files is not None:
        items = items[:max_files]

    print(f"Query returned {len(items)} result(s).")
    return items


def download_ncei_data(results, output_dir, overwrite=False, companion_extensions=None):
    """Download files returned by :func:`query_ncei_data`.

    Converts each result's ``CLOUD_PATH`` (an S3 URI) to a public HTTPS URL
    and streams the file to *output_dir*.  ``.tar`` archives are automatically
    extracted into a subfolder and the archive is removed.

    Args:
        results (list[dict]): Output from :func:`query_ncei_data`.
        output_dir (str | Path): Local directory to save downloaded files.
        overwrite (bool): If ``False`` (default), skip files that already
            exist in *output_dir*.
        companion_extensions (list[str] | None): Additional file extensions to
            download alongside each result.  For example,
            ``[".bot", ".idx"]`` will attempt to download companion files that
            share the same base name but have these extensions.  Missing
            companions (HTTP 404) are silently skipped.

    Returns:
        list[Path]: Paths to all downloaded (or extracted) files.

    Example::

        results = query_ncei_data(dataset_name="DY2207_EK80", max_files=3)
        paths = download_ncei_data(
            results,
            output_dir="./downloads",
            companion_extensions=[".bot", ".idx"],
        )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_paths = []
    total = len(results)

    for idx, item in enumerate(results, start=1):
        cloud_path = item.get("CLOUD_PATH")
        if not cloud_path:
            print(f"  [{idx}/{total}] Skipping result with no CLOUD_PATH.")
            continue

        url = _s3_to_https(cloud_path)
        filename = item.get("FILE_NAME") or url.rsplit("/", 1)[-1]

        # Probe the server to find the actual file extension if absent from the URL.
        file_ext = Path(filename).suffix
        url_ext = Path(url.rsplit("?", 1)[0]).suffix  # ignore query params
        if not url_ext:
            probed_ext = _probe_url_extension(url)
            if probed_ext:
                url += probed_ext
                # Correct the filename extension if it differs from reality
                if file_ext != probed_ext:
                    filename = Path(filename).stem + probed_ext
            elif file_ext:
                url += file_ext
            else:
                filename += ".tar"
                url += ".tar"

        dest = output_dir / filename

        # Check for already-extracted directory (tar case)
        extracted_dir = output_dir / Path(filename).stem
        if not overwrite and dest.exists():
            print(f"  [{idx}/{total}] Already exists, skipping: {filename}")
            all_paths.append(dest)
            # Still check companions even when main file exists
            if companion_extensions:
                all_paths.extend(
                    _download_companions(url, dest, output_dir, companion_extensions, overwrite)
                )
            continue
        if not overwrite and extracted_dir.is_dir():
            print(f"  [{idx}/{total}] Already extracted, skipping: {extracted_dir.name}/")
            existing = list(extracted_dir.rglob("*"))
            all_paths.extend(p for p in existing if p.is_file())
            continue

        print(f"  [{idx}/{total}] Downloading: {filename}")

        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()

        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                fh.write(chunk)

        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"           Saved ({size_mb:.1f} MB): {dest.name}")

        # Auto-extract tar archives
        if dest.suffix == ".tar" or tarfile.is_tarfile(dest):
            try:
                extracted = _extract_tar(dest, output_dir)
                all_paths.extend(extracted)
            except Exception as exc:
                print(f"           Warning: tar extraction failed: {exc}")
                all_paths.append(dest)
        else:
            all_paths.append(dest)

        # Download companion files (.bot, .idx, etc.)
        if companion_extensions:
            all_paths.extend(
                _download_companions(url, dest, output_dir, companion_extensions, overwrite)
            )

    print(f"Download complete. {len(all_paths)} file(s) ready.")
    return all_paths
