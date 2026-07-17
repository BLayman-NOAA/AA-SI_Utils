# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Runnable example for experimenting with :mod:`aa_si_utils.data_retrieval`.

Edit the parameters below and run the file directly, e.g. from the AA-SI
folder in a bash terminal:

    ./AA-SI_recipe_manager/.venv/Scripts/python.exe \
        AA-SI_Utils/src/aa_si_utils/data_retrieval_example.py

Querying is read-only (no files are fetched); downloading only happens if
you set DOWNLOAD = True below.
"""

from pprint import pprint

from aa_si_utils.data_retrieval import download_ncei_data, query_ncei_data

# --- Query parameters (edit these) -----------------------------------------

# Fine-grained time window, matched against the D{YYYYMMDD}-T{HHMMSS} stamp
# in each file name. ISO strings or datetime objects both work.
FILE_TIME_START = "2016-07-25T20:58"
FILE_TIME_END = "2016-07-25T21:45"

# Extra exact-match filters on any WCSD attribute field,
# e.g. {"CRUISE_NAME": "DY2207", "PLATFORM_NAME": "Oscar Dyson (DY)"}.
FILTERS = {"CRUISE_NAME": "HB1603"}

# Keep this small while experimenting; an unconstrained cruise query can
# return thousands of records.
MAX_FILES = 5

# Optional spatial filter: (west, south, east, north) in WGS 84 decimal
# degrees (longitude negative in the western hemisphere; west < east).
# Set to None to disable, e.g. BBOX = (-170.5, 52.0, -165.0, 58.5)
BBOX = (-69, 36, -64, 43)

# --- Download settings ------------------------------------------------------

# Querying alone never downloads anything. Set DOWNLOAD = True to fetch the
# matched files into OUTPUT_DIR / <query_label>/ (tar archives are
# auto-extracted; already-present files are skipped on re-runs).
DOWNLOAD = False
OUTPUT_DIR = "./downloads"
COMPANION_EXTENSIONS = [".bot", ".idx"]

# ---------------------------------------------------------------------------


def main():
    result = query_ncei_data(
        file_time_start=FILE_TIME_START,
        file_time_end=FILE_TIME_END,
        filters=FILTERS,
        bbox=BBOX,
    )

    records = result["records"]
    print()
    print(f"query_label: {result['query_label']}")
    print(f"records:     {len(records)}")

    if records:
        print("\nFirst record:")
        pprint(records[0])

    if DOWNLOAD and records:
        print()
        out = download_ncei_data(
            result,
            output_dir=OUTPUT_DIR,
            companion_extensions=COMPANION_EXTENSIONS,
        )
        print(f"\nDownloaded into: {out['download_dir']}")
        for path in out["downloaded_paths"]:
            print(f"  {path}")


if __name__ == "__main__":
    main()
