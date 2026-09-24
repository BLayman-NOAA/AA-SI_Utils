# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Compare seafloor detection methods on the HB1603 and HB2407 example data.

Six datasets of roughly two hours each: two from the HB1603 2016-07-25 leg
(EK60, 1500 to 2000 m, canyons and soft sediment, .bot reference) and four
from HB2407 (EK80, 21 to 70 m): 24 September and 12 October with Echoview
line references, 14 and 15 October (the herring days) without.

    python scripts/compare_seabed_methods.py --out-dir <folder> [--dataset NAME] [--skip-experimental]

Each dataset folder gets metrics.csv, pairwise.csv, lines.nc and overlay.png;
the output root gets summary.csv across datasets.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aa_si_utils.seabed.compare import Dataset, run_dataset  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[2] / "AA-SI_recipe_manager" / "examples"
HB1603 = EXAMPLES / "HB1603" / "raw_file_inputs" / "HB1603_2016-07-25_2058-2345"
HB2407 = EXAMPLES / "HB2407" / "raw"
HB2407_EVL = EXAMPLES / "HB2407" / "seabed_lines"


def _hb1603(name, stamps, notes):
    return Dataset(
        name=name,
        sonar_model="EK60",
        raw_paths=[str(HB1603 / f"D20160725-T{s}.raw") for s in stamps],
        r_min=800.0,
        r_max=2500.0,
        primary_khz=38.0,
        secondary_khz=18.0,
        regime="deep",
        hdbscan_bin_m=5.0,
        notes=notes,
    )


def _hb2407(name, stamps, evl, notes):
    return Dataset(
        name=name,
        sonar_model="EK80",
        raw_paths=[str(HB2407 / f"D{s}.raw") for s in stamps],
        # 15 m: the transducer is at 7 m and on 24 September the seabed was
        # at 21 to 30 m. A 30 m start excluded it and every detector locked
        # onto the first bottom multiple at 47 to 55 m.
        r_min=15.0,
        r_max=120.0,
        primary_khz=38.0,
        secondary_khz=18.0,
        evl_path=str(HB2407_EVL) if evl else None,
        regime="shelf",
        hdbscan_bin_m=1.0,
        notes=notes,
    )


DATASETS = [
    _hb1603("HB1603_A", ["203535", "205832", "212129", "214425"], "20:35 to 22:07, seabed 1650 to 1950 m, canyon walls"),
    _hb1603("HB1603_B", ["220721", "223017", "225314", "231505", "233217"], "22:07 to 23:55, seabed 1500 to 1750 m, soft flat section"),
    _hb2407("HB2407_0924", ["20240924-T194054", "20240924-T203624"], True, "19:40 to 21:31 on 2024-09-24, Echoview line available"),
    _hb2407("HB2407_1012", ["20241012-T045613", "20241012-T055116"], True, "04:56 to 06:46 on 2024-10-12, Echoview line available"),
    _hb2407("HB2407_1014", ["20241014-T073730", "20241014-T083236"], False, "07:37 to 09:27 on 2024-10-14, herring survey, no line"),
    _hb2407("HB2407_1015", ["20241015-T002915", "20241015-T012423"], False, "00:29 to 02:19 on 2024-10-15, herring survey, no line; the seabed leaves the 120 m window after ping 2600"),
]
# Same files with a window deep enough for the second half.
DATASETS.append(Dataset(**{**DATASETS[-1].__dict__, "name": "HB2407_1015_deep", "r_max": 400.0, "hdbscan_bin_m": 3.0,
                          "notes": "as HB2407_1015 with the window extended to 400 m"}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", action="append", help="run only this dataset (repeatable)")
    parser.add_argument("--methods", help="comma-separated method names to run")
    parser.add_argument("--skip-experimental", action="store_true")
    args = parser.parse_args(argv)

    chosen = [d for d in DATASETS if not args.dataset or d.name in args.dataset]
    methods = set(args.methods.split(",")) if args.methods else None
    tables = []
    for dataset in chosen:
        missing = [p for p in dataset.raw_paths if not Path(p).exists()]
        if missing:
            print(f"[{dataset.name}] skipped, missing {missing}")
            continue
        table = run_dataset(dataset, Path(args.out_dir) / dataset.name, methods, args.skip_experimental)
        print(table.to_string())
        tables.append(table)
    if tables:
        summary = pd.concat(tables)
        summary.to_csv(Path(args.out_dir) / "summary.csv")
        print(summary.to_string())


if __name__ == "__main__":
    main()
