# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Run the phase-aware seabed detector on one raw file against a reference.

Example, from the AA-SI_Utils folder with the recipe manager's venv:

    python scripts/validate_seabed_detection.py \
        --raw ../AA-SI_recipe_manager/examples/HB1603/raw_file_inputs/HB1603_2016-07-25_2058-2345/D20160725-T205832.raw \
        --sonar-model EK60 --reference bot --out-dir validation/hb1603

    python scripts/validate_seabed_detection.py \
        --raw ../AA-SI_recipe_manager/examples/HB2407/raw/D20241012-T064618.raw \
        --sonar-model EK80 --reference evl \
        --evl-path ../AA-SI_recipe_manager/examples/HB2407/seabed_lines --out-dir validation/hb2407

Writes metrics.json, overlay.png and diagnostics.zarr to the output folder.
The reference lines are not ground truth; read the overlay before trusting
either side of a disagreement.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aa_si_utils.seabed.validation import run_validation  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help="raw file")
    parser.add_argument("--sonar-model", required=True, choices=["EK60", "EK80"])
    parser.add_argument("--reference", required=True, choices=["bot", "evl"])
    parser.add_argument("--evl-path", help=".evl file or folder for the evl reference")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--channel", help="primary channel label, Hz or kHz; default auto")
    parser.add_argument("--r-min", type=float, help="search window lower bound (m)")
    parser.add_argument("--r-max", type=float, help="search window upper bound (m)")
    parser.add_argument("--regime", default="auto", choices=["auto", "shelf", "slope", "deep"])
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2], help="1 point features, 2 adds shape features")
    parser.add_argument("--line", default="leading_edge", choices=["leading_edge", "integration"])
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--max-slope-deg", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta-fraction", type=float, default=0.1)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--walkback-db", type=float, default=10.0)
    parser.add_argument("--phase-weight", type=float, default=1.0)
    parser.add_argument("--max-gap-s", type=float, default=30.0)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    args = parser.parse_args(argv)

    metrics, _, _, _ = run_validation(
        args.raw,
        args.sonar_model,
        args.reference,
        evl_path=args.evl_path,
        out_dir=args.out_dir,
        r_min=args.r_min,
        r_max=args.r_max,
        channel=args.channel,
        regime=args.regime,
        mode=args.mode,
        line=args.line,
        n_candidates=args.n_candidates,
        max_slope_deg=args.max_slope_deg,
        alpha=args.alpha,
        beta_fraction=args.beta_fraction,
        min_score=args.min_score,
        walkback_db=args.walkback_db,
        phase_weight=args.phase_weight,
        max_gap_s=args.max_gap_s,
        min_confidence=args.min_confidence,
    )
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
