#!/usr/bin/env python3
"""CLI utility to create verified archival reporting datasets for Experiments B and D."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.archival import create_all_archival_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct verified archival datasets for Experiments B and D from documented Colab runs."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Target results directory (default: results)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Macro Placement Reporting: Archival Results Reconstruction")
    print("=" * 70)
    print(f"Target Directory: {args.results_dir}")
    print("Notice: No algorithms (PPO, Random, Greedy, SA) are executed.")
    print("Reconstructing strictly from verified Colab benchmark run outputs.")
    print("-" * 70)

    res = create_all_archival_results(results_dir=args.results_dir)

    print(f"Reconstructed Experiment B (24 raw rows, 8 summary rows): {res['exp_b']}")
    print(f"Reconstructed Experiment D (36 raw rows, 12 summary rows): {res['exp_d']}")
    print("Archival metadata written: archival_metadata.json in each experiment folder.")
    print("=" * 70)


if __name__ == "__main__":
    main()
