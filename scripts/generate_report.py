#!/usr/bin/env python3
"""CLI Script to generate automated research reports for macro placement experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.report import generate_research_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication research report from macro placement benchmark outputs."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to results directory containing experiment folders (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/reports",
        help="Path to destination directory for generated report and figures (default: results/reports)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF compilation attempt",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Use relative paths for figures instead of base64 inline embedding",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("Macro Placement & Floorplanning: Research Report Generator")
    print("=" * 65)

    res = generate_research_report(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        generate_pdf=not args.no_pdf,
        embed_images=not args.no_embed,
    )

    print(f"Loaded Experiment A: {res['num_records_a']} rows")
    print(f"Loaded Experiment B: {res['num_records_b']} rows")
    print(f"Loaded Experiment D: {res['num_records_d']} rows")
    print("-" * 65)
    print("Generated:")
    print(f"  HTML:    {res['html_path']}")
    print(f"  PDF:     {res['pdf_status']}")
    print(f"  Figures: {len(res['figures'])}")
    print("=" * 65)


if __name__ == "__main__":
    main()
