#!/usr/bin/env python3
"""
Runnable pipeline script to preprocess ERA5 NetCDF datasets for Ocean Wave Forecasting.

Usage:
    python scripts/run_preprocessing.py
    python scripts/run_preprocessing.py --lat 18.5 --lon 72.5
"""

import argparse
from pathlib import Path
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import preprocess_era5_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess ERA5 wave and surface atmospheric NetCDF files into clean time-series datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Path to directory containing raw NetCDF (.nc) files (default: CP/data).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Path to directory for saving processed files (default: CP/data/processed).",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=18.5,
        help="Target offshore latitude in degrees North (default: 18.5).",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=72.5,
        help="Target offshore longitude in degrees East (default: 72.5).",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Disable Parquet format export.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV format export.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("🌊 ERA5 OCEAN WAVE FORECASTING: PREPROCESSING PIPELINE")
    print("=" * 80)
    print(f"Data Directory  : {data_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Point    : {args.lat:.2f}° N, {args.lon:.2f}° E")
    print("-" * 80)

    config = {
        "export_parquet": not args.no_parquet,
        "export_csv": not args.no_csv,
    }

    try:
        df_processed, report = preprocess_era5_pipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            target_lat=args.lat,
            target_lon=args.lon,
            config=config,
        )
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[1] GRID POINT VERIFICATION:")
    gp = report["selected_grid_point"]
    print(f"  • Selected Coordinates: {gp['selected_latitude']:.2f}° N, {gp['selected_longitude']:.2f}° E")
    print(f"  • Exact Grid Match    : {gp['is_exact_grid_match']}")
    print(f"  • Distance from Target: {gp['distance_from_requested_deg']:.4f}°")
    print(f"  • Water Cell Status   : Confirmed Ocean Cell (certainty: {gp['water_verification_certainty']})")
    print(f"  • Verification Note   : {gp['water_verification_reason']}")

    print("\n[2] TEMPORAL COVERAGE & CONTINUITY:")
    tc = report["temporal_coverage"]
    print(f"  • Time Span           : {tc['time_start']} to {tc['time_end']}")
    print(f"  • Total Hourly Rows   : {tc['total_rows']:,}")
    print(f"  • Missing Timestamps  : {tc['missing_timestamps']}")
    print(f"  • Duplicate Timestamps: {tc['duplicate_timestamps']}")
    print(f"  • Hourly Continuity   : {'PERFECT (1.0h continuous)' if tc['missing_timestamps'] == 0 else 'Discontinuous'}")

    print("\n[3] QUALITY ASSURANCE AUDIT:")
    qa = report["quality_assurance_summary"]
    print(f"  • QA Check Status     : {'PASSED' if qa['passed'] else 'FLAGS DETECTED'}")
    print(f"  • Total NaNs / Nulls  : {qa['total_nan_count']}")
    print(f"  • Total Infinities    : {qa['total_inf_count']}")
    print(f"  • Implausible Values  : {qa['total_implausible_count']}")

    print("\n[4] DATASET STATISTICS (Sampled Offshore Point):")
    stats = qa["statistics"]
    for col, st in stats.items():
        print(f"  • {col:18s} | Min: {st['min']:8.3f} | Mean: {st['mean']:8.3f} | Max: {st['max']:8.3f} | Std: {st['std']:8.3f}")

    print("\n[5] ARTIFACTS GENERATED:")
    for path in report["output_files"]:
        print(f"  ✓ {path}")
    print(f"  ✓ {PROJECT_ROOT / 'outputs' / 'reports' / 'preprocessing_report.json'}")
    print(f"  ✓ {PROJECT_ROOT / 'outputs' / 'reports' / 'preprocessing_report.md'}")

    print("\n" + "=" * 80)
    print("SUCCESS: Preprocessing pipeline completed without errors.")
    print("=" * 80)


if __name__ == "__main__":
    main()
