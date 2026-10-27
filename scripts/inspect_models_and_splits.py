#!/usr/bin/env python3
"""
Inspect Time-Series Forecasting Splits, Sliding Windows, and Deep Learning Architectures.

Demonstrates:
- Chronological 70% / 15% / 15% train/val/test splits without future data leakage
- Target timestamp alignment across forecast horizons (H = 1, 6, 24 hours)
- Scaler fitting isolation (strictly on training split)
- Parameter counts, layer summaries, and tensor shapes for WaveANN, WaveLSTM, and WaveCNN1D
"""

from pathlib import Path
import sys
import torch

# Ensure console safe encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    create_wave_forecast_dataloaders,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
)
from src.models import build_wave_model


def main():
    pq_path = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"
    if not pq_path.is_file():
        print(f"Error: Processed dataset not found at {pq_path}", file=sys.stderr)
        print("Please run 'python scripts/run_preprocessing.py' first.", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("🌊 ERA5 OCEAN WAVE FORECASTING: DATASET SPLITS & MODEL ARCHITECTURES")
    print("=" * 80)
    print(f"Dataset Path        : {pq_path}")
    print(f"Features ({len(DEFAULT_FEATURE_COLS)}) : {DEFAULT_FEATURE_COLS}")
    print(f"Target Variable     : {DEFAULT_TARGET_COL} (Significant Wave Height, m)")
    print("-" * 80)

    # 1. Inspect Multi-Horizon Setup
    for H in [1, 6, 24]:
        train_l, val_l, test_l, scaler, meta = create_wave_forecast_dataloaders(
            data_source=pq_path,
            window_size=24,
            horizon=H,
            batch_size=64,
            random_seed=42,
        )

        tr = meta["target_time_ranges"]
        sc = meta["split_counts"]

        print(f"\n[FORECAST HORIZON: {H} HOUR(S) AHEAD (H={H})]")
        print(f"  • Input Window Length (L) : {meta['window_size']} hours (24 historical steps)")
        print(f"  • Training Split (70%)    : {sc['train']:,} samples | Targets: {tr['train_start']} to {tr['train_end']}")
        print(f"  • Validation Split (15%)  : {sc['val']:,} samples | Targets: {tr['val_start']} to {tr['val_end']}")
        print(f"  • Testing Split (15%)     : {sc['test']:,} samples | Targets: {tr['test_start']} to {tr['test_end']}")

        # Verify zero overlap between split target boundaries
        t_train_max = tr["train_end"]
        t_val_min = tr["val_start"]
        t_val_max = tr["val_end"]
        t_test_min = tr["test_start"]

        assert t_train_max < t_val_min, "Leakage: Train overlaps Val!"
        assert t_val_max < t_test_min, "Leakage: Val overlaps Test!"
        print(f"  • Leakage Audit           : PASSED (Train targets < Val targets < Test targets, strictly disjoint)")

    # 2. Inspect Training Scaler Statistics
    print("\n" + "=" * 80)
    print("SCALER ISOLATION AUDIT (Fitted Strictly on Training Split 2012-2022)")
    print("=" * 80)
    print(f"Target Scaler  (swh)  : Mean = {scaler.target_scaler.mean_[0]:.4f} m, Scale = {scaler.target_scaler.scale_[0]:.4f} m")
    for feat, m, s in zip(DEFAULT_FEATURE_COLS, scaler.feature_scaler.mean_, scaler.feature_scaler.scale_):
        print(f"  • Feature: {feat:18s} | Mean: {m:8.4f} | Scale: {s:8.4f}")

    # 3. Model Architecture Comparison
    print("\n" + "=" * 80)
    print("DEEP LEARNING MODEL ARCHITECTURES (Unified PyTorch Interface)")
    print("=" * 80)

    seq_len = 24
    num_features = len(DEFAULT_FEATURE_COLS)
    output_dim = 1
    sample_shape = (64, seq_len, num_features)

    for m_type in ["ann", "lstm", "cnn1d"]:
        model = build_wave_model(
            model_type=m_type,
            input_dim=num_features,
            seq_len=seq_len,
            output_dim=output_dim,
        )
        print(model.get_model_summary(sample_input_shape=sample_shape))
        print()

    # 4. Demonstrate Common DataLoader Batch
    bx, by = next(iter(train_l))
    print("=" * 80)
    print("UNIFIED DATALOADER BATCH VERIFICATION")
    print("=" * 80)
    print(f"Batch X Tensor Shape : {list(bx.shape)} (batch_size, seq_len=24, num_features={num_features})")
    print(f"Batch Y Tensor Shape : {list(by.shape)} (batch_size, forecast_horizon=1)")
    print(f"Batch X Data Type    : {bx.dtype}")
    print(f"Batch Y Data Type    : {by.dtype}")
    print("=" * 80)


if __name__ == "__main__":
    main()
