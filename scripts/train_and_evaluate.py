#!/usr/bin/env python3
"""
End-to-End Training and Evaluation Pipeline for Ocean Wave Forecasting.
Benchmarks ANN, LSTM, and 1D-CNN architectures on ECMWF ERA5 reanalysis data.

Usage:
    python scripts/train_and_evaluate.py --model all
    python scripts/train_and_evaluate.py --model ann --epochs 15
    python scripts/train_and_evaluate.py --model lstm --horizon 6
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    create_wave_forecast_dataloaders,
    set_deterministic_seeds,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
)
from src.models import build_wave_model
from src.training.trainer import ModelTrainer
from src.training.checkpoint import ModelCheckpoint
from src.evaluation.evaluator import WaveModelEvaluator
from src.evaluation.metrics import measure_inference_latency
from src.visualization.plots import (
    plot_training_history,
    plot_actual_vs_predicted,
    plot_prediction_residuals,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate deep learning models (ANN, LSTM, 1D-CNN) for wave height forecasting."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "ann", "lstm", "cnn1d"],
        help="Model architecture to train (default: 'all').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of training epochs per model (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training and evaluation (default: 256).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate for Adam optimizer (default: 0.001).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping patience in epochs (default: 4).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        choices=[1, 6, 24],
        help="Forecast lead time horizon in hours ahead (default: 1).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=24,
        help="Historical input window length in hours (default: 24).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"),
        help="Path to preprocessed ERA5 Parquet file.",
    )
    return parser.parse_args()


def verify_prerequisites(data_path: Path):
    """Verify that processed dataset exists and contains required features before training."""
    if not data_path.is_file():
        print(f"\n[ERROR] Processed dataset not found at: {data_path}", file=sys.stderr)
        print("Please run the preprocessing pipeline first:", file=sys.stderr)
        print("    python scripts/run_preprocessing.py", file=sys.stderr)
        sys.exit(1)

    df_check = pd.read_parquet(data_path)
    required = ["swh", "mwp", "u10", "v10", "mwd_sin", "mwd_cos"]
    missing = [c for c in required if c not in df_check.columns]
    if missing:
        print(f"\n[ERROR] Required features missing from {data_path}: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Verified dataset: {data_path.name} ({len(df_check):,} rows, {len(df_check.columns)} columns)")


def main():
    args = parse_args()
    data_path = Path(args.data_path)

    print("=" * 80)
    print("🌊 ERA5 OCEAN WAVE FORECASTING: TRAINING & EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Target Variable      : {DEFAULT_TARGET_COL} (Significant Wave Height, meters)")
    print(f"Forecast Horizon (H) : {args.horizon} hour(s) ahead")
    print(f"Input Window (L)     : {args.window_size} historical hours")
    print(f"Models Selected      : {args.model.upper()}")
    print(f"Batch Size           : {args.batch_size} | Learning Rate: {args.lr} | Max Epochs: {args.epochs}")
    print(f"Early Stopping       : Patience = {args.patience} epochs")
    print("-" * 80)

    # 1. Verify Prerequisites
    verify_prerequisites(data_path)

    # 2. Build DataLoaders (Strict 70/15/15 chronological split with train-only scaling)
    print("\n[1] Constructing leak-free DataLoaders and fitting scalers...")
    train_loader, val_loader, test_loader, scaler, meta = create_wave_forecast_dataloaders(
        data_source=data_path,
        window_size=args.window_size,
        horizon=args.horizon,
        batch_size=args.batch_size,
        random_seed=args.seed,
    )

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    scaler.save(models_dir / "scaler.pkl")

    print(f"  • Training Samples   : {meta['split_counts']['train']:,} (70%)")
    print(f"  • Validation Samples : {meta['split_counts']['val']:,} (15%)")
    print(f"  • Testing Samples    : {meta['split_counts']['test']:,} (15%)")
    print(f"  • Features ({meta['num_features']})     : {meta['feature_cols']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  • Execution Device   : {device.upper()}")

    # 3. Determine Models to Train
    models_to_run = ["ann", "lstm", "cnn1d"] if args.model == "all" else [args.model.lower()]

    histories = {}
    test_predictions_raw = {}
    test_predictions_m = {}
    latency_results = {}
    trained_models = {}

    # 4. Train Models
    print("\n" + "=" * 80)
    print("[2] TRAINING MODEL BENCHMARK")
    print("=" * 80)

    for m_type in models_to_run:
        m_name = {"ann": "WaveANN", "lstm": "WaveLSTM", "cnn1d": "WaveCNN1D"}[m_type]
        print(f"\n>>> Initializing {m_name} (Horizon = {args.horizon}h, Window = {args.window_size}h)...")

        # Set seed for reproducible initialization
        set_deterministic_seeds(args.seed)

        model = build_wave_model(
            model_type=m_type,
            input_dim=meta["num_features"],
            seq_len=args.window_size,
            output_dim=1,
        )

        counts = model.count_parameters()
        print(f"    Architecture Parameters: {counts['trainable_params']:,} trainable")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
        )
        criterion = nn.MSELoss()

        model_name_ckpt = f"best_{m_type}.pt" if args.horizon == 1 else f"best_{m_type}_h{args.horizon}.pt"
        ckpt = ModelCheckpoint(
            checkpoint_dir=models_dir,
            best_model_name=model_name_ckpt,
            model_config=model.get_config(),
            dataset_config={
                "horizon": args.horizon,
                "window_size": args.window_size,
                "features": meta["feature_cols"],
                "target": meta["target_col"],
            },
        )

        trainer = ModelTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint=ckpt,
            scheduler=scheduler,
            patience=args.patience,
            min_delta=1e-4,
        )

        fit_res = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            verbose=True,
        )

        histories[m_name] = fit_res["history"]

        # Reload best saved weights
        print(f"    Loading best checkpoint (Epoch {fit_res['best_epoch']}, Val Loss: {fit_res['best_val_loss']:.5f})...")
        ckpt.load_best_checkpoint(model, device=device)
        trained_models[m_name] = model

        # Benchmark inference latency
        sample_batch, _ = next(iter(test_loader))
        lat_info = measure_inference_latency(model, sample_batch, num_iterations=100, warmup=20, device=device)
        latency_results[m_name] = lat_info

        # Generate test predictions
        preds_scaled = trainer.predict(test_loader)
        test_predictions_raw[m_name] = preds_scaled

        # INVERSE TRANSFORM TO METERS
        preds_m = scaler.inverse_transform_target(preds_scaled)
        test_predictions_m[m_name] = preds_m

    # 5. Extract Ground Truth Targets in Meters
    # Test dataset stores both scaled and raw targets
    test_ds = test_loader.dataset
    if test_ds.raw_targets is not None:
        y_true_m = test_ds.raw_targets[test_ds.anchors + test_ds.horizon].flatten()
    else:
        y_true_m = scaler.inverse_transform_target(test_ds.targets[test_ds.anchors + test_ds.horizon])

    test_timestamps = test_ds.timestamps[test_ds.anchors + test_ds.horizon]

    # 6. Evaluation and Benchmarking
    print("\n" + "=" * 80)
    print("[3] EVALUATION ON UNTOUCHED TEST SET (Physical Units: Meters)")
    print("=" * 80)

    evaluator = WaveModelEvaluator(y_true=y_true_m, target_units="meters")
    for m_name, preds_m in test_predictions_m.items():
        evaluator.add_model_result(
            model_name=m_name,
            y_pred=preds_m,
            latency_info=latency_results[m_name],
        )

    # Rank models by test RMSE
    best_name, best_metrics, comparison_df = evaluator.rank_models(criterion="rmse", ascending=True)

    # Print Comparison Table
    display_cols = [
        "model",
        "rmse",
        "mae",
        "r2",
        "scatter_index",
        "bias",
        "wape_pct",
        "latency_sample_ms",
        "throughput_fps",
    ]
    print("\n" + comparison_df[display_cols].to_string(index=False))

    print("\n" + "-" * 80)
    print(f"🏆 BEST MODEL SELECTED: {best_name}")
    print(f"   Selection Criterion : Lowest Test RMSE ({best_metrics['rmse']:.4f} m)")
    print(f"   Test MAE            : {best_metrics['mae']:.4f} m")
    print(f"   Test R² Score       : {best_metrics['r2']:.4f}")
    print(f"   Scatter Index (SI)  : {best_metrics['scatter_index']:.4f} (or {best_metrics['scatter_index']*100:.2f}%)")
    print(f"   Latency / Sample    : {best_metrics['latency_sample_ms']:.4f} ms ({best_metrics['throughput_fps']:.1f} samples/sec)")
    print("-" * 80)

    # 7. Generate and Save Outputs
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    preds_dir = PROJECT_ROOT / "outputs" / "predictions"
    for d in [metrics_dir, plots_dir, preds_dir]:
        d.mkdir(parents=True, exist_ok=True)

    suffix = f"_h{args.horizon}" if args.horizon != 1 else ""

    # A. outputs/metrics/model_comparison.csv
    csv_metrics_path = metrics_dir / f"model_comparison{suffix}.csv"
    comparison_df.to_csv(csv_metrics_path, index=False)
    print(f"\n[Artifact] Saved comparison table -> {csv_metrics_path}")

    # B. outputs/metrics/detailed_metrics.json
    detailed_json_path = metrics_dir / f"detailed_metrics{suffix}.json"
    detailed_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "forecast_horizon_hours": args.horizon,
        "input_window_hours": args.window_size,
        "best_model": best_name,
        "selection_criterion": "lowest_test_rmse",
        "models": comparison_df.to_dict(orient="records"),
        "latency_benchmarks": latency_results,
        "dataset_split_info": meta["target_time_ranges"],
    }
    with open(detailed_json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_payload, f, indent=2)
    print(f"[Artifact] Saved detailed metrics  -> {detailed_json_path}")

    # C. outputs/plots/training_history.png
    history_plot_path = plots_dir / f"training_history{suffix}.png"
    plot_training_history(histories, save_path=history_plot_path)
    print(f"[Artifact] Saved training curves   -> {history_plot_path}")

    # D. outputs/plots/actual_vs_predicted.png
    actual_pred_plot_path = plots_dir / f"actual_vs_predicted{suffix}.png"
    plot_actual_vs_predicted(
        y_true=y_true_m,
        predictions=test_predictions_m,
        timestamps=test_timestamps,
        save_path=actual_pred_plot_path,
        sample_window=300,
    )
    print(f"[Artifact] Saved forecast plot     -> {actual_pred_plot_path}")

    # E. outputs/plots/prediction_residuals.png
    residuals_plot_path = plots_dir / f"prediction_residuals{suffix}.png"
    plot_prediction_residuals(
        y_true=y_true_m,
        predictions=test_predictions_m,
        save_path=residuals_plot_path,
    )
    print(f"[Artifact] Saved residual plot     -> {residuals_plot_path}")

    # F. outputs/predictions/test_predictions.csv
    pred_df = pd.DataFrame({
        "valid_time": test_timestamps,
        "actual_swh_meters": y_true_m,
    })
    for m_name, preds_m in test_predictions_m.items():
        col_prefix = m_name.lower().replace("wave", "")
        pred_df[f"{col_prefix}_pred_meters"] = preds_m
        pred_df[f"{col_prefix}_error_meters"] = preds_m - y_true_m

    preds_csv_path = preds_dir / f"test_predictions{suffix}.csv"
    pred_df.to_csv(preds_csv_path, index=False)
    print(f"[Artifact] Saved test predictions  -> {preds_csv_path} ({len(pred_df):,} rows)")

    print("\n" + "=" * 80)
    print("SUCCESS: Training and evaluation pipeline finished with all artifacts saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
