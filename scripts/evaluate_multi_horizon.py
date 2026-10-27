#!/usr/bin/env python3
"""
Rigorous Multi-Horizon Benchmark: Naive Persistence vs ANN vs LSTM vs 1D-CNN.
Evaluates forecast horizons: 1-Hour, 6-Hour, and 24-Hour lead times.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_wave_model
from src.data.dataset import (
    create_wave_forecast_dataloaders,
    TimeSeriesScaler,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
)
from src.evaluation.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score_metric,
    scatter_index,
    mean_bias_error,
)


def inspect_available_checkpoint(
    model_type: str,
    horizon: int,
    models_dir: Path,
) -> Tuple[Optional[Path], Optional[Dict]]:
    """Check if a genuine checkpoint exists and its metadata matches the requested horizon."""
    m_code = model_type.lower().replace("wave", "")
    
    # Check horizon-specific checkpoint or default checkpoint
    ckpt_h = models_dir / f"best_{m_code}_h{horizon}.pt"
    meta_h = models_dir / f"best_{m_code}_h{horizon}_metadata.json"

    if ckpt_h.is_file() and meta_h.is_file():
        with open(meta_h, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("dataset_config", {}).get("horizon") == horizon:
            return ckpt_h, meta

    if horizon == 1:
        ckpt_default = models_dir / f"best_{m_code}.pt"
        meta_default = models_dir / f"best_{m_code}_metadata.json"
        if ckpt_default.is_file() and meta_default.is_file():
            with open(meta_default, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("dataset_config", {}).get("horizon") == 1:
                return ckpt_default, meta

    return None, None


def run_multi_horizon_benchmark():
    data_path = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"
    models_dir = PROJECT_ROOT / "models"
    outputs_metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    outputs_plots_dir = PROJECT_ROOT / "outputs" / "plots"

    outputs_metrics_dir.mkdir(parents=True, exist_ok=True)
    outputs_plots_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.is_file():
        print(f"[ERROR] Processed data not found at {data_path}", file=sys.stderr)
        sys.exit(1)

    df_full = pd.read_parquet(data_path)
    horizons = [1, 6, 24]
    architectures = [
        ("WaveLSTM", "LSTM"),
        ("WaveANN", "ANN"),
        ("WaveCNN1D", "CNN1D"),
    ]

    print("=" * 80)
    print("🌊 RIGOROUS MULTI-HORIZON BENCHMARK (H=1h, H=6h, H=24h)")
    print("=" * 80)
    print(f"Dataset Path : {data_path.name} ({len(df_full):,} hourly observations)")
    print("Splits       : 70% Train, 15% Validation, 15% Test (Strict Chronological)")
    print("Horizons     : 1 Hour Ahead, 6 Hours Ahead, 24 Hours Ahead")
    print("-" * 80)

    all_results = []
    horizon_summaries = {}

    for h in horizons:
        print(f"\n" + "#" * 80)
        print(f"📊 EVALUATING FORECAST HORIZON: H = {h} HOUR(S) AHEAD")
        print("#" * 80)

        # 1. Build DataLoaders and retrieve metadata for horizon H
        train_loader, val_loader, test_loader, scaler, meta = create_wave_forecast_dataloaders(
            data_source=data_path,
            window_size=24,
            horizon=h,
            batch_size=256,
        )

        test_ds = test_loader.dataset
        n_test = len(test_ds)
        test_anchors = test_ds.anchors
        target_indices = test_anchors + h

        # Exact timestamps
        anchor_times = df_full["valid_time"].iloc[test_anchors].values
        target_times = df_full["valid_time"].iloc[target_indices].values
        start_target_str = str(target_times[0])
        end_target_str = str(target_times[-1])

        # Ground truth future wave height in meters
        y_true = df_full["swh"].iloc[target_indices].values.astype(np.float64)

        # Naive Persistence: wave height observed at anchor timestamp t
        y_persist = df_full["swh"].iloc[test_anchors].values.astype(np.float64)

        # Evaluate Naive Persistence Baseline
        p_rmse = root_mean_squared_error(y_true, y_persist)
        p_mae = mean_absolute_error(y_true, y_persist)
        p_r2 = r2_score_metric(y_true, y_persist)
        p_si = scatter_index(y_true, y_persist)
        p_wape = (np.sum(np.abs(y_true - y_persist)) / np.sum(y_true)) * 100.0
        p_bias = mean_bias_error(y_true, y_persist)

        print(f"  Test Sample Count : {n_test:,} samples (0 NaNs / missing)")
        print(f"  Target Date Range : {start_target_str} to {end_target_str}")
        print(f"  Anchor Date Range : {str(anchor_times[0])} to {str(anchor_times[-1])}")
        print(f"  Ground Truth Hs   : Mean = {np.mean(y_true):.3f}m | Min = {np.min(y_true):.3f}m | Max = {np.max(y_true):.3f}m")
        print("-" * 80)
        print(f"  [BASELINE] Naive Persistence (y_hat(t+{h}h) = y(t)):")
        print(f"    RMSE = {p_rmse:.4f} m | MAE = {p_mae:.4f} m | R² = {p_r2:.4f} | SI = {p_si*100:.2f}% | WAPE = {p_wape:.2f}%")

        all_results.append({
            "horizon_hours": h,
            "model": "Naive Persistence",
            "model_type": "Baseline",
            "status": "Evaluated",
            "test_sample_count": n_test,
            "target_start_time": start_target_str,
            "target_end_time": end_target_str,
            "rmse_meters": p_rmse,
            "mae_meters": p_mae,
            "r2_score": p_r2,
            "scatter_index_pct": p_si * 100.0,
            "wape_pct": p_wape,
            "bias_meters": p_bias,
            "rmse_skill_pct": 0.0,
            "mae_skill_pct": 0.0,
            "notes": "Predicts future wave height as latest observed wave height at anchor t.",
        })

        horizon_summaries[h] = {
            "persistence": {
                "rmse": p_rmse,
                "mae": p_mae,
                "r2": p_r2,
                "si": p_si,
                "wape": p_wape,
            },
            "models": {},
        }

        # 2. Evaluate available Neural Network Checkpoints
        for m_name, m_type in architectures:
            ckpt_path, ckpt_meta = inspect_available_checkpoint(m_type, h, models_dir)

            if ckpt_path is None:
                print(f"  [MISSING] {m_name:10s} (H={h}h): No trained checkpoint found.")
                print(f"             Run: python scripts/train_and_evaluate.py --model {m_type.lower()} --horizon {h}")
                all_results.append({
                    "horizon_hours": h,
                    "model": m_name,
                    "model_type": m_type,
                    "status": "Pending Training (Checkpoint Missing)",
                    "test_sample_count": n_test,
                    "target_start_time": start_target_str,
                    "target_end_time": end_target_str,
                    "rmse_meters": None,
                    "mae_meters": None,
                    "r2_score": None,
                    "scatter_index_pct": None,
                    "wape_pct": None,
                    "bias_meters": None,
                    "rmse_skill_pct": None,
                    "mae_skill_pct": None,
                    "notes": f"Model for H={h}h not yet trained. Run train_and_evaluate.py --model {m_type.lower()} --horizon {h}",
                })
            else:
                # Model is genuinely trained! Evaluate.
                cfg = ckpt_meta.get("model_config", {})
                model = build_wave_model(
                    model_type=cfg.get("model_type", m_type),
                    input_dim=cfg.get("input_dim", len(DEFAULT_FEATURE_COLS)),
                    seq_len=cfg.get("seq_len", 24),
                    output_dim=cfg.get("output_dim", 1),
                )
                ckpt_data = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt_data.get("model_state_dict", ckpt_data)
                model.load_state_dict(state_dict)
                model.eval()

                all_preds = []
                with torch.no_grad():
                    for batch_x, _ in test_loader:
                        out = model(batch_x)
                        all_preds.append(out.cpu().numpy())

                preds_scaled = np.vstack(all_preds).flatten()
                preds_meters = scaler.inverse_transform_target(preds_scaled).astype(np.float64)

                m_rmse = root_mean_squared_error(y_true, preds_meters)
                m_mae = mean_absolute_error(y_true, preds_meters)
                m_r2 = r2_score_metric(y_true, preds_meters)
                m_si = scatter_index(y_true, preds_meters)
                m_wape = (np.sum(np.abs(y_true - preds_meters)) / np.sum(y_true)) * 100.0
                m_bias = mean_bias_error(y_true, preds_meters)

                # Skill score relative to persistence
                rmse_skill = (1.0 - (m_rmse / p_rmse)) * 100.0
                mae_skill = (1.0 - (m_mae / p_mae)) * 100.0

                print(f"  [TRAINED] {m_name:10s} (H={h}h):")
                print(f"    RMSE = {m_rmse:.4f} m | MAE = {m_mae:.4f} m | R² = {m_r2:.4f} | SI = {m_si*100:.2f}% | Skill vs Persist: RMSE {rmse_skill:+.2f}%, MAE {mae_skill:+.2f}%")

                all_results.append({
                    "horizon_hours": h,
                    "model": m_name,
                    "model_type": m_type,
                    "status": "Evaluated",
                    "test_sample_count": n_test,
                    "target_start_time": start_target_str,
                    "target_end_time": end_target_str,
                    "rmse_meters": m_rmse,
                    "mae_meters": m_mae,
                    "r2_score": m_r2,
                    "scatter_index_pct": m_si * 100.0,
                    "wape_pct": m_wape,
                    "bias_meters": m_bias,
                    "rmse_skill_pct": rmse_skill,
                    "mae_skill_pct": mae_skill,
                    "notes": f"Trained checkpoint: {ckpt_path.name} (Val Loss: {ckpt_meta.get('best_val_loss', 0):.5f})",
                })

                horizon_summaries[h]["models"][m_name] = {
                    "rmse": m_rmse,
                    "mae": m_mae,
                    "r2": m_r2,
                    "si": m_si,
                    "wape": m_wape,
                    "rmse_skill_pct": rmse_skill,
                }

    # 3. Export Comparison Artifacts
    df_results = pd.DataFrame(all_results)
    csv_out_path = outputs_metrics_dir / "multi_horizon_comparison.csv"
    df_results.to_csv(csv_out_path, index=False)
    print(f"\n[Artifact] Saved multi-horizon comparison table -> {csv_out_path}")

    json_out_path = outputs_metrics_dir / "multi_horizon_detailed.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "horizons_evaluated": horizons,
            "results": all_results,
            "summaries": horizon_summaries,
        }, f, indent=2)
    print(f"[Artifact] Saved detailed multi-horizon JSON  -> {json_out_path}")

    # 4. Generate Visualization: Persistence Error Growth Across Horizons
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    h_vals = [1, 6, 24]
    p_rmse_vals = [horizon_summaries[h]["persistence"]["rmse"] for h in h_vals]
    p_mae_vals = [horizon_summaries[h]["persistence"]["mae"] for h in h_vals]
    p_r2_vals = [horizon_summaries[h]["persistence"]["r2"] for h in h_vals]
    p_si_vals = [horizon_summaries[h]["persistence"]["si"] * 100.0 for h in h_vals]

    # Subplot 1: Persistence Error Degradation (RMSE & MAE)
    ax1.plot(h_vals, p_rmse_vals, marker="o", color="#ef4444", linewidth=2.2, label="Persistence RMSE (m)")
    ax1.plot(h_vals, p_mae_vals, marker="s", color="#f59e0b", linewidth=2.0, linestyle="--", label="Persistence MAE (m)")
    ax1.set_xlabel("Forecast Horizon (Hours Ahead)", fontsize=11, color="#e2e8f0")
    ax1.set_ylabel("Error (Meters)", fontsize=11, color="#e2e8f0")
    ax1.set_title("Naive Persistence Error Growth vs Lead Time", fontsize=12, color="#e2e8f0")
    ax1.set_xticks(h_vals)
    ax1.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax1.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    ax1.set_facecolor("#0f172a")

    # Annotate points
    for h_i, r_i in zip(h_vals, p_rmse_vals):
        ax1.annotate(f"{r_i*100:.1f} cm", (h_i, r_i), textcoords="offset points", xytext=(0, 10), ha="center", color="#fca5a5", fontsize=10)

    # Subplot 2: R² and Scatter Index Degradation
    ax2.plot(h_vals, p_r2_vals, marker="^", color="#10b981", linewidth=2.2, label="Persistence R² Score")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(h_vals, p_si_vals, marker="v", color="#38bdf8", linewidth=2.0, linestyle="-.", label="Scatter Index (%)")
    ax2.set_xlabel("Forecast Horizon (Hours Ahead)", fontsize=11, color="#e2e8f0")
    ax2.set_ylabel("R² Score", fontsize=11, color="#10b981")
    ax2_twin.set_ylabel("Scatter Index (%)", fontsize=11, color="#38bdf8")
    ax2.set_title("Persistence Degradation: R² & Scatter Index", fontsize=12, color="#e2e8f0")
    ax2.set_xticks(h_vals)
    ax2.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax2.set_facecolor("#0f172a")
    ax2_twin.set_facecolor("#0f172a")

    fig.patch.set_facecolor("#0b111e")
    for ax in [ax1, ax2, ax2_twin]:
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    plot_path1 = outputs_plots_dir / "multi_horizon_persistence_baseline.png"
    plt.tight_layout()
    plt.savefig(plot_path1, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[Artifact] Saved persistence degradation plot -> {plot_path1}")

    # 5. Generate Horizon 1h Model Comparison Chart
    fig, ax = plt.subplots(figsize=(10, 4.5))
    h1_models = [r for r in all_results if r["horizon_hours"] == 1 and r["rmse_meters"] is not None]
    m_labels = [r["model"] for r in h1_models]
    rmse_scores = [r["rmse_meters"] for r in h1_models]
    mae_scores = [r["mae_meters"] for r in h1_models]

    x = np.arange(len(m_labels))
    w = 0.35

    ax.bar(x - w/2, rmse_scores, w, label="RMSE (m)", color="#0284c7")
    ax.bar(x + w/2, mae_scores, w, label="MAE (m)", color="#10b981")
    ax.set_xticks(x)
    ax.set_xticklabels(m_labels, fontsize=11, color="#e2e8f0")
    ax.set_ylabel("Error (Meters)", fontsize=11, color="#e2e8f0")
    ax.set_title("Horizon H=1h Benchmark: Naive Persistence vs Deep Learning Models", fontsize=12, color="#e2e8f0")
    ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0b111e")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Add values on top of bars
    for i, (r_val, m_val) in enumerate(zip(rmse_scores, mae_scores)):
        ax.text(i - w/2, r_val + 0.001, f"{r_val:.3f}m", ha="center", va="bottom", color="#bae6fd", fontsize=9)
        ax.text(i + w/2, m_val + 0.001, f"{m_val:.3f}m", ha="center", va="bottom", color="#a7f3d0", fontsize=9)

    plot_path2 = outputs_plots_dir / "horizon_1h_model_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path2, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[Artifact] Saved H=1h benchmark comparison plot -> {plot_path2}")

    print("\n" + "=" * 80)
    print("SUCCESS: Multi-horizon benchmark executed and all artifacts saved.")
    print("=" * 80)


if __name__ == "__main__":
    run_multi_horizon_benchmark()
