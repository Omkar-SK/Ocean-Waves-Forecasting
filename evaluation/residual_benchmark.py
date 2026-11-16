"""
Residual Deep Learning Benchmark Script for OceanCast AI
=========================================================
Evaluates and compares all models on identical test-set timestamps:

Baselines:
  1. Naive Persistence (SWH(t))
  2. MA-6h (mean of last 6 anchor observations)
  3. MA-24h (mean of last 24 anchor observations)

Absolute models (from models/):
  4. WaveANN    absolute
  5. WaveLSTM   absolute
  6. WaveCNN1D  absolute

Residual models (from models/residual/):
  7. WaveANN    residual (ΔSWH)
  8. WaveLSTM   residual (ΔSWH)
  9. WaveCNN1D  residual (ΔSWH)

Outputs:
  - evaluation/residual_results.json
  - Console table

Usage:
    py -3.11 evaluation/residual_benchmark.py
"""

import sys
import os
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.models import build_wave_model
from src.data.dataset import TimeSeriesScaler

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("residual_bench")

# ── Config ───────────────────────────────────────────────────────────────────
HORIZONS     = [1, 6, 24]
MODEL_TYPES  = ["ann", "lstm", "cnn1d"]
SEQ_LEN      = 24
BATCH_SIZE   = 1024
DEVICE       = torch.device("cpu")

DATA_PATH        = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"
ORIG_SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
MODELS_DIR       = PROJECT_ROOT / "models"
RESIDUAL_DIR     = PROJECT_ROOT / "models" / "residual"
OUTPUT_PATH      = PROJECT_ROOT / "evaluation" / "residual_results.json"

FEATURE_COLS = [
    "swh", "mwp", "mwd_sin", "mwd_cos",
    "u10", "v10", "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]


# ── Metric functions ──────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Compute comprehensive regression metrics. No NaN filtering on skill."""
    residuals = y_true - y_pred
    mae  = float(np.mean(np.abs(residuals)))
    mse  = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Explained variance
    var_res = float(np.var(residuals))
    var_tot = float(np.var(y_true))
    explained_var = float(1 - var_res / var_tot) if var_tot > 0 else float("nan")

    # MAPE (filtered: only where y_true > 0.05 m)
    mask_filtered = y_true > 0.05
    if mask_filtered.sum() > 0:
        mape_filtered = float(
            100 * np.mean(np.abs(residuals[mask_filtered]) / y_true[mask_filtered])
        )
    else:
        mape_filtered = float("nan")

    # MAPE unfiltered
    with np.errstate(divide="ignore", invalid="ignore"):
        mape_raw = np.where(y_true != 0, np.abs(residuals) / np.abs(y_true), np.nan)
    mape_unfiltered = float(np.nanmean(mape_raw) * 100)

    return {
        "mae":               mae,
        "mse":               mse,
        "rmse":              rmse,
        "r2":                r2,
        "explained_var":     explained_var,
        "mape_filtered":     mape_filtered,
        "mape_unfiltered":   mape_unfiltered,
        "n":                 int(len(y_true)),
    }


def skill_score(rmse_model: float, rmse_ref: float) -> float:
    """RMSE Skill Score = (1 - RMSE_model / RMSE_ref) * 100. Negative = worse than ref."""
    if rmse_ref == 0:
        return float("nan")
    return float((1 - rmse_model / rmse_ref) * 100)


def mae_skill_score(mae_model: float, mae_ref: float) -> float:
    if mae_ref == 0:
        return float("nan")
    return float((1 - mae_model / mae_ref) * 100)


# ── Model inference helpers ───────────────────────────────────────────────────
@torch.no_grad()
def run_model_inference(
    model,
    feature_scaled: np.ndarray,
    anchor_indices: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Return raw model output (scaled) for all anchors."""
    model.eval()
    preds = []
    for start in range(0, len(anchor_indices), batch_size):
        batch_anchors = anchor_indices[start : start + batch_size]
        windows = np.stack([
            feature_scaled[t - SEQ_LEN + 1 : t + 1] for t in batch_anchors
        ])
        xb = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
        out = model(xb).cpu().numpy().flatten()
        preds.append(out)
    return np.concatenate(preds)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("OceanCast AI — Residual vs Absolute Model Benchmark")
    log.info("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    log.info(f"Loading data from {DATA_PATH} …")
    df = pd.read_parquet(DATA_PATH)
    if "hour_sin" not in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    n = len(df)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    swh_raw = df["swh"].values.astype(np.float32)
    log.info(f"  Total rows: {n:,} | test offset: {n_train + n_val:,}")

    # ── Load original feature scaler ─────────────────────────────────────────
    log.info(f"Loading original scaler from {ORIG_SCALER_PATH} …")
    with open(ORIG_SCALER_PATH, "rb") as f:
        orig_scaler: TimeSeriesScaler = pickle.load(f)

    feature_scaled, _ = orig_scaler.transform(df)   # [T, F]

    # ── Collect results ───────────────────────────────────────────────────────
    all_results = {}
    dataset_info = {
        "total_rows":     n,
        "n_train":        n_train,
        "n_val":          n_val,
        "n_test":         n - n_train - n_val,
        "feature_cols":   FEATURE_COLS,
        "seq_len":        SEQ_LEN,
        "swh_test_mean":  None,  # filled per horizon below
    }

    for H in HORIZONS:
        log.info("\n" + "=" * 70)
        log.info(f"HORIZON +{H}h")
        log.info("=" * 70)

        min_anchor = SEQ_LEN - 1
        max_anchor = n - H - 1
        all_valid  = np.arange(min_anchor, max_anchor + 1)

        test_mask    = all_valid >= n_train + n_val
        test_anchors = all_valid[test_mask]
        log.info(f"  Test anchors: {len(test_anchors):,}")

        # True targets in physical space
        y_true = swh_raw[test_anchors + H]
        log.info(f"  y_true stats — mean={y_true.mean():.4f} | std={y_true.std():.4f} | min={y_true.min():.4f} | max={y_true.max():.4f}")

        horizon_results = {}

        # ── 1. Naive Persistence ─────────────────────────────────────────────
        y_persist = swh_raw[test_anchors]
        m = compute_metrics(y_true, y_persist, "persistence")
        horizon_results["persistence"] = m
        log.info(f"  Persistence     | MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} | R²={m['r2']:.4f}")

        persist_rmse = m["rmse"]
        persist_mae  = m["mae"]

        # ── 2. MA-6h ─────────────────────────────────────────────────────────
        y_ma6 = np.array([
            swh_raw[max(0, t - 5) : t + 1].mean() for t in test_anchors
        ])
        m = compute_metrics(y_true, y_ma6)
        m["rmse_skill"] = skill_score(m["rmse"], persist_rmse)
        m["mae_skill"]  = mae_skill_score(m["mae"], persist_mae)
        horizon_results["ma6h"] = m
        log.info(f"  MA-6h           | MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} | R²={m['r2']:.4f} | Skill={m['rmse_skill']:+.2f}%")

        # ── 3. MA-24h ────────────────────────────────────────────────────────
        y_ma24 = np.array([
            swh_raw[max(0, t - 23) : t + 1].mean() for t in test_anchors
        ])
        m = compute_metrics(y_true, y_ma24)
        m["rmse_skill"] = skill_score(m["rmse"], persist_rmse)
        m["mae_skill"]  = mae_skill_score(m["mae"], persist_mae)
        horizon_results["ma24h"] = m
        log.info(f"  MA-24h          | MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} | R²={m['r2']:.4f} | Skill={m['rmse_skill']:+.2f}%")

        # ── 4–6. Absolute models ──────────────────────────────────────────────
        abs_ckpt_names = {
            "ann":   f"best_ann_h{H}.pt",
            "lstm":  f"best_lstm_h{H}.pt",
            "cnn1d": f"best_cnn1d_h{H}.pt",
        }
        # Naming: +1h = best_ann.pt, +6h/+24h = best_ann_h6.pt / best_ann_h24.pt
        h_suffix = "" if H == 1 else f"_h{H}"
        alt_patterns = {
            "ann":   [f"best_ann{h_suffix}.pt", f"best_ann_h{H}.pt", f"ann_h{H}_best.pt"],
            "lstm":  [f"best_lstm{h_suffix}.pt", f"best_lstm_h{H}.pt", f"lstm_h{H}_best.pt"],
            "cnn1d": [f"best_cnn1d{h_suffix}.pt", f"best_cnn1d_h{H}.pt", f"cnn1d_h{H}_best.pt"],
        }

        for m_type in MODEL_TYPES:
            # Find checkpoint
            ckpt_path = None
            for pattern in alt_patterns[m_type]:
                candidate = MODELS_DIR / pattern
                if candidate.exists():
                    ckpt_path = candidate
                    break

            key = f"abs_{m_type}"
            if ckpt_path is None:
                log.warning(f"  {m_type.upper()} absolute h{H}: checkpoint not found, skipping")
                horizon_results[key] = {"error": "checkpoint_not_found"}
                continue

            log.info(f"  Loading absolute {m_type.upper()} from {ckpt_path.name} …")
            model = build_wave_model(m_type, input_dim=len(FEATURE_COLS), seq_len=SEQ_LEN).to(DEVICE)
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            # Original checkpoints may be wrapped with model_state_dict key
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)

            # Inference: raw scaled output → inverse-transform target
            pred_scaled = run_model_inference(model, feature_scaled, test_anchors)
            y_pred = orig_scaler.inverse_transform_target(pred_scaled.reshape(-1, 1)).flatten()

            m = compute_metrics(y_true, y_pred)
            m["rmse_skill"] = skill_score(m["rmse"], persist_rmse)
            m["mae_skill"]  = mae_skill_score(m["mae"], persist_mae)
            m["checkpoint"] = str(ckpt_path.name)
            horizon_results[key] = m
            log.info(
                f"  Abs {m_type.upper():6s}      | MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} "
                f"| R²={m['r2']:.4f} | Skill={m['rmse_skill']:+.2f}%"
            )

        # ── 7–9. Residual models ──────────────────────────────────────────────
        # Load residual scaler for this horizon
        res_scaler_path = RESIDUAL_DIR / f"residual_scaler_h{H}.pkl"
        if not res_scaler_path.exists():
            log.warning(f"  Residual scaler not found: {res_scaler_path} — run train_residual_models.py first")
            for m_type in MODEL_TYPES:
                horizon_results[f"res_{m_type}"] = {"error": "residual_scaler_not_found"}
        else:
            with open(res_scaler_path, "rb") as f:
                res_scaler_info: dict = pickle.load(f)
            rt_mean = res_scaler_info["mean"]
            rt_std  = res_scaler_info["std"]
            log.info(f"  Residual scaler h{H}: mean={rt_mean:.4f}m std={rt_std:.4f}m")

            swh_anchor_test = swh_raw[test_anchors]

            for m_type in MODEL_TYPES:
                res_ckpt = RESIDUAL_DIR / f"best_{m_type}_residual_h{H}.pt"
                key = f"res_{m_type}"

                if not res_ckpt.exists():
                    log.warning(f"  {m_type.upper()} residual h{H}: checkpoint not found")
                    horizon_results[key] = {"error": "checkpoint_not_found"}
                    continue

                log.info(f"  Loading residual {m_type.upper()} from {res_ckpt.name} …")
                model = build_wave_model(m_type, input_dim=len(FEATURE_COLS), seq_len=SEQ_LEN).to(DEVICE)
                state = torch.load(res_ckpt, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state)

                # Inference → scaled delta → unscale → reconstruct SWH
                pred_delta_scaled = run_model_inference(model, feature_scaled, test_anchors)
                # Inverse-standardize residual
                pred_delta_m = pred_delta_scaled * rt_std + rt_mean
                # Reconstruct absolute SWH
                y_pred = swh_anchor_test + pred_delta_m

                m = compute_metrics(y_true, y_pred)
                m["rmse_skill"] = skill_score(m["rmse"], persist_rmse)
                m["mae_skill"]  = mae_skill_score(m["mae"], persist_mae)
                m["checkpoint"] = str(res_ckpt.name)
                horizon_results[key] = m
                log.info(
                    f"  Res {m_type.upper():6s}      | MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} "
                    f"| R²={m['r2']:.4f} | Skill={m['rmse_skill']:+.2f}%"
                )

        # Add persistence skill (reference = 0%)
        horizon_results["persistence"]["rmse_skill"] = 0.0
        horizon_results["persistence"]["mae_skill"]  = 0.0

        all_results[f"{H}h"] = horizon_results

    # ── Save JSON ─────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset":       dataset_info,
        "horizons":      all_results,
        "notes": [
            "Positive skill score = better than naive persistence.",
            "Negative skill score = worse than naive persistence.",
            "Skill scores are NOT clamped; negative values are retained.",
            "Absolute models use models/scaler.pkl for both input and output inverse-transform.",
            "Residual models use models/scaler.pkl for input features; residual_scaler_h{H}.pkl for output.",
            "Reconstruction: SWH_pred(t+H) = SWH(t) + Predicted_ΔSWH",
        ],
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\n✓ Results saved → {OUTPUT_PATH}")

    # ── Print summary table ───────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("BENCHMARK SUMMARY")
    log.info("=" * 70)
    model_keys = [
        ("persistence", "Persistence"),
        ("ma6h",        "MA-6h"),
        ("ma24h",       "MA-24h"),
        ("abs_ann",     "Abs ANN"),
        ("abs_lstm",    "Abs LSTM"),
        ("abs_cnn1d",   "Abs CNN1D"),
        ("res_ann",     "Res ANN"),
        ("res_lstm",    "Res LSTM"),
        ("res_cnn1d",   "Res CNN1D"),
    ]
    for H in HORIZONS:
        hkey = f"{H}h"
        log.info(f"\n  +{H}h Horizon:")
        log.info(f"  {'Model':<14} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Skill%':>9}")
        log.info(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
        for key, label in model_keys:
            rec = all_results[hkey].get(key, {})
            if "error" in rec:
                log.info(f"  {label:<14} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>9}")
                continue
            mae   = rec.get("mae", float("nan"))
            rmse  = rec.get("rmse", float("nan"))
            r2    = rec.get("r2", float("nan"))
            skill = rec.get("rmse_skill", 0.0)
            log.info(f"  {label:<14} {mae:8.4f} {rmse:8.4f} {r2:8.4f} {skill:+9.2f}%")

    log.info("\n✓ Benchmark complete.")


if __name__ == "__main__":
    main()
