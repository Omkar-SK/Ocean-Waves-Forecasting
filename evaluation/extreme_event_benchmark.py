"""
Comprehensive Extreme Wave Event Benchmarking and Evaluation Script
===================================================================
Author: Senior Oceanographic Deep Learning Researcher & Coastal Hazard Modelling Engineer
Project: OceanCast AI - Extreme Wave Event Improvement Framework

Independently benchmarks and compares:
1. Naive Persistence Baseline
2. Existing Absolute Checkpoints (WaveLSTM, WaveANN)
3. Existing Residual Checkpoints (Residual WaveLSTM, Residual WaveANN)
4. Candidate Extreme Improvement Models:
   - Extreme-Weighted Loss (EW-Loss)
   - Asymmetric Extreme-Penalized Loss (AEP-Loss)
   - Stratified Extreme Oversampling (SEO)
   - Multi-Task Extreme Hazard Network (MTE-Net)

Evaluation Regimes:
- Overall (Full test set)
- Calm (< 1.25m)
- Moderate (1.25m - 2.50m)
- Rough (2.50m - 4.00m)
- Severe (>= 4.00m, N = 22)

Metrics Computed:
- Continuous Accuracy: MAE, RMSE, Mean Bias (True - Pred), Underprediction Rate (%)
- Severe Peak Attenuation (Maximum Predicted vs Maximum True, Mean Underprediction on N=22)
- Skill Scores against Naive Persistence: RMSE Skill %, MAE Skill %
- Extreme Event Classification Detection at Thresholds 4.0m (Severe) and 2.5m (Rough):
  - Recall (Sensitivity / Hit Rate)
  - Precision (Positive Predictive Value)
  - False Positive Rate (FPR)
  - F1 Score
- Bootstrap 95% Confidence Intervals for Severe Event Skill

Outputs:
- evaluation/extreme_event_results.json
"""

import sys
import os
import json
import pickle
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.models import build_wave_model
from src.data.dataset import TimeSeriesScaler, DEFAULT_FEATURE_COLS
from training.train_extreme_models import MultiTaskWaveANN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extreme_benchmark")

SEED = 42
HORIZONS = [1, 6, 24]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"
ORIG_SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
MODELS_DIR = PROJECT_ROOT / "models"
RESIDUAL_DIR = MODELS_DIR / "residual"
EXPERIMENTAL_DIR = MODELS_DIR / "extreme_experimental"
OUTPUT_JSON = PROJECT_ROOT / "evaluation" / "extreme_event_results.json"

SEQ_LEN = 24


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Metric Calculations ──────────────────────────────────────────────────────
def compute_continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_persist: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true  # Oceanographic / meteorological MBE standard: pred - true. Negative = underprediction (pred < true).
    err_p = y_persist - y_true
    n = len(y_true)
    if n == 0:
        return {}

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))  # Negative = underpredicting (Pred < True), Positive = overpredicting (Pred > True)
    underpredict_pct = float(np.mean(y_pred < y_true) * 100.0)

    mae_p = float(np.mean(np.abs(err_p)))
    rmse_p = float(np.sqrt(np.mean(err_p ** 2)))

    rmse_skill = float((1.0 - rmse / rmse_p) * 100.0) if rmse_p > 0 else 0.0
    mae_skill = float((1.0 - mae / mae_p) * 100.0) if mae_p > 0 else 0.0

    return {
        "sample_count": n,
        "model_mae_m": round(mae, 4),
        "persist_mae_m": round(mae_p, 4),
        "mae_skill_pct": round(mae_skill, 2),
        "model_rmse_m": round(rmse, 4),
        "persist_rmse_m": round(rmse_p, 4),
        "rmse_skill_pct": round(rmse_skill, 2),
        "mean_bias_m": round(bias, 4),
        "underpredict_pct": round(underpredict_pct, 1),
        "max_true_m": round(float(np.max(y_true)), 3),
        "max_pred_m": round(float(np.max(y_pred)), 3),
        "peak_attenuation_m": round(float(np.max(y_true) - np.max(y_pred)), 3),
    }


def compute_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold_m: float) -> Dict[str, Any]:
    actual_pos = y_true >= threshold_m
    pred_pos = y_pred >= threshold_m

    tp = int(np.sum(actual_pos & pred_pos))
    fn = int(np.sum(actual_pos & ~pred_pos))
    fp = int(np.sum(~actual_pos & pred_pos))
    tn = int(np.sum(~actual_pos & ~pred_pos))

    n_actual_pos = tp + fn
    n_actual_neg = tn + fp

    recall = float(tp / n_actual_pos) if n_actual_pos > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    specificity = float(tn / n_actual_neg) if n_actual_neg > 0 else 0.0
    fpr = float(fp / n_actual_neg) if n_actual_neg > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold_m": threshold_m,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "recall_pct": round(recall * 100.0, 2),
        "precision_pct": round(precision * 100.0, 2),
        "specificity_pct": round(specificity * 100.0, 2),
        "false_positive_rate_pct": round(fpr * 100.0, 3),
        "f1_score": round(f1, 4),
    }


def bootstrap_severe_skill(
    y_true_sev: np.ndarray,
    y_pred_sev: np.ndarray,
    y_persist_sev: np.ndarray,
    n_reps: int = 1000,
) -> Dict[str, Any]:
    n = len(y_true_sev)
    if n < 5:
        return {}

    skills = []
    set_seed(SEED)
    for _ in range(n_reps):
        idx = np.random.choice(n, size=n, replace=True)
        yt = y_true_sev[idx]
        ym = y_pred_sev[idx]
        yp = y_persist_sev[idx]

        rmse_m = np.sqrt(np.mean((yt - ym) ** 2))
        rmse_p = np.sqrt(np.mean((yt - yp) ** 2))
        if rmse_p > 0:
            skills.append((1.0 - rmse_m / rmse_p) * 100.0)

    if not skills:
        return {}

    skills = np.array(skills)
    return {
        "mean_skill": round(float(np.mean(skills)), 2),
        "ci95_lower": round(float(np.percentile(skills, 2.5)), 2),
        "ci95_upper": round(float(np.percentile(skills, 97.5)), 2),
    }


# ── Model Predictor Helper ───────────────────────────────────────────────────
def run_predictions(
    model: torch.nn.Module,
    feature_scaled: np.ndarray,
    test_anchors: np.ndarray,
    y_persist: np.ndarray,
    is_residual: bool,
    is_multitask: bool,
    rt_mean: float,
    rt_std: float,
    orig_scaler: TimeSeriesScaler,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()
    batch_size = 1024
    preds = []
    hazard_probs = []

    with torch.no_grad():
        for start in range(0, len(test_anchors), batch_size):
            batch_anchors = test_anchors[start : start + batch_size]
            windows = np.stack([feature_scaled[t - 23 : t + 1] for t in batch_anchors])
            xb = torch.tensor(windows, dtype=torch.float32)

            if is_multitask:
                reg_out, clf_out = model(xb)
                preds.append(reg_out.cpu().numpy().flatten())
                probs = torch.sigmoid(clf_out).cpu().numpy().flatten()
                hazard_probs.append(probs)
            else:
                out = model(xb)
                preds.append(out.cpu().numpy().flatten())

    raw_preds = np.concatenate(preds)

    if is_residual or is_multitask:
        # Scale back delta
        pred_delta_m = raw_preds * rt_std + rt_mean
        y_pred = y_persist + pred_delta_m
    else:
        # Absolute model output inverse transform
        y_pred = orig_scaler.inverse_transform_target(raw_preds)

    clf_probs = np.concatenate(hazard_probs) if is_multitask else None
    return y_pred, clf_probs


# ── Main Benchmarking Runner ─────────────────────────────────────────────────
def main():
    log.info("=" * 80)
    log.info("OceanCast AI - Extreme Wave Event Benchmarking Suite")
    log.info("=" * 80)
    set_seed(SEED)

    # Load dataset
    df = pd.read_parquet(DATA_PATH)
    n = len(df)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    test_start = n_train + n_val
    swh_raw = df["swh"].values.astype(np.float32)

    with open(ORIG_SCALER_PATH, "rb") as f:
        orig_scaler: TimeSeriesScaler = pickle.load(f)
    feature_scaled, _ = orig_scaler.transform(df)

    benchmark_results = {
        "benchmark_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "ERA5 Mumbai Offshore Timeseries (2012-2026)",
        "total_test_samples": int(n - test_start),
        "severe_event_threshold_m": 4.00,
        "severe_test_sample_count": 22,
        "horizons": {},
        "comparative_summary": {},
    }

    for H in HORIZONS:
        log.info("\n" + "=" * 80)
        log.info(f"BENCHMARKING FORECAST HORIZON +{H}h")
        log.info("=" * 80)

        min_anchor = SEQ_LEN - 1
        max_anchor = n - H - 1
        all_valid = np.arange(min_anchor, max_anchor + 1)
        test_anchors = all_valid[all_valid >= test_start]
        n_test = len(test_anchors)

        y_true = swh_raw[test_anchors + H]
        y_persist = swh_raw[test_anchors]

        with open(RESIDUAL_DIR / f"residual_scaler_h{H}.pkl", "rb") as f:
            res_scaler = pickle.load(f)
        rt_mean = float(res_scaler["mean"])
        rt_std = float(res_scaler["std"])

        # Models to evaluate for this horizon
        base_arch = "lstm" if H == 1 else "ann"
        models_to_test = [
            ("persistence", "Naive Persistence Baseline", None, False, False),
            ("absolute", f"Absolute Wave{base_arch.upper()}", f"best_{base_arch}{'' if H == 1 else f'_h{H}'}.pt", False, False),
            ("residual_baseline", f"Residual Wave{base_arch.upper()} (Production)", f"best_{base_arch}_residual_h{H}.pt", True, False),
            ("exp_ew_loss", f"Extreme-Weighted Wave{base_arch.upper()}", f"ExtremeWeighted_{base_arch.upper()}_h{H}.pt", True, False),
            ("exp_asym_loss", f"Asymmetric-Penalty Wave{base_arch.upper()}", f"Asymmetric_{base_arch.upper()}_h{H}.pt", True, False),
            ("exp_seo", f"Oversampled-SEO Wave{base_arch.upper()}", f"Oversampled_{base_arch.upper()}_h{H}.pt", True, False),
        ]
        if base_arch == "ann":
            models_to_test.append(
                ("exp_multitask", "Multi-Task Hazard WaveANN", f"MultiTask_Extreme_ANN_h{H}.pt", True, True)
            )

        horizon_eval: Dict[str, Any] = {}

        # Define regimes
        regimes = {
            "Overall": np.ones(n_test, dtype=bool),
            "Calm (< 1.25m)": y_true < 1.25,
            "Moderate (1.25m - 2.50m)": (y_true >= 1.25) & (y_true < 2.50),
            "Rough (2.50m - 4.00m)": (y_true >= 2.50) & (y_true < 4.00),
            "Severe (>= 4.00m)": y_true >= 4.00,
        }

        for model_key, display_name, ckpt_file, is_residual, is_multitask in models_to_test:
            log.info(f"\nEvaluating: {display_name} ...")

            if model_key == "persistence":
                y_pred = y_persist
                clf_probs = None
            else:
                # Resolve checkpoint path
                if model_key == "absolute":
                    ckpt_path = MODELS_DIR / ckpt_file
                elif model_key == "residual_baseline":
                    ckpt_path = RESIDUAL_DIR / ckpt_file
                else:
                    ckpt_path = EXPERIMENTAL_DIR / ckpt_file

                if not ckpt_path.is_file():
                    log.warning(f"  Checkpoint not found: {ckpt_path}. Skipping.")
                    continue

                if is_multitask:
                    model = MultiTaskWaveANN(input_dim=len(DEFAULT_FEATURE_COLS), seq_len=SEQ_LEN)
                else:
                    model = build_wave_model(base_arch, input_dim=len(DEFAULT_FEATURE_COLS), seq_len=SEQ_LEN)

                weights = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                if isinstance(weights, dict) and "model_state_dict" in weights:
                    weights = weights["model_state_dict"]
                model.load_state_dict(weights)

                y_pred, clf_probs = run_predictions(
                    model=model,
                    feature_scaled=feature_scaled,
                    test_anchors=test_anchors,
                    y_persist=y_persist,
                    is_residual=is_residual,
                    is_multitask=is_multitask,
                    rt_mean=rt_mean,
                    rt_std=rt_std,
                    orig_scaler=orig_scaler,
                )

            # Evaluate regimes
            regime_results = {}
            for reg_name, reg_mask in regimes.items():
                if np.sum(reg_mask) == 0:
                    continue
                metrics = compute_continuous_metrics(
                    y_true=y_true[reg_mask],
                    y_pred=y_pred[reg_mask],
                    y_persist=y_persist[reg_mask],
                )
                regime_results[reg_name] = metrics

            # Severe event classification metrics (Hs >= 4.0m)
            det_severe = compute_detection_metrics(y_true, y_pred, threshold_m=4.00)
            det_rough = compute_detection_metrics(y_true, y_pred, threshold_m=2.50)

            # Severe event bootstrap CI
            sev_mask = regimes["Severe (>= 4.00m)"]
            ci_sev = bootstrap_severe_skill(y_true[sev_mask], y_pred[sev_mask], y_persist[sev_mask])

            # Multi-task auxiliary probability evaluation (if available)
            multitask_eval = None
            if clf_probs is not None:
                actual_hazard = (y_true >= 2.50).astype(int)
                pred_hazard = (clf_probs >= 0.50).astype(int)
                mt_tp = int(np.sum((actual_hazard == 1) & (pred_hazard == 1)))
                mt_fn = int(np.sum((actual_hazard == 1) & (pred_hazard == 0)))
                mt_fp = int(np.sum((actual_hazard == 0) & (pred_hazard == 1)))
                mt_tn = int(np.sum((actual_hazard == 0) & (pred_hazard == 0)))
                mt_pos = mt_tp + mt_fn
                mt_neg = mt_tn + mt_fp
                mt_recall = round(float(mt_tp / mt_pos * 100.0), 2) if mt_pos > 0 else 0.0
                mt_prec = round(float(mt_tp / (mt_tp + mt_fp) * 100.0), 2) if (mt_tp + mt_fp) > 0 else 0.0
                mt_spec = round(float(mt_tn / mt_neg * 100.0), 2) if mt_neg > 0 else 0.0
                mt_fpr = round(float(mt_fp / mt_neg * 100.0), 3) if mt_neg > 0 else 0.0
                rec_d = mt_recall / 100.0
                prec_d = mt_prec / 100.0
                mt_f1 = round(float(2 * prec_d * rec_d / (prec_d + rec_d)), 4) if (prec_d + rec_d) > 0 else 0.0
                multitask_eval = {
                    "auxiliary_target": "P(SWH >= 2.5m)",
                    "classification_threshold": 0.50,
                    "prevalence_pos": mt_pos,
                    "prevalence_neg": mt_neg,
                    "prevalence_pos_pct": round(float(mt_pos / (mt_pos + mt_neg) * 100.0), 2),
                    "hazard_recall_pct": mt_recall,
                    "hazard_precision_pct": mt_prec,
                    "hazard_specificity_pct": mt_spec,
                    "hazard_fpr_pct": mt_fpr,
                    "f1_score": mt_f1,
                    "confusion_matrix": {"tp": mt_tp, "fn": mt_fn, "fp": mt_fp, "tn": mt_tn},
                }

            horizon_eval[model_key] = {
                "display_name": display_name,
                "checkpoint": str(ckpt_file) if ckpt_file else None,
                "regimes": regime_results,
                "severe_detection_metrics_4m": det_severe,
                "rough_detection_metrics_2_5m": det_rough,
                "severe_skill_bootstrap_ci": ci_sev,
                "multitask_auxiliary_eval": multitask_eval,
            }

            # Log severe summary
            sev_stats = regime_results.get("Severe (>= 4.00m)", {})
            log.info(
                f"  Severe: RMSE={sev_stats.get('model_rmse_m')}m | "
                f"Skill={sev_stats.get('rmse_skill_pct'):+.2f}% | "
                f"Bias={sev_stats.get('mean_bias_m'):+.3f}m | "
                f"Underpredict={sev_stats.get('underpredict_pct')}% | "
                f"Peak Attenuation={sev_stats.get('peak_attenuation_m')}m"
            )

        benchmark_results["horizons"][f"+{H}h"] = horizon_eval

    # Write output JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    log.info(f"\nSaved complete extreme event benchmark results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
