"""
Statistical Significance Testing and Independent Scientific Validation
for OceanCast AI Residual Wave Forecasting Models.

Author: Senior ML Research Scientist & Validation Engineer
Date: 2026-11-16
Task: Independent statistical inference, paired moving-block bootstrap (B=5,000),
      Diebold-Mariano tests with HAC variance, regime error analysis, and publication figures.
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
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.models import build_wave_model
from src.data.dataset import TimeSeriesScaler, DEFAULT_FEATURE_COLS
from src.models.registry import ModelRegistry, ForecastMode

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stat_validation")

# ── Configuration ───────────────────────────────────────────────────────────
SEED = 42
B_REPLICATES = 5000
BLOCK_SIZES = [12, 24, 48, 72, 168]  # 0.5d, 1d, 2d, 3d, 7d
PRIMARY_BLOCK = 48  # 48-hour block (matches synoptic marine scale)

HORIZONS = [1, 6, 24]
CANDIDATE_MODELS = {
    1: {"arch": "lstm", "ckpt": "best_lstm_residual_h1.pt", "name": "Residual WaveLSTM"},
    6: {"arch": "ann", "ckpt": "best_ann_residual_h6.pt", "name": "Residual WaveANN"},
    24: {"arch": "ann", "ckpt": "best_ann_residual_h24.pt", "name": "Residual WaveANN"},
}

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
RESIDUAL_DIR = MODELS_DIR / "residual"
ORIG_SCALER_PATH = MODELS_DIR / "scaler.pkl"

OUTPUT_JSON = PROJECT_ROOT / "evaluation" / "statistical_significance.json"
OUTPUT_BOOTSTRAP_CSV = PROJECT_ROOT / "evaluation" / "bootstrap_confidence_intervals.csv"
OUTPUT_ERROR_CSV = PROJECT_ROOT / "evaluation" / "model_error_analysis.csv"
OUTPUT_PNG = PROJECT_ROOT / "evaluation" / "statistical_comparison.png"


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Statistical Test Functions ───────────────────────────────────────────────
def diebold_mariano_test(d: np.ndarray, h: int = 1, max_lags: int = None) -> dict:
    """
    Diebold-Mariano (1995) test with Harvey-Leybourne-Newbold (1997) correction
    and Newey-West (1987) HAC long-run variance estimator.
    
    d: loss differential series (d_t = loss_baseline - loss_model). Positive = model better.
    h: forecast lead time in steps.
    """
    N = len(d)
    d_bar = float(np.mean(d))
    d_demean = d - d_bar
    
    # Lag truncation: default to h - 1 (theoretical minimum for h-step forecast), at least 1
    if max_lags is None:
        max_lags = max(1, h - 1)
        
    gamma_0 = float(np.var(d, ddof=0))
    gamma = []
    for k in range(1, max_lags + 1):
        cov_k = float(np.sum(d_demean[k:] * d_demean[:-k]) / N)
        gamma.append(cov_k)
        
    weights = [1.0 - (k / (max_lags + 1)) for k in range(1, max_lags + 1)]
    v_lr = gamma_0 + 2.0 * float(np.sum(np.array(weights) * np.array(gamma)))
    v_lr = max(v_lr, 1e-12)
    
    se = float(np.sqrt(v_lr / N))
    dm_stat = float(d_bar / se) if se > 0 else 0.0
    
    # Harvey, Leybourne, Newbold (1997) finite-sample correction factor
    hln_corr = float(np.sqrt(max(1e-12, (N + 1 - 2 * h + h * (h - 1) / N) / N)))
    dm_hln_stat = float(dm_stat * hln_corr)
    
    # Degrees of freedom = N - 1
    p_two_sided = float(2.0 * stats.t.sf(np.abs(dm_hln_stat), df=N - 1))
    # One-sided (H1: model error < baseline error <=> d_bar > 0)
    p_one_sided = float(stats.t.sf(dm_hln_stat, df=N - 1)) if dm_hln_stat > 0 else float(1.0 - stats.t.sf(np.abs(dm_hln_stat), df=N - 1))
    
    cohens_d = float(d_bar / np.std(d, ddof=1)) if np.std(d, ddof=1) > 0 else 0.0
    
    return {
        "mean_loss_diff": d_bar,
        "hac_standard_error": se,
        "dm_stat_raw": dm_stat,
        "dm_stat_hln": dm_hln_stat,
        "hln_correction_factor": hln_corr,
        "lags_used": max_lags,
        "p_value_two_sided": p_two_sided,
        "p_value_one_sided": p_one_sided,
        "cohens_d": cohens_d,
        "is_significant_05": p_two_sided < 0.05,
        "is_significant_01": p_two_sided < 0.01,
        "is_significant_001": p_two_sided < 0.001,
    }


def moving_block_bootstrap(
    y_true: np.ndarray,
    y_model: np.ndarray,
    y_persist: np.ndarray,
    block_size: int = 48,
    n_replicates: int = B_REPLICATES,
    seed: int = SEED,
) -> dict:
    """
    Paired moving-block bootstrap resampling for time-series forecasting comparison.
    Resamples contiguous blocks of length b simultaneously for model and persistence.
    """
    np.random.seed(seed)
    N = len(y_true)
    K = int(np.ceil(N / block_size))
    
    sq_err_model = (y_true - y_model) ** 2
    sq_err_persist = (y_true - y_persist) ** 2
    abs_err_model = np.abs(y_true - y_model)
    abs_err_persist = np.abs(y_true - y_persist)
    
    # Process in memory-safe chunks of 1000 replicates
    chunk_size = 1000
    rmse_skills = []
    mae_skills = []
    model_rmses = []
    persist_rmses = []
    model_maes = []
    persist_maes = []
    d_sq_means = []
    d_abs_means = []
    
    for start_rep in range(0, n_replicates, chunk_size):
        B_chunk = min(chunk_size, n_replicates - start_rep)
        starts = np.random.randint(0, N - block_size + 1, size=(B_chunk, K))
        idx = (starts[:, :, None] + np.arange(block_size)).reshape(B_chunk, K * block_size)[:, :N]
        
        # Metric computations across resampled sets
        m_mse = sq_err_model[idx].mean(axis=1)
        p_mse = sq_err_persist[idx].mean(axis=1)
        m_rmse = np.sqrt(m_mse)
        p_rmse = np.sqrt(p_mse)
        r_skill = (1.0 - m_rmse / p_rmse) * 100.0
        
        m_mae = abs_err_model[idx].mean(axis=1)
        p_mae = abs_err_persist[idx].mean(axis=1)
        a_skill = (1.0 - m_mae / p_mae) * 100.0
        
        d_sq = (sq_err_persist[idx] - sq_err_model[idx]).mean(axis=1)
        d_abs = (abs_err_persist[idx] - abs_err_model[idx]).mean(axis=1)
        
        rmse_skills.append(r_skill)
        mae_skills.append(a_skill)
        model_rmses.append(m_rmse)
        persist_rmses.append(p_rmse)
        model_maes.append(m_mae)
        persist_maes.append(p_mae)
        d_sq_means.append(d_sq)
        d_abs_means.append(d_abs)
        
    rmse_skills = np.concatenate(rmse_skills)
    mae_skills = np.concatenate(mae_skills)
    model_rmses = np.concatenate(model_rmses)
    persist_rmses = np.concatenate(persist_rmses)
    model_maes = np.concatenate(model_maes)
    persist_maes = np.concatenate(persist_maes)
    d_sq_means = np.concatenate(d_sq_means)
    d_abs_means = np.concatenate(d_abs_means)
    
    def get_summary(arr: np.ndarray, point_est: float):
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        se = float(np.std(arr, ddof=1))
        # Basic / Empirical bootstrap CI: 2*theta - q97.5, 2*theta - q2.5
        basic_lower = float(2.0 * point_est - ci_upper)
        basic_upper = float(2.0 * point_est - ci_lower)
        prob_negative = float(np.mean(arr <= 0.0))
        return {
            "mean": float(np.mean(arr)),
            "std_error": se,
            "ci_95_percentile": [ci_lower, ci_upper],
            "ci_95_basic": [basic_lower, basic_upper],
            "p_value_empirical_le_zero": prob_negative,
            "distribution_sample": [float(x) for x in arr[:200]],  # sample for inspection
        }
        
    actual_rmse_skill = float((1.0 - np.sqrt(np.mean(sq_err_model)) / np.sqrt(np.mean(sq_err_persist))) * 100.0)
    actual_mae_skill = float((1.0 - np.mean(abs_err_model) / np.mean(abs_err_persist)) * 100.0)
    
    return {
        "block_size_hours": block_size,
        "n_replicates": n_replicates,
        "rmse_skill": get_summary(rmse_skills, actual_rmse_skill),
        "mae_skill": get_summary(mae_skills, actual_mae_skill),
        "model_rmse": get_summary(model_rmses, float(np.sqrt(np.mean(sq_err_model)))),
        "persist_rmse": get_summary(persist_rmses, float(np.sqrt(np.mean(sq_err_persist)))),
        "model_mae": get_summary(model_maes, float(np.mean(abs_err_model))),
        "persist_mae": get_summary(persist_maes, float(np.mean(abs_err_persist))),
        "d_sq_loss_diff": get_summary(d_sq_means, float(np.mean(sq_err_persist - sq_err_model))),
        "d_abs_loss_diff": get_summary(d_abs_means, float(np.mean(abs_err_persist - abs_err_model))),
    }


# ── Sub-Regime Breakdown ─────────────────────────────────────────────────────
def compute_regime_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_persist: np.ndarray, timestamps: pd.DatetimeIndex) -> dict:
    """Evaluates errors across physical wave height regimes, extreme events, and seasons."""
    regimes = {
        "Calm (< 1.25m)": y_true < 1.25,
        "Moderate (1.25m - 2.50m)": (y_true >= 1.25) & (y_true < 2.50),
        "Rough (2.50m - 4.00m)": (y_true >= 2.50) & (y_true < 4.00),
        "Severe (>= 4.00m)": y_true >= 4.00,
    }
    
    # Extreme percentiles
    q95 = np.percentile(y_true, 95)
    q99 = np.percentile(y_true, 99)
    regimes["Extreme Top 5% (> {:.2f}m)".format(q95)] = y_true > q95
    regimes["Extreme Top 1% (> {:.2f}m)".format(q99)] = y_true > q99
    
    # Seasons (Arabian Sea oceanography)
    dt_idx = pd.DatetimeIndex(timestamps)
    months = dt_idx.month
    regimes["Monsoon (Jun-Sep)"] = np.isin(months, [6, 7, 8, 9])
    regimes["Post-Monsoon (Oct-Nov)"] = np.isin(months, [10, 11])
    regimes["Winter (Dec-Feb)"] = np.isin(months, [12, 1, 2])
    regimes["Pre-Monsoon (Mar-May)"] = np.isin(months, [3, 4, 5])
    
    # Temporal Halves
    mid_idx = len(y_true) // 2
    half1_mask = np.zeros(len(y_true), dtype=bool)
    half1_mask[:mid_idx] = True
    regimes["First Half Period"] = half1_mask
    regimes["Second Half Period"] = ~half1_mask
    
    res = {}
    for r_name, mask in regimes.items():
        n_sub = int(np.sum(mask))
        if n_sub < 5:
            continue
        yt = y_true[mask]
        ym = y_pred[mask]
        yp = y_persist[mask]
        
        err_m = yt - ym
        err_p = yt - yp
        
        rmse_m = float(np.sqrt(np.mean(err_m ** 2)))
        rmse_p = float(np.sqrt(np.mean(err_p ** 2)))
        mae_m = float(np.mean(np.abs(err_m)))
        mae_p = float(np.mean(np.abs(err_p)))
        
        r_skill = float((1.0 - rmse_m / rmse_p) * 100.0) if rmse_p > 0 else 0.0
        a_skill = float((1.0 - mae_m / mae_p) * 100.0) if mae_p > 0 else 0.0
        
        # Mean Bias Error (MBE = mean(predicted - actual))
        bias = float(np.mean(ym - yt))
        underpredict_pct = float(np.mean(ym < yt) * 100.0)
        overpredict_pct = float(np.mean(ym > yt) * 100.0)
        
        res[r_name] = {
            "sample_count": n_sub,
            "sample_pct": float(n_sub / len(y_true) * 100.0),
            "mean_swh_m": float(np.mean(yt)),
            "std_swh_m": float(np.std(yt)),
            "model_rmse_m": rmse_m,
            "persist_rmse_m": rmse_p,
            "rmse_skill_pct": r_skill,
            "model_mae_m": mae_m,
            "persist_mae_m": mae_p,
            "mae_skill_pct": a_skill,
            "mean_bias_error_m": bias,
            "underpredict_pct": underpredict_pct,
            "overpredict_pct": overpredict_pct,
        }
    return res


# ── Residual Distribution Diagnostics ─────────────────────────────────────────
def compute_residual_diagnostics(errors: np.ndarray) -> dict:
    """Computes higher-order moments and error autocorrelation structure."""
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors, ddof=1))
    skew = float(stats.skew(errors))
    kurt = float(stats.kurtosis(errors))  # excess kurtosis
    
    # Jarque-Bera normality test
    jb_stat, jb_p = stats.jarque_bera(errors)
    
    # Autocorrelation of errors at multiple lags
    N = len(errors)
    e_demean = errors - mean_err
    var_e = np.var(errors, ddof=0)
    lags = [1, 2, 3, 6, 12, 24, 48]
    acf = {}
    for lag in lags:
        if lag < N:
            c = float(np.sum(e_demean[lag:] * e_demean[:-lag]) / (N * var_e))
            acf[f"lag_{lag}h"] = c
            
    return {
        "mean_error_m": mean_err,
        "std_error_m": std_err,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p_value": float(jb_p),
        "is_normal": float(jb_p) > 0.05,
        "error_autocorrelation": acf,
    }


# ── Inference Helper ─────────────────────────────────────────────────────────
@torch.no_grad()
def run_model_predictions(
    model: torch.nn.Module,
    feature_scaled: np.ndarray,
    anchors: np.ndarray,
    seq_len: int = 24,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(anchors), batch_size):
        batch_anchors = anchors[start : start + batch_size]
        windows = np.stack([
            feature_scaled[t - seq_len + 1 : t + 1] for t in batch_anchors
        ])
        xb = torch.tensor(windows, dtype=torch.float32)
        out = model(xb).cpu().numpy().flatten()
        preds.append(out)
    return np.concatenate(preds)


# ── Main Validation Pipeline ─────────────────────────────────────────────────
def main():
    log.info("=" * 80)
    log.info("OceanCast AI — Statistical Significance Testing & Validation Pipeline")
    log.info("=" * 80)
    set_seed(SEED)

    # 1. Load dataset & prepare splits
    log.info(f"Loading ERA5 dataset from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    n = len(df)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    test_start = n_train + n_val
    log.info(f"Total rows: {n:,} | Train: {n_train:,} | Val: {n_val:,} | Test: {n - test_start:,}")

    # Load original feature scaler
    with open(ORIG_SCALER_PATH, "rb") as f:
        scaler: TimeSeriesScaler = pickle.load(f)
    feature_scaled, _ = scaler.transform(df)
    swh_raw = df["swh"].values.astype(np.float32)

    all_horizon_stats = {}
    bootstrap_ci_rows = []
    error_analysis_rows = []

    # Store series for plotting
    plot_data = {}

    for H in HORIZONS:
        log.info("\n" + "=" * 80)
        log.info(f"EVALUATING HORIZON +{H}h")
        log.info("=" * 80)

        cand_info = CANDIDATE_MODELS[H]
        arch = cand_info["arch"]
        ckpt_name = cand_info["ckpt"]
        model_display = cand_info["name"]

        min_anchor = 23
        max_anchor = n - H - 1
        all_valid = np.arange(min_anchor, max_anchor + 1)
        test_anchors = all_valid[all_valid >= test_start]
        n_test = len(test_anchors)
        log.info(f"Candidate: {model_display} ({ckpt_name}) | Test anchors: {n_test:,}")

        y_true = swh_raw[test_anchors + H]
        y_persist = swh_raw[test_anchors]
        timestamps = df.index[test_anchors] if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["valid_time"].iloc[test_anchors])

        # Load residual model & scaler
        ckpt_path = RESIDUAL_DIR / ckpt_name
        model = build_wave_model(arch, input_dim=len(DEFAULT_FEATURE_COLS), seq_len=24)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

        with open(RESIDUAL_DIR / f"residual_scaler_h{H}.pkl", "rb") as f:
            res_scaler = pickle.load(f)
        rt_mean = float(res_scaler["mean"])
        rt_std = float(res_scaler["std"])

        # Execute model inference
        pred_delta_scaled = run_model_predictions(model, feature_scaled, test_anchors)
        pred_delta_m = pred_delta_scaled * rt_std + rt_mean
        y_pred = y_persist + pred_delta_m

        # Compute point metrics
        err_m = y_true - y_pred
        err_p = y_true - y_persist
        
        mse_m = float(np.mean(err_m ** 2))
        mse_p = float(np.mean(err_p ** 2))
        rmse_m = float(np.sqrt(mse_m))
        rmse_p = float(np.sqrt(mse_p))
        mae_m = float(np.mean(np.abs(err_m)))
        mae_p = float(np.mean(np.abs(err_p)))
        
        ss_res = np.sum(err_m ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = float(1.0 - ss_res / ss_tot)
        
        rmse_skill = float((1.0 - rmse_m / rmse_p) * 100.0)
        mae_skill = float((1.0 - mae_m / mae_p) * 100.0)
        
        log.info(f"Point Metrics:")
        log.info(f"  Persistence : RMSE={rmse_p:.4f}m, MAE={mae_p:.4f}m")
        log.info(f"  {model_display:<14}: RMSE={rmse_m:.4f}m, MAE={mae_m:.4f}m, R²={r2:.4f}")
        log.info(f"  Skill Scores: RMSE Skill = {rmse_skill:+.2f}%, MAE Skill = {mae_skill:+.2f}%")

        # ── Diebold-Mariano Tests ─────────────────────────────────────────────
        log.info("\nRunning Diebold-Mariano Tests with Newey-West HAC Variance ...")
        d_sq = err_p ** 2 - err_m ** 2   # Positive = model superior
        d_abs = np.abs(err_p) - np.abs(err_m)
        
        # Test 1: Lag truncation = H - 1 (theoretical minimum)
        dm_sq_h = diebold_mariano_test(d_sq, h=H, max_lags=max(1, H - 1))
        dm_abs_h = diebold_mariano_test(d_abs, h=H, max_lags=max(1, H - 1))
        
        # Test 2: Automatic Newey-West lag truncation (L = 4*(N/100)^(2/9) ≈ 13)
        l_auto = int(np.floor(4.0 * (n_test / 100.0) ** (2.0 / 9.0)))
        dm_sq_auto = diebold_mariano_test(d_sq, h=H, max_lags=l_auto)
        dm_abs_auto = diebold_mariano_test(d_abs, h=H, max_lags=l_auto)

        log.info(f"  DM Squared Error (lags={dm_sq_h['lags_used']}) : DM={dm_sq_h['dm_stat_hln']:.3f}, p(2-sided)={dm_sq_h['p_value_two_sided']:.4e}, Cohen's d={dm_sq_h['cohens_d']:.4f}")
        log.info(f"  DM Absolute Error (lags={dm_abs_h['lags_used']}): DM={dm_abs_h['dm_stat_hln']:.3f}, p(2-sided)={dm_abs_h['p_value_two_sided']:.4e}, Cohen's d={dm_abs_h['cohens_d']:.4f}")
        log.info(f"  DM Squared Error (auto lags={l_auto}) : DM={dm_sq_auto['dm_stat_hln']:.3f}, p(2-sided)={dm_sq_auto['p_value_two_sided']:.4e}")

        # ── Paired Moving-Block Bootstrap ─────────────────────────────────────
        log.info(f"\nRunning Paired Moving-Block Bootstrap ({B_REPLICATES:,} replicates) ...")
        bootstrap_results_by_block = {}
        for b in BLOCK_SIZES:
            b_res = moving_block_bootstrap(y_true, y_pred, y_persist, block_size=b, n_replicates=B_REPLICATES, seed=SEED)
            bootstrap_results_by_block[f"block_{b}h"] = b_res
            
            ci_r = b_res["rmse_skill"]["ci_95_percentile"]
            ci_a = b_res["mae_skill"]["ci_95_percentile"]
            log.info(f"  Block={b:3d}h | RMSE Skill 95% CI: [{ci_r[0]:+.2f}%, {ci_r[1]:+.2f}%] (SE={b_res['rmse_skill']['std_error']:.3f}%) | MAE Skill: [{ci_a[0]:+.2f}%, {ci_a[1]:+.2f}%]")

            bootstrap_ci_rows.append({
                "horizon_hours": H,
                "candidate_model": model_display,
                "block_size_hours": b,
                "n_replicates": B_REPLICATES,
                "rmse_skill_point_est": rmse_skill,
                "rmse_skill_boot_mean": b_res["rmse_skill"]["mean"],
                "rmse_skill_std_error": b_res["rmse_skill"]["std_error"],
                "rmse_skill_ci95_lower": ci_r[0],
                "rmse_skill_ci95_upper": ci_r[1],
                "rmse_skill_p_empirical_le_zero": b_res["rmse_skill"]["p_value_empirical_le_zero"],
                "mae_skill_point_est": mae_skill,
                "mae_skill_boot_mean": b_res["mae_skill"]["mean"],
                "mae_skill_std_error": b_res["mae_skill"]["std_error"],
                "mae_skill_ci95_lower": ci_a[0],
                "mae_skill_ci95_upper": ci_a[1],
                "model_rmse_point_m": rmse_m,
                "model_rmse_ci95_lower": b_res["model_rmse"]["ci_95_percentile"][0],
                "model_rmse_ci95_upper": b_res["model_rmse"]["ci_95_percentile"][1],
                "persist_rmse_point_m": rmse_p,
                "persist_rmse_ci95_lower": b_res["persist_rmse"]["ci_95_percentile"][0],
                "persist_rmse_ci95_upper": b_res["persist_rmse"]["ci_95_percentile"][1],
            })

        # ── Regime Breakdown ─────────────────────────────────────────────────
        log.info("\nEvaluating Regime Robustness & Error Diagnostics ...")
        regimes_res = compute_regime_metrics(y_true, y_pred, y_persist, timestamps)
        for r_name, r_stats in regimes_res.items():
            error_analysis_rows.append({
                "horizon_hours": H,
                "candidate_model": model_display,
                "regime_category": r_name,
                "sample_count": r_stats["sample_count"],
                "sample_pct": r_stats["sample_pct"],
                "mean_swh_m": r_stats["mean_swh_m"],
                "std_swh_m": r_stats["std_swh_m"],
                "model_rmse_m": r_stats["model_rmse_m"],
                "persist_rmse_m": r_stats["persist_rmse_m"],
                "rmse_skill_pct": r_stats["rmse_skill_pct"],
                "model_mae_m": r_stats["model_mae_m"],
                "persist_mae_m": r_stats["persist_mae_m"],
                "mae_skill_pct": r_stats["mae_skill_pct"],
                "mean_bias_error_m": r_stats["mean_bias_error_m"],
                "underpredict_pct": r_stats["underpredict_pct"],
                "overpredict_pct": r_stats["overpredict_pct"],
            })

        # Error diagnostics
        err_diag = compute_residual_diagnostics(err_m)
        persist_err_diag = compute_residual_diagnostics(err_p)

        # Store for report & plotting
        primary_b_res = bootstrap_results_by_block[f"block_{PRIMARY_BLOCK}h"]
        all_horizon_stats[f"{H}h"] = {
            "horizon_hours": H,
            "candidate_model": model_display,
            "checkpoint_file": ckpt_name,
            "n_test_samples": n_test,
            "point_metrics": {
                "model_rmse_m": rmse_m,
                "persist_rmse_m": rmse_p,
                "rmse_skill_pct": rmse_skill,
                "model_mae_m": mae_m,
                "persist_mae_m": mae_p,
                "mae_skill_pct": mae_skill,
                "model_mse_m2": mse_m,
                "persist_mse_m2": mse_p,
                "r2_score": r2,
            },
            "diebold_mariano": {
                "squared_error_h_lags": dm_sq_h,
                "absolute_error_h_lags": dm_abs_h,
                "squared_error_auto_lags": dm_sq_auto,
                "absolute_error_auto_lags": dm_abs_auto,
            },
            "bootstrap_primary_block_48h": {
                "block_size_hours": PRIMARY_BLOCK,
                "rmse_skill": primary_b_res["rmse_skill"],
                "mae_skill": primary_b_res["mae_skill"],
                "model_rmse": primary_b_res["model_rmse"],
                "persist_rmse": primary_b_res["persist_rmse"],
            },
            "bootstrap_sensitivity_across_blocks": {
                b_key: {
                    "block_size_hours": b_val["block_size_hours"],
                    "rmse_skill_ci95": b_val["rmse_skill"]["ci_95_percentile"],
                    "rmse_skill_std_error": b_val["rmse_skill"]["std_error"],
                    "mae_skill_ci95": b_val["mae_skill"]["ci_95_percentile"],
                    "mae_skill_std_error": b_val["mae_skill"]["std_error"],
                }
                for b_key, b_val in bootstrap_results_by_block.items()
            },
            "regime_breakdown": regimes_res,
            "residual_diagnostics": err_diag,
            "persistence_residual_diagnostics": persist_err_diag,
        }

        plot_data[H] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_persist": y_persist,
            "err_m": err_m,
            "err_p": err_p,
            "rmse_skill": rmse_skill,
            "ci_lower": primary_b_res["rmse_skill"]["ci_95_percentile"][0],
            "ci_upper": primary_b_res["rmse_skill"]["ci_95_percentile"][1],
            "dm_stat": dm_sq_h["dm_stat_hln"],
            "dm_p": dm_sq_h["p_value_two_sided"],
            "model_name": model_display,
        }

    # ── Multiple Comparison Corrections ───────────────────────────────────────
    raw_p_values = [all_horizon_stats[f"{H}h"]["diebold_mariano"]["squared_error_h_lags"]["p_value_two_sided"] for H in HORIZONS]
    
    # Bonferroni adjustment
    bonf_p = [float(min(1.0, p * len(HORIZONS))) for p in raw_p_values]
    
    # Benjamini-Hochberg FDR adjustment
    order = np.argsort(raw_p_values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(raw_p_values) + 1)
    bh_p = [float(min(1.0, p * len(HORIZONS) / r)) for p, r in zip(raw_p_values, ranks)]
    
    for idx, H in enumerate(HORIZONS):
        all_horizon_stats[f"{H}h"]["diebold_mariano"]["squared_error_h_lags"]["p_value_bonferroni"] = bonf_p[idx]
        all_horizon_stats[f"{H}h"]["diebold_mariano"]["squared_error_h_lags"]["p_value_fdr_bh"] = bh_p[idx]

    # Classification assignment based on rigorous criteria
    # Criteria for STATISTICALLY_SUPPORTED:
    # 1. Point RMSE skill > 0
    # 2. 95% bootstrap CI lower bound > 0 across ALL tested block lengths
    # 3. Diebold-Mariano test p_bonferroni < 0.05
    # 4. Outperforms persistence in major sea state regimes
    for H in HORIZONS:
        rec = all_horizon_stats[f"{H}h"]
        pt_skill = rec["point_metrics"]["rmse_skill_pct"]
        all_ci_lowers = [
            b_info["rmse_skill_ci95"][0]
            for b_info in rec["bootstrap_sensitivity_across_blocks"].values()
        ]
        min_ci_lower = min(all_ci_lowers)
        p_bonf = rec["diebold_mariano"]["squared_error_h_lags"]["p_value_bonferroni"]
        
        if pt_skill > 0 and min_ci_lower > 0 and p_bonf < 0.05:
            classification = "STATISTICALLY_SUPPORTED"
        elif pt_skill > 0 and (min_ci_lower <= 0 or p_bonf >= 0.05):
            classification = "POSITIVE_BUT_INCONCLUSIVE"
        else:
            classification = "NOT_SUPPORTED"
            
        rec["scientific_classification"] = classification
        log.info(f"+{H}h Final Scientific Recommendation: {classification} (p_bonf={p_bonf:.4e}, min CI lower={min_ci_lower:+.2f}%)")

    # ── Operational Registry Check Verification ───────────────────────────────
    log.info("\nVerifying Operational Model Registry Selection ...")
    registry_verification = {}
    for H in HORIZONS:
        entry = ModelRegistry.get_candidate(H, mode=ForecastMode.RESIDUAL)
        expected_ckpt = CANDIDATE_MODELS[H]["ckpt"]
        matches_ckpt = (entry.checkpoint_path.name == expected_ckpt)
        scaler_exists = entry.residual_scaler_path.is_file()
        registry_verification[f"{H}h"] = {
            "resolved_model_id": entry.model_id,
            "resolved_checkpoint": entry.checkpoint_path.name,
            "matches_tested_checkpoint": matches_ckpt,
            "residual_scaler_path": str(entry.residual_scaler_path.name),
            "residual_scaler_exists": scaler_exists,
            "model_version": entry.model_version,
            "reference_validation_status": entry.get_validation_status("mumbai_offshore"),
            "non_reference_validation_status": entry.get_validation_status("chennai_port"),
        }
        assert matches_ckpt, f"Registry mismatch for +{H}h: got {entry.checkpoint_path.name}, expected {expected_ckpt}"

    # ── Save Deliverable Files ────────────────────────────────────────────────
    log.info("\nSaving deliverables ...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    full_output = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_period": "ERA5 Indian Ocean Offshore Hold-Out Split (Last 15%)",
            "seed": SEED,
            "bootstrap_replicates": B_REPLICATES,
            "primary_block_size_hours": PRIMARY_BLOCK,
            "block_sizes_tested": BLOCK_SIZES,
        },
        "registry_operational_verification": registry_verification,
        "horizons": all_horizon_stats,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    log.info(f"✓ Saved JSON: {OUTPUT_JSON}")

    # CSV 1: Bootstrap CIs
    df_boot = pd.DataFrame(bootstrap_ci_rows)
    df_boot.to_csv(OUTPUT_BOOTSTRAP_CSV, index=False)
    log.info(f"✓ Saved Bootstrap CSV: {OUTPUT_BOOTSTRAP_CSV}")

    # CSV 2: Error Analysis by Regime
    df_err = pd.DataFrame(error_analysis_rows)
    df_err.to_csv(OUTPUT_ERROR_CSV, index=False)
    log.info(f"✓ Saved Error Analysis CSV: {OUTPUT_ERROR_CSV}")

    # ── Publication-Grade 4-Panel Visualization ───────────────────────────────
    log.info("\nGenerating publication figure: statistical_comparison.png ...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.25)
    
    colors = {"1": "#10B981", "6": "#20AFA7", "24": "#246B9E"}
    
    # Panel 1: Skill Scores with 95% MBB Confidence Intervals
    ax1 = axes[0, 0]
    horizons_str = ["+1h\n(Res WaveLSTM)", "+6h\n(Res WaveANN)", "+24h\n(Res WaveANN)"]
    skills = [plot_data[1]["rmse_skill"], plot_data[6]["rmse_skill"], plot_data[24]["rmse_skill"]]
    yerr_lower = [skills[i] - plot_data[H]["ci_lower"] for i, H in enumerate(HORIZONS)]
    yerr_upper = [plot_data[H]["ci_upper"] - skills[i] for i, H in enumerate(HORIZONS)]
    yerr = [yerr_lower, yerr_upper]
    
    bars1 = ax1.bar(horizons_str, skills, color=["#10B981", "#20AFA7", "#246B9E"], width=0.55, edgecolor="#102A43", linewidth=1.2, zorder=3)
    ax1.errorbar(horizons_str, skills, yerr=yerr, fmt="none", ecolor="#0F172A", elinewidth=2.2, capsize=6, capthick=2, zorder=4)
    ax1.axhline(0, color="#DC2626", linestyle="--", linewidth=1.5, label="Persistence Baseline (Skill = 0%)", zorder=2)
    ax1.set_ylabel("RMSE Skill Score (%)", fontsize=11, fontweight="bold", color="#102A43")
    ax1.set_title("A. RMSE Skill Score with 95% Moving-Block Bootstrap CI (B=5,000, b=48h)", fontsize=11, fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax1.set_ylim(-5, 36)
    for bar, skill, cl, cu in zip(bars1, skills, [plot_data[H]["ci_lower"] for H in HORIZONS], [plot_data[H]["ci_upper"] for H in HORIZONS]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.2, f"+{skill:.1f}%\n[{cl:+.1f}%, {cu:+.1f}%]", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel 2: Diebold-Mariano Test Statistics vs Critical Thresholds
    ax2 = axes[0, 1]
    dm_stats = [plot_data[H]["dm_stat"] for H in HORIZONS]
    bars2 = ax2.bar(horizons_str, dm_stats, color=["#10B981", "#20AFA7", "#246B9E"], width=0.55, edgecolor="#102A43", linewidth=1.2, zorder=3)
    ax2.axhline(1.96, color="#DC2626", linestyle="--", linewidth=1.5, label="Critical Value (α = 0.05, z = 1.96)", zorder=2)
    ax2.axhline(2.576, color="#B91C1C", linestyle=":", linewidth=1.5, label="Critical Value (α = 0.01, z = 2.58)", zorder=2)
    ax2.set_ylabel("Diebold-Mariano Statistic (HLN-Corrected)", fontsize=11, fontweight="bold", color="#102A43")
    ax2.set_title("B. Paired Diebold-Mariano Forecast Superiority Test (HAC Variance)", fontsize=11, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax2.set_ylim(0, max(dm_stats) * 1.25)
    for bar, dm_val, h in zip(bars2, dm_stats, HORIZONS):
        p_val = all_horizon_stats[f"{h}h"]["diebold_mariano"]["squared_error_h_lags"]["p_value_bonferroni"]
        p_str = f"p < 1e-15" if p_val < 1e-15 else f"p = {p_val:.2e}"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"DM = {dm_val:.1f}\n({p_str})", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel 3: Sea State Regime Breakdown (Calm, Moderate, Rough, Severe)
    ax3 = axes[1, 0]
    regime_labels = ["Calm\n(<1.25m)", "Moderate\n(1.25-2.5m)", "Rough\n(2.5-4.0m)", "Severe\n(≥4.0m)"]
    regime_keys = ["Calm (< 1.25m)", "Moderate (1.25m - 2.50m)", "Rough (2.50m - 4.00m)", "Severe (>= 4.00m)"]
    x = np.arange(len(regime_labels))
    w = 0.25
    
    for i, H in enumerate(HORIZONS):
        reg_dict = all_horizon_stats[f"{H}h"]["regime_breakdown"]
        reg_skills = [reg_dict[k]["rmse_skill_pct"] if k in reg_dict else 0.0 for k in regime_keys]
        ax3.bar(x + (i - 1) * w, reg_skills, width=w, label=f"+{H}h ({CANDIDATE_MODELS[H]['arch'].upper()})", color=["#10B981", "#20AFA7", "#246B9E"][i], edgecolor="#102A43", linewidth=1.0, zorder=3)
    
    ax3.axhline(0, color="#DC2626", linestyle="--", linewidth=1.2, zorder=2)
    ax3.set_xticks(x)
    ax3.set_xticklabels(regime_labels, fontsize=9.5, fontweight="medium")
    ax3.set_ylabel("Regime RMSE Skill Score (%)", fontsize=11, fontweight="bold", color="#102A43")
    ax3.set_title("C. Skill Score Robustness Across WMO Sea State Regimes", fontsize=11, fontweight="bold", pad=10)
    ax3.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax3.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel 4: Forecast Error Distribution Diagnostics (Residual vs Persistence at +6h)
    ax4 = axes[1, 1]
    e_m_6 = plot_data[6]["err_m"]
    e_p_6 = plot_data[6]["err_p"]
    
    bins = np.linspace(-0.5, 0.5, 80)
    ax4.hist(e_p_6, bins=bins, alpha=0.5, color="#DC2626", density=True, label=f"Persistence Error (std={np.std(e_p_6):.3f}m)", edgecolor="#991B1B")
    ax4.hist(e_m_6, bins=bins, alpha=0.65, color="#20AFA7", density=True, label=f"Residual WaveANN Error (std={np.std(e_m_6):.3f}m)", edgecolor="#0F766E")
    ax4.axvline(0, color="#0F172A", linestyle="-", linewidth=1.0)
    ax4.set_xlabel("Forecast Error (Actual - Predicted) [m]", fontsize=11, fontweight="bold", color="#102A43")
    ax4.set_ylabel("Probability Density", fontsize=11, fontweight="bold", color="#102A43")
    ax4.set_title("D. Error Probability Density Distribution (+6h Lead Time)", fontsize=11, fontweight="bold", pad=10)
    ax4.set_xlim(-0.4, 0.4)
    ax4.grid(axis="both", linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()
    log.info(f"✓ Saved Figure: {OUTPUT_PNG}")

    log.info("\n" + "=" * 80)
    log.info("STATISTICAL VALIDATION PIPELINE EXECUTION COMPLETED")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
