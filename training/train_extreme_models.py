"""
Extreme Wave Event Forecasting Model Training Script
===================================================
Author: Senior Oceanographic Deep Learning Researcher & Coastal Hazard Modelling Engineer
Project: OceanCast AI - Extreme Wave Event Improvement Framework

Trains scientifically justified experimental deep learning models to address peak
attenuation during severe wave events (SWH >= 4.0m) at lead times +1h, +6h, +24h.

Experimental Architectures & Formulations:
1. Extreme-Weighted Loss (EW-Loss): Smooth height-dependent loss weighting w(y).
2. Asymmetric Extreme-Penalized Loss (AEP-Loss): Height-conditioned penalty on underprediction.
3. Stratified Extreme Oversampling (SEO): Controlled oversampling of elevated sea states in training.
4. Multi-Task Extreme Hazard Network (MTE-Net): Shared representation with continuous residual
   regression and auxiliary extreme event classification heads.

Guarantees:
- Saves all experimental checkpoints exclusively to models/extreme_experimental/.
- Existing production models in models/ and models/residual/ are untouched.
- Chronological split: Train (70%), Val (15%), Test (15%) strictly preserved.
- All oversampling and scalers fit strictly on training split data.
"""

import sys
import os
import json
import pickle
import random
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
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models import build_wave_model
from src.models.base import BaseWaveModel
from src.data.dataset import TimeSeriesScaler, DEFAULT_FEATURE_COLS

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extreme_train")

# ── Configuration ───────────────────────────────────────────────────────────
SEED = 42
HORIZONS = [1, 6, 24]
BATCH_SIZE = 512
MAX_EPOCHS = 18
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_DELTA = 1e-4
DEVICE = torch.device("cpu")

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"
ORIG_SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
RESIDUAL_DIR = PROJECT_ROOT / "models" / "residual"
OUTPUT_DIR = PROJECT_ROOT / "models" / "extreme_experimental"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 24


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Custom Multi-Task Network ────────────────────────────────────────────────
class MultiTaskWaveANN(BaseWaveModel):
    """
    Multi-Task Neural Network that outputs both:
    1. Continuous scaled residual wave height change ΔSWH(t+H).
    2. Auxiliary hazard classification logit P(SWH(t+H) >= 2.5m).
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 24,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.flattened_dim = seq_len * input_dim
        layers = []
        prev_dim = self.flattened_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.trunk = nn.Sequential(*layers)
        
        # Regression head (predicts scaled delta)
        self.reg_head = nn.Linear(prev_dim, 1)
        # Hazard classification head (predicts rough/severe wave crossing)
        self.clf_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        feats = self.trunk(x_flat)
        reg_out = self.reg_head(feats)
        clf_out = self.clf_head(feats)
        return reg_out, clf_out


# ── Dataset Formulations ─────────────────────────────────────────────────────
class ExtremeWaveDataset(torch.utils.data.Dataset):
    """
    Standard dataset returning:
    - feature_window: [SEQ_LEN, F]
    - delta_scaled: float tensor
    - current_swh: float tensor (for dynamic loss weighting)
    - future_swh: float tensor
    - is_high_wave: float tensor (1.0 if future SWH >= 2.5m)
    """

    def __init__(
        self,
        feature_arr: np.ndarray,
        swh_raw: np.ndarray,
        anchor_indices: np.ndarray,
        horizon: int,
        rt_mean: float,
        rt_std: float,
        seq_len: int = SEQ_LEN,
    ):
        self.feature_arr = feature_arr
        self.swh_raw = swh_raw
        self.anchor_indices = anchor_indices
        self.horizon = horizon
        self.rt_mean = rt_mean
        self.rt_std = rt_std
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def __getitem__(self, idx: int):
        t = self.anchor_indices[idx]
        x = self.feature_arr[t - self.seq_len + 1 : t + 1]
        cur_swh = self.swh_raw[t]
        fut_swh = self.swh_raw[t + self.horizon]
        delta = fut_swh - cur_swh
        delta_scaled = (delta - self.rt_mean) / (self.rt_std + 1e-8)
        is_hazard = 1.0 if fut_swh >= 2.50 else 0.0

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(delta_scaled, dtype=torch.float32),
            torch.tensor(cur_swh, dtype=torch.float32),
            torch.tensor(fut_swh, dtype=torch.float32),
            torch.tensor(is_hazard, dtype=torch.float32),
        )


# ── Loss Functions ───────────────────────────────────────────────────────────
def extreme_weighted_loss(pred: torch.Tensor, target: torch.Tensor, cur_swh: torch.Tensor) -> torch.Tensor:
    """
    Loss weighted by current wave severity.
    w(y) = 1.0 + 3.0 * max(0, (y - 2.0)/2.0)^2
    Samples with rough/severe conditions receive higher weight.
    """
    excess = torch.clamp((cur_swh - 2.0) / 2.0, min=0.0)
    weights = 1.0 + 3.0 * (excess ** 2)
    diff = (pred.squeeze(1) - target) ** 2
    weighted_diff = weights * diff
    return torch.mean(weighted_diff)


def asymmetric_extreme_penalty_loss(pred: torch.Tensor, target: torch.Tensor, cur_swh: torch.Tensor) -> torch.Tensor:
    """
    Asymmetric underprediction penalty.
    When true delta > predicted delta (underpredicting wave growth/surge) in rough conditions (y >= 2.0m),
    penalize with extra factor lambda = 3.0.
    """
    diff = target - pred.squeeze(1)
    is_underpredict = (diff > 0).float()
    elevation_factor = torch.clamp((cur_swh - 1.5) / 1.5, min=0.0, max=2.0)
    penalty = 1.0 + 3.0 * is_underpredict * elevation_factor
    loss = penalty * (diff ** 2)
    return torch.mean(loss)


# ── Training Helpers ─────────────────────────────────────────────────────────
def train_epoch_single_task(model, loader, optimizer, loss_fn_type: str):
    model.train()
    total_loss = 0.0
    for xb, yb, cur_swh, fut_swh, _ in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        cur_swh = cur_swh.to(DEVICE)

        optimizer.zero_grad()
        pred = model(xb)
        if loss_fn_type == "ew_loss":
            loss = extreme_weighted_loss(pred, yb, cur_swh)
        elif loss_fn_type == "asym_loss":
            loss = asymmetric_extreme_penalty_loss(pred, yb, cur_swh)
        else:
            loss = nn.MSELoss()(pred.squeeze(1), yb)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch_single_task(model, loader):
    model.eval()
    total_loss = 0.0
    mse_fn = nn.MSELoss()
    with torch.no_grad():
        for xb, yb, _, _, _ in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = mse_fn(pred.squeeze(1), yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def train_epoch_multitask(model, loader, optimizer, gamma: float = 0.5):
    model.train()
    total_loss = 0.0
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()
    for xb, yb, _, _, is_hazard in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        is_hazard = is_hazard.to(DEVICE)

        optimizer.zero_grad()
        reg_pred, clf_pred = model(xb)
        l_reg = mse_fn(reg_pred.squeeze(1), yb)
        l_clf = bce_fn(clf_pred.squeeze(1), is_hazard)
        loss = l_reg + gamma * l_clf

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch_multitask(model, loader):
    model.eval()
    total_loss = 0.0
    mse_fn = nn.MSELoss()
    with torch.no_grad():
        for xb, yb, _, _, _ in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            reg_pred, _ = model(xb)
            loss = mse_fn(reg_pred.squeeze(1), yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


# ── Main Training Pipeline ───────────────────────────────────────────────────
def main():
    log.info("=" * 80)
    log.info("OceanCast AI - Extreme Wave Event Forecasting Model Training Pipeline")
    log.info("=" * 80)
    set_seed(SEED)

    # 1. Load dataset
    log.info(f"Loading ERA5 offshore timeseries from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    n = len(df)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    test_start = n_train + n_val
    log.info(f"Data rows: {n:,} | Train: {n_train:,} | Val: {n_val:,} | Test: {n - test_start:,}")

    # Load original feature scaler
    with open(ORIG_SCALER_PATH, "rb") as f:
        scaler: TimeSeriesScaler = pickle.load(f)
    feature_scaled, _ = scaler.transform(df)
    swh_raw = df["swh"].values.astype(np.float32)

    training_metadata = {}

    for H in HORIZONS:
        log.info("\n" + "=" * 80)
        log.info(f"TRAINING EXPERIMENTAL MODELS FOR HORIZON +{H}h")
        log.info("=" * 80)

        min_anchor = SEQ_LEN - 1
        max_anchor = n - H - 1
        all_valid = np.arange(min_anchor, max_anchor + 1)

        train_anchors = all_valid[all_valid < n_train]
        val_anchors = all_valid[(all_valid >= n_train) & (all_valid < test_start)]
        test_anchors = all_valid[all_valid >= test_start]

        # Load existing residual scaler info
        with open(RESIDUAL_DIR / f"residual_scaler_h{H}.pkl", "rb") as f:
            res_scaler = pickle.load(f)
        rt_mean = float(res_scaler["mean"])
        rt_std = float(res_scaler["std"])

        log.info(f"Residual Scaler: Mean={rt_mean:.4f}m, Std={rt_std:.4f}m")

        # Datasets
        train_ds = ExtremeWaveDataset(feature_scaled, swh_raw, train_anchors, H, rt_mean, rt_std)
        val_ds = ExtremeWaveDataset(feature_scaled, swh_raw, val_anchors, H, rt_mean, rt_std)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Build Stratified Oversampled Loader for SEO experiment (train set only!)
        rough_anchors = train_anchors[swh_raw[train_anchors] >= 2.50]
        calm_anchors = train_anchors[swh_raw[train_anchors] < 2.50]
        # Repeat rough anchors 3x to increase extreme representation in training
        oversampled_train_anchors = np.concatenate([train_anchors, rough_anchors, rough_anchors])
        np.random.shuffle(oversampled_train_anchors)
        oversampled_ds = ExtremeWaveDataset(feature_scaled, swh_raw, oversampled_train_anchors, H, rt_mean, rt_std)
        oversampled_train_loader = torch.utils.data.DataLoader(oversampled_ds, batch_size=BATCH_SIZE, shuffle=True)
        log.info(f"SEO Augmented train samples: {len(oversampled_train_anchors):,} (Rough boosted 3x)")

        # Determine architecture for this horizon (matching baseline selection: LSTM for 1h, ANN for 6h and 24h)
        base_arch = "lstm" if H == 1 else "ann"

        experiments = [
            ("ew_loss", f"ExtremeWeighted_{base_arch.upper()}_h{H}", "Extreme-Weighted Loss"),
            ("asym_loss", f"Asymmetric_{base_arch.upper()}_h{H}", "Asymmetric Penalty Loss"),
            ("seo", f"Oversampled_{base_arch.upper()}_h{H}", "Stratified Extreme Oversampling"),
        ]
        if base_arch == "ann":
            experiments.append(("multitask", f"MultiTask_Extreme_ANN_h{H}", "Multi-Task Hazard Network"))

        for exp_key, model_id, exp_name in experiments:
            log.info(f"\nTraining Experiment: {exp_name} ({model_id}) ...")
            set_seed(SEED)

            is_multitask = (exp_key == "multitask")
            if is_multitask:
                model = MultiTaskWaveANN(input_dim=len(DEFAULT_FEATURE_COLS), seq_len=SEQ_LEN).to(DEVICE)
            else:
                model = build_wave_model(base_arch, input_dim=len(DEFAULT_FEATURE_COLS), seq_len=SEQ_LEN).to(DEVICE)

            optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)

            current_train_loader = oversampled_train_loader if exp_key == "seo" else train_loader

            best_val_loss = float("inf")
            patience = 0
            best_weights = None

            t0 = time.time()
            for ep in range(1, MAX_EPOCHS + 1):
                if is_multitask:
                    tr_loss = train_epoch_multitask(model, current_train_loader, optimizer)
                    va_loss = eval_epoch_multitask(model, val_loader)
                else:
                    tr_loss = train_epoch_single_task(model, current_train_loader, optimizer, loss_fn_type=exp_key)
                    va_loss = eval_epoch_single_task(model, val_loader)

                scheduler.step(va_loss)
                if va_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
                    best_val_loss = va_loss
                    patience = 0
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1

                if ep % 5 == 0 or ep == 1:
                    log.info(f"  Epoch {ep:>2}/{MAX_EPOCHS} | Train Loss: {tr_loss:.5f} | Val MSE: {va_loss:.5f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

                if patience >= EARLY_STOP_PATIENCE:
                    log.info(f"  Early stopping triggered at epoch {ep}")
                    break

            elapsed = time.time() - t0
            ckpt_path = OUTPUT_DIR / f"{model_id}.pt"
            torch.save(best_weights, ckpt_path)
            log.info(f"  Saved checkpoint to {ckpt_path} (Best Val Loss: {best_val_loss:.5f}, Elapsed: {elapsed:.1f}s)")

            training_metadata[model_id] = {
                "horizon": H,
                "experiment_key": exp_key,
                "experiment_name": exp_name,
                "architecture": base_arch if not is_multitask else "multitask_ann",
                "checkpoint_path": str(ckpt_path),
                "best_val_loss": float(best_val_loss),
                "elapsed_seconds": round(elapsed, 2),
            }

    # Save training metadata
    meta_path = OUTPUT_DIR / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(training_metadata, f, indent=2)
    log.info(f"\nAll experimental models trained successfully. Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
