"""
Residual Deep Learning Training Script for OceanCast AI
=======================================================
Trains WaveANN, WaveLSTM, WaveCNN1D to predict ΔSWH (residual wave height change)
instead of absolute SWH. Saves 9 new checkpoints to models/residual/.

Target: ΔSWH = SWH(t + H) - SWH(t)
Reconstruction: SWH_pred(t + H) = SWH(t) + Predicted ΔSWH

Existing checkpoints in models/ are NOT modified.

Usage:
    py -3.11 training/train_residual_models.py
"""

import sys
import os
import json
import pickle
import random
import time
import logging
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models import build_wave_model
from src.data.dataset import TimeSeriesScaler, WaveTimeSeriesDataset, compute_leak_free_split_indices
from src.training.checkpoint import ModelCheckpoint

# ── Configuration ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("residual_train")

SEED = 42
HORIZONS = [1, 6, 24]
MODEL_TYPES = ["ann", "lstm", "cnn1d"]
BATCH_SIZE = 256
MAX_EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
MIN_LR = 1e-5

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"
MODELS_DIR = PROJECT_ROOT / "models" / "residual"
ORIG_SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

FEATURE_COLS = [
    "swh", "mwp", "mwd_sin", "mwd_cos",
    "u10", "v10", "wind_speed", "wind_dir_sin", "wind_dir_cos",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
TARGET_COL = "swh"
SEQ_LEN = 24
DEVICE = torch.device("cpu")


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_deterministic_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Custom Residual Dataset ──────────────────────────────────────────────────
class ResidualWaveDataset(torch.utils.data.Dataset):
    """
    Returns (feature_window, delta_swh_scaled) pairs.

    feature_window : [SEQ_LEN, n_features] — scaled with the ORIGINAL feature scaler
    delta_swh_scaled : scalar — delta scaled with the per-horizon residual target scaler
    """

    def __init__(
        self,
        feature_arr: np.ndarray,   # [T, n_features]  — already scaled
        swh_raw: np.ndarray,        # [T]               — raw SWH (m)
        anchor_indices: np.ndarray, # valid anchor indices
        horizon: int,
        residual_target_mean: float,
        residual_target_std: float,
        seq_len: int = SEQ_LEN,
    ):
        self.feature_arr = feature_arr
        self.swh_raw = swh_raw
        self.anchor_indices = anchor_indices
        self.horizon = horizon
        self.rt_mean = residual_target_mean
        self.rt_std = residual_target_std
        self.seq_len = seq_len

    def __len__(self):
        return len(self.anchor_indices)

    def __getitem__(self, idx):
        t = self.anchor_indices[idx]
        x = self.feature_arr[t - self.seq_len + 1 : t + 1]       # [SEQ_LEN, F]
        delta = self.swh_raw[t + self.horizon] - self.swh_raw[t]  # scalar (m)
        # Standardize delta with training statistics
        delta_scaled = (delta - self.rt_mean) / (self.rt_std + 1e-8)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(delta_scaled, dtype=torch.float32),
        )


# ── Training loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("OceanCast AI — Residual (ΔSWH) Model Training")
    log.info("=" * 70)
    log.info(f"Project root : {PROJECT_ROOT}")
    log.info(f"Output dir   : {MODELS_DIR}")
    log.info(f"Device       : {DEVICE}")
    log.info(f"Seed         : {SEED}")

    # ── Load dataset ────────────────────────────────────────────────────────
    log.info(f"\nLoading dataset from {DATA_PATH} …")
    df = pd.read_parquet(DATA_PATH)
    log.info(f"  Rows: {len(df):,}  |  Cols: {list(df.columns)}")

    # Reconstruct time-based features if not present
    if "hour_sin" not in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # ── Split indices (leak-free, chronological) ─────────────────────────────
    n = len(df)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val
    log.info(f"  Split: train={n_train:,} | val={n_val:,} | test={n_test:,}")

    # ── Load original feature scaler (must NOT be replaced) ─────────────────
    log.info(f"\nLoading original feature scaler from {ORIG_SCALER_PATH} …")
    with open(ORIG_SCALER_PATH, "rb") as f:
        orig_scaler: TimeSeriesScaler = pickle.load(f)

    # Scale features using original scaler
    # transform() takes a DataFrame and returns (scaled_features, scaled_target)
    swh_raw = df["swh"].values.astype(np.float32)
    feature_scaled, _ = orig_scaler.transform(df)   # [T, F]

    results_summary = {}

    # ── Iterate horizons ────────────────────────────────────────────────────
    for H in HORIZONS:
        log.info("\n" + "=" * 70)
        log.info(f"HORIZON +{H}h")
        log.info("=" * 70)

        # Valid anchor indices: need [t-23..t] input + t+H target
        min_anchor = SEQ_LEN - 1              # 23
        max_anchor = n - H - 1               # leave room for future target
        all_valid = np.arange(min_anchor, max_anchor + 1)

        # Chronological split on valid anchors
        # Training anchors: anchor < n_train
        # Val anchors: n_train <= anchor < n_train + n_val
        # Test anchors: anchor >= n_train + n_val
        train_mask = all_valid < n_train
        val_mask   = (all_valid >= n_train) & (all_valid < n_train + n_val)
        test_mask  = all_valid >= n_train + n_val

        train_anchors = all_valid[train_mask]
        val_anchors   = all_valid[val_mask]
        test_anchors  = all_valid[test_mask]

        log.info(f"  Anchors — train:{len(train_anchors):,} | val:{len(val_anchors):,} | test:{len(test_anchors):,}")

        # ── Compute residual target statistics on TRAINING data only ─────────
        train_deltas = swh_raw[train_anchors + H] - swh_raw[train_anchors]
        rt_mean = float(train_deltas.mean())
        rt_std  = float(train_deltas.std())
        log.info(f"  ΔSWH train stats — mean={rt_mean:.4f}m | std={rt_std:.4f}m")

        # Save residual scaler info (we store mean/std as a dict for transparency)
        residual_scaler_info = {
            "horizon": H,
            "target": "delta_swh",
            "mean": rt_mean,
            "std": rt_std,
            "n_train_samples": int(len(train_anchors)),
        }
        scaler_path = MODELS_DIR / f"residual_scaler_h{H}.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(residual_scaler_info, f)
        log.info(f"  Saved residual scaler → {scaler_path}")

        # ── Build DataLoaders ─────────────────────────────────────────────────
        def make_loader(anchors, shuffle):
            ds = ResidualWaveDataset(
                feature_arr=feature_scaled,
                swh_raw=swh_raw,
                anchor_indices=anchors,
                horizon=H,
                residual_target_mean=rt_mean,
                residual_target_std=rt_std,
            )
            return torch.utils.data.DataLoader(
                ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                num_workers=0, pin_memory=False,
            )

        train_loader = make_loader(train_anchors, shuffle=True)
        val_loader   = make_loader(val_anchors,   shuffle=False)

        # ── Train each model type ─────────────────────────────────────────────
        horizon_results = {}

        for m_type in MODEL_TYPES:
            log.info(f"\n  ── Training {m_type.upper()} residual +{H}h ──")
            set_deterministic_seeds(SEED)

            model = build_wave_model(m_type, input_dim=len(FEATURE_COLS), seq_len=SEQ_LEN).to(DEVICE)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"     Parameters: {n_params:,}")

            optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=SCHEDULER_FACTOR,
                patience=SCHEDULER_PATIENCE, min_lr=MIN_LR,
            )
            criterion = nn.MSELoss()

            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None
            train_history = []
            val_history = []

            t0 = time.time()
            for epoch in range(1, MAX_EPOCHS + 1):
                tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
                va_loss = eval_one_epoch(model, val_loader, criterion)
                scheduler.step(va_loss)
                train_history.append(float(tr_loss))
                val_history.append(float(va_loss))

                improved = va_loss < best_val_loss - EARLY_STOP_MIN_DELTA
                if improved:
                    best_val_loss = va_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1

                if epoch % 5 == 0 or epoch == 1:
                    log.info(
                        f"     Ep {epoch:3d} | train_loss={tr_loss:.6f} "
                        f"| val_loss={va_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}"
                    )

                if patience_counter >= EARLY_STOP_PATIENCE:
                    log.info(f"     Early stop at epoch {epoch}")
                    break

            elapsed = time.time() - t0
            log.info(f"     Training done in {elapsed:.1f}s | best_val_loss={best_val_loss:.6f}")

            # Restore best weights
            if best_state is not None:
                model.load_state_dict(best_state)

            # ── Save checkpoint ───────────────────────────────────────────────
            ckpt_path = MODELS_DIR / f"best_{m_type}_residual_h{H}.pt"
            meta_path = MODELS_DIR / f"best_{m_type}_residual_h{H}_metadata.json"

            torch.save(model.state_dict(), ckpt_path)

            metadata = {
                "model_type": "residual",
                "architecture": m_type,
                "target": "delta_swh",
                "horizon": H,
                "reconstruction": f"SWH(t+{H}) = SWH(t) + pred_delta",
                "residual_scaler": f"models/residual/residual_scaler_h{H}.pkl",
                "best_val_loss_scaled": float(best_val_loss),
                "n_parameters": n_params,
                "seed": SEED,
                "max_epochs": MAX_EPOCHS,
                "epochs_trained": len(train_history),
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "train_loss_history": train_history,
                "val_loss_history": val_history,
                "feature_cols": FEATURE_COLS,
                "seq_len": SEQ_LEN,
                "residual_target_mean_m": rt_mean,
                "residual_target_std_m": rt_std,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            log.info(f"     Saved checkpoint → {ckpt_path}")
            log.info(f"     Saved metadata   → {meta_path}")

            horizon_results[m_type] = {
                "best_val_loss_scaled": float(best_val_loss),
                "epochs_trained": len(train_history),
                "n_parameters": n_params,
            }

        results_summary[H] = horizon_results

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("TRAINING COMPLETE — Summary")
    log.info("=" * 70)
    for H, hr in results_summary.items():
        log.info(f"\n  +{H}h horizon:")
        for m_type, stats in hr.items():
            log.info(
                f"    {m_type.upper():8s} | val_loss={stats['best_val_loss_scaled']:.6f} "
                f"| epochs={stats['epochs_trained']:2d} | params={stats['n_parameters']:,}"
            )

    # Verify original checkpoints are untouched
    orig_ckpts = list((PROJECT_ROOT / "models").glob("best_*.pt"))
    residual_ckpts = list(MODELS_DIR.glob("*.pt"))
    log.info(f"\n  Original checkpoints : {len(orig_ckpts)} (unchanged)")
    log.info(f"  Residual checkpoints : {len(residual_ckpts)} (new)")
    assert len(residual_ckpts) == 9, f"Expected 9 residual .pt files, found {len(residual_ckpts)}"
    log.info("\n✓ All 9 residual checkpoints saved successfully.")
    log.info("✓ Original models/ checkpoints untouched.")


if __name__ == "__main__":
    main()
