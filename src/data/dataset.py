"""
PyTorch Time-Series Forecasting Dataset and DataLoaders for Ocean Wave Prediction.
Handles:
- Configurable historical input window (L) and forecast horizon (H)
- Strict chronological 70% / 15% / 15% train/val/test splitting
- Zero-leakage scaler fitting on training data only
- Memory-efficient sliding window indexing
"""

from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


DEFAULT_FEATURE_COLS = [
    "swh",
    "mwp",
    "mwd_sin",
    "mwd_cos",
    "u10",
    "v10",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

DEFAULT_TARGET_COL = "swh"


def set_deterministic_seeds(seed: int = 42):
    """Set random seeds across NumPy and PyTorch for deterministic execution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimeSeriesScaler:
    """
    Manages standardization for features and target variables.
    Strictly fits statistics (mean, std) ONLY on the training split
    to prevent future information leakage into validation or testing.
    """

    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.target_name: str = ""

    def fit(self, df_train: pd.DataFrame, feature_cols: List[str], target_col: str):
        """Fit scaler parameters solely using training split data."""
        self.feature_names = list(feature_cols)
        self.target_name = target_col

        self.feature_scaler.fit(df_train[feature_cols].values)
        self.target_scaler.fit(df_train[[target_col]].values)
        self.is_fitted = True

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform features and target using fitted training parameters."""
        if not self.is_fitted:
            raise RuntimeError("TimeSeriesScaler must be fitted on training data before transforming.")
        scaled_x = self.feature_scaler.transform(df[self.feature_names].values).astype(np.float32)
        scaled_y = self.target_scaler.transform(df[[self.target_name]].values).astype(np.float32)
        return scaled_x, scaled_y

    def inverse_transform_target(self, y_scaled: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Invert normalized target predictions back to original physical units (meters)."""
        if isinstance(y_scaled, torch.Tensor):
            y_scaled = y_scaled.detach().cpu().numpy()
        y_arr = np.asarray(y_scaled).reshape(-1, 1)
        return self.target_scaler.inverse_transform(y_arr).flatten()

    def save(self, filepath: Union[str, Path]):
        """Persist fitted scaler to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Union[str, Path]) -> "TimeSeriesScaler":
        """Load fitted scaler from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class WaveTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time-series wave height forecasting.

    Given anchor time index t:
    - Input window X: observations from (t - window_size + 1) to t (length L).
    - Future target y: target variable at future step (t + horizon).
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        timestamps: np.ndarray,
        anchors: np.ndarray,
        window_size: int = 24,
        horizon: int = 1,
        raw_targets: Optional[np.ndarray] = None,
    ):
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.timestamps = np.asarray(timestamps)
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.raw_targets = np.asarray(raw_targets, dtype=np.float32) if raw_targets is not None else None

        # Sanity check anchor bounds
        if len(self.anchors) > 0:
            min_anchor = self.anchors.min()
            max_anchor = self.anchors.max()
            assert min_anchor - self.window_size + 1 >= 0, "Anchor index requires history before index 0"
            assert max_anchor + self.horizon < len(self.targets), "Anchor + horizon exceeds dataset bounds"

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor = self.anchors[idx]
        # Historical window of length L: [anchor - window_size + 1, ..., anchor]
        x_window = self.features[anchor - self.window_size + 1 : anchor + 1]
        # Target at future step: anchor + horizon
        y_val = self.targets[anchor + self.horizon]

        return torch.from_numpy(x_window), torch.from_numpy(y_val)

    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Return human-readable timestamp and label metadata for a given index."""
        anchor = self.anchors[idx]
        return {
            "index": idx,
            "anchor_index": int(anchor),
            "input_start_time": str(self.timestamps[anchor - self.window_size + 1]),
            "input_end_time": str(self.timestamps[anchor]),
            "target_time": str(self.timestamps[anchor + self.horizon]),
            "target_scaled": float(self.targets[anchor + self.horizon][0]),
            "target_raw": float(self.raw_targets[anchor + self.horizon][0]) if self.raw_targets is not None else None,
        }


def compute_leak_free_split_indices(
    total_len: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    window_size: int = 24,
    horizon: int = 1,
) -> Dict[str, Any]:
    """
    Compute anchor indices for train, val, and test splits with zero future leakage.

    Rules:
    1. Training targets are strictly within [0, n_train - 1].
    2. Validation targets are strictly within [n_train, n_train + n_val - 1].
    3. Testing targets are strictly within [n_train + n_val, total_len - 1].
    4. Target timestamps for Train < Val < Test are strictly non-overlapping and disjoint.
    """
    n_train = int(total_len * train_ratio)
    n_val = int(total_len * val_ratio)
    n_test = total_len - n_train - n_val

    # Anchors definition: target index is (anchor + horizon)
    # Train: target in [window_size - 1 + horizon, n_train - 1]
    train_anchors = np.arange(window_size - 1, n_train - horizon)

    # Val: target in [n_train, n_train + n_val - 1]
    val_anchors = np.arange(n_train - horizon, n_train + n_val - horizon)

    # Test: target in [n_train + n_val, total_len - 1]
    test_anchors = np.arange(n_train + n_val - horizon, total_len - horizon)

    # Verify boundary assertions
    train_targets = train_anchors + horizon
    val_targets = val_anchors + horizon
    test_targets = test_anchors + horizon

    assert train_targets.max() < val_targets.min(), "Leakage detected: train targets overlap validation targets!"
    assert val_targets.max() < test_targets.min(), "Leakage detected: validation targets overlap testing targets!"
    assert test_targets.max() == total_len - 1, "Testing targets do not reach the end of the series!"

    return {
        "n_train_series": n_train,
        "n_val_series": n_val,
        "n_test_series": n_test,
        "train_anchors": train_anchors,
        "val_anchors": val_anchors,
        "test_anchors": test_anchors,
        "train_sample_count": len(train_anchors),
        "val_sample_count": len(val_anchors),
        "test_sample_count": len(test_anchors),
        "train_target_index_range": (int(train_targets.min()), int(train_targets.max())),
        "val_target_index_range": (int(val_targets.min()), int(val_targets.max())),
        "test_target_index_range": (int(test_targets.min()), int(test_targets.max())),
    }


def create_wave_forecast_dataloaders(
    data_source: Union[str, Path, pd.DataFrame],
    window_size: int = 24,
    horizon: int = 1,
    feature_cols: Optional[List[str]] = None,
    target_col: str = DEFAULT_TARGET_COL,
    batch_size: int = 64,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, TimeSeriesScaler, Dict[str, Any]]:
    """
    Factory function to build leak-free PyTorch DataLoaders for wave forecasting.

    Parameters
    ----------
    data_source : Union[str, Path, pd.DataFrame]
        DataFrame or path to preprocessed Parquet/CSV file.
    window_size : int
        Historical input sequence length in hours (default 24).
    horizon : int
        Future forecast lead time in hours (supports 1, 6, 24, etc.).
    feature_cols : Optional[List[str]]
        List of feature column names (defaults to DEFAULT_FEATURE_COLS).
    target_col : str
        Target column name (default 'swh').
    batch_size : int
        Batch size for DataLoaders (default 64).
    train_ratio : float
        Proportion of series for training (default 0.70).
    val_ratio : float
        Proportion of series for validation (default 0.15).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader, TimeSeriesScaler, Dict[str, Any]]
        (train_loader, val_loader, test_loader, fitted_scaler, metadata)
    """
    set_deterministic_seeds(random_seed)

    # 1. Load DataFrame
    if isinstance(data_source, (str, Path)):
        path = Path(data_source)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    else:
        df = data_source.copy()

    # Ensure chronological sort
    time_col = "valid_time" if "valid_time" in df.columns else "time"
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    if feature_cols is None:
        feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    total_len = len(df)
    n_train = int(total_len * train_ratio)

    # 2. Fit Scaler ONLY on Training Split
    df_train_raw = df.iloc[:n_train]
    scaler = TimeSeriesScaler()
    scaler.fit(df_train_raw, feature_cols=feature_cols, target_col=target_col)

    # 3. Transform Entire Series Using Training Parameters
    scaled_x, scaled_y = scaler.transform(df)
    raw_y = df[[target_col]].values.astype(np.float32)
    timestamps = df[time_col].values

    # 4. Compute Split Indices
    splits = compute_leak_free_split_indices(
        total_len=total_len,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        window_size=window_size,
        horizon=horizon,
    )

    # 5. Build Datasets
    train_dataset = WaveTimeSeriesDataset(
        features=scaled_x,
        targets=scaled_y,
        timestamps=timestamps,
        anchors=splits["train_anchors"],
        window_size=window_size,
        horizon=horizon,
        raw_targets=raw_y,
    )

    val_dataset = WaveTimeSeriesDataset(
        features=scaled_x,
        targets=scaled_y,
        timestamps=timestamps,
        anchors=splits["val_anchors"],
        window_size=window_size,
        horizon=horizon,
        raw_targets=raw_y,
    )

    test_dataset = WaveTimeSeriesDataset(
        features=scaled_x,
        targets=scaled_y,
        timestamps=timestamps,
        anchors=splits["test_anchors"],
        window_size=window_size,
        horizon=horizon,
        raw_targets=raw_y,
    )

    # 6. Build DataLoaders
    # Generator for deterministic shuffling
    g = torch.Generator()
    g.manual_seed(random_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    metadata = {
        "total_series_rows": total_len,
        "window_size": window_size,
        "horizon": horizon,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "num_features": len(feature_cols),
        "batch_size": batch_size,
        "random_seed": random_seed,
        "split_counts": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "target_time_ranges": {
            "train_start": str(timestamps[splits["train_target_index_range"][0]]),
            "train_end": str(timestamps[splits["train_target_index_range"][1]]),
            "val_start": str(timestamps[splits["val_target_index_range"][0]]),
            "val_end": str(timestamps[splits["val_target_index_range"][1]]),
            "test_start": str(timestamps[splits["test_target_index_range"][0]]),
            "test_end": str(timestamps[splits["test_target_index_range"][1]]),
        },
        "scaler_stats": {
            "feature_means": scaler.feature_scaler.mean_.tolist(),
            "feature_scales": scaler.feature_scaler.scale_.tolist(),
            "target_mean": float(scaler.target_scaler.mean_[0]),
            "target_scale": float(scaler.target_scaler.scale_[0]),
        },
    }

    return train_loader, val_loader, test_loader, scaler, metadata
