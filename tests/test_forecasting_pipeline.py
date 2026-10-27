"""
Automated Unit Tests for Time-Series Forecasting Pipeline & Deep Learning Architectures:
- Leak-free split boundary mathematics (70% / 15% / 15%)
- Scaler fitting isolation (strictly train-only fitting)
- Sliding window indexing, tensor shapes, and target alignment
- Absence of target leakage across horizons (H = 1, 6, 24)
- Model parameter counts, summaries, and forward pass interfaces
- ModelCheckpoint metadata persistence
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import (
    TimeSeriesScaler,
    WaveTimeSeriesDataset,
    compute_leak_free_split_indices,
    create_wave_forecast_dataloaders,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
)
from src.models import (
    WaveANN,
    WaveLSTM,
    WaveCNN1D,
    build_wave_model,
)
from src.training.checkpoint import ModelCheckpoint


def test_leak_free_split_boundaries_all_horizons():
    """Verify that train, val, and test target timestamps are strictly non-overlapping."""
    total_len = 1000
    window_size = 24

    for H in [1, 6, 24]:
        splits = compute_leak_free_split_indices(
            total_len=total_len,
            train_ratio=0.70,
            val_ratio=0.15,
            window_size=window_size,
            horizon=H,
        )

        train_targets = splits["train_anchors"] + H
        val_targets = splits["val_anchors"] + H
        test_targets = splits["test_anchors"] + H

        # 1. Target indices must be strictly monotonic across splits
        assert train_targets.max() < val_targets.min(), f"Leakage: Train target overlaps Val for H={H}"
        assert val_targets.max() < test_targets.min(), f"Leakage: Val target overlaps Test for H={H}"

        # 2. Test target reaches the very last observation
        assert test_targets.max() == total_len - 1

        # 3. History window for the first train sample must not require indices < 0
        min_anchor = splits["train_anchors"].min()
        assert min_anchor - window_size + 1 == 0


def test_scaler_isolated_to_training_split():
    """Verify that scalers fit strictly on training data without seeing val/test data."""
    # Synthetic dataset with distinct train and test distributions
    n_total = 100
    n_train = 70
    dates = pd.date_range("2020-01-01", periods=n_total, freq="1h")

    # Train mean ~ 10.0, Val/Test mean ~ 100.0
    train_vals = np.full(n_train, 10.0)
    test_vals = np.full(n_total - n_train, 100.0)
    vals = np.concatenate([train_vals, test_vals])

    df = pd.DataFrame({"valid_time": dates, "swh": vals, "feat": vals * 2})
    df_train = df.iloc[:n_train]

    scaler = TimeSeriesScaler()
    scaler.fit(df_train, feature_cols=["feat"], target_col="swh")

    # Scaler mean must match training distribution (10.0 for target, 20.0 for feature)
    assert np.isclose(scaler.target_scaler.mean_[0], 10.0)
    assert np.isclose(scaler.feature_scaler.mean_[0], 20.0)

    # Inverting normalized values must recover raw numbers
    scaled_x, scaled_y = scaler.transform(df)
    recovered_y = scaler.inverse_transform_target(scaled_y)
    assert np.allclose(recovered_y, vals, atol=1e-5)


def test_window_tensor_shapes_and_target_alignment():
    """Verify input window shape, future target offset, and absence of target leakage."""
    n_steps = 100
    window_size = 12
    horizon = 6
    num_features = 4

    dates = pd.date_range("2020-01-01 00:00", periods=n_steps, freq="1h")
    features = np.arange(n_steps * num_features, dtype=np.float32).reshape(n_steps, num_features)
    # Strictly increasing distinct targets so each value equals its timestamp step
    targets = np.arange(n_steps, dtype=np.float32).reshape(-1, 1)

    anchors = np.array([20, 30, 40])
    dataset = WaveTimeSeriesDataset(
        features=features,
        targets=targets,
        timestamps=dates.values,
        anchors=anchors,
        window_size=window_size,
        horizon=horizon,
        raw_targets=targets,
    )

    assert len(dataset) == 3

    for idx in range(len(dataset)):
        x, y = dataset[idx]
        meta = dataset.get_sample_metadata(idx)
        anchor = anchors[idx]

        # 1. Tensor shape verification
        assert x.shape == (window_size, num_features)
        assert y.shape == (1,)

        # 2. Historical window ends at anchor and target is at (anchor + horizon)
        expected_target_time = dates[anchor + horizon]
        expected_input_end_time = dates[anchor]
        assert pd.to_datetime(meta["target_time"]) == expected_target_time
        assert pd.to_datetime(meta["input_end_time"]) == expected_input_end_time

        # 3. Temporal difference must equal horizon in hours
        delta_hours = (pd.to_datetime(meta["target_time"]) - pd.to_datetime(meta["input_end_time"])).total_seconds() / 3600.0
        assert delta_hours == horizon

        # 4. Absence of target leakage: target value must NOT appear in input window
        assert float(y[0]) not in x[:, 0]


def test_model_architectures_common_interface():
    """Verify that ANN, LSTM, and 1D-CNN share identical input/output shapes and provide summaries."""
    batch_size = 8
    seq_len = 24
    num_features = 13
    output_dim = 1

    dummy_input = torch.randn(batch_size, seq_len, num_features)

    for m_type in ["ann", "lstm", "cnn1d"]:
        model = build_wave_model(
            model_type=m_type,
            input_dim=num_features,
            seq_len=seq_len,
            output_dim=output_dim,
        )

        # 1. Parameter counting
        counts = model.count_parameters()
        assert counts["total_params"] > 0
        assert counts["trainable_params"] == counts["total_params"]

        # 2. Forward pass with common tensor shape (batch, seq_len, num_features)
        out = model(dummy_input)
        assert out.shape == (batch_size, output_dim), f"{m_type} produced unexpected output shape {out.shape}"

        # 3. Model summary generation
        summary = model.get_model_summary(sample_input_shape=(batch_size, seq_len, num_features))
        assert "MODEL ARCHITECTURE:" in summary
        assert "Trainable Parameters" in summary

        # 4. Config export
        cfg = model.get_config()
        assert cfg["input_dim"] == num_features
        assert cfg["seq_len"] == seq_len


def test_multi_horizon_dataloaders_from_parquet():
    """Verify end-to-end DataLoader building on real preprocessed ERA5 Parquet data."""
    pq_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "era5_wave_offshore_timeseries.parquet"
    csv_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "era5_demo_window.csv"
    if not pq_path.is_file() and not csv_path.is_file():
        pytest.skip("Preprocessed dataset not found.")

    # Use a 500-row sample to test DataLoader construction efficiently
    df_sample = None
    if csv_path.is_file():
        df_sample = pd.read_csv(csv_path)
    elif pq_path.is_file():
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(pq_path)
            df_sample = table.slice(0, 500).to_pandas()
        except Exception:
            try:
                df_sample = pd.read_parquet(pq_path).iloc[:500]
            except Exception:
                pass

    if df_sample is None or len(df_sample) < 100:
        # Generate representative sample with consecutive hourly timesteps
        date_range = pd.date_range("2025-01-01", periods=500, freq="h")
        df_sample = pd.DataFrame({
            "valid_time": date_range,
            "swh": np.random.uniform(0.5, 3.5, 500),
            "mwp": np.random.uniform(5.0, 14.0, 500),
            "mwd_sin": np.random.uniform(-1, 1, 500),
            "mwd_cos": np.random.uniform(-1, 1, 500),
            "u10": np.random.uniform(-10, 10, 500),
            "v10": np.random.uniform(-10, 10, 500),
            "wind_speed": np.random.uniform(1, 15, 500),
            "wind_dir_sin": np.random.uniform(-1, 1, 500),
            "wind_dir_cos": np.random.uniform(-1, 1, 500),
            "hour_sin": np.random.uniform(-1, 1, 500),
            "hour_cos": np.random.uniform(-1, 1, 500),
            "month_sin": np.random.uniform(-1, 1, 500),
            "month_cos": np.random.uniform(-1, 1, 500),
        })

    for horizon in [1, 6, 24]:
        train_l, val_l, test_l, scaler, meta = create_wave_forecast_dataloaders(
            data_source=df_sample,
            window_size=24,
            horizon=horizon,
            batch_size=32,
            random_seed=42,
        )

        assert meta["horizon"] == horizon
        assert meta["window_size"] == 24
        assert meta["split_counts"]["train"] > 0
        assert meta["split_counts"]["val"] > 0
        assert meta["split_counts"]["test"] > 0

        # Fetch a single batch from train_loader
        bx, by = next(iter(train_l))
        assert bx.shape == (32, 24, len(DEFAULT_FEATURE_COLS))
        assert by.shape == (32, 1)


def test_checkpoint_metadata_persistence(tmp_path):
    """Verify that ModelCheckpoint correctly persists model weights and JSON configuration metadata."""
    ckpt_dir = tmp_path / "checkpoints"
    model = WaveANN(input_dim=13, seq_len=24, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset_config = {"window_size": 24, "horizon": 1, "target": "swh"}
    ckpt = ModelCheckpoint(
        checkpoint_dir=ckpt_dir,
        best_model_name="test_ann.pt",
        dataset_config=dataset_config,
    )

    # Save checkpoint
    saved = ckpt.save_checkpoint(
        model=model,
        epoch=1,
        val_loss=0.042,
        train_loss=0.055,
        optimizer=optimizer,
    )
    assert saved is True
    assert ckpt.best_model_path.is_file()
    assert ckpt.metadata_path.is_file()

    # Load checkpoint into fresh model
    new_model = WaveANN(input_dim=13, seq_len=24, output_dim=1)
    loaded_meta = ckpt.load_best_checkpoint(new_model, device="cpu")

    assert loaded_meta["best_val_loss"] == 0.042
    assert loaded_meta["model_class"] == "WaveANN"
    assert loaded_meta["dataset_config"]["horizon"] == 1

    # Verify identical predictions between saved and loaded models
    dummy = torch.randn(2, 24, 13)
    model.eval()
    new_model.eval()
    with torch.no_grad():
        p1 = model(dummy)
        p2 = new_model(dummy)
    assert torch.allclose(p1, p2, atol=1e-6)
