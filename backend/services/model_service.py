"""
Model inference and checkpoint caching service.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from backend.config import (
    MODELS_DIR,
    SCALER_PATH,
    SUPPORTED_HORIZONS,
    SUPPORTED_MODELS,
    FEATURE_COLS,
    WINDOW_SIZE,
)
from backend.schemas.forecast import SingleModelForecast
from src.data.dataset import TimeSeriesScaler
from src.data.live_service import get_sea_state_classification
from src.models import build_wave_model


class ModelService:
    """Manages lazy-loading and caching of PyTorch models and scalers in memory."""

    def __init__(self):
        self._scaler: Optional[TimeSeriesScaler] = None
        self._models: Dict[str, torch.nn.Module] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def get_scaler(self) -> TimeSeriesScaler:
        """Returns the pre-fitted TimeSeriesScaler, loading it once."""
        if self._scaler is None:
            if not SCALER_PATH.is_file():
                raise FileNotFoundError(f"Scaler artifact not found at {SCALER_PATH}")
            self._scaler = TimeSeriesScaler.load(SCALER_PATH)
        return self._scaler

    def get_model(self, model_type: str, horizon: int) -> Tuple[torch.nn.Module, Dict[str, Any], str]:
        """Loads and caches the PyTorch model for a given architecture and horizon."""
        m_code = model_type.lower().replace("wave", "").strip()
        if m_code not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type '{model_type}'. Choose from {SUPPORTED_MODELS}.")
        if horizon not in SUPPORTED_HORIZONS:
            raise ValueError(f"Unsupported horizon {horizon}h. Supported horizons: {SUPPORTED_HORIZONS}.")

        cache_key = f"{m_code}_h{horizon}"
        if cache_key in self._models:
            ckpt_name = f"best_{m_code}.pt" if horizon == 1 else f"best_{m_code}_h{horizon}.pt"
            return self._models[cache_key], self._metadata[cache_key], ckpt_name

        if horizon == 1:
            ckpt_name = f"best_{m_code}.pt"
            meta_name = f"best_{m_code}_metadata.json"
        else:
            ckpt_name = f"best_{m_code}_h{horizon}.pt"
            meta_name = f"best_{m_code}_h{horizon}_metadata.json"

        ckpt_path = MODELS_DIR / ckpt_name
        meta_path = MODELS_DIR / meta_name

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file '{ckpt_name}' missing from {MODELS_DIR}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Metadata file '{meta_name}' missing from {MODELS_DIR}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg = meta.get("model_config", {})
        model = build_wave_model(
            model_type=cfg.get("model_type", m_code),
            input_dim=cfg.get("input_dim", len(FEATURE_COLS)),
            seq_len=cfg.get("seq_len", WINDOW_SIZE),
            output_dim=cfg.get("output_dim", 1),
        )

        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
            model.load_state_dict(ckpt_data["model_state_dict"])
        else:
            model.load_state_dict(ckpt_data)

        model.eval()
        self._models[cache_key] = model
        self._metadata[cache_key] = meta
        return model, meta, ckpt_name

    def predict_single(
        self,
        prepared_df: pd.DataFrame,
        horizon: int,
        model_type: str,
        persistence_swh: float,
        external_nwp: Optional[float] = None,
    ) -> SingleModelForecast:
        """Executes forward pass inference for a single architecture."""
        scaler = self.get_scaler()
        if len(prepared_df) != WINDOW_SIZE:
            raise ValueError(f"Input sequence must have {WINDOW_SIZE} rows, found {len(prepared_df)}.")

        # Scaling feature inputs (NEVER refitting scaler!)
        scaled_feats = scaler.feature_scaler.transform(prepared_df[scaler.feature_names].values).astype(np.float32)
        tensor_x = torch.tensor(scaled_feats, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 24, 13]

        model, meta, ckpt_name = self.get_model(model_type, horizon)

        with torch.no_grad():
            out_scaled = model(tensor_x).cpu().numpy().squeeze()

        forecast_m = float(scaler.inverse_transform_target(np.array([[out_scaled]]))[0])
        delta_persist = forecast_m - persistence_swh
        delta_nwp = (forecast_m - external_nwp) if external_nwp is not None else None
        sea_desc, sea_col = get_sea_state_classification(forecast_m)

        model_display_names = {
            "lstm": "WaveLSTM",
            "ann": "WaveANN",
            "cnn1d": "WaveCNN1D",
        }
        m_key = model_type.lower().replace("wave", "")

        return SingleModelForecast(
            model_name=model_display_names.get(m_key, meta.get("model_class", model_type)),
            model_type=m_key.upper(),
            checkpoint_file=ckpt_name,
            forecast_swh_m=round(forecast_m, 4),
            delta_vs_persistence_m=round(delta_persist, 4),
            delta_vs_nwp_m=round(delta_nwp, 4) if delta_nwp is not None else None,
            sea_state_desc=sea_desc,
            sea_state_color=sea_col,
        )

    def verify_checkpoints_status(self) -> Dict[str, bool]:
        """Checks availability of all 9 trained checkpoints."""
        status = {}
        for m in SUPPORTED_MODELS:
            for h in SUPPORTED_HORIZONS:
                name = f"best_{m}.pt" if h == 1 else f"best_{m}_h{h}.pt"
                status[f"{m}_h{h}"] = (MODELS_DIR / name).is_file()
        return status


model_service = ModelService()
