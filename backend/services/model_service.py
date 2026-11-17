"""
Deep Learning Inference and Model Management Service.
Supports both Validated Residual (ΔSWH) Forecasting and Original Absolute Inference.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
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
from src.models.registry import ForecastMode, ModelRegistry, ModelRegistryEntry


class ModelService:
    """
    Manages lazy-loading, caching, and inference for:
    - 9 Validated Residual Deep Learning Models (WaveANN, WaveLSTM, WaveCNN1D x H=1, 6, 24)
    - 9 Original Absolute Baseline Models
    - Pre-fitted Input Scaler and Horizon-specific Residual Target Scalers
    """

    def __init__(self):
        self._scaler: Optional[TimeSeriesScaler] = None
        self._models: Dict[str, torch.nn.Module] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._residual_scalers: Dict[int, Dict[str, Any]] = {}

    def get_scaler(self) -> TimeSeriesScaler:
        """Returns the pre-fitted TimeSeriesScaler for input features, loading once."""
        if self._scaler is None:
            if not SCALER_PATH.is_file():
                raise FileNotFoundError(f"Scaler artifact not found at {SCALER_PATH}")
            self._scaler = TimeSeriesScaler.load(SCALER_PATH)
        return self._scaler

    def get_residual_scaler(self, horizon: int) -> Dict[str, Any]:
        """Loads and caches the per-horizon residual target scaler."""
        if horizon not in SUPPORTED_HORIZONS:
            raise ValueError(f"Unsupported horizon {horizon}h. Supported: {SUPPORTED_HORIZONS}")
        if horizon not in self._residual_scalers:
            scaler_path = MODELS_DIR / "residual" / f"residual_scaler_h{horizon}.pkl"
            if not scaler_path.is_file():
                raise FileNotFoundError(f"Residual scaler file missing: {scaler_path}")
            with open(scaler_path, "rb") as f:
                self._residual_scalers[horizon] = pickle.load(f)
        return self._residual_scalers[horizon]

    def get_model(
        self,
        model_type: str,
        horizon: int,
        mode: Union[ForecastMode, str] = ForecastMode.RESIDUAL,
    ) -> Tuple[torch.nn.Module, Dict[str, Any], str, ModelRegistryEntry]:
        """
        Loads and caches PyTorch model architecture and weights from the central registry.
        """
        f_mode = ForecastMode(mode) if isinstance(mode, str) else mode
        reg_entry = ModelRegistry.resolve_model(horizon, model_type, mode=f_mode)
        cache_key = reg_entry.model_id

        if cache_key in self._models:
            return (
                self._models[cache_key],
                self._metadata[cache_key],
                reg_entry.checkpoint_path.name,
                reg_entry,
            )

        if not reg_entry.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file '{reg_entry.checkpoint_path.name}' missing at {reg_entry.checkpoint_path}"
            )
        if not reg_entry.metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata file '{reg_entry.metadata_path.name}' missing at {reg_entry.metadata_path}"
            )

        with open(reg_entry.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg = meta.get("model_config", {})
        arch_type = cfg.get("model_type", reg_entry.architecture)

        model = build_wave_model(
            model_type=arch_type,
            input_dim=cfg.get("input_dim", len(FEATURE_COLS)),
            seq_len=cfg.get("seq_len", WINDOW_SIZE),
            output_dim=cfg.get("output_dim", 1),
        )

        ckpt_data = torch.load(reg_entry.checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
            model.load_state_dict(ckpt_data["model_state_dict"])
        else:
            model.load_state_dict(ckpt_data)

        model.eval()
        self._models[cache_key] = model
        self._metadata[cache_key] = meta

        return model, meta, reg_entry.checkpoint_path.name, reg_entry

    def predict_single(
        self,
        prepared_df: pd.DataFrame,
        horizon: int,
        model_type: str,
        persistence_swh: float,
        external_nwp: Optional[float] = None,
        mode: Union[ForecastMode, str] = ForecastMode.RESIDUAL,
        location_id: str = "mumbai_offshore",
    ) -> SingleModelForecast:
        """
        Executes forward-pass wave forecasting.
        
        When mode == 'residual':
            1. Scales input features with training scaler.
            2. Predicts normalized ΔSWH.
            3. Inverts residual scaling using horizon-specific residual scaler.
            4. Reconstructs: SWH(t+H) = SWH(t) + Predicted_ΔSWH.
            5. Enforces physical oceanographic plausibility (SWH >= 0, finite).
            
        When mode == 'absolute':
            1. Scales input features with training scaler.
            2. Predicts normalized SWH.
            3. Inverts scaling to meters directly.
        """
        f_mode = ForecastMode(mode) if isinstance(mode, str) else mode
        scaler = self.get_scaler()

        if len(prepared_df) != WINDOW_SIZE:
            raise ValueError(f"Input sequence must have {WINDOW_SIZE} rows, found {len(prepared_df)}.")

        # Extract current observed wave height at latest step t
        current_swh = float(prepared_df["swh"].iloc[-1])

        # Step 1: Feature normalization using training scaler (NEVER refitting!)
        scaled_feats = scaler.feature_scaler.transform(
            prepared_df[scaler.feature_names].values
        ).astype(np.float32)
        tensor_x = torch.tensor(scaled_feats, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 24, 13]

        # Step 2: Resolve model from registry
        model, meta, ckpt_name, reg_entry = self.get_model(model_type, horizon, mode=f_mode)

        # Step 3: Forward inference pass
        with torch.no_grad():
            out_scaled = float(model(tensor_x).cpu().numpy().squeeze())

        # Step 4: Reconstruct forecast depending on mode
        if f_mode == ForecastMode.RESIDUAL:
            # Load horizon-specific residual scaler
            res_scaler = self.get_residual_scaler(horizon)
            rt_mean = float(res_scaler["mean"])
            rt_std = float(res_scaler["std"])

            # Inverse-transform ΔSWH
            pred_delta_m = out_scaled * rt_std + rt_mean

            # Reconstruct SWH: SWH(t+H) = SWH(t) + ΔSWH
            forecast_m = current_swh + pred_delta_m

            # Physical sanity verification
            if not np.isfinite(forecast_m):
                raise ValueError(f"Model prediction produced non-finite SWH value: {forecast_m}")
            # Wave height physical non-negativity constraint
            forecast_m = max(0.0, forecast_m)
            # Physical ocean wave ceiling check
            if forecast_m > 25.0:
                forecast_m = 25.0
        else:
            # Absolute model direct inverse transform
            forecast_m = float(scaler.inverse_transform_target(np.array([[out_scaled]]))[0])
            pred_delta_m = forecast_m - current_swh
            if not np.isfinite(forecast_m):
                raise ValueError(f"Model prediction produced non-finite SWH value: {forecast_m}")
            forecast_m = max(0.0, forecast_m)

        delta_persist = forecast_m - persistence_swh
        delta_nwp = (forecast_m - external_nwp) if external_nwp is not None else None
        sea_desc, sea_col = get_sea_state_classification(forecast_m)

        # Check if this model is the recommended candidate for this horizon
        candidate_entry = ModelRegistry.get_candidate(horizon, mode=f_mode)
        is_candidate = (reg_entry.model_id == candidate_entry.model_id)

        # Validation status for this location
        validation_status = reg_entry.get_validation_status(location_id)

        return SingleModelForecast(
            model_name=reg_entry.display_name,
            model_type=reg_entry.architecture.upper(),
            checkpoint_file=ckpt_name,
            forecast_swh_m=round(forecast_m, 4),
            delta_vs_persistence_m=round(delta_persist, 4),
            delta_vs_nwp_m=round(delta_nwp, 4) if delta_nwp is not None else None,
            sea_state_desc=sea_desc,
            sea_state_color=sea_col,
            # Phase 5 additions
            current_swh_m=round(current_swh, 4),
            predicted_residual_m=round(pred_delta_m, 4),
            forecast_mode=f_mode.value,
            model_version=reg_entry.model_version,
            validation_status=validation_status,
            rmse_skill_score=reg_entry.rmse_skill_score_vs_persistence,
            is_candidate_model=is_candidate,
            reconstruction_formula=reg_entry.reconstruction_formula,
            # Explicit aliases
            predicted_swh=round(forecast_m, 4),
            current_swh=round(current_swh, 4),
            predicted_residual=round(pred_delta_m, 4),
            selected_model_architecture=reg_entry.display_name,
            model_validation_status=validation_status,
        )

    def verify_checkpoints_status(self) -> Dict[str, bool]:
        """Checks availability of all 9 original trained checkpoints."""
        status = {}
        for m in SUPPORTED_MODELS:
            for h in SUPPORTED_HORIZONS:
                name = f"best_{m}.pt" if h == 1 else f"best_{m}_h{h}.pt"
                status[f"{m}_h{h}"] = (MODELS_DIR / name).is_file()
        return status

    def verify_residual_checkpoints_status(self) -> Dict[str, bool]:
        """Checks availability of all 9 residual trained checkpoints."""
        status = {}
        for m in SUPPORTED_MODELS:
            for h in SUPPORTED_HORIZONS:
                name = f"best_{m}_residual_h{h}.pt"
                status[f"residual_{m}_h{h}"] = (MODELS_DIR / "residual" / name).is_file()
        return status

    def verify_all_models_status(self) -> Dict[str, bool]:
        """Checks availability of all 18 models (9 absolute + 9 residual)."""
        status = {}
        for entry in ModelRegistry.list_all():
            status[entry.model_id] = entry.exists()
        return status


model_service = ModelService()
