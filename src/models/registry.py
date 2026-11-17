"""
Central Deep Learning Model Registry for OceanCast AI.

Maintains metadata, checkpoint paths, scalers, and validation status for:
- 9 Original Absolute Models (WaveANN, WaveLSTM, WaveCNN1D for H=1, 6, 24)
- 9 Residual Models (predicting ΔSWH = SWH(t+H) - SWH(t))

Provides candidate model selection based on empirical benchmark results:
- +1h: Residual WaveLSTM (RMSE Skill: +29.76% vs persistence)
- +6h: Residual WaveANN  (RMSE Skill: +18.85% vs persistence)
- +24h: Residual WaveANN (RMSE Skill: +11.09% vs persistence)

Scientific Validation Boundary:
All models are strictly calibrated on the Mumbai Offshore reference location (18.50°N, 72.50°E).
Transfer to other coastal stations is explicitly marked as EXPERIMENTAL_UNVALIDATED.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
RESIDUAL_MODELS_DIR = MODELS_DIR / "residual"
ORIG_SCALER_PATH = MODELS_DIR / "scaler.pkl"


class ForecastMode(str, Enum):
    RESIDUAL = "residual"
    ABSOLUTE = "absolute"


class ModelArchitecture(str, Enum):
    ANN = "ann"
    LSTM = "lstm"
    CNN1D = "cnn1d"


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Immutable metadata and path configuration for a trained forecasting model."""
    model_id: str
    architecture: str
    display_name: str
    forecast_mode: ForecastMode
    horizon_hours: int
    checkpoint_path: Path
    metadata_path: Path
    input_scaler_path: Path
    residual_scaler_path: Optional[Path] = None
    target_type: str = "swh"  # "swh" or "delta_swh"
    reconstruction_formula: str = "Direct absolute prediction"
    model_version: str = "1.0.0"
    reference_location_id: str = "mumbai_offshore"
    rmse_skill_score_vs_persistence: Optional[float] = None
    description: str = ""

    def exists(self) -> bool:
        """Verify that checkpoint, metadata, and scalers exist on disk."""
        if not self.checkpoint_path.is_file():
            return False
        if not self.metadata_path.is_file():
            return False
        if not self.input_scaler_path.is_file():
            return False
        if self.forecast_mode == ForecastMode.RESIDUAL:
            if self.residual_scaler_path is None or not self.residual_scaler_path.is_file():
                return False
        return True

    def get_validation_status(self, location_id: str) -> str:
        """
        Geographic validation gating.
        Only the reference offshore station is scientifically validated.
        """
        if location_id == self.reference_location_id:
            return "reference_validated"
        return "experimental_unvalidated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "display_name": self.display_name,
            "forecast_mode": self.forecast_mode.value,
            "horizon_hours": self.horizon_hours,
            "checkpoint_file": self.checkpoint_path.name,
            "metadata_file": self.metadata_path.name,
            "input_scaler_file": self.input_scaler_path.name,
            "residual_scaler_file": self.residual_scaler_path.name if self.residual_scaler_path else None,
            "target_type": self.target_type,
            "reconstruction_formula": self.reconstruction_formula,
            "model_version": self.model_version,
            "reference_location_id": self.reference_location_id,
            "rmse_skill_score_vs_persistence": self.rmse_skill_score_vs_persistence,
            "description": self.description,
            "is_ready": self.exists(),
        }


# ==============================================================================
# Model Registry Catalog
# ==============================================================================

# Candidate recommended residual models based on rigorous held-out test benchmarks
CANDIDATE_RESIDUAL_MODELS: Dict[int, str] = {
    1: "residual_lstm_h1",
    6: "residual_ann_h6",
    24: "residual_ann_h24",
}

# Candidate recommended absolute models (prior benchmarks)
CANDIDATE_ABSOLUTE_MODELS: Dict[int, str] = {
    1: "absolute_lstm_h1",
    6: "absolute_lstm_h6",
    24: "absolute_ann_h24",
}


def _build_registry() -> Dict[str, ModelRegistryEntry]:
    """Build complete catalog of 9 absolute and 9 residual models."""
    entries: Dict[str, ModelRegistryEntry] = {}

    # 1. Nine Original Absolute Models
    abs_benchmarks = {
        ("lstm", 1): -27.20,
        ("ann", 1): -118.80,
        ("cnn1d", 1): -176.08,
        ("lstm", 6): 13.88,
        ("ann", 6): 8.77,
        ("cnn1d", 6): -5.42,
        ("ann", 24): -2.89,
        ("lstm", 24): -6.07,
        ("cnn1d", 24): -1.91,
    }

    for arch, display in [("ann", "WaveANN"), ("lstm", "WaveLSTM"), ("cnn1d", "WaveCNN1D")]:
        for h in [1, 6, 24]:
            model_id = f"absolute_{arch}_h{h}"
            ckpt_name = f"best_{arch}.pt" if h == 1 else f"best_{arch}_h{h}.pt"
            meta_name = f"best_{arch}_metadata.json" if h == 1 else f"best_{arch}_h{h}_metadata.json"

            entries[model_id] = ModelRegistryEntry(
                model_id=model_id,
                architecture=arch,
                display_name=display,
                forecast_mode=ForecastMode.ABSOLUTE,
                horizon_hours=h,
                checkpoint_path=MODELS_DIR / ckpt_name,
                metadata_path=MODELS_DIR / meta_name,
                input_scaler_path=ORIG_SCALER_PATH,
                residual_scaler_path=None,
                target_type="swh",
                reconstruction_formula="Direct absolute prediction: SWH(t+H)",
                model_version="1.0.0-absolute",
                rmse_skill_score_vs_persistence=abs_benchmarks.get((arch, h)),
                description=f"Standard absolute wave height forecaster ({display} +{h}h)",
            )

    # 2. Nine Validated Residual Models
    res_benchmarks = {
        ("lstm", 1): 29.76,
        ("cnn1d", 1): 28.39,
        ("ann", 1): 27.30,
        ("ann", 6): 18.85,
        ("cnn1d", 6): 17.16,
        ("lstm", 6): 12.26,
        ("ann", 24): 11.09,
        ("cnn1d", 24): 5.34,
        ("lstm", 24): 0.46,
    }

    for arch, display in [("ann", "Residual WaveANN"), ("lstm", "Residual WaveLSTM"), ("cnn1d", "Residual WaveCNN1D")]:
        for h in [1, 6, 24]:
            model_id = f"residual_{arch}_h{h}"
            ckpt_name = f"best_{arch}_residual_h{h}.pt"
            meta_name = f"best_{arch}_residual_h{h}_metadata.json"
            scaler_name = f"residual_scaler_h{h}.pkl"

            entries[model_id] = ModelRegistryEntry(
                model_id=model_id,
                architecture=arch,
                display_name=display,
                forecast_mode=ForecastMode.RESIDUAL,
                horizon_hours=h,
                checkpoint_path=RESIDUAL_MODELS_DIR / ckpt_name,
                metadata_path=RESIDUAL_MODELS_DIR / meta_name,
                input_scaler_path=ORIG_SCALER_PATH,
                residual_scaler_path=RESIDUAL_MODELS_DIR / scaler_name,
                target_type="delta_swh",
                reconstruction_formula=f"SWH(t+{h}) = SWH(t) + Predicted_ΔSWH",
                model_version="2.0.0-residual",
                rmse_skill_score_vs_persistence=res_benchmarks.get((arch, h)),
                description=f"Residual wave height change forecaster ({display} +{h}h)",
            )

    return entries


_REGISTRY: Dict[str, ModelRegistryEntry] = _build_registry()


class ModelRegistry:
    """Central registry lookup and management interface."""

    @staticmethod
    def get(model_id: str) -> Optional[ModelRegistryEntry]:
        """Fetch model configuration by unique model_id."""
        return _REGISTRY.get(model_id)

    @staticmethod
    def get_or_raise(model_id: str) -> ModelRegistryEntry:
        entry = _REGISTRY.get(model_id)
        if not entry:
            raise KeyError(f"Model ID '{model_id}' not found in registry. Known: {list(_REGISTRY.keys())}")
        return entry

    @staticmethod
    def get_candidate(horizon: int, mode: Union[ForecastMode, str] = ForecastMode.RESIDUAL) -> ModelRegistryEntry:
        """
        Returns the top-performing candidate model for a given horizon.

        Empirical benchmark winners:
        - +1h: Residual WaveLSTM (+29.76% RMSE skill)
        - +6h: Residual WaveANN  (+18.85% RMSE skill)
        - +24h: Residual WaveANN (+11.09% RMSE skill)
        """
        f_mode = ForecastMode(mode) if isinstance(mode, str) else mode
        if horizon not in [1, 6, 24]:
            raise ValueError(f"Unsupported horizon {horizon}h. Supported: [1, 6, 24].")

        if f_mode == ForecastMode.RESIDUAL:
            candidate_id = CANDIDATE_RESIDUAL_MODELS[horizon]
        else:
            candidate_id = CANDIDATE_ABSOLUTE_MODELS[horizon]

        return ModelRegistry.get_or_raise(candidate_id)

    @staticmethod
    def resolve_model(
        horizon: int,
        model_type: Optional[str] = None,
        mode: Union[ForecastMode, str] = ForecastMode.RESIDUAL,
    ) -> ModelRegistryEntry:
        """
        Resolve a specific architecture for a horizon, or default to the candidate winner.
        """
        f_mode = ForecastMode(mode) if isinstance(mode, str) else mode
        if horizon not in [1, 6, 24]:
            raise ValueError(f"Unsupported horizon {horizon}h. Supported: [1, 6, 24].")

        if not model_type or model_type.lower() in ["all", "best", "candidate", "auto"]:
            return ModelRegistry.get_candidate(horizon, mode=f_mode)

        clean_arch = model_type.lower().replace("residual", "").replace("wave", "").strip()
        if clean_arch not in ["ann", "lstm", "cnn1d"]:
            clean_arch = "lstm" if horizon == 1 else "ann"

        model_id = f"{f_mode.value}_{clean_arch}_h{horizon}"
        return ModelRegistry.get_or_raise(model_id)

    @staticmethod
    def list_all(mode: Optional[Union[ForecastMode, str]] = None) -> List[ModelRegistryEntry]:
        """List all registered models, optionally filtered by mode."""
        if mode is None:
            return list(_REGISTRY.values())
        f_mode = ForecastMode(mode) if isinstance(mode, str) else mode
        return [e for e in _REGISTRY.values() if e.forecast_mode == f_mode]

    @staticmethod
    def get_status_overview() -> Dict[str, Any]:
        """Returns readiness status of all 18 models."""
        total = len(_REGISTRY)
        ready = sum(1 for e in _REGISTRY.values() if e.exists())
        return {
            "total_models": total,
            "ready_models": ready,
            "all_ready": total == ready,
            "candidate_residual_models": {
                h: ModelRegistry.get_candidate(h, ForecastMode.RESIDUAL).to_dict()
                for h in [1, 6, 24]
            },
            "candidate_absolute_models": {
                h: ModelRegistry.get_candidate(h, ForecastMode.ABSOLUTE).to_dict()
                for h in [1, 6, 24]
            },
        }
