"""
Model checkpointing and state persistence with comprehensive metadata tracking.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch


class ModelCheckpoint:
    """
    Handles saving and restoring model states, optimizer states, training history,
    and complete dataset/model configurations.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        best_model_name: str = "best_model.pt",
        model_config: Optional[Dict[str, Any]] = None,
        dataset_config: Optional[Dict[str, Any]] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / best_model_name
        self.metadata_path = self.checkpoint_dir / f"{Path(best_model_name).stem}_metadata.json"
        self.best_loss = float("inf")
        self.model_config = model_config or {}
        self.dataset_config = dataset_config or {}

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        val_loss: float,
        train_loss: Optional[float] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save model and complete metadata if validation loss improves.
        """
        is_best = val_loss < self.best_loss
        if is_best:
            self.best_loss = val_loss

            # Extract model config if available
            m_config = dict(self.model_config)
            if hasattr(model, "get_config"):
                m_config.update(model.get_config())

            param_counts = {}
            if hasattr(model, "count_parameters"):
                param_counts = model.count_parameters()

            metadata = {
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "epoch": epoch,
                "best_val_loss": float(val_loss),
                "train_loss": float(train_loss) if train_loss is not None else None,
                "model_class": model.__class__.__name__,
                "model_config": m_config,
                "param_counts": param_counts,
                "dataset_config": self.dataset_config,
                "extra": extra_meta or {},
            }

            payload = {
                "epoch": epoch,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "metadata": metadata,
            }

            # Save binary checkpoint
            torch.save(payload, self.best_model_path)

            # Save readable JSON metadata
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        return is_best

    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load weights from best checkpoint into provided model and optional optimizer.
        """
        if not self.best_model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.best_model_path}")
        checkpoint = torch.load(self.best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("metadata", {})
