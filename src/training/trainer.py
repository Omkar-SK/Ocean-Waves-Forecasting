"""
PyTorch Training Loop Engine with Validation, Early Stopping, and Checkpointing.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .checkpoint import ModelCheckpoint


class ModelTrainer:
    """
    Standard training runner for deep learning wave forecasting models.
    Supports:
    - Validation loss monitoring
    - Early stopping with configurable patience and min_delta
    - Model checkpointing with metadata tracking
    - Learning rate scheduling
    - Out-of-sample prediction generation
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        device: Optional[str] = None,
        checkpoint: Optional[ModelCheckpoint] = None,
        scheduler: Optional[Any] = None,
        patience: int = 5,
        min_delta: float = 1e-4,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.MSELoss()
        self.checkpoint = checkpoint
        self.scheduler = scheduler
        self.patience = patience
        self.min_delta = min_delta

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch_duration_sec": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(batch_x)
            loss = self.criterion(preds, batch_y)
            loss.backward()

            # Gradient clipping to stabilize recurrent and deep layers
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            batch_count = len(batch_y)
            total_loss += loss.item() * batch_count
            n_samples += batch_count

        return total_loss / n_samples if n_samples > 0 else 0.0

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate loss on validation or test DataLoader."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)

                batch_count = len(batch_y)
                total_loss += loss.item() * batch_count
                n_samples += batch_count

        return total_loss / n_samples if n_samples > 0 else 0.0

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """Generate model predictions across an entire DataLoader."""
        self.model.eval()
        preds_list = []

        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x)
                preds_list.append(preds.detach().cpu().numpy())

        return np.concatenate(preds_list, axis=0).flatten()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute training loop with validation, early stopping, and checkpointing.
        """
        if verbose:
            print(f"\nTraining {self.model.__class__.__name__} on {self.device.upper()} for up to {epochs} epochs...")
            print(f"Early Stopping: Patience = {self.patience}, Min Delta = {self.min_delta}")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            t_loss = self.train_epoch(train_loader)
            v_loss = self.evaluate(val_loader)
            ep_duration = time.time() - ep_start

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(float(t_loss))
            self.history["val_loss"].append(float(v_loss))
            self.history["learning_rate"].append(float(current_lr))
            self.history["epoch_duration_sec"].append(float(ep_duration))

            # Step scheduler if present
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(v_loss)
                else:
                    self.scheduler.step()

            # Checkpoint & Early stopping evaluation
            improved = (self.best_val_loss - v_loss) > self.min_delta
            if improved:
                self.best_val_loss = v_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                if self.checkpoint is not None:
                    self.checkpoint.save_checkpoint(
                        model=self.model,
                        epoch=epoch,
                        val_loss=v_loss,
                        train_loss=t_loss,
                        optimizer=self.optimizer,
                    )
                best_marker = " [BEST SAVED]"
            else:
                self.epochs_without_improvement += 1
                best_marker = ""

            if verbose:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f} | "
                    f"LR: {current_lr:.1e} | Time: {ep_duration:.1f}s{best_marker}"
                )

            # Early stopping check
            if self.epochs_without_improvement >= self.patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}. Best validation loss: {self.best_val_loss:.5f} at epoch {self.best_epoch}.")
                break

        total_duration = time.time() - start_time
        if verbose:
            print(f"Training completed in {total_duration:.1f}s. Best Epoch: {self.best_epoch} (Val Loss: {self.best_val_loss:.5f})")

        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_duration_sec": total_duration,
        }
