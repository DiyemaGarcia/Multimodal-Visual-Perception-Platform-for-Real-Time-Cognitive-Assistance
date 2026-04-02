from __future__ import annotations
import torch
import shutil
from pathlib import Path
from typing import Dict, List, Optional


class EarlyStopping:
    """
    Monitors a validation metric and stops training when
    it stops improving for `patience` consecutive epochs.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        mode: str = "min",
        monitor: str = "val_loss",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.best_value: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, metrics: Dict[str, float]) -> bool:
        """
        Update the early stopping counter.

        Args:
            metrics: Dictionary of metric_name -> value.

        Returns:
            True if training should stop.
        """
        value = metrics.get(self.monitor)
        if value is None:
            return False

        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """
    Saves model checkpoints during training.
    Keeps the N best checkpoints and always saves the latest.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_elbo",
        mode: str = "min",
        keep_last: int = 3,
        save_best: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.keep_last = keep_last
        self.save_best = save_best
        self.best_value: Optional[float] = None
        self._saved_checkpoints: List[Path] = []

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Save a checkpoint if the monitored metric improved.

        Args:
            epoch:     Current epoch number.
            model:     The VAE model.
            optimizer: The optimizer.
            metrics:   Validation metrics dictionary.
            extra:     Additional data to include in the checkpoint.
        """
        value = metrics.get(self.monitor)
        is_best = False

        if value is not None:
            if self.best_value is None:
                is_best = True
            elif self.mode == "min" and value < self.best_value:
                is_best = True
            elif self.mode == "max" and value > self.best_value:
                is_best = True

        if is_best:
            self.best_value = value

        # Always save latest
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            **(extra or {}),
        }
        latest_path = self.save_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save epoch checkpoint
        epoch_path = self.save_dir / f"epoch_{epoch:04d}.pt"
        torch.save(checkpoint, epoch_path)
        self._saved_checkpoints.append(epoch_path)

        # Save best separately
        if is_best and self.save_best:
            best_path = self.save_dir / "best.pt"
            shutil.copy(epoch_path, best_path)

        # Prune old checkpoints
        self._prune()

    def _prune(self) -> None:
        """Remove oldest checkpoints beyond keep_last."""
        while len(self._saved_checkpoints) > self.keep_last:
            oldest = self._saved_checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()

    def load_best(self, device: torch.device) -> Dict:
        """Load the best saved checkpoint."""
        best_path = self.save_dir / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError("No best checkpoint found.")
        return torch.load(best_path, map_location=device)

    def load_latest(self, device: torch.device) -> Dict:
        """Load the latest saved checkpoint."""
        latest_path = self.save_dir / "latest.pt"
        if not latest_path.exists():
            raise FileNotFoundError("No latest checkpoint found.")
        return torch.load(latest_path, map_location=device)