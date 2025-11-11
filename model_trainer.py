"""Model training class for AltaStata ChRIS demo.

This module contains the ModelTrainer class that encapsulates model definition,
instantiation, and training logic.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Training parameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 10
PATIENCE = 5
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.2
SEED = 42

# Default training parameters (for backward compatibility)
DEFAULT_LEARNING_RATE = LEARNING_RATE
DEFAULT_WEIGHT_DECAY = WEIGHT_DECAY
DEFAULT_EPOCHS = EPOCHS
DEFAULT_PATIENCE = PATIENCE


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    seconds: float


class ModelTrainer:
    """Encapsulates model definition, instantiation, and training logic."""

    class SimpleCNN(nn.Module):
        """Same architecture used in the AltaStata reference notebook."""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 12 * 12, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        epochs: int = DEFAULT_EPOCHS,
        patience: int = DEFAULT_PATIENCE,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the model trainer with data loaders and training parameters.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            learning_rate: Learning rate for optimizer (default: 1e-4)
            weight_decay: Weight decay for optimizer (default: 1e-4)
            epochs: Maximum number of epochs to train (default: 25)
            patience: Early stopping patience (default: 5)
            device: Device to train on (default: cuda if available, else cpu)
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        self.model = self.SimpleCNN().to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info("Model contains %s trainable parameters.", f"{total_params:,}")

        # Create criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Training state
        self.metrics: List[EpochMetrics] = []
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _run_epoch(self, loader: DataLoader, is_training: bool) -> Tuple[float, float]:
        """Run a single epoch (training or validation).

        Args:
            loader: DataLoader for the epoch
            is_training: Whether this is a training epoch

        Returns:
            Tuple of (average_loss, accuracy_percentage)
        """
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            if is_training:
                self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            if is_training:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        average_loss = total_loss / max(total_samples, 1)
        accuracy = 100.0 * total_correct / max(total_samples, 1)
        return average_loss, accuracy

    def train(self) -> Tuple[Optional[Dict[str, torch.Tensor]], List[EpochMetrics]]:
        """Run the training loop with early stopping.

        Returns:
            Tuple of (best_model_state_dict, list_of_epoch_metrics)
        """
        logging.info("Starting training for %d epochs (patience %d).", self.epochs, self.patience)

        for epoch in range(1, self.epochs + 1):
            start = time.time()
            train_loss, train_acc = self._run_epoch(self.train_loader, is_training=True)
            with torch.no_grad():
                val_loss, val_acc = self._run_epoch(self.val_loader, is_training=False)

            elapsed = time.time() - start
            self.metrics.append(EpochMetrics(epoch, train_loss, train_acc, val_loss, val_acc, elapsed))

            logging.info(
                "Epoch %03d | train_loss=%.4f train_acc=%.1f%% | val_loss=%.4f val_acc=%.1f%% | %.1fs",
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                elapsed,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = self.model.state_dict()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logging.info("Early stopping triggered after %d epochs.", epoch)
                    break

        return self.best_state, self.metrics


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def set_random_seed() -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def split_indices(dataset_size: int) -> Tuple[List[int], List[int]]:
    """Split dataset indices into training and validation sets.

    Args:
        dataset_size: Total number of samples in the dataset

    Returns:
        Tuple of (train_indices, val_indices)
    """
    indices = np.arange(dataset_size)
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    return indices[split:].tolist(), indices[:split].tolist()


def write_summary(filename: str, metrics: List[EpochMetrics], dataset_size: int, output_dir: Optional[Path] = None) -> None:
    """Write training summary to a JSON file.

    Args:
        filename: Name of the summary file
        metrics: List of epoch metrics
        dataset_size: Total dataset size
        output_dir: Directory to write the summary file (default: current script directory)
    """
    if output_dir is None:
        # Write to root directory (where the script is located)
        output_dir = Path(__file__).parent
    summary_path = output_dir / filename
    payload = {
        "dataset_size": dataset_size,
        "validation_split": VALIDATION_SPLIT,
        "epochs_recorded": len(metrics),
        "metrics": [asdict(item) for item in metrics],
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    logging.info("Training summary written to %s", summary_path)

