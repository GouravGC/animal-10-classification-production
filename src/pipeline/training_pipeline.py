"""
Training pipeline (reproducibility only).

This module faithfully mirrors the authoritative notebook's training logic
(optimizer, scheduler, early stopping, training/evaluation loops, model saving,
history saving, plotting). It is NOT executed automatically.

It exists so that a user can reproduce the notebook's experiments if the raw
dataset is available. The notebook's fixed hyper-parameters are preserved
exactly:

- Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau(mode='min', factor=0.1, patience=2, min_lr=1e-6)
- Loss: CrossEntropyLoss
- Early stopping: patience=5
- Batch size: 32, Epochs: 20 (used in the notebook cells that trained models),
  Image size: 224, seed: 42.

IMPORTANT: This module does NOT retrain anything automatically. It provides the
code path only if explicitly invoked by the user for reproducibility.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.components.model import AlexNet, LeNet, MiniCNN, initialize_weights
from src.constants import (
    CHECKPOINTS_DIR,
    EPOCHS,
    HISTORY_DIR,
    LEARNING_RATE,
    MODELS_DIR,
    PLOTS_DIR,
    SEED,
    EARLY_STOPPING_PATIENCE,
    OPTIMIZER_WEIGHT_DECAY,
    SCHEDULER_MODE,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SCHEDULER_MIN_LR,
)
from src.logger import get_logger

logger = get_logger(__name__)

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
# Helpers (mirror the notebook)
# ----------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_history(history: Dict[str, Any], file_name: str) -> None:
    history_df = pd.DataFrame(history)
    path = os.path.join(HISTORY_DIR, file_name + ".csv")
    history_df.to_csv(path, index=False)
    logger.info("Saved history to %s", path)


def save_pickle(data: Any, file_name: str) -> None:
    import joblib

    path = os.path.join(HISTORY_DIR, file_name + ".pkl")
    joblib.dump(data, path)
    logger.info("Saved pickle to %s", path)


def plot_history(history: Dict[str, Any], model_name: str, experiment: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Validation Loss")
    ax.set_title(f"{model_name} ({experiment}) Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_{experiment}_loss.png"), dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_acc"], label="Train Accuracy")
    ax.plot(history["val_acc"], label="Validation Accuracy")
    ax.set_title(f"{model_name} ({experiment}) Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_{experiment}_accuracy.png"), dpi=300)
    plt.close()

    logger.info("Plotted history for %s (%s)", model_name, experiment)


# ----------------------------------------------------------------------------
# Optimizer / scheduler / loss (exact from notebook)
# ----------------------------------------------------------------------------
def get_optimizer(model: nn.Module) -> optim.AdamW:
    return optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=OPTIMIZER_WEIGHT_DECAY)


def get_scheduler(optimizer: optim.Optimizer) -> optim.lr_scheduler.ReduceLROnPlateau:
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=SCHEDULER_MODE,
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )


class EarlyStopping:
    """Exact copy of the notebook's EarlyStopping class."""

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ----------------------------------------------------------------------------
# Training / validation loops (exact from notebook)
# ----------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


# ----------------------------------------------------------------------------
# Complete training function (exact from notebook)
# ----------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    model_name: str,
    experiment: str,
    device: torch.device,
    epochs: int = 20,
) -> tuple:
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        logger.info("%s | %s | Epoch %d/%d", model_name, experiment, epoch + 1, epochs)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(lr)

        logger.info(
            "Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f | LR: %.8f",
            train_loss, train_acc, val_loss, val_acc, lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

            torch.save(
                best_model_state,
                os.path.join(MODELS_DIR, f"{model_name}_{experiment}.pth"),
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                os.path.join(CHECKPOINTS_DIR, f"{model_name}_{experiment}_BEST.pt"),
            )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            },
            os.path.join(CHECKPOINTS_DIR, f"{model_name}_{experiment}_LAST.pt"),
        )

        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early Stopping Triggered")
            break

    model.load_state_dict(best_model_state)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss

    save_history(history, f"{model_name}_{experiment}")
    save_pickle(history, f"{model_name}_{experiment}")
    plot_history(history, model_name, experiment)

    logger.info(
        "Training finished for %s (%s). Best Epoch: %d, Best Val Loss: %.4f",
        model_name, experiment, best_epoch, best_val_loss,
    )
    return model, history


# ----------------------------------------------------------------------------
# Model factory (for reproducibility)
# ----------------------------------------------------------------------------
MODEL_REGISTRY = {
    "MiniCNN": MiniCNN,
    "LeNet": LeNet,
    "AlexNet": AlexNet,
}


def create_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """Create a model from the registry and initialize weights (notebook behaviour)."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[model_name](num_classes=num_classes)
    initialize_weights(model)
    return model.to(device)


__all__ = [
    "set_seed",
    "count_parameters",
    "save_history",
    "save_pickle",
    "plot_history",
    "get_optimizer",
    "get_scheduler",
    "EarlyStopping",
    "train_one_epoch",
    "validate_one_epoch",
    "train_model",
    "create_model",
]
