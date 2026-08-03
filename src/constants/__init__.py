"""
Centralized project constants and path definitions.

These constants mirror the configuration used in the authoritative notebook
(notebooks/animals-10-classification-updated.ipynb) and the saved
artifacts/reports/config.json. They are the single place where project-level
paths and fixed hyper-parameters are defined.
"""

from __future__ import annotations

import os

# ----------------------------------------------------------------------------
# Project root (directory containing the `src` package)
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# ----------------------------------------------------------------------------
# Artifacts directory (pre-trained outputs from the notebook - READ ONLY)
# ----------------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")
HISTORY_DIR = os.path.join(ARTIFACTS_DIR, "history")
CHECKPOINTS_DIR = os.path.join(ARTIFACTS_DIR, "checkpoints")

# ----------------------------------------------------------------------------
# Pre-trained artifact file paths (loaded, never regenerated)
# ----------------------------------------------------------------------------
BEST_MODEL_PTH = os.path.join(ARTIFACTS_DIR, "best_model.pth")
BEST_MODEL_ONNX = os.path.join(ARTIFACTS_DIR, "best_model.onnx")
CONFIG_JSON = os.path.join(REPORTS_DIR, "config.json")
CLASS_NAMES_JSON = os.path.join(REPORTS_DIR, "class_names.json")
MODEL_COMPARISON_CSV = os.path.join(REPORTS_DIR, "model_comparison.csv")

# ----------------------------------------------------------------------------
# Inference folder (for user-supplied sample images during testing)
# ----------------------------------------------------------------------------
INFERENCE_DIR = os.path.join(PROJECT_ROOT, "inference")
SAMPLE_IMAGES_DIR = os.path.join(INFERENCE_DIR, "sample_images")

# ----------------------------------------------------------------------------
# Fixed hyper-parameters / pre-processing values (from the notebook)
# WARNING: These MUST NOT be changed. They define the exact behavior of the
# validated experiment and are required to reproduce identical predictions.
# ----------------------------------------------------------------------------
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
NUM_WORKERS = 2
NUM_CLASSES = 10

# Normalization statistics (ImageNet) used by the notebook transforms.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Optimizer / scheduler configuration used during training (reproducibility).
OPTIMIZER_WEIGHT_DECAY = 1e-4
SCHEDULER_MODE = "min"
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 2
SCHEDULER_MIN_LR = 1e-6
EARLY_STOPPING_PATIENCE = 5

# Available inference backends.
BACKEND_PYTORCH = "pytorch"
BACKEND_ONNX = "onnx"
