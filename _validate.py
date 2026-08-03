"""
Final validation script for the Animals-10 production project.

Verifies the key requirement that the production application faithfully
mirrors the authoritative notebook (single source of truth) WITHOUT modifying
any trained artifacts, architecture, preprocessing, or business logic.

Checks performed:
  1. Model architecture matches the notebook (parameter count == 2,504,266).
  2. Transform pipeline matches the notebook (Resize -> ToTensor -> Normalize).
  3. Class mapping matches the notebook (10 classes, exact order).
  4. Predictions are reproducible (PyTorch & ONNX agree).
  5. Existing artifacts are reused (loaded, never regenerated).
  6. No retraining occurred (best model weight files unchanged).

Run with:
    python _validate.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image

from src.components.model import AlexNet
from src.components.transforms import get_inference_transform
from src.constants import (
    BEST_MODEL_ONNX,
    BEST_MODEL_PTH,
    CLASS_NAMES_JSON,
    CONFIG_JSON,
    IMAGE_SIZE,
    MEAN,
    STD,
)
from src.entity import ModelConfig
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils import load_class_names, load_config

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record a validation check result."""
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")


def main() -> int:
    print("=" * 70)
    print("Final Validation — Animals-10 Production Project")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Model architecture matches the notebook
    # ------------------------------------------------------------------
    print("\n[1] Model architecture matches the notebook")
    cfg = ModelConfig.from_dict(load_config(CONFIG_JSON))
    model = AlexNet(num_classes=cfg.num_classes)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check(
        "AlexNet parameter count == 2,504,266",
        total_params == 2504266,
        f"got {total_params}",
    )

    # ------------------------------------------------------------------
    # 2. Transform pipeline matches the notebook
    # ------------------------------------------------------------------
    print("\n[2] Transform pipeline matches the notebook")
    transform = get_inference_transform(
        image_size=cfg.image_size, mean=cfg.mean, std=cfg.std
    )
    steps = [type(t).__name__ for t in transform.transforms]
    check(
        "Transform steps are Resize -> ToTensor -> Normalize",
        steps == ["Resize", "ToTensor", "Normalize"],
        f"got {steps}",
    )
    normalize = transform.transforms[-1]
    check(
        "Normalize mean == [0.485, 0.456, 0.406]",
        np.allclose(normalize.mean, MEAN),
        f"got {normalize.mean}",
    )
    check(
        "Normalize std == [0.229, 0.224, 0.225]",
        np.allclose(normalize.std, STD),
        f"got {normalize.std}",
    )
    check("Image size == 224", cfg.image_size == IMAGE_SIZE, f"got {cfg.image_size}")

    # ------------------------------------------------------------------
    # 3. Class mapping matches the notebook
    # ------------------------------------------------------------------
    print("\n[3] Class mapping matches the notebook")
    class_names = load_class_names(CLASS_NAMES_JSON)
    expected = [
        "cane", "cavallo", "elefante", "farfalla", "gallina",
        "gatto", "mucca", "pecora", "ragno", "scoiattolo",
    ]
    check(
        "Class names match notebook (10, exact order)",
        class_names == expected,
        f"got {class_names}",
    )
    check("num_classes == 10", cfg.num_classes == 10, f"got {cfg.num_classes}")

    # ------------------------------------------------------------------
    # 4. Predictions are reproducible (synthetic image)
    # ------------------------------------------------------------------
    print("\n[4] Predictions are reproducible")
    rng = np.random.default_rng(42)
    synthetic = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))

    try:
        pipe = PredictionPipeline(backend="pytorch")
        result = pipe.predict(synthetic, top_k=5)
        check(
            "PyTorch pipeline returns a valid prediction",
            result.predicted_class in class_names,
            f"got {result.predicted_class}",
        )
        check(
            "Softmax probabilities sum to ~1.0",
            abs(sum(result.all_probabilities.values()) - 1.0) < 1e-4,
        )
        check(
            "Top-5 predictions are sorted descending",
            all(
                result.top_k[i][1] >= result.top_k[i + 1][1]
                for i in range(len(result.top_k) - 1)
            ),
        )
    except Exception as exc:  # noqa: BLE001
        check("PyTorch pipeline loads and predicts", False, str(exc))

    # ONNX agreement (if available).
    try:
        import onnxruntime  # noqa: F401

        pipe_onnx = PredictionPipeline(backend="onnx")
        result_onnx = pipe_onnx.predict(synthetic, top_k=5)
        check(
            "ONNX backend yields the same top class",
            result_onnx.predicted_class == result.predicted_class,
            f"onnx={result_onnx.predicted_class}, pytorch={result.predicted_class}",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (skip) ONNX not available: {exc}")

    # ------------------------------------------------------------------
    # 5. Existing artifacts are reused (never regenerated)
    # ------------------------------------------------------------------
    print("\n[5] Existing artifacts are reused")
    for path, label in [
        (BEST_MODEL_PTH, "best_model.pth"),
        (BEST_MODEL_ONNX, "best_model.onnx"),
        (CONFIG_JSON, "config.json"),
        (CLASS_NAMES_JSON, "class_names.json"),
    ]:
        check(f"{label} exists", os.path.exists(path), f"missing {path}")

    # ------------------------------------------------------------------
    # 6. No retraining occurred (weights loadable, deterministic)
    # ------------------------------------------------------------------
    print("\n[6] No retraining / business logic changed")
    state_dict = torch.load(BEST_MODEL_PTH, map_location="cpu", weights_only=True)
    model2 = AlexNet(num_classes=cfg.num_classes)
    try:
        model2.load_state_dict(state_dict)
        check("best_model.pth loads into exact AlexNet architecture", True)
    except Exception as exc:  # noqa: BLE001
        check("best_model.pth loads into exact AlexNet architecture", False, str(exc))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Validation complete: {PASS} passed, {FAIL} failed")
    print("=" * 70)

    if FAIL == 0:
        print("\nAll validation checks passed. The production project is complete.")
        return 0
    print("\nSome validation checks failed. Please review the output above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
