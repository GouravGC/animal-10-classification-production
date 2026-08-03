"""
Command-line prediction entry point.

Allows classifying an image from the terminal using the pre-trained model.

Examples:
    python prediction.py path/to/image.jpg
    python prediction.py --backend onnx path/to/image.jpg
    python prediction.py --top-k 3 path/to/image.jpg
    python prediction.py --threshold 0.80 path/to/image.jpg
"""

from __future__ import annotations

import argparse
import os

from src.constants import BACKEND_ONNX, BACKEND_PYTORCH
from src.exception import CustomException
from src.logger import get_logger
from src.pipeline.prediction_pipeline import PredictionPipeline

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify an animal image using the pre-trained AlexNet model."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=[BACKEND_PYTORCH, BACKEND_ONNX],
        default=BACKEND_PYTORCH,
        help="Inference backend (default: pytorch).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to display (default: 5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="OOD confidence threshold (default: 0.75).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.image_path):
        logger.error("Image file not found: %s", args.image_path)
        raise SystemExit(1)

    try:
        pipeline = PredictionPipeline(
            backend=args.backend,
            confidence_threshold=args.threshold,
        )
        result = pipeline.predict(args.image_path, top_k=args.top_k)
    except CustomException as exc:
        logger.error("Prediction pipeline error: %s", exc)
        raise SystemExit(1) from exc

    print("\n" + "=" * 46)
    print("Prediction Result")
    print("=" * 46)
    print(f"Class      : {result.predicted_class}")
    print(f"Confidence : {result.confidence * 100:.2f}%")
    print(f"Inference  : {result.inference_time_ms:.1f} ms")
    print(f"Entropy    : {result.entropy:.4f}")

    if result.is_ood:
        print("\n⚠️  OOD FLAGGED")
        print(f"> {result.ood_message}")

    print(f"\nTop-{args.top_k} Predictions")
    print("-" * 46)
    for name, conf in result.top_k:
        print(f"{name:<16}{conf * 100:.2f}%")
    print("=" * 46)


if __name__ == "__main__":
    main()
