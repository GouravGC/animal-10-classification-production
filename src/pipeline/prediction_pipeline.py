"""
Prediction pipeline.

Loads the pre-trained best model (AlexNet - `best_model.pth`) and produces
predictions for input images. The logic is a faithful production version of the
notebook's inference behaviour:

- Pre-processing: Resize((224,224)) -> ToTensor() -> Normalize(mean, std)
- Forward pass: AlexNet (custom architecture from the notebook)
- Output: class indices + softmax probabilities

Two backends are supported:
- PyTorch (default): Uses `best_model.pth` with the custom AlexNet architecture.
- ONNX: Uses `best_model.onnx` via onnxruntime (optional, useful for deployment).

The model and artifacts are only LOADED, never regenerated or retrained.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.components.model import AlexNet
from src.components.transforms import get_inference_transform
from src.constants import (
    BACKEND_ONNX,
    BACKEND_PYTORCH,
    BEST_MODEL_ONNX,
    BEST_MODEL_PTH,
    CLASS_NAMES_JSON,
    CONFIG_JSON,
)
from src.entity import ModelConfig, PredictionResult
from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_class_names, load_config

logger = get_logger(__name__)

# Default Out-of-Distribution (OOD) confidence threshold. Below this confidence
# the image is treated as not belonging to any supported class. This is a pure
# inference-time safeguard and does NOT modify the trained model.
DEFAULT_OOD_THRESHOLD = 0.75


def _softmax_entropy(probs: np.ndarray) -> float:
    """
    Compute the Shannon entropy of a probability distribution (normalised).

    Used as an additional OOD signal. Higher entropy signals a more uncertain
    / out-of-distribution prediction.

    Args:
        probs (np.ndarray): 1D probability vector.

    Returns:
        float: Normalised entropy in [0, 1].
    """
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    # Normalise by log(n_classes) so the value lies in [0, 1].
    n_classes = probs.shape[0] if probs.shape[0] > 0 else 1
    return float(entropy / np.log(n_classes))


class PredictionPipeline:
    """
    Production inference pipeline for the Animals-10 classifier.

    Supports both PyTorch and ONNX backends. The default backend is PyTorch.
    """

    def __init__(
        self,
        backend: str = BACKEND_PYTORCH,
        model_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
        config_path: Optional[str] = None,
        class_names_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_OOD_THRESHOLD,
        auto_fallback: bool = True,
    ) -> None:
        """
        Initialize the prediction pipeline.

        Args:
            backend (str): Inference backend ('pytorch' or 'onnx').
            model_path (Optional[str]): Path to the PyTorch model weights.
            onnx_path (Optional[str]): Path to the ONNX model.
            config_path (Optional[str]): Path to config.json.
            class_names_path (Optional[str]): Path to class_names.json.
            confidence_threshold (float): OOD confidence threshold in [0,1].
            auto_fallback (bool): If True, fall back to PyTorch when the ONNX
                backend is unavailable. Defaults to True.
        """
        self.requested_backend = backend.lower()
        self.backend = self.requested_backend
        self.model_path = model_path or BEST_MODEL_PTH
        self.onnx_path = onnx_path or BEST_MODEL_ONNX
        self.config_path = config_path or CONFIG_JSON
        self.class_names_path = class_names_path or CLASS_NAMES_JSON
        self.confidence_threshold = confidence_threshold
        self.auto_fallback = auto_fallback

        if self.backend not in (BACKEND_PYTORCH, BACKEND_ONNX):
            raise CustomException(
                f"Unsupported backend '{self.backend}'. Choose 'pytorch' or 'onnx'."
            )

        # Load configuration and class names from the artifacts.
        self.class_names: List[str] = load_class_names(self.class_names_path)
        config_data = load_config(self.config_path)
        self.config = ModelConfig.from_dict(config_data)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Deterministic inference.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Build the inference transform (exact from the notebook).
        self.transform = get_inference_transform(
            image_size=self.config.image_size,
            mean=self.config.mean,
            std=self.config.std,
        )

        self._model = None
        self._onnx_session = None

        if self.backend == BACKEND_PYTORCH:
            self._load_pytorch_model()
        else:
            try:
                self._load_onnx_model()
            except CustomException as exc:
                if self.auto_fallback:
                    logger.warning(
                        "ONNX backend unavailable (%s). Falling back to PyTorch.",
                        exc,
                    )
                    self.backend = BACKEND_PYTORCH
                    self._load_pytorch_model()
                else:
                    raise

        logger.info(
            "Prediction pipeline initialized (backend=%s, device=%s)",
            self.backend,
            self.device,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_pytorch_model(self) -> None:
        """Load the PyTorch model from best_model.pth."""
        if not os.path.exists(self.model_path):
            raise CustomException(
                f"PyTorch model weights not found at {self.model_path}. "
                "Expected the artifact best_model.pth."
            )
        try:
            model = AlexNet(num_classes=self.config.num_classes)
            state_dict = torch.load(
                self.model_path, map_location=self.device, weights_only=True
            )
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model.to(self.device)
            logger.info("Loaded PyTorch model from %s", self.model_path)
        except Exception as exc:
            raise CustomException(f"Failed to load PyTorch model: {exc}")

    def _load_onnx_model(self) -> None:
        """Load the ONNX model via onnxruntime (optional)."""
        if not os.path.exists(self.onnx_path):
            raise CustomException(
                f"ONNX model not found at {self.onnx_path}. "
                "Expected the artifact best_model.onnx."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise CustomException(
                "onnxruntime is not installed. Install it via "
                "`pip install onnxruntime` or use the PyTorch backend."
            ) from exc

        try:
            self._onnx_session = ort.InferenceSession(
                self.onnx_path, providers=["CPUExecutionProvider"]
            )
            logger.info("Loaded ONNX model from %s", self.onnx_path)
        except Exception as exc:
            raise CustomException(f"Failed to load ONNX model: {exc}")

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    def _preprocess_image(
        self, image: Union[str, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """
        Pre-process a single image for inference.

        Args:
            image: Image path, PIL Image, or numpy array.

        Returns:
            torch.Tensor: A batched tensor ready for the model (1, C, H, W).
        """
        try:
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise CustomException(f"Image file not found: {image}")
                pil_image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image).convert("RGB")
            elif isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            else:
                raise CustomException(
                    f"Unsupported image type: {type(image)}. "
                    "Expected a path, PIL Image, or numpy array."
                )
        except CustomException:
            raise
        except Exception as exc:
            raise CustomException(f"Failed to open image: {exc}")

        tensor = self.transform(pil_image)  # (C, H, W)
        tensor = tensor.unsqueeze(0)  # (1, C, H, W)
        return tensor

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _predict_pytorch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run inference using the PyTorch model."""
        device_tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self._model(device_tensor)
        return logits.cpu()

    def _predict_onnx(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run inference using the ONNX session."""
        input_name = self._onnx_session.get_inputs()[0].name
        output_name = self._onnx_session.get_outputs()[0].name
        result = self._onnx_session.run([output_name], {input_name: tensor.numpy()})[0]
        return torch.from_numpy(result)

    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray],
        top_k: int = 5,
    ) -> PredictionResult:
        """
        Run inference on a single image and return a detailed result.

        Args:
            image: Image path, PIL Image, or numpy array.
            top_k (int): Number of top predictions to include.

        Returns:
            PredictionResult: Detailed prediction result.
        """
        tensor = self._preprocess_image(image)

        start_time = time.perf_counter()
        if self.backend == BACKEND_PYTORCH:
            logits = self._predict_pytorch(tensor)
        else:
            logits = self._predict_onnx(tensor)
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        probs = F.softmax(logits, dim=1).squeeze(0).numpy()

        predicted_index = int(np.argmax(probs))
        predicted_class = self.class_names[predicted_index]
        confidence = float(probs[predicted_index])
        entropy = _softmax_entropy(probs)

        # Out-of-Distribution safeguard: when the maximum softmax probability is
        # below the threshold we treat the input as not belonging to any of the
        # supported classes. This does NOT modify the trained model - it only
        # improves inference-time UX.
        is_ood = confidence < self.confidence_threshold

        # Build all probabilities dict.
        all_probs = {
            name: float(prob) for name, prob in zip(self.class_names, probs)
        }

        # Build top-k sorted list.
        sorted_indices = np.argsort(probs)[::-1][:top_k]
        top_k_list = [
            (self.class_names[int(i)], float(probs[int(i)])) for i in sorted_indices
        ]

        return PredictionResult(
            predicted_index=predicted_index,
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probs,
            top_k=top_k_list,
            is_ood=is_ood,
            confidence_threshold=self.confidence_threshold,
            entropy=entropy,
            inference_time_ms=inference_time_ms,
        )

    def predict_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        top_k: int = 5,
    ) -> List[PredictionResult]:
        """Run inference on a list of images."""
        return [self.predict(image, top_k=top_k) for image in images]


__all__ = ["PredictionPipeline"]

