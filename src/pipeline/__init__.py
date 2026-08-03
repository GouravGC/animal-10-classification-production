"""
Pipeline package.

Contains:
- prediction_pipeline.py: Production inference pipeline (PyTorch + optional ONNX).
- training_pipeline.py: Reproducibility training pipeline (not auto-executed).
"""

from src.pipeline.prediction_pipeline import PredictionPipeline

__all__ = ["PredictionPipeline"]
