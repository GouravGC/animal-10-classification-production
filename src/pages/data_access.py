"""
Cached data-access helpers for the Streamlit application.

These functions load the shared, read-only artifacts (config, class names,
model comparison, plots, reports) once and cache them so that the UI does not
reload them on every interaction. No artifacts are ever regenerated.
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
import streamlit as st

from src.constants import (
    CLASS_NAMES_JSON,
    CONFIG_JSON,
    MODEL_COMPARISON_CSV,
    PLOTS_DIR,
    REPORTS_DIR,
)
from src.entity import ModelConfig
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils import load_class_names, load_config


@st.cache_data
def load_cached_config() -> Dict:
    """Load the config.json (cached)."""
    return load_config(CONFIG_JSON)


@st.cache_data
def load_cached_class_names() -> List[str]:
    """Load the class_names.json (cached)."""
    return load_class_names(CLASS_NAMES_JSON)


@st.cache_resource
def get_cached_pipeline(backend: str, threshold: float = 0.75) -> PredictionPipeline:
    """Create (and cache) the prediction pipeline for a given backend."""
    return PredictionPipeline(backend=backend, confidence_threshold=threshold)


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    """Load the model comparison CSV (cached)."""
    return pd.read_csv(MODEL_COMPARISON_CSV)


@st.cache_data
def load_best_metrics() -> Dict:
    """Return the best-model metrics (AlexNet Raw) from the comparison CSV."""
    df = load_model_comparison()
    best = df.sort_values("F1 Score", ascending=False).iloc[0]
    return {
        "model": best["Model"],
        "dataset": best["Dataset"],
        "test_accuracy": float(best["Test Accuracy"]),
        "precision": float(best["Precision"]),
        "recall": float(best["Recall"]),
        "f1": float(best["F1 Score"]),
        "params": int(best["Parameters"]),
    }


@st.cache_data
def get_model_config() -> ModelConfig:
    cfg = load_cached_config()
    return ModelConfig.from_dict(cfg)


def get_plot_path(filename: str) -> str:
    """Return the absolute path to a plot file in the plots directory."""
    return os.path.join(PLOTS_DIR, filename)


def get_report_path(filename: str) -> str:
    """Return the absolute path to a report file in the reports directory."""
    return os.path.join(REPORTS_DIR, filename)


def get_plot(filename: str) -> bool:
    """Check if a plot file exists (used to avoid regenerating artifacts)."""
    return os.path.exists(get_plot_path(filename))
