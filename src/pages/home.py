"""
🏠 Home page - project overview, model summary, quick metrics and quick demo.
"""

from __future__ import annotations

import os

import streamlit as st

from src.constants import PLOTS_DIR, SAMPLE_IMAGES_DIR
from src.pages.data_access import (
    get_model_config,
    load_best_metrics,
    load_cached_class_names,
    load_model_comparison,
)
from src.pages.ui_utils import hero, metric_card, page_header


def render_home() -> None:
    """Render the Home page."""
    page_header("🏠 Home", "Animals-10 Image Classification — Project Overview")

    hero(
        "🐾 Animals-10 Image Classification",
        "A custom AlexNet CNN trained to classify 10 animal species with "
        "production-grade Streamlit deployment.",
    )

    # ------------------------------------------------------------------
    # Model summary metrics
    # ------------------------------------------------------------------
    st.markdown("### 📊 Model Summary")
    cfg = get_model_config()
    metrics = load_best_metrics()

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card("Architecture", f"{metrics['model']} ({metrics['dataset']})")
    metric_card("Classes", str(cfg.num_classes))
    metric_card("Image Size", f"{cfg.image_size}×{cfg.image_size}")
    metric_card("Test Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
    metric_card("F1 Score", f"{metrics['f1']:.4f}")

    c6, c7, c8, c9 = st.columns(4)
    metric_card("Precision", f"{metrics['precision']*100:.2f}%")
    metric_card("Recall", f"{metrics['recall']*100:.2f}%")
    metric_card("Parameters", f"{metrics['params']/1e6:.2f}M")
    metric_card("Backend", "PyTorch / ONNX")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Two-column layout: description + quick demo
    # ------------------------------------------------------------------
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🎯 Project Objective")
        st.markdown(
            """
            This project classifies images of animals into **10 classes** using a
            **custom AlexNet** convolutional neural network. The model was trained
            on the **Animals-10** dataset (KAGGLE) and is served here as a
            production-ready Streamlit application.

            **Key features:**
            - Custom AlexNet architecture (2.50M parameters)
            - Exact preprocessing from the notebook (`Resize → ToTensor → Normalize`)
            - Out-of-Distribution (OOD) safeguard for unknown images
            - PyTorch & ONNX inference backends
            - Multi-page professional UI

            **Author:** Gourav Chhatwani
            """
        )

        st.markdown("### 🧬 Architecture")
        st.markdown(
            """
            | Layer | Details |
            |-------|---------|
            | Conv2d | 64 filters, 11×11, stride 4, pad 2 |
            | MaxPool | 3×3, stride 2 |
            | Conv2d | 192 filters, 5×5, pad 2 + BatchNorm |
            | MaxPool | 3×3, stride 2 |
            | Conv2d | 384 filters, 3×3, pad 1 |
            | Conv2d | 256 filters, 3×3, pad 1 |
            | Conv2d | 256 filters, 3×3, pad 1 |
            | MaxPool | 3×3, stride 2 |
            | AdaptiveAvgPool | (1,1) |
            | Classifier | Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→10) |
            """
        )

    with col_right:
        st.markdown("### 🚀 Quick Demo")
        st.markdown(
            """
            **Try the model right now:**
            - Go to **🔍 Single Image Prediction** to upload an image.
            - Use **📂 Batch Prediction** to classify multiple images.
            - Explore **📊 Model Performance** for evaluation results.
            - View **📁 Dataset Information** and **🧠 Model Details**.
            """
        )

        # Dataset summary.
        st.markdown("### 📁 Dataset Summary")
        st.markdown(
            f"""
            | Property | Value |
            |----------|-------|
            | **Dataset** | Animals-10 (Kaggle) |
            | **Classes** | {cfg.num_classes} |
            | **Images** | ~26,179 |
            | **Split** | 70/15/15 (seed 42) |
            | **Image Size** | {cfg.image_size}×{cfg.image_size} |
            | **Preprocessing** | Resize → ToTensor → Normalize |
            """
        )

        # Sample images if available.
        sample_files = []
        if os.path.isdir(SAMPLE_IMAGES_DIR):
            sample_files = [
                os.path.join(SAMPLE_IMAGES_DIR, f)
                for f in sorted(os.listdir(SAMPLE_IMAGES_DIR))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        if sample_files:
            st.markdown("**Sample images available:**")
            for f in sample_files[:4]:
                st.image(f, caption=os.path.basename(f), use_column_width=True)
        else:
            st.info("Add images to `inference/sample_images/` for quick demos.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Model comparison table
    # ------------------------------------------------------------------
    st.markdown("### 📋 Model Comparison")
    comparison = load_model_comparison()
    st.dataframe(comparison, use_container_width=True)

    st.markdown("---")

    # Sample training images.
    st.markdown("### 🖼️ Sample Training Images")
    sample_path = os.path.join(PLOTS_DIR, "sample_training_images.png")
    if os.path.exists(sample_path):
        st.image(sample_path, use_column_width=True)
    else:
        st.info("Sample training images not found.")
