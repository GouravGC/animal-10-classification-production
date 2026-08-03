"""
📁 Dataset Information page.

Displays dataset metadata, class names, sample images, split details, and
pre-processing/augmentation information. All data is from the notebook config
and existing artifacts.
"""

from __future__ import annotations

import os

import streamlit as st

from src.constants import PLOTS_DIR
from src.pages.data_access import (
    get_model_config,
    load_cached_class_names,
)
from src.pages.ui_utils import metric_card, page_header


def render_dataset_info() -> None:
    """Render the Dataset Information page."""
    page_header(
        "📁 Dataset Information",
        "Details about the Animals-10 dataset and pre-processing used.",
    )

    cfg = get_model_config()
    class_names = load_cached_class_names()

    st.markdown("### 🗂️ Dataset Description")
    st.markdown(
        """
        **Animals-10** is a dataset of animal images from the **KAGGLE** platform.
        It contains **10 different animal classes** with images of varying sizes.

        The dataset is split using a **70% / 15% / 15%** train / validation / test
        split (with a fixed seed of 42 for reproducibility).
        """
    )

    # ------------------------------------------------------------------
    # Class summary
    # ------------------------------------------------------------------
    st.markdown("### 🐾 Classes")
    c1, c2, c3 = st.columns(3)
    metric_card("Number of Classes", str(cfg.num_classes))
    metric_card("Image Size", f"{cfg.image_size}×{cfg.image_size}")
    metric_card("Batch Size", str(cfg.batch_size))

    st.markdown("**Class names:**")
    st.markdown(" | ".join(f"`{c}`" for c in class_names))

    # Class distribution plot (existing artifact).
    st.markdown("### 📊 Class Distribution")
    dist_path = os.path.join(PLOTS_DIR, "class_distribution.png")
    if os.path.exists(dist_path):
        st.image(dist_path, use_column_width=True)
    else:
        st.info("Class distribution plot not found.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Split details
    # ------------------------------------------------------------------
    st.markdown("### ✂️ Train / Validation / Test Split")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training", "70%")
    col2.metric("Validation", "15%")
    col3.metric("Testing", "15%")

    st.caption("Split is performed with a fixed seed (42) for reproducibility.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    st.markdown("### 🧪 Pre-processing / Normalization")
    st.markdown(
        f"""
        **Inference transform** (exact from the notebook):
        ```
        Resize(({cfg.image_size}, {cfg.image_size}))
        → ToTensor()
        → Normalize(mean={cfg.mean}, std={cfg.std})
        ```
        """
    )

    st.markdown("### 🎨 Data Augmentation (Training only)")
    st.markdown(
        """
        The **augmented** experiments used additional training-time augmentations:
        - `Resize((256, 256))`
        - `RandomResizedCrop(224)`
        - `RandomHorizontalFlip()`
        - `RandomVerticalFlip(p=0.2)`
        - `RandomRotation(15)`
        - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`
        - `ToTensor()`
        - `Normalize(mean, std)`

        > Note: The **production inference** uses the *raw* (non-augmented) test
        > transform, exactly as the notebook does for evaluation.
        """
    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Sample images
    # ------------------------------------------------------------------
    st.markdown("### 🖼️ Sample Training Images")
    sample_path = os.path.join(PLOTS_DIR, "sample_training_images.png")
    if os.path.exists(sample_path):
        st.image(sample_path, use_column_width=True)
    else:
        st.info("Sample training images not found.")
