"""
📖 About page.

Project information: objective, source of truth, dataset, technologies, folder
structure, author and links.
"""

from __future__ import annotations

import os

import streamlit as st

from src.constants import PROJECT_ROOT
from src.pages.ui_utils import page_header


def _tree_structure(root: str, prefix: str = "", max_depth: int = 3, depth: int = 0):
    """Build a simple text representation of the folder structure."""
    if depth >= max_depth:
        return
    entries = sorted(os.listdir(root))
    dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(root, e))]
    for d in dirs:
        yield f"{prefix}{d}/"
        yield from _tree_structure(
            os.path.join(root, d), prefix + "    ", max_depth, depth + 1
        )
    for f in files[:20]:
        yield f"{prefix}{f}"


def render_about() -> None:
    """Render the About page."""
    page_header(
        "📖 About",
        "About this project, the dataset, and the technologies used.",
    )

    st.markdown("### 🎯 Project Objective")
    st.markdown(
        """
        This project builds a **production-ready image classification application**
        for the **Animals-10** dataset. A custom **AlexNet** convolutional neural
        network classifies animal images into 10 categories. The application
        showcases best practices in Deep Learning engineering: modular code,
        configuration management, logging, exception handling, multi-backend
        inference (PyTorch / ONNX), and a professional Streamlit UI.
        """
    )

    st.markdown("### 📓 Notebook as Single Source of Truth")
    st.markdown(
        """
        The **Jupyter Notebook** in `notebooks/` is the **single source of truth**.
        All model architecture, preprocessing, augmentations, hyper-parameters,
        training pipeline and inference logic in this production app faithfully
        mirror the notebook.

        **The production app:**
        - Does **NOT** retrain the model.
        - Does **NOT** modify the architecture.
        - Does **NOT** change preprocessing / normalization / transforms.
        - Reuses the existing artifacts in `artifacts/` (loaded, never regenerated).
        """
    )

    st.markdown("### 📚 Dataset Source")
    st.markdown(
        """
        **Animals-10** dataset from [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10).
        It contains 10 classes:
        `cane` (dog), `cavallo` (horse), `elefante` (elephant), `farfalla` (butterfly),
        `gallina` (chicken), `gatto` (cat), `mucca` (cow), `pecora` (sheep),
        `ragno` (spider), `scoiattolo` (squirrel).
        """
    )

    st.markdown("### 🛠️ Technologies Used")
    st.markdown(
        """
        | Technology | Purpose |
        |-----------|---------|
        | Python | Primary language |
        | PyTorch | Model definition, training, inference |
        | Torchvision | Transforms, datasets |
        | ONNX Runtime | Optional accelerated inference |
        | Streamlit | Web UI |
        | Pandas | Data handling, results |
        | Plotly | Interactive charts |
        | Scikit-learn | Evaluation metrics |
        | OpenCV | Grad-CAM visualization |
        """
    )

    st.markdown("### 📂 Folder Structure")
    st.markdown(
        """
        ```
        Animals_10_classification/
        ├── app.py                 # Streamlit entry point
        ├── prediction.py          # CLI prediction entrypoint
        ├── requirements.txt
        ├── README.md
        ├── artifacts/             # Pre-trained outputs (read-only)
        ├── inference/             # Sample images for testing
        ├── notebooks/             # Authoritative notebook
        └── src/
            ├── components/        # Model, transforms, data loading
            ├── constants/         # Paths & hyper-parameters
            ├── entity/            # Dataclasses (config, results)
            ├── exception/         # Custom exceptions
            ├── logger/            # Logging setup
            ├── pages/             # Streamlit page modules
            ├── pipeline/          # Prediction & training pipelines
            └── utils/             # Helpers
        """
    )
    st.markdown(
        "**Key training hyper-parameters:** Epochs 25, Batch 32, LR 1e-3, "
        "AdamW (wd 1e-4), ReduceLROnPlateau, CrossEntropyLoss."
    )

    st.markdown("### 👤 Author & Links")
    st.markdown(
        """
        **Gourav Chhatwani** · Deep Learning Engineer

        | Link | URL |
        |------|-----|
        | 🌐 **GitHub** | [github.com/GouravGC](https://github.com/GouravGC) |
        | 💼 **LinkedIn** | [linkedin.com/in/gourav-chhatwani-9a301134a](https://www.linkedin.com/in/gourav-chhatwani-9a301134a/) |
        | 🚀 **Live App Demo** | [animal-10-classification-pytorch.streamlit.app](https://animal-10-classification-pytorch.streamlit.app/) |
        | 📊 **Kaggle Dataset** | [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) |

        **Model:** Custom AlexNet (2.50M params), Test Acc ~73.93%
        """
    )

