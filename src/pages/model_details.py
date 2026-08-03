"""
🧠 Model Details page.

Displays architecture details, parameter counts, hyper-parameters, optimizer,
scheduler and loss function. All values are read from the notebook config and
the loaded model.
"""

from __future__ import annotations

import streamlit as st
import torch

from src.constants import (
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    OPTIMIZER_WEIGHT_DECAY,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_MODE,
    SCHEDULER_PATIENCE,
    SEED,
)
from src.pages.data_access import get_model_config, get_cached_pipeline, load_best_metrics
from src.pages.ui_utils import metric_card, page_header
from src.components.model import AlexNet
from src.utils import count_parameters


def render_model_details() -> None:
    """Render the Model Details page."""
    page_header(
        "🧠 Model Details",
        "Architecture, parameters, hyper-parameters and training configuration.",
    )

    cfg = get_model_config()
    metrics = load_best_metrics()

    backend = st.session_state.get("backend", "pytorch")

    # ------------------------------------------------------------------
    # Model summary metrics
    # ------------------------------------------------------------------
    st.markdown("### 📦 Model Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card("Architecture", f"{metrics['model']}")
    metric_card("Dataset", metrics["dataset"])
    metric_card("Framework", "PyTorch")
    metric_card("Image Input", f"{cfg.image_size}×{cfg.image_size}")
    metric_card("Classes", str(cfg.num_classes))

    c6, c7, c8, c9 = st.columns(4)
    metric_card("Total Params", f"{metrics['params']:,}")
    metric_card("Trainable Params", f"{metrics['params']:,}")
    metric_card("Device", torch.device("cuda" if torch.cuda.is_available() else "cpu").type.upper())
    metric_card("Seed", str(SEED))

    st.markdown("---")

    # ------------------------------------------------------------------
    # Architecture diagram (text based)
    # ------------------------------------------------------------------
    st.markdown("### 🧬 CNN Architecture (AlexNet)")
    st.markdown(
        """
        ```
        Input: (3, 224, 224)
        ┌─────────────────────────────────────────────┐
        │ Conv2d(3→64, kernel=11, stride=4, pad=2)   │
        │ ReLU                                        │
        │ MaxPool2d(3, stride=2)                      │
        │ Conv2d(64→192, kernel=5, pad=2)            │
        │ BatchNorm2d(192)                           │
        │ ReLU                                        │
        │ MaxPool2d(3, stride=2)                      │
        │ Conv2d(192→384, kernel=3, pad=1)           │
        │ ReLU                                        │
        │ Conv2d(384→256, kernel=3, pad=1)           │
        │ ReLU                                        │
        │ Conv2d(256→256, kernel=3, pad=1)           │
        │ ReLU                                        │
        │ MaxPool2d(3, stride=2)                      │
        │ AdaptiveAvgPool2d((1,1))                    │
        └─────────────────────────────────────────────┘
        Classifier:
        ┌─────────────────────────────────────────────┐
        │ Flatten                                     │
        │ Linear(256→128) + ReLU + Dropout(0.5)       │
        │ Linear(128→10)                              │
        └─────────────────────────────────────────────┘
        Output: 10 logits (one per class)
        ```
        """
    )

    # Render the actual model summary.
    st.markdown("### 📄 PyTorch Model Summary")
    try:
        model = AlexNet(num_classes=cfg.num_classes)
        # Build a string summary.
        lines = [f"{'Layer':<28}{'Type':<28}{'Shape':<28}{'Params':<12}"]
        total = 0
        for name, module in model.named_children():
            sub = 0
            if hasattr(module, "parameters"):
                for p in module.parameters():
                    if p.requires_grad:
                        sub += p.numel()
            total += sub
            lines.append(f"{name:<28}{type(module).__name__:<28}{'-':<28}{sub:<12,}")
        lines.append(f"{'Total':<28}{'':<28}{'-':<28}{total:<12,}")
        st.code("\n".join(lines), language="text")
        st.caption(f"Total trainable parameters: **{count_parameters(model):,}**")
    except Exception as exc:
        st.warning(f"Could not render model summary: {exc}")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Hyper-parameters
    # ------------------------------------------------------------------
    st.markdown("### ⚙️ Hyper-parameters")
    hyp_cols = st.columns(4)
    hyp_cols[0].metric("Epochs", str(EPOCHS))
    hyp_cols[1].metric("Batch Size", str(cfg.batch_size))
    hyp_cols[2].metric("Learning Rate", f"{LEARNING_RATE:.1e}")
    hyp_cols[3].metric("Workers", str(NUM_WORKERS))

    st.markdown("### 🧮 Optimizer")
    opt_cols = st.columns(3)
    opt_cols[0].metric("Optimizer", "AdamW")
    opt_cols[1].metric("Learning Rate", f"{LEARNING_RATE:.1e}")
    opt_cols[2].metric("Weight Decay", f"{OPTIMIZER_WEIGHT_DECAY:.0e}")

    st.markdown("### 📉 LR Scheduler")
    sch_cols = st.columns(4)
    sch_cols[0].metric("Scheduler", "ReduceLROnPlateau")
    sch_cols[1].metric("Mode", SCHEDULER_MODE)
    sch_cols[2].metric("Factor", str(SCHEDULER_FACTOR))
    sch_cols[3].metric("Patience", str(SCHEDULER_PATIENCE))
    sch_cols_2 = st.columns(3)
    sch_cols_2[0].metric("Min LR", f"{SCHEDULER_MIN_LR:.0e}")
    sch_cols_2[1].metric("Early Stopping", f"{EARLY_STOPPING_PATIENCE}")
    sch_cols_2[2].metric("Loss Function", "CrossEntropyLoss")
