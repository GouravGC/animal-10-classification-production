"""
⚙ Settings page.

Allows the user to select the inference backend (PyTorch / ONNX) and adjust the
Out-of-Distribution (OOD) confidence threshold. If ONNX Runtime is unavailable,
the app shows a friendly warning and automatically falls back to PyTorch.
"""

from __future__ import annotations

import streamlit as st

from src.constants import BACKEND_ONNX, BACKEND_PYTORCH
from src.pages.data_access import get_cached_pipeline, load_cached_config
from src.pages.ui_utils import friendly_error, page_header


def render_settings() -> None:
    """Render the Settings page."""
    page_header(
        "⚙ Settings",
        "Configure the inference backend and OOD confidence threshold.",
    )

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------
    st.markdown("### 🚀 Inference Backend")
    st.markdown(
        """
        - **PyTorch** (default): Uses `best_model.pth` with the exact AlexNet
          architecture from the notebook.
        - **ONNX**: Uses `best_model.onnx` via onnxruntime (faster on CPU).
        """
    )

    saved_backend = st.session_state.get("backend", BACKEND_PYTORCH)
    backend = st.radio(
        "Select backend",
        options=[BACKEND_PYTORCH, BACKEND_ONNX],
        index=0 if saved_backend == BACKEND_PYTORCH else 1,
        format_func=lambda x: (
            "✔ PyTorch (default)" if x == BACKEND_PYTORCH else "✔ ONNX"
        ),
        help="ONNX requires onnxruntime. If unavailable, PyTorch is used.",
    )

    if backend == BACKEND_ONNX:
        # Check if ONNX runtime is actually available.
        try:
            import onnxruntime  # noqa: F401

            st.success("✅ ONNX Runtime is available. ONNX backend can be used.")
        except ImportError:
            st.warning(
                "⚠️ **ONNX Runtime is not installed.**\n\n"
                "The app will automatically fall back to the PyTorch backend. "
                "To use ONNX, install it with:\n\n"
                "```bash\npip install onnxruntime\n```"
            )
            backend = BACKEND_PYTORCH

    # ------------------------------------------------------------------
    # OOD threshold
    # ------------------------------------------------------------------
    st.markdown("### 🛡️ Out-of-Distribution (OOD) Threshold")
    st.markdown(
        """
        The OOD safeguard flags images whose top-class confidence is **below**
        the threshold as **unknown / out-of-distribution**. This prevents
        unrelated images (cars, laptops, etc.) from receiving over-confident
        predictions. This does **NOT** modify the trained model.
        """
    )
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.05,
        help="Images with confidence below this are flagged as OOD.",
    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Apply / save
    # ------------------------------------------------------------------
    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        st.session_state["backend"] = backend
        st.session_state["confidence_threshold"] = threshold
        st.success(
            f"Settings saved! Backend: **{backend}**, "
            f"OOD threshold: **{threshold*100:.0f}%**."
        )

    # ------------------------------------------------------------------
    # Current settings summary
    # ------------------------------------------------------------------
    st.markdown("### 📌 Current Settings")
    cur_backend = st.session_state.get("backend", BACKEND_PYTORCH)
    cur_threshold = st.session_state.get("confidence_threshold", 0.75)

    c1, c2 = st.columns(2)
    c1.metric("Backend", cur_backend)
    c2.metric("OOD Threshold", f"{cur_threshold*100:.0f}%")

    # Validate pipeline initialises.
    try:
        pipeline = get_cached_pipeline(cur_backend, cur_threshold)
        c3, c4 = st.columns(2)
        c3.metric("Active Backend", pipeline.backend)
        c4.metric("Device", str(pipeline.device).upper())
        if pipeline.backend != cur_backend:
            st.info(
                f"ℹ️ The selected backend **{cur_backend}** was unavailable, so the "
                f"app is using **{pipeline.backend}** instead."
            )
        st.success("✅ Prediction pipeline is ready.")
    except Exception as exc:
        friendly_error(
            "Could not initialize the prediction pipeline. Please check that the "
            "model artifacts are present in `artifacts/`.",
            exc,
        )
