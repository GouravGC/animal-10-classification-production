"""
🔍 Single Image Prediction page.

Allows the user to upload a single image and view:
- The uploaded image
- Predicted class & confidence
- Top-5 predictions with confidence bars
- Inference time, backend, image size
- Full probability chart
- Prediction explanation
- Grad-CAM visualization (best-effort, optional)
"""

from __future__ import annotations

import io
import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from src.constants import SAMPLE_IMAGES_DIR
from src.pages.data_access import get_cached_pipeline, get_model_config, load_cached_class_names
from src.pages.ui_utils import confidence_bars, fmt_pct, friendly_error, page_header


def _load_image(uploaded) -> Optional[Image.Image]:
    """Load an image from an uploaded file with friendly error handling."""
    try:
        img = Image.open(uploaded).convert("RGB")
        return img
    except Exception:
        st.error("⚠️ Could not read the uploaded file. Please upload a valid image.")
        return None


def _grad_cam(pipeline, pil_image: Image.Image) -> Optional[Image.Image]:
    """
    Best-effort Grad-CAM visualization using the model's last conv layer.

    Works only with the PyTorch backend. If anything fails, returns None
    (the UI shows a friendly message instead of crashing).
    """
    try:
        import torch

        if pipeline.backend != "pytorch":
            return None

        model = pipeline._model
        model.eval()

        # Register forward hook on the last conv layer.
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module

        if target_layer is None:
            return None

        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            activations["value"] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            gradients["value"] = grad_output[0].detach()

        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_full_backward_hook(backward_hook)

        # Preprocess image.
        tensor = pipeline._preprocess_image(pil_image)
        device_tensor = tensor.to(pipeline.device)

        model.zero_grad()
        output = model(device_tensor)
        score = output[0, output.argmax(dim=1)]
        score.backward()

        act = activations["value"].squeeze(0)
        grad = gradients["value"].squeeze(0)
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=0)).cpu().numpy()

        handle_f.remove()
        handle_b.remove()

        # Normalize CAM.
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize CAM to image size and overlay.
        import cv2

        cam = cv2.resize(cam, (pil_image.size[0], pil_image.size[1]))
        cam = np.uint8(255 * cam)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

        base = np.array(pil_image.convert("RGB"))
        overlay = 0.5 * base + 0.5 * cam
        return Image.fromarray(np.uint8(overlay))
    except Exception:
        return None


def render_single_prediction() -> None:
    """Render the Single Image Prediction page."""
    page_header(
        "🔍 Single Image Prediction",
        "Upload an image and get a detailed prediction with confidence scores.",
    )

    cfg = get_model_config()
    class_names = load_cached_class_names()

    # Backend + threshold from session state (set in Settings page).
    backend = st.session_state.get("backend", "pytorch")
    threshold = st.session_state.get("confidence_threshold", 0.75)

    # ------------------------------------------------------------------
    # Image input
    # ------------------------------------------------------------------
    col_up, col_opt = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload an animal image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Supported formats: JPG, JPEG, PNG, WEBP, BMP.",
        )
    with col_opt:
        top_k = st.slider("Top-K predictions", 1, 10, 5, help="How many top predictions to show.")

    # Sample image picker.
    sample_files = []
    if os.path.isdir(SAMPLE_IMAGES_DIR):
        sample_files = [
            os.path.join(SAMPLE_IMAGES_DIR, f)
            for f in sorted(os.listdir(SAMPLE_IMAGES_DIR))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    if sample_files:
        sample_names = ["(none)"] + [os.path.basename(f) for f in sample_files]
        selected = st.selectbox("Or pick a sample image", sample_names)
        if selected != "(none)":
            idx = sample_names.index(selected) - 1
            uploaded_file = sample_files[idx]

    image = None
    if uploaded_file is not None:
        image = _load_image(uploaded_file)
        if image is not None:
            st.image(image, caption="Uploaded Image", use_column_width=True)

    if image is None:
        st.info("👆 Please upload an image or select a sample to begin.")
        return

    # ------------------------------------------------------------------
    # Predict button
    # ------------------------------------------------------------------
    if st.button("🔮 Run Prediction", type="primary", use_container_width=True):
        try:
            pipeline = get_cached_pipeline(backend, threshold)
            with st.spinner("Running inference..."):
                result = pipeline.predict(image, top_k=top_k)

            # ------------------------------------------------------------------
            # Result display
            # ------------------------------------------------------------------
            st.markdown("## ✅ Prediction Result")

            if result.is_ood:
                st.warning(
                    "⚠️ **Unknown / Out-of-Distribution image detected.**\n\n"
                    f"{result.ood_message}"
                )
            else:
                st.success(
                    f"🎯 **Predicted Class:** {result.predicted_class} "
                    f"(confidence {fmt_pct(result.confidence)})"
                )

            # Metric cards.
            display_class = "Unknown / OOD" if result.is_ood else result.predicted_class
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Predicted Class", display_class)
            c2.metric("Confidence", fmt_pct(result.confidence))
            c3.metric("Inference Time", f"{result.inference_time_ms:.1f} ms")
            c4.metric("Backend", "PyTorch" if backend == "pytorch" else "ONNX")
            c5.metric("Entropy", f"{result.entropy:.3f}")

            # Original image size.
            st.caption(f"Input image size: {image.width}×{image.height} px → resized to {cfg.image_size}×{cfg.image_size}")

            # ------------------------------------------------------------------
            # Top-5 + probability chart
            # ------------------------------------------------------------------
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown("### 🏆 Top Predictions")
                confidence_bars(result.top_k)

            with col_b:
                st.markdown("### 📊 Probability Distribution")
                prob_df = pd.DataFrame(
                    {
                        "Class": list(result.all_probabilities.keys()),
                        "Probability": list(result.all_probabilities.values()),
                    }
                ).sort_values("Probability", ascending=True)
                fig = px.bar(
                    prob_df,
                    x="Probability",
                    y="Class",
                    orientation="h",
                    color="Probability",
                    color_continuous_scale="Viridis",
                    height=360,
                )
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#CBD5E1"),
                )
                st.plotly_chart(fig, use_container_width=True)

            # ------------------------------------------------------------------
            # Grad-CAM
            # ------------------------------------------------------------------
            st.markdown("### 🔥 Grad-CAM Visualization")
            cam_img = _grad_cam(pipeline, image)
            if cam_img is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_column_width=True)
                with col2:
                    st.image(cam_img, caption="Grad-CAM Overlay", use_column_width=True)
                st.caption(
                    "Grad-CAM highlights the regions the model focused on for its prediction. "
                    "This is an optional visualization and does not affect the prediction."
                )
            else:
                st.info(
                    "Grad-CAM is available with the PyTorch backend and OpenCV. "
                    "Switch to PyTorch in ⚙ Settings to enable it."
                )

            # ------------------------------------------------------------------
            # Prediction explanation
            # ------------------------------------------------------------------
            with st.expander("📖 Prediction Explanation"):
                st.markdown(
                    f"""
                    - The model predicted **{result.predicted_class}** with a confidence of **{fmt_pct(result.confidence)}**.
                    - Confidence is the softmax probability of the top class.
                    - Entropy (**{result.entropy:.3f}**) measures prediction uncertainty; values near 0 indicate high certainty.
                    - If the confidence is below the threshold (**{threshold*100:.0f}%**), the image is flagged as **out-of-distribution**.
                    - The image was resized to **{cfg.image_size}×{cfg.image_size}** and normalized using the notebook's exact mean/std.
                    """
                )

            with st.expander("🐾 Supported Classes"):
                st.markdown(" | ".join(class_names))

        except Exception as exc:
            friendly_error("Prediction failed. Please try a different image.", exc)
