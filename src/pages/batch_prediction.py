"""
📂 Batch Prediction page.

Allows the user to upload multiple images, classify them all, preview the
results in a table, and download the results as a CSV file.
"""

from __future__ import annotations

import io
from typing import List

import pandas as pd
import streamlit as st
from PIL import Image

from src.pages.data_access import get_cached_pipeline
from src.pages.ui_utils import fmt_pct, friendly_error, page_header


def _csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to UTF-8 CSV bytes."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def render_batch_prediction() -> None:
    """Render the Batch Prediction page."""
    page_header(
        "📂 Batch Prediction",
        "Upload multiple images, classify them all, and download the results.",
    )

    backend = st.session_state.get("backend", "pytorch")
    threshold = st.session_state.get("confidence_threshold", 0.75)

    top_k = st.slider("Top-K predictions per image", 1, 10, 5)

    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        help="You can select several images at once.",
    )

    if not uploaded_files:
        st.info("👆 Upload one or more images to begin batch prediction.")
        return

    st.markdown(f"**{len(uploaded_files)} image(s) uploaded.**")

    if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
        try:
            pipeline = get_cached_pipeline(backend, threshold)
            rows: List[dict] = []
            progress = st.progress(0.0)

            preview_labels = {}
            for idx, uploaded in enumerate(uploaded_files):
                try:
                    img = Image.open(uploaded).convert("RGB")
                    result = pipeline.predict(img, top_k=top_k)

                    top_classes = ", ".join(result.top_classes[:3])
                    label = (
                        result.predicted_class if not result.is_ood else "Unknown / OOD"
                    )
                    preview_labels[uploaded.name] = (
                        f"{label} · {fmt_pct(result.confidence)}"
                    )
                    rows.append(
                        {
                            "File": uploaded.name,
                            "Predicted Class": (
                                result.predicted_class if not result.is_ood else "Unknown"
                            ),
                            "Confidence": fmt_pct(result.confidence),
                            "OOD Flagged": "Yes" if result.is_ood else "No",
                            "Top-3": top_classes,
                            "Inference (ms)": round(result.inference_time_ms, 2),
                        }
                    )
                except Exception as exc:
                    preview_labels[uploaded.name] = "Error"
                    rows.append(
                        {
                            "File": uploaded.name,
                            "Predicted Class": "Error",
                            "Confidence": "-",
                            "OOD Flagged": "-",
                            "Top-3": "Could not process image",
                            "Inference (ms)": "-",
                        }
                    )
                progress.progress((idx + 1) / len(uploaded_files))

            progress.empty()

            df = pd.DataFrame(rows)
            st.markdown("### 📋 Batch Results")
            st.dataframe(df, use_container_width=True)

            # Summary counts.
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Images", len(df))
            col2.metric("OOD Flagged", int((df["OOD Flagged"] == "Yes").sum()))
            col3.metric("Errors", int((df["Predicted Class"] == "Error").sum()))

            # Download button.
            st.download_button(
                "⬇️ Download CSV",
                data=_csv_bytes(df),
                file_name="batch_predictions.csv",
                mime="text/csv",
                type="primary",
            )

            # Preview of predictions.
            st.markdown("### 🖼️ Prediction Preview")
            preview_cols = st.columns(3)
            for i, uploaded in enumerate(uploaded_files[:6]):
                col = preview_cols[i % 3]
                try:
                    img = Image.open(uploaded).convert("RGB")
                    caption = preview_labels.get(
                        uploaded.name, uploaded.name
                    )
                    with col:
                        st.image(img, caption=caption, use_column_width=True)
                except Exception:
                    with col:
                        st.error(f"Failed to display {uploaded.name}")

        except Exception as exc:
            friendly_error("Batch prediction failed. Please check the backend settings.", exc)
