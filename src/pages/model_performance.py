"""
📊 Model Performance page.

Displays the EXISTING notebook outputs only (plots, metrics, confusion
matrices, classification reports). Nothing is regenerated.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.constants import HISTORY_DIR
from src.pages.data_access import (
    get_plot,
    get_plot_path,
    get_report_path,
    load_model_comparison,
)
from src.pages.ui_utils import page_header

# All six trained models (used to display per-model plots).
MODEL_EXPTS = [
    ("MiniCNN", "Raw"),
    ("MiniCNN", "Augmented"),
    ("LeNet", "Raw"),
    ("LeNet", "Augmented"),
    ("AlexNet", "Raw"),
    ("AlexNet", "Augmented"),
]


def render_model_performance() -> None:
    """Render the Model Performance page."""
    page_header(
        "📊 Model Performance",
        "Existing notebook outputs — displayed only, never regenerated.",
    )

    st.info(
        "These figures and reports are loaded directly from `artifacts/plots` and "
        "`artifacts/reports`. They are the exact outputs produced by the notebook."
    )

    # ------------------------------------------------------------------
    # Metrics overview
    # ------------------------------------------------------------------
    st.markdown("### 🎯 Evaluation Metrics")
    comparison = load_model_comparison()
    st.dataframe(comparison, use_container_width=True)

    # Highlight best model.
    best = comparison.sort_values("F1 Score", ascending=False).iloc[0]
    st.success(
        f"🏆 **Best Model:** {best['Model']} ({best['Dataset']}) — "
        f"F1 Score: {best['F1 Score']:.4f}, Accuracy: {best['Test Accuracy']*100:.2f}%"
    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Model selector for plots
    # ------------------------------------------------------------------
    st.markdown("### 📈 Training Curves & Confusion Matrices")
    selected = st.selectbox(
        "Select a model to view its plots",
        [f"{m} ({e})" for m, e in MODEL_EXPTS],
    )
    model_name, experiment = selected.split(" (")
    experiment = experiment.rstrip(")")

    tab_acc, tab_loss, tab_cm = st.tabs(["Accuracy", "Loss", "Confusion Matrix"])

    with tab_acc:
        acc_path = get_plot_path(f"{model_name}_{experiment}_accuracy.png")
        if get_plot(f"{model_name}_{experiment}_accuracy.png"):
            st.image(acc_path, use_column_width=True)
        else:
            st.info("Accuracy plot not available for this model.")

    with tab_loss:
        loss_path = get_plot_path(f"{model_name}_{experiment}_loss.png")
        if get_plot(f"{model_name}_{experiment}_loss.png"):
            st.image(loss_path, use_column_width=True)
        else:
            st.info("Loss plot not available for this model.")

    with tab_cm:
        cm_path = get_plot_path(f"{model_name}_{experiment}_confusion_matrix.png")
        if get_plot(f"{model_name}_{experiment}_confusion_matrix.png"):
            st.image(cm_path, use_column_width=True)
        else:
            st.info("Confusion matrix not available for this model.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Learning curves (interactive, from history CSV)
    # ------------------------------------------------------------------
    st.markdown("### 📉 Learning Curves")
    history_path = os.path.join(
        HISTORY_DIR, f"{model_name}_{experiment}.csv"
    )
    if os.path.exists(history_path):
        hist = pd.read_csv(history_path)
        epochs = list(range(1, len(hist) + 1))

        tab_lc_loss, tab_lc_acc = st.tabs(["Loss", "Accuracy"])

        with tab_lc_loss:
            fig_loss = go.Figure()
            fig_loss.add_trace(
                go.Scatter(
                    x=epochs, y=hist["train_loss"], mode="lines+markers",
                    name="Training Loss", line=dict(color="#4F46E5", width=2),
                )
            )
            fig_loss.add_trace(
                go.Scatter(
                    x=epochs, y=hist["val_loss"], mode="lines+markers",
                    name="Validation Loss", line=dict(color="#EF4444", width=2),
                )
            )
            fig_loss.update_layout(
                title="Training vs Validation Loss",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with tab_lc_acc:
            fig_acc = go.Figure()
            fig_acc.add_trace(
                go.Scatter(
                    x=epochs, y=hist["train_acc"], mode="lines+markers",
                    name="Training Accuracy", line=dict(color="#10B981", width=2),
                )
            )
            fig_acc.add_trace(
                go.Scatter(
                    x=epochs, y=hist["val_acc"], mode="lines+markers",
                    name="Validation Accuracy", line=dict(color="#0EA5E9", width=2),
                )
            )
            fig_acc.update_layout(
                title="Training vs Validation Accuracy",
                xaxis_title="Epoch",
                yaxis_title="Accuracy",
                yaxis_tickformat=".0%",
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.info("Learning-curve history not available for this model.")

    # ------------------------------------------------------------------
    # ROC / PR curves (not produced by the notebook)
    # ------------------------------------------------------------------
    st.markdown("### 📈 ROC / Precision-Recall Curves")
    st.info(
        "The notebook does not produce ROC / Precision-Recall curve artifacts. "
        "These are not available for display. The available evaluation outputs "
        "are the confusion matrix, classification report, and training curves above."
    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Classification report for selected model
    # ------------------------------------------------------------------
    st.markdown("### 📝 Classification Report")
    report_path = get_report_path(f"{model_name}_{experiment}_classification_report.txt")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    else:
        st.info("Classification report not available.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # All confusion matrices
    # ------------------------------------------------------------------
    st.markdown("### 🧮 All Confusion Matrices")
    cm_cols = st.columns(3)
    for i, (m, e) in enumerate(MODEL_EXPTS):
        filename = f"{m}_{e}_confusion_matrix.png"
        cm_path = get_plot_path(filename)
        if get_plot(filename):
            with cm_cols[i % 3]:
                st.markdown(f"**{m} ({e})**")
                st.image(cm_path, use_column_width=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Training history (CSV from notebook)
    # ------------------------------------------------------------------
    st.markdown("### 📚 Training History")
    history_files = [
        f for f in os.listdir(HISTORY_DIR) if f.endswith(".csv")
    ]
    if history_files:
        history_files.sort()
        selected_h = st.selectbox("Select a training history CSV", history_files)
        history_path = os.path.join(HISTORY_DIR, selected_h)
        if os.path.exists(history_path):
            df = pd.read_csv(history_path)
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No training history CSVs found.")
