"""
Animals-10 Classification — Professional Multi-Page Streamlit Application.

This is the main entry point of the production application. It provides a
sidebar-based navigation with the following pages:

    🏠 Home
    🔍 Single Image Prediction
    📂 Batch Prediction
    📊 Model Performance
    📁 Dataset Information
    🧠 Model Details
    ⚙ Settings
    📖 About

The application faithfully mirrors the authoritative notebook and uses the
existing trained artifacts (loaded, never regenerated).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.logger import get_logger
from src.pages import (
    render_about,
    render_batch_prediction,
    render_dataset_info,
    render_home,
    render_model_details,
    render_model_performance,
    render_settings,
    render_single_prediction,
)
from src.pages.ui_utils import inject_css

logger = get_logger(__name__)

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Animals-10 Classification",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global custom CSS / theme.
inject_css()

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
PAGES = {
    "🏠 Home": "home",
    "🔍 Single Image Prediction": "single",
    "📂 Batch Prediction": "batch",
    "📊 Model Performance": "performance",
    "📁 Dataset Information": "dataset",
    "🧠 Model Details": "model_details",
    "⚙ Settings": "settings",
    "📖 About": "about",
}


def init_session_state() -> None:
    """Initialise default session-state values."""
    if "backend" not in st.session_state:
        st.session_state["backend"] = "pytorch"
    if "confidence_threshold" not in st.session_state:
        st.session_state["confidence_threshold"] = 0.75


def render_sidebar() -> str:
    """Render the sidebar navigation and return the selected page key."""
    with st.sidebar:
        st.markdown("## 🐾 Animals-10")
        st.markdown("**Image Classification**")
        st.markdown("---")

        selected = st.radio(
            "Navigation",
            options=list(PAGES.keys()),
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption(
            f"Backend: **{st.session_state.get('backend', 'pytorch')}**  \n"
            f"OOD Threshold: **{st.session_state.get('confidence_threshold', 0.75)*100:.0f}%**"
        )
        st.caption("Artifacts loaded from `artifacts/` — never regenerated.")
        st.caption("👤 **Gourav Chhatwani**")

    return PAGES[selected]


def render_page(page_key: str) -> None:
    """Dispatch to the selected page render function."""
    if page_key == "home":
        render_home()
    elif page_key == "single":
        render_single_prediction()
    elif page_key == "batch":
        render_batch_prediction()
    elif page_key == "performance":
        render_model_performance()
    elif page_key == "dataset":
        render_dataset_info()
    elif page_key == "model_details":
        render_model_details()
    elif page_key == "settings":
        render_settings()
    elif page_key == "about":
        render_about()


def main() -> None:
    """Main entry point."""
    init_session_state()
    page_key = render_sidebar()
    render_page(page_key)


if __name__ == "__main__":
    main()

