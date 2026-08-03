"""
Shared UI helpers for the Streamlit application.

These helpers provide a consistent, professional look across all pages:
custom CSS, metric cards, confidence bars, and colour helpers.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from src.logger import get_logger

logger = get_logger(__name__)


# Brand colour palette.
PRIMARY = "#4F46E5"  # indigo
ACCENT = "#10B981"   # emerald
WARNING = "#F59E0B"  # amber
DANGER = "#EF4444"   # red
INFO = "#0EA5E9"     # sky
BG = "#0F172A"       # slate-900
CARD = "#1E293B"     # slate-800
TEXT = "#F8FAFC"     # slate-50


def inject_css() -> None:
    """
    Inject global custom CSS for a polished, modern dark theme.

    The theme uses a dark background with light text everywhere. Every Streamlit
    component (selectbox, radio, slider, text inputs, captions, dataframes,
    tabs, expanders, metrics, warnings/info boxes) is explicitly styled so that
    ALL text is clearly visible against the dark background.
    """
    st.markdown(
        f"""
        <style>
        /* ============ GLOBAL ============ */
        .stApp {{
            background-color: {BG};
            color: {TEXT};
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }}

        html, body, .stApp, p, li, span, div, label, ul, ol {{
            color: {TEXT};
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT};
            font-weight: 700;
        }}

        /* ============ MARKDOWN / TEXT ============ */
        .stMarkdown p,
        .stMarkdown li,
        .stMarkdown span,
        .stMarkdown div {{
            color: #E2E8F0;
        }}
        .stMarkdown strong {{
            color: {TEXT};
        }}
        .stMarkdown a {{
            color: {INFO};
        }}

        /* ============ CAPTIONS / HINTS ============ */
        .stCaption, [data-testid="stCaptionContainer"] p,
        .stCaptionContainer p {{
            color: #CBD5E1 !important;
        }}

        /* ============ WIDGET LABELS ============ */
        .stRadio label,
        .stSelectbox label,
        .stSlider label,
        .stFileUploader label,
        .stTextInput label,
        .stNumberInput label,
        .stDateInput label,
        .stCheckbox label {{
            color: {TEXT} !important;
            font-weight: 600;
        }}
        .stCaptionContainer p {{
            color: #CBD5E1 !important;
        }}

        /* ============ SELECTBOX ============ */
        .stSelectbox [data-baseweb="select"] > div {{
            background-color: {CARD};
            border: 1px solid #334155;
            color: {TEXT};
        }}
        .stSelectbox [data-baseweb="select"] span {{
            color: {TEXT};
        }}
        .stSelectbox [data-baseweb="select"] [data-baseweb="popover"] div {{
            background-color: {CARD};
            color: {TEXT};
        }}
        .stSelectbox div[role="listbox"] ul li {{
            color: {TEXT};
        }}
        .stSelectbox div[role="listbox"] ul li:hover {{
            background-color: {PRIMARY};
            color: white;
        }}

        /* ============ RADIO ============ */
        .stRadio div[role="radiogroup"] label {{
            color: {TEXT};
        }}
        .stRadio div[role="radiogroup"] label > div:first-child {{
            color: {TEXT};
        }}
        .stRadio [data-testid="stMarkdownContainer"] p {{
            color: {TEXT};
        }}

        /* ============ SLIDER ============ */
        .stSlider [data-baseweb="slider"] div {{
            color: {TEXT};
        }}
        .stSlider [data-testid="stSliderValue"] {{
            color: {TEXT};
        }}
        .stSlider input[type="range"] {{
            accent-color: {PRIMARY};
        }}

        /* ============ BUTTONS ============ */
        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {{
            background: {PRIMARY};
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.5rem 1.25rem;
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background: #6366F1;
            color: white;
        }}
        .stButton > button:focus,
        .stDownloadButton > button:focus {{
            color: white;
        }}

        /* ============ METRIC CARDS ============ */
        div[data-testid="stMetric"] {{
            background: {CARD};
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1rem 1.25rem;
        }}
        div[data-testid="stMetric"] label {{
            color: #94A3B8;
        }}
        div[data-testid="stMetricValue"] {{
            color: {TEXT};
            font-size: 1.4rem;
        }}
        div[data-testid="stMetricDelta"] {{
            color: #10B981;
        }}

        /* ============ SIDEBAR ============ */
        section[data-testid="stSidebar"] {{
            background: {CARD};
        }}
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown span,
        section[data-testid="stSidebar"] .stMarkdown li {{
            color: {TEXT};
        }}
        section[data-testid="stSidebar"] .stRadio label {{
            color: {TEXT};
        }}
        section[data-testid="stSidebar"] .stCaption {{
            color: #CBD5E1;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: #334155;
        }}

        /* ============ TABS ============ */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: {CARD};
            border-radius: 8px;
            padding: 0.4rem 1rem;
            color: #CBD5E1;
        }}
        .stTabs [aria-selected="true"] {{
            background: {PRIMARY} !important;
            color: white !important;
        }}
        .stTabs [data-baseweb="tab"] div {{
            color: #CBD5E1;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            color: {TEXT};
        }}

        /* ============ EXPANDERS ============ */
        .stExpander {{
            background: {CARD};
            border: 1px solid #334155;
            border-radius: 10px;
        }}
        .stExpander summary,
        .stExpander [data-testid="stExpanderHeader"] {{
            color: {TEXT};
        }}
        .stExpander summary p {{
            color: {TEXT};
        }}
        .stExpander [data-testid="stExpanderDetails"] p {{
            color: #E2E8F0;
        }}

        /* ============ INFO / SUCCESS / WARNING / ERROR ============ */
        .stAlert {{ 
            border-radius: 10px;
        }}
        .stAlert [data-testid="stMarkdownContainer"] p {{
            color: #E2E8F0;
        }}
        .stAlert [data-testid="stMarkdownContainer"] strong {{
            color: {TEXT};
        }}
        div[data-testid="stAlertSuccess"] {{
            background-color: #064E3B;
            border: 1px solid #10B981;
        }}
        div[data-testid="stAlertError"] {{
            background-color: #7F1D1D;
            border: 1px solid #EF4444;
        }}
        div[data-testid="stAlertWarning"] {{
            background-color: #78350F;
            border: 1px solid #F59E0B;
        }}
        div[data-testid="stAlertInfo"] {{
            background-color: #0C4A6E;
            border: 1px solid #0EA5E9;
        }}

        /* ============ DATAFRAME / TABLES ============ */
        .stDataFrame {{
            color: {TEXT};
        }}
        .stDataFrame [data-testid="stDataFrame"] {{
            color: {TEXT};
        }}
        .stDataFrame thead th {{
            color: {TEXT};
            background-color: {CARD};
        }}
        .stDataFrame tbody td {{
            color: {TEXT};
        }}

        /* ============ CODE BLOCKS ============ */
        .stCodeBlock, .stCode pre, pre {{
            background-color: #0B1120;
            color: #E2E8F0;
            border: 1px solid #334155;
            border-radius: 8px;
        }}
        .stCodeBlock code, .stCode pre code, pre code {{
            color: #E2E8F0;
        }}

        /* ============ FILE UPLOADER ============ */
        .stFileUploader section[data-testid="stFileUploaderDropzone"] {{
            background-color: {CARD};
            border: 1px dashed #334155;
            color: {TEXT};
        }}
        .stFileUploader section[data-testid="stFileUploaderDropzone"] button {{
            color: {TEXT};
        }}
        .stFileUploader [data-testid="stFileUploaderFile"] {{
            color: {TEXT};
        }}

        /* ============ MISC ============ */
        hr {{
            border-color: #334155;
        }}
        .stprogress > div > div > div > div {{
            background-color: {PRIMARY};
        }}

        /* ============ CUSTOM COMPONENTS ============ */
        .page-header {{
            background: linear-gradient(135deg, {PRIMARY}, {INFO});
            border-radius: 16px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            color: white;
        }}
        .page-header h1 {{
            color: white;
            margin: 0;
        }}
        .page-header p {{
            color: #E0E7FF;
            margin: 0.25rem 0 0 0;
        }}

        .conf-bar {{
            background: #1E293B;
            border-radius: 8px;
            height: 26px;
            margin: 4px 0 10px 0;
            overflow: hidden;
            position: relative;
        }}
        .conf-bar-fill {{
            height: 100%;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 8px;
            color: white;
            font-weight: 600;
            font-size: 0.8rem;
        }}
        .conf-label {{
            color: #CBD5E1;
            font-weight: 600;
            margin-bottom: 2px;
        }}

        .hero {{
            background: linear-gradient(135deg, #1E293B, #312E81);
            border-radius: 20px;
            padding: 2.5rem;
            text-align: center;
            margin-bottom: 1.5rem;
            border: 1px solid #334155;
        }}
        .hero h1 {{
            font-size: 2.6rem;
            color: white;
            margin: 0 0 0.5rem 0;
        }}
        .hero p {{
            color: #A5B4FC;
            font-size: 1.1rem;
        }}

        /* Ensure raw markdown tables are readable */
        .stMarkdown table {{
            color: {TEXT};
        }}
        .stMarkdown table th {{
            background-color: {CARD};
            color: {TEXT};
        }}
        .stMarkdown table td {{
            color: #E2E8F0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    """Render a prominent page header banner."""
    st.markdown(
        f"""
        <div class="page-header">
            <h1>{title}</h1>
            {f"<p>{subtitle}</p>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    """Render a large hero banner (used on the Home page)."""
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str = None) -> None:
    """Render a styled metric card using st.metric with a custom caption."""
    st.metric(label=label, value=value, delta=delta)


def confidence_bars(top_k: List[Tuple[str, float]]) -> None:
    """
    Render a list of confidence bars for the top-k predictions.

    Args:
        top_k (List[Tuple[str, float]]): List of (class_name, probability).
    """
    for name, prob in top_k:
        pct = prob * 100.0
        bar_colour = PRIMARY if pct >= 50 else ("#10B981" if pct >= 25 else INFO)
        st.markdown(
            f"""
            <div class="conf-label">{name} — {pct:.2f}%</div>
            <div class="conf-bar">
                <div class="conf-bar-fill" style="width:{pct:.1f}%; background:{bar_colour};">
                    {pct:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def fmt_pct(value: float) -> str:
    """Format a probability as a percentage string."""
    return f"{value * 100:.2f}%"


def friendly_error(message: str, exc: Exception = None) -> None:
    """
    Render a clean, user-friendly error without exposing tracebacks.

    The raw exception details are logged (not shown to the user) so the UI
    never leaks Python internals. The user only sees a helpful message.

    Args:
        message (str): A friendly, human-readable error message.
        exc (Exception, optional): The underlying exception (logged only).
    """
    if exc is not None:
        logger.warning("User-facing error: %s | %r", message, exc)
    st.error(f"❌ {message}")
    st.info(
        "Please try again with a different image or check the backend settings. "
        "If the problem persists, verify that the model artifacts are present "
        "in `artifacts/`."
    )
