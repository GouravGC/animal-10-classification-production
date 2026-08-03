"""
Streamlit page modules.

Each page is a self-contained function that renders a section of the multi-page
application. The pages are orchestrated by `app.py` via a sidebar navigation.
"""

from src.pages.home import render_home
from src.pages.single_prediction import render_single_prediction
from src.pages.batch_prediction import render_batch_prediction
from src.pages.model_performance import render_model_performance
from src.pages.dataset_info import render_dataset_info
from src.pages.model_details import render_model_details
from src.pages.settings import render_settings
from src.pages.about import render_about

__all__ = [
    "render_home",
    "render_single_prediction",
    "render_batch_prediction",
    "render_model_performance",
    "render_dataset_info",
    "render_model_details",
    "render_settings",
    "render_about",
]
