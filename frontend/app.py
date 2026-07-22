"""Main entry point for the ODNN Streamlit frontend."""

import sys
from pathlib import Path

import streamlit as st

# Make adapters / services / utils importable from any page
sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="ODNN",
    layout="wide",
    initial_sidebar_state="expanded",
)

designer_page     = st.Page("pages/dataset_designer.py", title="Dataset & Label Designer")
training_wl_page  = st.Page("pages/training_wl.py",  title="Multi-WL Training")
testing_wl_page   = st.Page("pages/testing_wl.py",   title="Multi-WL Testing")
training_page     = st.Page("pages/training.py",     title="Training")
testing_page      = st.Page("pages/testing.py",      title="Testing")
analysis_page     = st.Page("pages/analysis.py",     title="Analysis")
gpu_monitor_page  = st.Page("pages/gpu_monitor.py",  title="GPU Monitor")
settings_page     = st.Page("pages/settings.py",     title="Settings")

pg = st.navigation([
    designer_page,
    training_wl_page, testing_wl_page,
    #training_page, testing_page,
    #analysis_page, 
    gpu_monitor_page, settings_page,
], position="sidebar")
pg.run()
