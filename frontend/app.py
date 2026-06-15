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

training_page     = st.Page("pages/training.py",     title="Training")
testing_page      = st.Page("pages/testing.py",      title="Testing")
training_wl_page  = st.Page("pages/training_wl.py",  title="Multi-WL Training")
testing_wl_page   = st.Page("pages/testing_wl.py",   title="Multi-WL Testing")
analysis_page     = st.Page("pages/analysis.py",     title="Analysis")

pg = st.navigation([
    training_page, testing_page, training_wl_page, testing_wl_page, analysis_page
], position="sidebar")
pg.run()
