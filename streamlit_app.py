"""
Streamlit app entry point for Streamlit Cloud deployment.

This file ensures the tep package is importable and forces the Python backend.
"""
import sys
import os

# Add the repo root to path so tep package can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force Python backend (no Fortran on Streamlit Cloud)
os.environ['TEP_BACKEND'] = 'python'

# Import and run the dashboard
from tep.dashboard_streamlit import main

if __name__ == "__main__":
    main()
