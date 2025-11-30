"""
Streamlit app entry point for Streamlit Cloud and Hugging Face Spaces.

This file ensures the tep package is importable from local files
and forces the Python backend (no Fortran compilation needed).
"""
import sys
import os

# Force Python backend (no Fortran on cloud platforms)
os.environ['TEP_BACKEND'] = 'python'

# Remove any existing tep package/finder from a previous install attempt
modules_to_remove = [key for key in sys.modules if key == 'tep' or key.startswith('tep.')]
for mod in modules_to_remove:
    del sys.modules[mod]

# Add the repo root to path FIRST so local tep package is found
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

# Import and run the dashboard
from tep.dashboard_streamlit import main

if __name__ == "__main__":
    main()
else:
    # Streamlit runs without __name__ == "__main__"
    main()
