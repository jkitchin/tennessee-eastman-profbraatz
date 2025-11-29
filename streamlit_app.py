"""
Streamlit app entry point for Streamlit Cloud deployment.

This file ensures the tep package is importable from local files
and forces the Python backend (no Fortran compilation needed).
"""
import sys
import os

# Force Python backend (no Fortran on Streamlit Cloud)
os.environ['TEP_BACKEND'] = 'python'

# Remove any existing tep package/finder from a previous install attempt
# This handles the case where an editable install was cached
modules_to_remove = [key for key in sys.modules if key == 'tep' or key.startswith('tep.')]
for mod in modules_to_remove:
    del sys.modules[mod]

# Remove any custom finders that might interfere
sys.meta_path = [finder for finder in sys.meta_path
                 if not (hasattr(finder, '_name') and 'tep' in str(getattr(finder, '_name', '')))]

# Add the repo root to path FIRST so local tep package is found
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

# Debug: show where tep is being imported from
import tep
print(f"[streamlit_app] TEP loaded from: {tep.__file__}")
print(f"[streamlit_app] TEP version: {tep.__version__}")
print(f"[streamlit_app] TEP_BACKEND env: {os.environ.get('TEP_BACKEND')}")
print(f"[streamlit_app] Available backends: {tep.get_available_backends()}")
print(f"[streamlit_app] Default backend: {tep.get_default_backend()}")

# Now import and run the dashboard
from tep.dashboard_streamlit import main

# Verify dashboard_streamlit location
from tep import dashboard_streamlit
print(f"[streamlit_app] dashboard_streamlit loaded from: {dashboard_streamlit.__file__}")

if __name__ == "__main__":
    main()
else:
    # Streamlit Cloud runs without __name__ == "__main__"
    main()
