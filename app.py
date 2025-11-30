"""
Entry point for Hugging Face Spaces Docker deployment.

HuggingFace Spaces with Docker SDK looks for app.py as the default entry point.
This file launches the Dash dashboard server.

The Dockerfile sets TEP_BACKEND=fortran after building with Fortran support.
"""
from tep.dashboard_dash import app

# Expose the server for gunicorn
server = app.server

if __name__ == "__main__":
    # Get port from environment (HF Spaces uses 7860)
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
