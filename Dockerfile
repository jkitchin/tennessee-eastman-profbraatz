# Dockerfile for deploying TEP Dash Dashboard
# Compatible with: Hugging Face Spaces, Render, Railway, Fly.io

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal for pure Python backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy requirements first for better caching
COPY --chown=user requirements-dash.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire package
COPY --chown=user . /app

# Install the TEP package in development mode
RUN pip install --no-cache-dir -e .

# Expose port (HF Spaces uses 7860, Render uses 10000)
# The app will bind to the PORT environment variable if set
EXPOSE 7860

# Set environment variables
ENV TEP_BACKEND=python
ENV PORT=7860

# Run the Dash app with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "tep.dashboard_dash:server"]
