# Dockerfile for deploying TEP Dash Dashboard
# Compatible with: Hugging Face Spaces, Render, Railway, Fly.io

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (gfortran for Fortran backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-dash.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install build dependencies for Fortran backend
RUN pip install --no-cache-dir meson meson-python ninja numpy

# Copy the entire package
COPY . /app

# Install the TEP package with Fortran backend
RUN pip install --no-cache-dir . --config-settings=setup-args=-Dfortran=enabled

# Remove source src/tep/ directory to avoid shadowing the installed package
# The installed package in site-packages has the compiled .so file
RUN rm -rf /app/src/tep

# Verify Fortran extension after installation
RUN python -c "from tep import is_fortran_available, get_available_backends; print(f'Fortran available: {is_fortran_available()}'); print(f'Backends: {get_available_backends()}')"

# Create non-root user for security (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Set environment variables (use Fortran backend for speed)
ENV TEP_BACKEND=fortran
ENV PORT=7860

# Run the Dash app with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "tep.dashboard_dash:server"]
