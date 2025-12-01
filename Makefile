# Makefile for Tennessee Eastman Process Python Simulator
# Usage: make [target]

.PHONY: help install install-dev install-web test test-verbose test-coverage \
        lint format dashboard run-example clean clean-pyc clean-build \
        docs build publish backend-info backend-test hf-upload install-skill

# Default target
help:
	@echo "Tennessee Eastman Process Simulator - Available targets:"
	@echo ""
	@echo "  Installation:"
	@echo "    install        Install the package (requires Fortran compiler)"
	@echo "    install-dev    Install with development dependencies"
	@echo "    install-web    Install with web dashboard support"
	@echo "    install-all    Install with all optional dependencies"
	@echo ""
	@echo "  Testing:"
	@echo "    test           Run all tests"
	@echo "    test-verbose   Run tests with verbose output"
	@echo "    test-coverage  Run tests with coverage report"
	@echo "    test-fast      Run tests without slow integration tests"
	@echo ""
	@echo "  Code Quality:"
	@echo "    lint           Run linting checks (ruff)"
	@echo "    format         Format code (ruff)"
	@echo "    typecheck      Run type checking (mypy)"
	@echo ""
	@echo "  Running:"
	@echo "    dashboard      Launch the web dashboard (tep-web)"
	@echo "    run-example    Run basic simulation example"
	@echo "    run-examples   Run all examples"
	@echo ""
	@echo "  Building:"
	@echo "    build          Build distribution packages"
	@echo "    clean          Clean all build artifacts"
	@echo ""
	@echo "  Backends:"
	@echo "    backend-info   Show available simulation backends"
	@echo "    backend-test   Test Fortran vs Python backend parity"
	@echo ""
	@echo "  Fortran (standalone executable):"
	@echo "    fortran-build  Compile original Fortran code as standalone"
	@echo "    fortran-run    Build and run standalone Fortran simulation"
	@echo "    fortran-clean  Clean Fortran build artifacts"
	@echo ""
	@echo "  Claude Code:"
	@echo "    install-skill  Install TEP skill to .claude/skills/"
	@echo ""

# ============================================================================
# Installation targets
# ============================================================================

install:
	pip install .

install-dev:
	pip install ".[dev]"

install-web:
	pip install ".[web]"

install-all:
	pip install ".[dev,web,plot]"

# ============================================================================
# Testing targets
# ============================================================================

test:
	python -m pytest tests/ -v

test-verbose:
	python -m pytest tests/ -v --tb=long

test-coverage:
	python -m pytest tests/ -v --cov=tep --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

test-fast:
	python -m pytest tests/ -v -m "not slow"

# Run specific test file
test-constants:
	python -m pytest tests/test_constants.py -v

test-controllers:
	python -m pytest tests/test_controllers.py -v

test-simulator:
	python -m pytest tests/test_simulator.py -v

# ============================================================================
# Code quality targets
# ============================================================================

lint:
	@echo "Running ruff linter..."
	-ruff check tep/ tests/ examples/
	@echo ""
	@echo "Running ruff format check..."
	-ruff format --check tep/ tests/ examples/

format:
	@echo "Formatting code with ruff..."
	ruff format tep/ tests/ examples/
	ruff check --fix tep/ tests/ examples/

typecheck:
	@echo "Running mypy type checker..."
	-mypy tep/ --ignore-missing-imports

# ============================================================================
# Running targets
# ============================================================================

dashboard:
	@echo "Launching TEP Web Dashboard..."
	@echo "(Requires web dependencies: make install-web)"
	tep-web

run-example:
	@echo "Running basic simulation example..."
	python examples/basic_simulation.py

run-examples:
	@echo "Running all examples..."
	@echo ""
	@echo "=== Basic Simulation ==="
	python examples/basic_simulation.py
	@echo ""
	@echo "=== Disturbance Simulation ==="
	python examples/disturbance_simulation.py
	@echo ""
	@echo "=== Custom Controller ==="
	python examples/custom_controller.py
	@echo ""
	@echo "=== Data Generation ==="
	python examples/data_generation.py
	@echo ""
	@echo "All examples completed!"

# Quick simulation from command line
simulate:
	@echo "Running 1-hour closed-loop simulation..."
	python -c "\
from tep import TEPSimulator; \
sim = TEPSimulator(); \
sim.initialize(); \
result = sim.simulate(duration_hours=1.0); \
print(f'Simulation complete: {len(result.time)} steps'); \
print(f'Final reactor temp: {result.measurements[-1, 8]:.1f} C'); \
print(f'Shutdown: {result.shutdown}')"

# ============================================================================
# Building targets
# ============================================================================

build: clean-build
	python -m build

clean: clean-pyc clean-build clean-coverage clean-examples

clean-pyc:
	@echo "Removing Python bytecode..."
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

clean-build:
	@echo "Removing build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

clean-coverage:
	@echo "Removing coverage data..."
	rm -rf htmlcov/
	rm -f .coverage
	rm -f coverage.xml

clean-examples:
	@echo "Removing example output files..."
	rm -f normal_data.npy fault*_data.npy normal_data_sample.csv

# ============================================================================
# Development helpers
# ============================================================================

# Show package version
version:
	python -c "from tep import __version__; print(__version__)"

# Show package info
info:
	@echo "Package Information:"
	@python -c "import tep; print(f'  Version: {tep.__version__}')"
	@python -c "from tep.constants import NUM_STATES, NUM_MEASUREMENTS, NUM_MANIPULATED_VARS; \
		print(f'  States: {NUM_STATES}'); \
		print(f'  Measurements: {NUM_MEASUREMENTS}'); \
		print(f'  MVs: {NUM_MANIPULATED_VARS}')"

# Check if all imports work
check-imports:
	@echo "Checking package imports..."
	cd /tmp && python -c "from tep import TEPSimulator, ControlMode"
	cd /tmp && python -c "from tep.controllers import PIController, DecentralizedController"
	cd /tmp && python -c "from tep.fortran_backend import FortranTEProcess"
	cd /tmp && python -c "from tep.detector_base import BaseFaultDetector, FaultDetectorRegistry"
	cd /tmp && python -c "from tep.controller_base import BaseController, ControllerRegistry"
	@echo "All imports successful!"

# Interactive Python with TEP pre-imported
shell:
	cd /tmp && python -i -c "\
from tep import TEPSimulator, ControlMode; \
from tep.controllers import PIController, DecentralizedController; \
import numpy as np; \
print('TEP Simulator loaded. Available: TEPSimulator, ControlMode, PIController, np')"

# ============================================================================
# Documentation targets
# ============================================================================

# View documentation
docs:
	@echo "Documentation files:"
	@echo "  - README.md"
	@echo "  - docs/api.md"
	@echo "  - docs/dashboard.md"
	@echo "  - examples/README.md"

# ============================================================================
# Fortran targets (for comparison with original)
# ============================================================================

fortran-build:
	@echo "Building Fortran code..."
	@if command -v gfortran >/dev/null 2>&1; then \
		sed 's|~/|./|g' temain_mod.f > temain_mod_local.f; \
		gfortran -o tep_fortran temain_mod_local.f teprob.f; \
		rm -f temain_mod_local.f; \
		echo "Built: tep_fortran"; \
	else \
		echo "gfortran not found. Install with: brew install gcc (macOS) or apt install gfortran (Linux)"; \
	fi

fortran-run: fortran-build
	@echo "Running Fortran TEP simulation..."
	@echo "Note: Will prompt for disturbance (1=no disturbance, then 0=none)"
	@if [ -f tep_fortran ]; then \
		echo "1 0" | ./tep_fortran; \
		echo ""; \
		echo "Output files generated:"; \
		ls -la TE_data_*.dat 2>/dev/null || echo "  (no output files found)"; \
	else \
		echo "Error: tep_fortran not found. Run 'make fortran-build' first."; \
	fi

fortran-clean:
	rm -f tep_fortran temain_mod_local.f a.out TE_data_*.dat

# ============================================================================
# Backend targets
# ============================================================================

backend-info:
	@echo "TEP Simulator Backend Information"
	@echo "================================="
	@cd /tmp && python -c "\
from tep import get_available_backends, get_default_backend; \
print(f'Available backends: {get_available_backends()}'); \
print(f'Default backend: {get_default_backend()}'); \
from tep import TEPSimulator; \
sim = TEPSimulator(); \
print(f'TEPSimulator uses: {sim.backend}')"

backend-test:
	@echo "Testing Backend Parity (Python vs Fortran)"
	@echo "==========================================="
	cd /tmp && python $(CURDIR)/examples/compare_fortran_python.py

# ============================================================================
# Deployment targets
# ============================================================================

hf-upload:
	@echo "Uploading to Hugging Face Spaces..."
	hf upload jkitchin/tennessee-eastman-process . --repo-type=space --commit-message "update"

# ============================================================================
# Claude Code skill targets
# ============================================================================

install-skill:
	@echo "Installing TEP Claude Code skill..."
	@mkdir -p .claude/skills
	@rm -rf .claude/skills/TEP
	@cp -r skillz/TEP .claude/skills/
	@echo "Installed skill to .claude/skills/TEP"
	@echo ""
	@echo "Files installed:"
	@ls -la .claude/skills/TEP/
	@echo ""
	@echo "Claude Code will automatically discover this skill when working in this project."
