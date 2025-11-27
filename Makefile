# Makefile for Tennessee Eastman Process Python Simulator
# Usage: make [target]

.PHONY: help install install-dev install-gui test test-verbose test-coverage \
        lint format dashboard run-example clean clean-pyc clean-build \
        docs build publish

# Default target
help:
	@echo "Tennessee Eastman Process Simulator - Available targets:"
	@echo ""
	@echo "  Installation:"
	@echo "    install        Install the package"
	@echo "    install-dev    Install with development dependencies"
	@echo "    install-gui    Install with GUI dashboard support"
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
	@echo "    dashboard      Launch the interactive GUI dashboard"
	@echo "    run-example    Run basic simulation example"
	@echo "    run-examples   Run all examples"
	@echo ""
	@echo "  Building:"
	@echo "    build          Build distribution packages"
	@echo "    clean          Clean all build artifacts"
	@echo ""

# ============================================================================
# Installation targets
# ============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-gui:
	pip install -e ".[gui]"

install-all:
	pip install -e ".[dev,gui]"

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

test-thermo:
	python -m pytest tests/test_thermodynamics.py -v

test-disturbances:
	python -m pytest tests/test_disturbances.py -v

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
	@echo "Launching TEP Dashboard..."
	@echo "(Requires GUI dependencies: make install-gui)"
	python -c "from tep import run_dashboard; run_dashboard()"

# Alternative dashboard launch
dashboard-script:
	tep-dashboard

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
	python -c "from tep import TEPSimulator, TEProcess"
	python -c "from tep.simulator import ControlMode"
	python -c "from tep.controllers import PIController, DecentralizedController"
	python -c "from tep.thermodynamics import calculate_enthalpy, calculate_temperature"
	python -c "from tep.disturbances import DisturbanceManager, RandomGenerator"
	python -c "from tep.integrators import Integrator, IntegratorType"
	@echo "All imports successful!"

# Interactive Python with TEP pre-imported
shell:
	python -i -c "\
from tep import TEPSimulator, TEProcess; \
from tep.simulator import ControlMode; \
from tep.controllers import PIController, DecentralizedController; \
import numpy as np; \
print('TEP Simulator loaded. Available: TEPSimulator, TEProcess, ControlMode, PIController, np')"

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
		gfortran -o tep_fortran temain_mod.f teprob.f; \
		echo "Built: tep_fortran"; \
	else \
		echo "gfortran not found. Install with: apt install gfortran"; \
	fi

fortran-clean:
	rm -f tep_fortran a.out TE_data_*.dat
