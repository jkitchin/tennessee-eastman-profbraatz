"""Tests for backend selection functionality.

This module tests the auto-selection of backends when Python is the default
but Fortran may optionally be available.
"""

import pytest
import numpy as np


class TestBackendFunctions:
    """Test backend utility functions."""

    def test_get_available_backends_includes_python(self):
        """Test that Python backend is always available."""
        from tep import get_available_backends
        backends = get_available_backends()
        assert "python" in backends

    def test_get_available_backends_returns_list(self):
        """Test that get_available_backends returns a list."""
        from tep import get_available_backends
        backends = get_available_backends()
        assert isinstance(backends, list)
        assert len(backends) >= 1

    def test_get_default_backend_returns_string(self):
        """Test that get_default_backend returns a string."""
        from tep import get_default_backend
        backend = get_default_backend()
        assert isinstance(backend, str)
        assert backend in ["python", "fortran"]

    def test_is_fortran_available_returns_bool(self):
        """Test that is_fortran_available returns a boolean."""
        from tep import is_fortran_available
        available = is_fortran_available()
        assert isinstance(available, bool)

    def test_default_backend_is_in_available(self):
        """Test that default backend is in available backends."""
        from tep import get_available_backends, get_default_backend
        backends = get_available_backends()
        default = get_default_backend()
        assert default in backends


class TestSimulatorBackendSelection:
    """Test TEPSimulator backend selection."""

    def test_explicit_python_backend(self):
        """Test creating simulator with explicit Python backend."""
        from tep import TEPSimulator
        sim = TEPSimulator(backend="python")
        assert sim.backend == "python"

    def test_auto_backend_selection(self):
        """Test that None backend auto-selects."""
        from tep import TEPSimulator, get_default_backend
        sim = TEPSimulator(backend=None)
        expected = get_default_backend()
        assert sim.backend == expected

    def test_default_backend_selection(self):
        """Test that default (no argument) auto-selects."""
        from tep import TEPSimulator, get_default_backend
        sim = TEPSimulator()
        expected = get_default_backend()
        assert sim.backend == expected

    def test_invalid_backend_raises(self):
        """Test that invalid backend name raises error."""
        from tep import TEPSimulator
        with pytest.raises((ValueError, ImportError)):
            TEPSimulator(backend="invalid_backend")

    def test_simulation_runs_with_auto_backend(self):
        """Test that simulation runs with auto-selected backend."""
        from tep import TEPSimulator
        sim = TEPSimulator()  # Auto-select backend
        result = sim.simulate(duration_hours=0.01)
        assert result.time.shape[0] > 0
        assert result.measurements.shape[1] == 41


class TestBackendConsistency:
    """Test that backends behave consistently."""

    def test_python_backend_produces_results(self):
        """Test that Python backend produces valid results."""
        from tep import TEPSimulator
        sim = TEPSimulator(backend="python")
        result = sim.simulate(duration_hours=0.1)

        assert result.time.shape[0] > 0
        assert result.measurements.shape[1] == 41
        assert result.manipulated_vars.shape[1] == 12
        assert result.states.shape[1] == 50

    def test_python_backend_initialization(self):
        """Test Python backend initializes correctly."""
        from tep import TEPSimulator
        sim = TEPSimulator(backend="python")
        sim.initialize()

        assert sim.initialized
        assert sim.time == 0.0

    def test_python_backend_reproducibility(self):
        """Test that Python backend is reproducible with same seed."""
        from tep import TEPSimulator

        # First run
        sim1 = TEPSimulator(backend="python", random_seed=12345)
        result1 = sim1.simulate(duration_hours=0.1)

        # Second run with same seed
        sim2 = TEPSimulator(backend="python", random_seed=12345)
        result2 = sim2.simulate(duration_hours=0.1)

        # Results should match
        np.testing.assert_array_almost_equal(
            result1.measurements, result2.measurements
        )


class TestBackendExport:
    """Test that backends are properly exported in __init__."""

    def test_python_backend_exported(self):
        """Test PythonTEProcess is exported."""
        from tep import PythonTEProcess
        assert PythonTEProcess is not None

    def test_fortran_backend_exported_or_none(self):
        """Test FortranTEProcess is exported (may be None if not available)."""
        from tep import FortranTEProcess, is_fortran_available
        if is_fortran_available():
            assert FortranTEProcess is not None
        else:
            assert FortranTEProcess is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
