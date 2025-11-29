"""Tests for the pure Python backend implementation.

This module validates that the Python backend produces statistically
similar results to the Fortran backend across various scenarios.
"""

import pytest
import numpy as np
from scipy import stats

# Import backends directly
from tep.python_backend import PythonTEProcess
from tep import TEPSimulator, ControlMode


class TestPythonBackendBasic:
    """Basic functionality tests for Python backend."""

    def test_initialization(self):
        """Test that Python backend initializes correctly."""
        process = PythonTEProcess()
        process.initialize()

        assert process._initialized
        assert process.time == 0.0
        assert len(process.yy) == 50
        assert len(process.yp) == 50
        assert len(process.xmeas) == 41
        assert len(process.xmv) == 12

    def test_initial_state_values(self):
        """Test initial state values match expected steady-state."""
        process = PythonTEProcess()
        process.initialize()

        # Check first few state values (reactor holdups)
        assert abs(process.yy[0] - 10.40491389) < 1e-6
        assert abs(process.yy[1] - 4.363996017) < 1e-6
        assert abs(process.yy[2] - 7.570059737) < 1e-6

        # Check valve positions (XMV 1)
        assert abs(process.yy[38] - 63.05263039) < 1e-6

    def test_step_advances_time(self):
        """Test that stepping advances simulation time."""
        process = PythonTEProcess()
        process.initialize()

        initial_time = process.time
        process.step()
        assert process.time > initial_time
        assert abs(process.time - 1.0/3600.0) < 1e-10

    def test_measurements_change_over_time(self):
        """Test that measurements evolve during simulation."""
        process = PythonTEProcess()
        process.initialize()

        initial_meas = process.xmeas.copy()

        # Step for 100 seconds
        for _ in range(100):
            process.step()

        # Some measurements should have changed (due to noise at minimum)
        assert not np.allclose(initial_meas, process.xmeas, rtol=1e-6)

    def test_disturbance_setting(self):
        """Test setting and clearing disturbances."""
        process = PythonTEProcess()
        process.initialize()

        # Set disturbance
        process.set_idv(1, 1)
        assert process.idv[0] == 1

        # Clear disturbance
        process.set_idv(1, 0)
        assert process.idv[0] == 0

        # Clear all
        process.set_idv(5, 1)
        process.set_idv(10, 1)
        process.clear_disturbances()
        assert np.sum(process.idv) == 0

    def test_manipulated_variable_setting(self):
        """Test setting manipulated variables."""
        process = PythonTEProcess()
        process.initialize()

        # Set MV
        process.set_xmv(1, 75.0)
        assert abs(process.xmv[0] - 75.0) < 1e-10

        # Test clipping
        process.set_xmv(2, 150.0)  # Should clip to 100
        assert abs(process.xmv[1] - 100.0) < 1e-10

        process.set_xmv(3, -10.0)  # Should clip to 0
        assert abs(process.xmv[2] - 0.0) < 1e-10

    def test_random_seed_reproducibility(self):
        """Test that same seed produces same trajectory."""
        seed = 12345

        # First run
        p1 = PythonTEProcess(random_seed=seed)
        p1.initialize()
        for _ in range(1000):
            p1.step()
        meas1 = p1.xmeas.copy()

        # Second run with same seed
        p2 = PythonTEProcess(random_seed=seed)
        p2.initialize()
        for _ in range(1000):
            p2.step()
        meas2 = p2.xmeas.copy()

        np.testing.assert_array_almost_equal(meas1, meas2)

    def test_different_seeds_different_trajectories(self):
        """Test that different seeds produce different trajectories."""
        # First run
        p1 = PythonTEProcess(random_seed=111)
        p1.initialize()
        for _ in range(1000):
            p1.step()
        meas1 = p1.xmeas.copy()

        # Second run with different seed
        p2 = PythonTEProcess(random_seed=222)
        p2.initialize()
        for _ in range(1000):
            p2.step()
        meas2 = p2.xmeas.copy()

        # Should be different
        assert not np.allclose(meas1, meas2, rtol=1e-6)


class TestPythonBackendWithSimulator:
    """Test Python backend through TEPSimulator interface."""

    def test_simulator_with_python_backend(self):
        """Test creating simulator with Python backend."""
        sim = TEPSimulator(backend="python")
        assert sim.backend == "python"
        sim.initialize()
        assert sim.initialized

    def test_simulate_basic(self):
        """Test basic simulation with Python backend."""
        sim = TEPSimulator(backend="python", control_mode=ControlMode.CLOSED_LOOP)
        result = sim.simulate(duration_hours=0.1)

        assert result.time.shape[0] > 0
        assert result.measurements.shape[1] == 41
        assert result.manipulated_vars.shape[1] == 12
        assert result.states.shape[1] == 50
        assert not result.shutdown

    def test_simulate_with_disturbance(self):
        """Test simulation with disturbance activation."""
        sim = TEPSimulator(backend="python", control_mode=ControlMode.CLOSED_LOOP)
        sim.initialize()

        # Activate fault 1 immediately
        sim.set_disturbance(1, 1)

        result = sim.simulate(duration_hours=0.5)
        assert not result.shutdown  # Should not shut down for fault 1

    def test_open_loop_stability(self):
        """Test open-loop simulation doesn't immediately crash."""
        sim = TEPSimulator(backend="python", control_mode=ControlMode.OPEN_LOOP)

        # Open loop should run for at least a short time
        result = sim.simulate(duration_hours=0.01)
        assert result.time.shape[0] > 0


class TestPythonVsFortranComparison:
    """Compare Python backend against Fortran backend for statistical similarity."""

    @pytest.fixture
    def fortran_available(self):
        """Check if Fortran backend is available."""
        from tep import is_fortran_available
        return is_fortran_available()

    def test_initial_states_match(self, fortran_available):
        """Test that initial states match between backends."""
        if not fortran_available:
            pytest.skip("Fortran backend not available")

        from tep.fortran_backend import FortranTEProcess

        py_proc = PythonTEProcess()
        py_proc.initialize()

        f_proc = FortranTEProcess()
        f_proc.initialize()

        # Initial states should match closely (small floating-point differences expected)
        np.testing.assert_array_almost_equal(
            py_proc.yy, f_proc.yy, decimal=5,
            err_msg="Initial states differ between backends"
        )

    def test_initial_measurements_similar(self, fortran_available):
        """Test that initial measurements are similar between backends."""
        if not fortran_available:
            pytest.skip("Fortran backend not available")

        from tep.fortran_backend import FortranTEProcess

        py_proc = PythonTEProcess()
        py_proc.initialize()

        f_proc = FortranTEProcess()
        f_proc.initialize()

        # Measurements should be close (may have small differences due to
        # floating point and noise generation)
        np.testing.assert_array_almost_equal(
            py_proc.xmeas, f_proc.xmeas, decimal=3,
            err_msg="Initial measurements differ significantly"
        )

    def test_short_trajectory_similar(self, fortran_available):
        """Test that short trajectories are similar between backends."""
        if not fortran_available:
            pytest.skip("Fortran backend not available")

        from tep.fortran_backend import FortranTEProcess

        seed = 4651207995  # Default Fortran seed

        py_proc = PythonTEProcess(random_seed=seed)
        py_proc.initialize()

        f_proc = FortranTEProcess(random_seed=seed)
        f_proc.initialize()

        # Run for 100 seconds
        for _ in range(100):
            py_proc.step()
            f_proc.step()

        # States should remain close
        max_state_diff = np.max(np.abs(py_proc.yy - f_proc.yy))
        assert max_state_diff < 0.1, f"States diverged: max diff = {max_state_diff}"

    def test_statistical_similarity_open_loop(self, fortran_available):
        """Test statistical similarity of trajectories in open loop."""
        if not fortran_available:
            pytest.skip("Fortran backend not available")

        n_runs = 5
        duration = 0.5  # hours
        n_steps = int(duration * 3600)

        py_means = []
        f_means = []

        for seed in range(n_runs):
            # Python backend
            py_sim = TEPSimulator(backend="python", random_seed=seed * 1000,
                                  control_mode=ControlMode.OPEN_LOOP)
            py_result = py_sim.simulate(duration_hours=duration)
            py_means.append(py_result.measurements.mean(axis=0))

            # Fortran backend
            f_sim = TEPSimulator(backend="fortran", random_seed=seed * 1000,
                                 control_mode=ControlMode.OPEN_LOOP)
            f_result = f_sim.simulate(duration_hours=duration)
            f_means.append(f_result.measurements.mean(axis=0))

        py_means = np.array(py_means)
        f_means = np.array(f_means)

        # Compare means across runs for each measurement
        for i in range(41):
            # Calculate relative difference in means
            py_mean = py_means[:, i].mean()
            f_mean = f_means[:, i].mean()

            if abs(f_mean) > 1e-6:
                rel_diff = abs(py_mean - f_mean) / abs(f_mean)
                assert rel_diff < 0.1, (
                    f"XMEAS({i+1}): Python mean={py_mean:.4f}, "
                    f"Fortran mean={f_mean:.4f}, rel_diff={rel_diff:.4f}"
                )

    def test_statistical_similarity_closed_loop(self, fortran_available):
        """Test statistical similarity of trajectories in closed loop."""
        if not fortran_available:
            pytest.skip("Fortran backend not available")

        n_runs = 5
        duration = 1.0  # hours

        py_final_states = []
        f_final_states = []

        for seed in range(n_runs):
            # Python backend
            py_sim = TEPSimulator(backend="python", random_seed=seed * 1000,
                                  control_mode=ControlMode.CLOSED_LOOP)
            py_result = py_sim.simulate(duration_hours=duration)
            py_final_states.append(py_result.measurements[-1, :])

            # Fortran backend
            f_sim = TEPSimulator(backend="fortran", random_seed=seed * 1000,
                                 control_mode=ControlMode.CLOSED_LOOP)
            f_result = f_sim.simulate(duration_hours=duration)
            f_final_states.append(f_result.measurements[-1, :])

        py_final = np.array(py_final_states)
        f_final = np.array(f_final_states)

        # Compare mean final states
        py_mean = py_final.mean(axis=0)
        f_mean = f_final.mean(axis=0)

        # Most measurements should be within 10% of each other
        n_similar = 0
        for i in range(41):
            if abs(f_mean[i]) > 1e-6:
                rel_diff = abs(py_mean[i] - f_mean[i]) / abs(f_mean[i])
                if rel_diff < 0.1:
                    n_similar += 1
            else:
                if abs(py_mean[i] - f_mean[i]) < 0.1:
                    n_similar += 1

        # At least 90% of measurements should be similar
        assert n_similar >= 37, f"Only {n_similar}/41 measurements are similar"

    def test_variance_similarity(self, fortran_available):
        """Test that variance of trajectories is similar between backends."""
        if not fortran_available:
            pytest.skip("Fortran backend not available")

        duration = 1.0  # hours

        # Python backend
        py_sim = TEPSimulator(backend="python", random_seed=12345,
                              control_mode=ControlMode.CLOSED_LOOP)
        py_result = py_sim.simulate(duration_hours=duration)
        py_var = py_result.measurements.var(axis=0)

        # Fortran backend
        f_sim = TEPSimulator(backend="fortran", random_seed=12345,
                             control_mode=ControlMode.CLOSED_LOOP)
        f_result = f_sim.simulate(duration_hours=duration)
        f_var = f_result.measurements.var(axis=0)

        # Compare variances - should be same order of magnitude
        for i in range(41):
            if f_var[i] > 1e-10:
                ratio = py_var[i] / f_var[i]
                assert 0.1 < ratio < 10, (
                    f"XMEAS({i+1}): variance ratio = {ratio:.4f}"
                )


class TestDisturbanceResponses:
    """Test process response to various disturbances."""

    @pytest.mark.parametrize("fault_id", [1, 2, 3, 4, 5, 6, 7])
    def test_step_disturbance_response(self, fault_id):
        """Test that step disturbances cause measurable changes."""
        sim = TEPSimulator(backend="python", control_mode=ControlMode.CLOSED_LOOP)
        sim.initialize()

        # Get baseline measurements
        baseline = sim.get_measurements().copy()

        # Apply disturbance
        sim.set_disturbance(fault_id, 1)

        # Simulate for 10 minutes
        sim.simulate(duration_hours=10/60)

        # Get new measurements
        after_fault = sim.get_measurements()

        # At least some measurements should have changed
        max_change = np.max(np.abs(after_fault - baseline))
        assert max_change > 0.01, f"Fault {fault_id} caused no measurable change"

    def test_multiple_disturbances(self):
        """Test simultaneous disturbances."""
        sim = TEPSimulator(backend="python", control_mode=ControlMode.CLOSED_LOOP)
        sim.initialize()

        # Apply multiple disturbances
        sim.set_disturbance(1, 1)
        sim.set_disturbance(4, 1)
        sim.set_disturbance(8, 1)

        # Should be able to simulate
        result = sim.simulate(duration_hours=0.5)
        assert not result.shutdown


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_time_step(self):
        """Test behavior with very small time step."""
        process = PythonTEProcess()
        process.initialize()

        state_before = process.yy.copy()
        process.step(dt=1e-10)

        # State should barely change with tiny time step
        assert np.allclose(state_before, process.yy, rtol=1e-6)

    def test_large_time_step(self):
        """Test stability with larger time step."""
        process = PythonTEProcess()
        process.initialize()

        # Larger time step (10 seconds)
        process.step(dt=10.0/3600.0)

        # Process should still be running (not shutdown)
        assert not process.is_shutdown()

    def test_extreme_mv_values(self):
        """Test behavior with extreme MV values."""
        sim = TEPSimulator(backend="python", control_mode=ControlMode.MANUAL)
        sim.initialize()

        # Set all MVs to maximum
        for i in range(1, 13):
            sim.set_mv(i, 100.0)

        # Simulate briefly - might cause shutdown
        result = sim.simulate(duration_hours=0.1)
        # Just verify it runs (may or may not shutdown)
        assert result.time.shape[0] > 0

    def test_process_state_save_restore(self):
        """Test saving and restoring process state."""
        sim = TEPSimulator(backend="python")
        sim.initialize()

        # Run for a bit
        sim.simulate(duration_hours=0.1)

        # Save state
        saved_state = sim.get_states().copy()
        saved_meas = sim.get_measurements().copy()

        # Run more
        sim.simulate(duration_hours=0.1)

        # State should have changed
        assert not np.allclose(saved_state, sim.get_states())

        # Restore state
        sim.process.set_state(saved_state)

        # State should match saved
        np.testing.assert_array_almost_equal(saved_state, sim.get_states())


class TestThermodynamicFunctions:
    """Test individual thermodynamic functions."""

    def test_tesub1_liquid_enthalpy(self):
        """Test liquid enthalpy calculation."""
        process = PythonTEProcess()

        # Test with pure component at 100 deg C
        z = np.zeros(8)
        z[3] = 1.0  # Pure component D

        h = process._tesub1(z, 100.0, 0)
        assert h > 0  # Enthalpy should be positive at 100 C

    def test_tesub2_temperature_iteration(self):
        """Test temperature from enthalpy iteration."""
        process = PythonTEProcess()

        # Test round-trip
        z = np.zeros(8)
        z[3] = 0.5
        z[4] = 0.3
        z[5] = 0.2

        t_original = 80.0
        h = process._tesub1(z, t_original, 0)
        t_recovered = process._tesub2(z, 50.0, h, 0)

        assert abs(t_recovered - t_original) < 0.01

    def test_tesub4_density(self):
        """Test liquid density calculation."""
        process = PythonTEProcess()

        # Test with typical liquid composition
        x = np.zeros(8)
        x[3] = 0.3  # D
        x[4] = 0.25  # E
        x[5] = 0.25  # F
        x[6] = 0.1  # G
        x[7] = 0.1  # H

        rho = process._tesub4(x, 80.0)
        # Density is in kmol/m^3, organic liquids typically 0.2-2 kmol/m^3
        assert 0.1 < rho < 5, f"Density {rho} kmol/m^3 outside reasonable range"

    def test_tesub7_random_range(self):
        """Test random number generator produces correct range."""
        process = PythonTEProcess()

        # Test positive range [0, 1)
        for _ in range(100):
            r = process._tesub7(1)
            assert 0 <= r < 1

        # Test symmetric range (-1, 1)
        vals = [process._tesub7(-1) for _ in range(1000)]
        assert min(vals) < 0  # Should have negative values
        assert max(vals) > 0  # Should have positive values
        assert all(-1 <= v <= 1 for v in vals)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
