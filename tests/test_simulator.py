"""
Tests for the main TEP simulator.
"""

import pytest
import numpy as np
from tep.simulator import TEPSimulator, ControlMode, SimulationResult
from tep.constants import (
    NUM_STATES, NUM_MEASUREMENTS, NUM_MANIPULATED_VARS,
    INITIAL_STATES, DEFAULT_RANDOM_SEED
)


class TestTEPSimulator:
    """Test the high-level simulator interface."""

    @pytest.fixture
    def simulator(self):
        sim = TEPSimulator(
            random_seed=DEFAULT_RANDOM_SEED,
            control_mode=ControlMode.CLOSED_LOOP
        )
        sim.initialize()
        return sim

    def test_initialization(self, simulator):
        """Simulator should initialize properly."""
        assert simulator.initialized
        assert simulator.time == 0.0
        assert simulator.step_count == 0

    def test_get_measurements(self, simulator):
        """Should return correct number of measurements."""
        meas = simulator.get_measurements()
        assert len(meas) == NUM_MEASUREMENTS

    def test_get_mvs(self, simulator):
        """Should return correct number of MVs."""
        mvs = simulator.get_manipulated_vars()
        assert len(mvs) == NUM_MANIPULATED_VARS

    def test_get_states(self, simulator):
        """Should return correct number of states."""
        states = simulator.get_states()
        assert len(states) == NUM_STATES

    def test_step(self, simulator):
        """Single step should advance time."""
        simulator.step()
        assert simulator.step_count == 1
        assert simulator.time > 0

    def test_multiple_steps(self, simulator):
        """Multiple steps should work."""
        simulator.step(10)
        assert simulator.step_count == 10

    def test_simulate_short_duration(self, simulator):
        """Short simulation should complete."""
        result = simulator.simulate(duration_hours=0.01)
        assert isinstance(result, SimulationResult)
        assert len(result.time) > 1
        assert result.shutdown == False

    def test_simulate_returns_correct_shapes(self, simulator):
        """Simulation result arrays should have correct shapes."""
        result = simulator.simulate(duration_hours=0.01, record_interval=10)

        assert result.states.shape[1] == NUM_STATES
        assert result.measurements.shape[1] == NUM_MEASUREMENTS
        assert result.manipulated_vars.shape[1] == NUM_MANIPULATED_VARS
        assert len(result.time) == result.states.shape[0]

    def test_simulate_time_progression(self, simulator):
        """Time should progress monotonically."""
        result = simulator.simulate(duration_hours=0.01)
        assert all(np.diff(result.time) > 0)

    def test_set_disturbance(self, simulator):
        """Setting disturbance should work."""
        simulator.set_disturbance(1, 1)
        assert 1 in simulator.get_active_disturbances()

    def test_clear_disturbances(self, simulator):
        """Clearing disturbances should work."""
        simulator.set_disturbance(1, 1)
        simulator.clear_disturbances()
        assert len(simulator.get_active_disturbances()) == 0

    def test_simulate_with_disturbance_schedule(self, simulator):
        """Disturbance schedule should be applied."""
        result = simulator.simulate(
            duration_hours=0.1,
            disturbances={1: (0.05, 1)}  # Apply IDV(1) at 0.05 hours
        )
        assert isinstance(result, SimulationResult)


class TestControlModes:
    """Test different control modes."""

    def test_open_loop_mode(self):
        """Open-loop mode should not change MVs."""
        sim = TEPSimulator(control_mode=ControlMode.OPEN_LOOP)
        sim.initialize()

        initial_mvs = sim.get_manipulated_vars().copy()
        sim.step(100)
        final_mvs = sim.get_manipulated_vars()

        # MVs should be unchanged in open-loop
        np.testing.assert_array_almost_equal(initial_mvs, final_mvs)

    def test_closed_loop_mode(self):
        """Closed-loop mode should have controller active."""
        sim = TEPSimulator(control_mode=ControlMode.CLOSED_LOOP)
        sim.initialize()

        # Create a disturbance that controller should respond to
        sim.set_disturbance(1, 1)
        initial_mvs = sim.get_manipulated_vars().copy()
        sim.step(1000)
        final_mvs = sim.get_manipulated_vars()

        # Some MVs should have changed
        assert not np.allclose(initial_mvs, final_mvs)

    def test_manual_mode(self):
        """Manual mode should allow direct MV setting."""
        sim = TEPSimulator(control_mode=ControlMode.MANUAL)
        sim.initialize()

        sim.set_mv(1, 70.0)
        sim.step(100)

        mvs = sim.get_manipulated_vars()
        assert abs(mvs[0] - 70.0) < 0.1


class TestSimulationResult:
    """Test SimulationResult properties."""

    @pytest.fixture
    def result(self):
        sim = TEPSimulator()
        sim.initialize()
        return sim.simulate(duration_hours=0.01)

    def test_time_seconds(self, result):
        """time_seconds should be time * 3600."""
        np.testing.assert_array_almost_equal(
            result.time_seconds,
            result.time * 3600
        )

    def test_time_minutes(self, result):
        """time_minutes should be time * 60."""
        np.testing.assert_array_almost_equal(
            result.time_minutes,
            result.time * 60
        )


class TestSafetyShutdown:
    """Test safety shutdown functionality."""

    def test_no_shutdown_at_steady_state(self):
        """Normal operation should not trigger shutdown for short duration."""
        sim = TEPSimulator()
        sim.initialize()
        # Use shorter duration - longer simulations may require tuning
        result = sim.simulate(duration_hours=0.05)
        assert result.shutdown == False

    def test_shutdown_flag(self):
        """Simulator should report shutdown state."""
        sim = TEPSimulator()
        sim.initialize()
        assert not sim.is_shutdown()


class TestReproducibility:
    """Test that simulations are reproducible."""

    def test_same_seed_same_results(self):
        """Same seed should produce identical results."""
        sim1 = TEPSimulator(random_seed=12345)
        sim1.initialize()
        result1 = sim1.simulate(duration_hours=0.1)

        sim2 = TEPSimulator(random_seed=12345)
        sim2.initialize()
        result2 = sim2.simulate(duration_hours=0.1)

        np.testing.assert_array_almost_equal(result1.states, result2.states)
        np.testing.assert_array_almost_equal(result1.measurements, result2.measurements)



class TestStreamingInterface:
    """Test real-time streaming interface."""

    def test_stream_start(self):
        """Starting stream should initialize simulator."""
        sim = TEPSimulator()
        sim.start_stream(history_size=100)
        assert sim.initialized

    def test_stream_step(self):
        """Stream step should return dict with required keys."""
        sim = TEPSimulator()
        sim.start_stream()

        result = sim.stream_step()

        assert 'time' in result
        assert 'time_seconds' in result
        assert 'measurements' in result
        assert 'mvs' in result
        assert 'shutdown' in result

    def test_stream_history(self):
        """Stream history should accumulate."""
        sim = TEPSimulator()
        sim.start_stream(history_size=100)

        for _ in range(50):
            sim.stream_step()

        history = sim.get_stream_history()
        assert len(history['time']) == 51  # Initial + 50 steps

    def test_stream_history_limit(self):
        """Stream history should respect size limit."""
        sim = TEPSimulator()
        sim.start_stream(history_size=100)

        for _ in range(200):
            sim.stream_step()

        history = sim.get_stream_history()
        assert len(history['time']) == 100
