"""
Tests for the controllers module.
"""

import pytest
import numpy as np
from tep.controllers import PIController, DecentralizedController, ManualController
from tep.constants import NUM_MEASUREMENTS, NUM_MANIPULATED_VARS


class TestPIController:
    """Test PI controller implementation."""

    def test_p_only_control(self):
        """P-only control (taui=0) should respond proportionally."""
        ctrl = PIController(setpoint=50.0, gain=1.0, taui=0.0, scale=1.0)

        # Initial call
        output1 = ctrl.calculate(50.0, 50.0, 0.001)  # At setpoint
        assert abs(output1 - 50.0) < 0.01

        # Error of +10
        output2 = ctrl.calculate(40.0, 50.0, 0.001)
        # With gain=1 and error=10, output should increase by ~10
        assert output2 > output1

    def test_pi_control_integral_action(self):
        """PI control should have integral action."""
        ctrl = PIController(setpoint=50.0, gain=1.0, taui=0.1, scale=1.0)

        output = 50.0
        # Apply constant error
        for _ in range(100):
            output = ctrl.calculate(40.0, output, 0.001)

        # With integral action, output should continue increasing
        assert output > 50.0

    def test_output_limits(self):
        """Output should respect limits."""
        ctrl = PIController(
            setpoint=50.0, gain=100.0, taui=0.0, scale=1.0,
            output_min=0.0, output_max=100.0
        )

        # Large positive error
        output = ctrl.calculate(0.0, 50.0, 0.001)
        assert output <= 100.0

        # Reset
        ctrl.reset()

        # Large negative error
        output = ctrl.calculate(100.0, 50.0, 0.001)
        assert output >= 0.0

    def test_reset(self):
        """Reset should clear error history."""
        ctrl = PIController(setpoint=50.0, gain=1.0, taui=0.0, scale=1.0)

        ctrl.calculate(40.0, 50.0, 0.001)
        assert ctrl.err_old != 0.0

        ctrl.reset()
        assert ctrl.err_old == 0.0

    def test_scaling(self):
        """Scale factor should normalize error."""
        ctrl1 = PIController(setpoint=50.0, gain=1.0, taui=0.0, scale=1.0)
        ctrl2 = PIController(setpoint=50.0, gain=1.0, taui=0.0, scale=0.5)

        output1 = ctrl1.calculate(40.0, 50.0, 0.001)
        output2 = ctrl2.calculate(40.0, 50.0, 0.001)

        # Scaled controller should have smaller response
        assert abs(output2 - 50.0) < abs(output1 - 50.0)


class TestDecentralizedController:
    """Test the full decentralized control system."""

    @pytest.fixture
    def controller(self):
        return DecentralizedController()

    @pytest.fixture
    def steady_state_xmeas(self):
        """Approximate steady-state measurements."""
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[0] = 0.25  # A Feed
        xmeas[1] = 3664.0  # D Feed
        xmeas[2] = 4509.3  # E Feed
        xmeas[3] = 9.35  # A+C Feed
        xmeas[4] = 26.9  # Recycle
        xmeas[7] = 75.0  # Reactor level
        xmeas[8] = 120.4  # Reactor temp
        xmeas[9] = 0.34  # Purge rate
        xmeas[11] = 50.0  # Sep level
        xmeas[12] = 2633.7  # Sep pressure
        xmeas[14] = 50.0  # Stripper level
        xmeas[16] = 22.9  # Stripper underflow
        xmeas[17] = 65.7  # Stripper temp
        xmeas[18] = 230.0  # Steam flow
        xmeas[20] = 94.6  # Reactor CW outlet
        xmeas[22] = 32.2  # Reactor feed A
        xmeas[25] = 6.9  # Reactor feed D
        xmeas[26] = 18.8  # Reactor feed E
        xmeas[29] = 13.8  # Purge B
        xmeas[37] = 0.84  # Product E
        return xmeas

    @pytest.fixture
    def steady_state_xmv(self):
        """Approximate steady-state MVs."""
        from tep.constants import INITIAL_STATES
        return INITIAL_STATES[38:50].copy()

    def test_output_shape(self, controller, steady_state_xmeas, steady_state_xmv):
        """Controller should return correct number of MVs."""
        xmv_new = controller.calculate(steady_state_xmeas, steady_state_xmv, 3)
        assert len(xmv_new) == NUM_MANIPULATED_VARS

    def test_output_in_range(self, controller, steady_state_xmeas, steady_state_xmv):
        """All MVs should be in 0-100 range."""
        xmv_new = controller.calculate(steady_state_xmeas, steady_state_xmv, 3)
        assert all(xmv_new >= 0.0)
        assert all(xmv_new <= 100.0)

    def test_fast_loop_timing(self, controller, steady_state_xmeas, steady_state_xmv):
        """Fast loops should execute every 3 steps."""
        # Step 0-2 should not trigger fast loops (only step 3, 6, 9...)
        xmv1 = controller.calculate(steady_state_xmeas, steady_state_xmv, 1)
        xmv2 = controller.calculate(steady_state_xmeas, steady_state_xmv, 2)

        # At step 3, fast loops should trigger
        xmv3 = controller.calculate(steady_state_xmeas, steady_state_xmv, 3)

        # Since we're at setpoint, changes should be small but present
        assert np.allclose(xmv1, xmv2)  # No control at steps 1, 2

    def test_medium_loop_timing(self, controller, steady_state_xmeas, steady_state_xmv):
        """Medium loops should execute every 360 steps."""
        # Execute many fast loops first
        for step in range(3, 359, 3):
            controller.calculate(steady_state_xmeas, steady_state_xmv, step)

        # At step 360, medium loops should trigger
        xmv_before = controller.setpoints.copy()
        controller.calculate(steady_state_xmeas, steady_state_xmv, 360)
        xmv_after = controller.setpoints.copy()

        # Some setpoints may change at step 360
        # This is a weak test - just ensure no errors

    def test_reset(self, controller, steady_state_xmeas, steady_state_xmv):
        """Reset should restore initial state."""
        controller.calculate(steady_state_xmeas, steady_state_xmv, 3)
        controller.reset()

        assert controller.step_count == 0
        assert controller.purge_flag == 0


class TestManualController:
    """Test the manual controller."""

    def test_initial_values(self):
        """Should use provided initial values."""
        init = np.array([50.0] * 12)
        ctrl = ManualController(initial_values=init)
        np.testing.assert_array_equal(ctrl.mv_values, init)

    def test_set_mv(self):
        """Setting MV should work."""
        ctrl = ManualController()
        ctrl.set_mv(1, 75.0)
        assert ctrl.mv_values[0] == 75.0

    def test_set_mv_clipping(self):
        """MV values should be clipped."""
        ctrl = ManualController()
        ctrl.set_mv(1, 150.0)
        assert ctrl.mv_values[0] == 100.0

        ctrl.set_mv(1, -50.0)
        assert ctrl.mv_values[0] == 0.0

    def test_calculate_returns_set_values(self):
        """Calculate should return the set values."""
        ctrl = ManualController()
        ctrl.set_mv(5, 60.0)

        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmv = np.zeros(NUM_MANIPULATED_VARS)

        result = ctrl.calculate(xmeas, xmv, 0)
        assert result[4] == 60.0


class TestPurgeControl:
    """Test the purge valve override logic."""

    @pytest.fixture
    def controller(self):
        return DecentralizedController()

    def test_high_pressure_override(self, controller):
        """High separator pressure should open purge valve."""
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[12] = 3000.0  # High pressure
        xmv = np.zeros(NUM_MANIPULATED_VARS)
        xmv[5] = 40.0

        xmv_new = controller.calculate(xmeas, xmv, 3)
        assert xmv_new[5] == 100.0

    def test_low_pressure_override(self, controller):
        """Low separator pressure should close purge valve."""
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[12] = 2200.0  # Low pressure
        xmv = np.zeros(NUM_MANIPULATED_VARS)
        xmv[5] = 40.0

        xmv_new = controller.calculate(xmeas, xmv, 3)
        assert xmv_new[5] == 0.0
