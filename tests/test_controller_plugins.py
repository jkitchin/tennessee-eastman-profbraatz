"""
Tests for the controller plugin system.

This module tests:
1. Base controller interface
2. Controller registry
3. Individual plugin controllers
4. Composite controllers
5. Integration with simulator
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from tep.controller_base import (
    BaseController,
    ControllerRegistry,
    register_controller,
    CompositeController,
)
from tep.controllers import (
    DecentralizedController,
    ManualController,
    PIController,
)
from tep.controller_plugins import (
    ReactorTemperatureController,
    SeparatorLevelController,
    StripperLevelController,
    ReactorSubsystemController,
    SeparatorSubsystemController,
    FeedSubsystemController,
    ProductQualityController,
    ReactorCompositionController,
    ProportionalOnlyController,
    EconomicMPCController,
    PassthroughController,
    create_composite_controller,
)
from tep.simulator import TEPSimulator, ControlMode
from tep.constants import NUM_MANIPULATED_VARS, NUM_MEASUREMENTS


# =============================================================================
# BASE CONTROLLER TESTS
# =============================================================================

class TestBaseController:
    """Tests for BaseController abstract class."""

    def test_cannot_instantiate_directly(self):
        """BaseController cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseController()

    def test_subclass_must_implement_calculate(self):
        """Subclasses must implement calculate method."""
        class IncompleteController(BaseController):
            def reset(self):
                pass

        with pytest.raises(TypeError):
            IncompleteController()

    def test_subclass_must_implement_reset(self):
        """Subclasses must implement reset method."""
        class IncompleteController(BaseController):
            def calculate(self, xmeas, xmv, step):
                return xmv

        with pytest.raises(TypeError):
            IncompleteController()

    def test_valid_subclass(self):
        """Valid subclass can be instantiated."""
        class ValidController(BaseController):
            name = "valid"
            description = "A valid controller"

            def calculate(self, xmeas, xmv, step):
                return xmv.copy()

            def reset(self):
                pass

        ctrl = ValidController()
        assert ctrl.name == "valid"
        assert ctrl.description == "A valid controller"

    def test_get_info(self):
        """get_info returns controller metadata."""
        class TestController(BaseController):
            name = "test_ctrl"
            description = "Test controller"
            version = "2.0.0"
            controlled_mvs = [1, 2, 3]

            def calculate(self, xmeas, xmv, step):
                return xmv

            def reset(self):
                pass

        ctrl = TestController()
        info = ctrl.get_info()

        assert info["name"] == "test_ctrl"
        assert info["description"] == "Test controller"
        assert info["version"] == "2.0.0"
        assert info["controlled_mvs"] == [1, 2, 3]


# =============================================================================
# CONTROLLER REGISTRY TESTS
# =============================================================================

class TestControllerRegistry:
    """Tests for ControllerRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        # Save existing registrations
        self._saved = ControllerRegistry._controllers.copy()

    def teardown_method(self):
        """Restore registry after each test."""
        ControllerRegistry._controllers = self._saved

    def test_register_controller(self):
        """Can register a controller class."""
        class MyController(BaseController):
            name = "my_ctrl"
            description = "My controller"

            def calculate(self, xmeas, xmv, step):
                return xmv

            def reset(self):
                pass

        ControllerRegistry.register(MyController)
        assert "my_ctrl" in ControllerRegistry.list_available()

    def test_register_with_custom_name(self):
        """Can register with a custom name."""
        class MyController(BaseController):
            name = "original_name"
            description = "My controller"

            def calculate(self, xmeas, xmv, step):
                return xmv

            def reset(self):
                pass

        ControllerRegistry.register(MyController, name="custom_name")
        assert "custom_name" in ControllerRegistry.list_available()

    def test_register_non_basecontroller_fails(self):
        """Registering non-BaseController class raises TypeError."""
        class NotAController:
            pass

        with pytest.raises(TypeError):
            ControllerRegistry.register(NotAController)

    def test_get_controller(self):
        """Can get a registered controller class."""
        class MyController(BaseController):
            name = "get_test"
            description = "Test"

            def calculate(self, xmeas, xmv, step):
                return xmv

            def reset(self):
                pass

        ControllerRegistry.register(MyController)
        cls = ControllerRegistry.get("get_test")
        assert cls is MyController

    def test_get_unknown_controller_fails(self):
        """Getting unknown controller raises KeyError."""
        with pytest.raises(KeyError):
            ControllerRegistry.get("nonexistent_controller")

    def test_create_controller(self):
        """Can create a controller instance."""
        class MyController(BaseController):
            name = "create_test"
            description = "Test"

            def __init__(self, param1=10):
                self.param1 = param1

            def calculate(self, xmeas, xmv, step):
                return xmv

            def reset(self):
                pass

        ControllerRegistry.register(MyController)
        ctrl = ControllerRegistry.create("create_test", param1=42)

        assert isinstance(ctrl, MyController)
        assert ctrl.param1 == 42

    def test_create_with_default_params(self):
        """Create uses registered default params."""
        class MyController(BaseController):
            name = "defaults_test"
            description = "Test"

            def __init__(self, value=0):
                self.value = value

            def calculate(self, xmeas, xmv, step):
                return xmv

            def reset(self):
                pass

        ControllerRegistry.register(
            MyController,
            default_params={"value": 100}
        )
        ctrl = ControllerRegistry.create("defaults_test")

        assert ctrl.value == 100

    def test_list_available(self):
        """list_available returns registered controller names."""
        # Should include built-in controllers
        available = ControllerRegistry.list_available()
        assert "decentralized" in available
        assert "manual" in available

    def test_get_info(self):
        """get_info returns controller metadata."""
        info = ControllerRegistry.get_info("decentralized")
        assert info["name"] == "decentralized"
        assert "description" in info

    def test_unregister(self):
        """Can unregister a controller."""
        class TempController(BaseController):
            name = "temp_ctrl"
            description = "Temporary"

            def calculate(self, xmeas, xmv, step):
                return xmv

            def reset(self):
                pass

        ControllerRegistry.register(TempController)
        assert "temp_ctrl" in ControllerRegistry.list_available()

        ControllerRegistry.unregister("temp_ctrl")
        assert "temp_ctrl" not in ControllerRegistry.list_available()


class TestRegisterDecorator:
    """Tests for @register_controller decorator."""

    def setup_method(self):
        self._saved = ControllerRegistry._controllers.copy()

    def teardown_method(self):
        ControllerRegistry._controllers = self._saved

    def test_decorator_registers_controller(self):
        """Decorator registers the controller."""
        @register_controller(name="decorated_ctrl")
        class DecoratedController(BaseController):
            name = "decorated_ctrl"
            description = "Decorated"

            def calculate(self, xmeas, xmv, step):
                return xmv

            def reset(self):
                pass

        assert "decorated_ctrl" in ControllerRegistry.list_available()


# =============================================================================
# SINGLE-LOOP CONTROLLER TESTS
# =============================================================================

class TestReactorTemperatureController:
    """Tests for ReactorTemperatureController."""

    def test_initialization(self):
        """Controller initializes with correct parameters."""
        ctrl = ReactorTemperatureController(setpoint=125.0, gain=-2.0)
        assert ctrl.setpoint == 125.0
        assert ctrl.gain == -2.0

    def test_calculate_returns_correct_shape(self):
        """calculate returns 12-element array."""
        ctrl = ReactorTemperatureController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        result = ctrl.calculate(xmeas, xmv, step=3)
        assert result.shape == (NUM_MANIPULATED_VARS,)

    def test_only_modifies_mv10(self):
        """Only modifies XMV 10 (reactor cooling water)."""
        ctrl = ReactorTemperatureController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[8] = 130.0  # High reactor temp
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        result = ctrl.calculate(xmeas, xmv, step=3)

        # Only MV 10 (index 9) should change
        for i in range(NUM_MANIPULATED_VARS):
            if i == 9:
                assert result[i] != 50.0  # Should have changed
            else:
                assert result[i] == 50.0  # Should be unchanged

    def test_reset_clears_state(self):
        """reset clears controller state."""
        ctrl = ReactorTemperatureController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[8] = 130.0
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        # Run a few steps to build up state
        for step in range(0, 12, 3):
            ctrl.calculate(xmeas, xmv, step)

        ctrl.reset()
        assert ctrl._controller.err_old == 0.0


class TestSeparatorLevelController:
    """Tests for SeparatorLevelController."""

    def test_only_modifies_mv7(self):
        """Only modifies XMV 7 (separator underflow)."""
        ctrl = SeparatorLevelController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[11] = 60.0  # High separator level
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        result = ctrl.calculate(xmeas, xmv, step=3)

        # Only MV 7 (index 6) should change
        for i in range(NUM_MANIPULATED_VARS):
            if i == 6:
                assert result[i] != 50.0
            else:
                assert result[i] == 50.0


class TestStripperLevelController:
    """Tests for StripperLevelController."""

    def test_only_modifies_mv8(self):
        """Only modifies XMV 8 (stripper product flow)."""
        ctrl = StripperLevelController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[14] = 60.0  # High stripper level
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        result = ctrl.calculate(xmeas, xmv, step=3)

        # Only MV 8 (index 7) should change
        for i in range(NUM_MANIPULATED_VARS):
            if i == 7:
                assert result[i] != 50.0
            else:
                assert result[i] == 50.0


# =============================================================================
# SUBSYSTEM CONTROLLER TESTS
# =============================================================================

class TestReactorSubsystemController:
    """Tests for ReactorSubsystemController."""

    def test_modifies_correct_mvs(self):
        """Modifies XMV 4 and 10."""
        ctrl = ReactorSubsystemController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[3] = 10.0   # A+C feed flow
        xmeas[7] = 80.0   # Reactor level
        xmeas[8] = 125.0  # Reactor temp
        xmeas[20] = 95.0  # CW outlet temp
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        result = ctrl.calculate(xmeas, xmv, step=3)

        # Check that MVs 4 and 10 are potentially modified
        controlled = [3, 9]  # 0-indexed
        for i in range(NUM_MANIPULATED_VARS):
            if i not in controlled:
                assert result[i] == 50.0, f"MV {i+1} should not change"

    def test_cascade_structure(self):
        """Controller has proper cascade structure."""
        ctrl = ReactorSubsystemController()
        assert hasattr(ctrl, 'cw_temp_setpoint')
        assert hasattr(ctrl, 'ac_feed_setpoint')


class TestFeedSubsystemController:
    """Tests for FeedSubsystemController."""

    def test_modifies_mvs_1_to_4(self):
        """Modifies XMV 1, 2, 3, 4."""
        ctrl = FeedSubsystemController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[0] = 0.3     # A feed
        xmeas[1] = 4000.0  # D feed
        xmeas[2] = 5000.0  # E feed
        xmeas[3] = 10.0    # A+C feed
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        result = ctrl.calculate(xmeas, xmv, step=3)

        controlled = [0, 1, 2, 3]
        for i in range(NUM_MANIPULATED_VARS):
            if i not in controlled:
                assert result[i] == 50.0, f"MV {i+1} should not change"

    def test_set_setpoints(self):
        """Can update setpoints dynamically."""
        ctrl = FeedSubsystemController()
        ctrl.set_setpoints(d_feed=4000.0, e_feed=5000.0)

        assert ctrl.d_feed_sp == 4000.0
        assert ctrl.e_feed_sp == 5000.0


# =============================================================================
# COMPOSITION CONTROLLER TESTS
# =============================================================================

class TestProductQualityController:
    """Tests for ProductQualityController."""

    def test_only_modifies_mv9(self):
        """Only modifies XMV 9 (steam valve)."""
        ctrl = ProductQualityController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[17] = 66.0   # Stripper temp
        xmeas[18] = 240.0  # Steam flow
        xmeas[37] = 0.85   # Product E composition
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        result = ctrl.calculate(xmeas, xmv, step=3)

        for i in range(NUM_MANIPULATED_VARS):
            if i == 8:
                pass  # MV 9 may or may not change on this step
            else:
                assert result[i] == 50.0

    def test_slow_loop_timing(self):
        """Slow loop only executes every 900 steps."""
        ctrl = ProductQualityController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[37] = 0.9  # High product E - should trigger outer loop
        xmeas[17] = 66.0  # Stripper temp
        xmeas[18] = 230.0  # Steam flow
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        initial_sp = ctrl.stripper_temp_setpoint

        # Run only at steps that are NOT divisible by 900
        # The fast loop (every 3 steps) will run but slow loop should not
        for step in [3, 6, 9, 300, 600]:
            ctrl.calculate(xmeas, xmv, step)

        # Slow loop setpoint should not have changed (only runs at step % 900 == 0)
        # Note: step=0 would trigger it, so we avoid that
        assert ctrl.stripper_temp_setpoint == initial_sp, \
            f"Stripper temp setpoint changed before step 900: {ctrl.stripper_temp_setpoint}"

        # Now run at step 900
        ctrl.calculate(xmeas, xmv, step=900)

        # Slow loop setpoint should have changed
        assert ctrl.stripper_temp_setpoint != initial_sp, \
            f"Stripper temp setpoint should change at step 900"


# =============================================================================
# FULL PROCESS CONTROLLER TESTS
# =============================================================================

class TestProportionalOnlyController:
    """Tests for ProportionalOnlyController."""

    def test_controls_mvs_1_to_11(self):
        """Controls MVs 1-11."""
        ctrl = ProportionalOnlyController()
        assert ctrl.controlled_mvs == list(range(1, 12))

    def test_calculate_updates_multiple_mvs(self):
        """calculate updates multiple MVs."""
        ctrl = ProportionalOnlyController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        # Set some non-zero measurements
        xmeas[1] = 3664.0   # D feed
        xmeas[2] = 4509.3   # E feed
        xmeas[11] = 60.0    # Sep level (high)
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        result = ctrl.calculate(xmeas, xmv, step=3)

        # Multiple MVs should have changed
        changed_count = sum(1 for i in range(11) if result[i] != 50.0)
        assert changed_count > 0


class TestEconomicMPCController:
    """Tests for EconomicMPCController."""

    def test_safety_override_high_temp(self):
        """Safety layer activates on high reactor temperature."""
        ctrl = EconomicMPCController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[8] = 148.0  # Near max temp (150)
        xmeas[6] = 2800.0  # Normal pressure
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        result = ctrl.calculate(xmeas, xmv, step=3)

        # Cooling water should increase
        assert result[9] > 50.0

    def test_safety_override_high_pressure(self):
        """Safety layer activates on high reactor pressure."""
        ctrl = EconomicMPCController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[8] = 120.0   # Normal temp
        xmeas[6] = 2860.0  # Near max pressure (2900)
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        result = ctrl.calculate(xmeas, xmv, step=3)

        # Purge should increase
        assert result[5] > 50.0

    def test_production_rate_adjustment(self):
        """Can adjust production rate target."""
        ctrl = EconomicMPCController()
        ctrl.set_production_rate(0.8)

        assert ctrl.production_rate_target == 0.8


class TestPassthroughController:
    """Tests for PassthroughController."""

    def test_returns_unchanged_mvs(self):
        """Returns MVs unchanged."""
        ctrl = PassthroughController()
        xmeas = np.random.rand(NUM_MEASUREMENTS)
        xmv = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 50, 50], dtype=float)

        result = ctrl.calculate(xmeas, xmv, step=3)

        np.testing.assert_array_equal(result, xmv)


# =============================================================================
# COMPOSITE CONTROLLER TESTS
# =============================================================================

class TestCompositeController:
    """Tests for CompositeController."""

    def test_combines_multiple_controllers(self):
        """Combines outputs from multiple sub-controllers."""
        composite = CompositeController()

        # Add reactor temp controller (MV 10)
        composite.add_controller(ReactorTemperatureController(), mvs=[10])

        # Add separator level controller (MV 7)
        composite.add_controller(SeparatorLevelController(), mvs=[7])

        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[8] = 130.0   # High reactor temp
        xmeas[11] = 60.0   # High sep level
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        result = composite.calculate(xmeas, xmv, step=3)

        # Both MVs should be modified
        assert result[9] != 50.0  # MV 10
        assert result[6] != 50.0  # MV 7

    def test_mv_conflict_raises_error(self):
        """Adding conflicting MV assignments raises error."""
        composite = CompositeController()
        composite.add_controller(ReactorTemperatureController(), mvs=[10])

        # Try to add another controller for MV 10
        with pytest.raises(ValueError):
            composite.add_controller(SeparatorLevelController(), mvs=[10])

    def test_fallback_controller(self):
        """Fallback controller handles unassigned MVs."""
        # Use a passthrough as fallback
        composite = CompositeController(
            fallback_controller=PassthroughController()
        )
        composite.add_controller(ReactorTemperatureController(), mvs=[10])

        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        result = composite.calculate(xmeas, xmv, step=3)

        # Uncontrolled MVs should be passed through
        assert result[0] == 50.0  # MV 1 (not controlled)

    def test_reset_resets_all(self):
        """reset calls reset on all sub-controllers."""
        composite = CompositeController()
        composite.add_controller(ReactorTemperatureController(), mvs=[10])
        composite.add_controller(SeparatorLevelController(), mvs=[7])

        # Build up state
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[8] = 130.0
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0
        for step in range(0, 30, 3):
            composite.calculate(xmeas, xmv, step)

        composite.reset()

        # Check sub-controllers were reset
        for ctrl, _ in composite._sub_controllers:
            assert ctrl._controller.err_old == 0.0


class TestCreateCompositeController:
    """Tests for create_composite_controller factory function."""

    def test_creates_composite_from_names(self):
        """Creates composite controller from subsystem names."""
        ctrl = create_composite_controller(
            ["reactor_temp", "separator_level"],
            fallback="passthrough"
        )

        assert isinstance(ctrl, CompositeController)
        assert len(ctrl._sub_controllers) == 2


# =============================================================================
# SIMULATOR INTEGRATION TESTS
# =============================================================================

class TestSimulatorIntegration:
    """Tests for controller integration with TEPSimulator."""

    @pytest.fixture
    def fortran_available(self):
        """Check if Fortran backend is available."""
        try:
            from tep._fortran import teprob  # noqa: F401
            return True
        except ImportError:
            return False

    def test_decentralized_controller_registered(self):
        """DecentralizedController is registered."""
        assert "decentralized" in ControllerRegistry.list_available()

    def test_manual_controller_registered(self):
        """ManualController is registered."""
        assert "manual" in ControllerRegistry.list_available()

    def test_custom_controller_with_simulator(self, fortran_available):
        """Custom controller works with simulator."""
        if not fortran_available:
            pytest.skip("Fortran backend not available")

        sim = TEPSimulator(control_mode=ControlMode.CLOSED_LOOP)
        sim.initialize()

        # Replace with custom controller
        custom_ctrl = ReactorTemperatureController()
        sim.controller = custom_ctrl

        # Run a few steps
        for _ in range(10):
            sim.step()

        # Should not have crashed
        assert sim.step_count == 10

    def test_simulate_with_controller_method(self, fortran_available):
        """simulate_with_controller accepts custom controllers."""
        if not fortran_available:
            pytest.skip("Fortran backend not available")

        sim = TEPSimulator(control_mode=ControlMode.CLOSED_LOOP)
        sim.initialize()

        # Use simulate_with_controller
        result = sim.simulate_with_controller(
            duration_hours=0.01,  # Short simulation
            controller=PassthroughController(),
            record_interval=36
        )

        assert result.time[-1] > 0

    def test_plugin_controller_via_registry(self, fortran_available):
        """Can use registry to create controller for simulator."""
        if not fortran_available:
            pytest.skip("Fortran backend not available")

        ctrl = ControllerRegistry.create("reactor_temp")

        sim = TEPSimulator(control_mode=ControlMode.CLOSED_LOOP)
        sim.initialize()
        sim.controller = ctrl

        # Run briefly
        result = sim.simulate(duration_hours=0.01, record_interval=36)
        assert not result.shutdown


# =============================================================================
# BUILT-IN CONTROLLER INHERITANCE TESTS
# =============================================================================

class TestDecentralizedControllerInheritance:
    """Tests that DecentralizedController properly inherits from BaseController."""

    def test_is_base_controller(self):
        """DecentralizedController inherits from BaseController."""
        ctrl = DecentralizedController()
        assert isinstance(ctrl, BaseController)

    def test_has_required_methods(self):
        """Has calculate and reset methods."""
        ctrl = DecentralizedController()
        assert hasattr(ctrl, 'calculate')
        assert hasattr(ctrl, 'reset')
        assert callable(ctrl.calculate)
        assert callable(ctrl.reset)

    def test_has_class_attributes(self):
        """Has required class attributes."""
        ctrl = DecentralizedController()
        assert ctrl.name == "decentralized"
        assert ctrl.description is not None
        assert ctrl.version is not None

    def test_get_parameters(self):
        """get_parameters returns controller state."""
        ctrl = DecentralizedController()
        params = ctrl.get_parameters()

        assert "setpoints" in params
        assert "purge_flag" in params


class TestManualControllerInheritance:
    """Tests that ManualController properly inherits from BaseController."""

    def test_is_base_controller(self):
        """ManualController inherits from BaseController."""
        ctrl = ManualController()
        assert isinstance(ctrl, BaseController)

    def test_has_required_methods(self):
        """Has calculate and reset methods."""
        ctrl = ManualController()
        assert hasattr(ctrl, 'calculate')
        assert hasattr(ctrl, 'reset')

    def test_reset_restores_defaults(self):
        """reset restores default MV values."""
        ctrl = ManualController()
        ctrl.set_mv(1, 100.0)

        ctrl.reset()

        # Should be back to default
        assert ctrl.mv_values[0] != 100.0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_controller_with_zero_step(self):
        """Controllers handle step=0."""
        ctrl = ReactorTemperatureController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        # Should not raise
        result = ctrl.calculate(xmeas, xmv, step=0)
        assert result.shape == (NUM_MANIPULATED_VARS,)

    def test_controller_with_negative_measurements(self):
        """Controllers handle negative measurements gracefully."""
        ctrl = ReactorTemperatureController()
        xmeas = np.ones(NUM_MEASUREMENTS) * -10.0
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        # Should not crash
        result = ctrl.calculate(xmeas, xmv, step=3)
        assert result.shape == (NUM_MANIPULATED_VARS,)

    def test_controller_with_nan_measurements(self):
        """Controllers produce output even with NaN measurements."""
        ctrl = ReactorTemperatureController()
        xmeas = np.ones(NUM_MEASUREMENTS) * np.nan
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        # Should not crash (though output may be NaN)
        result = ctrl.calculate(xmeas, xmv, step=3)
        assert result.shape == (NUM_MANIPULATED_VARS,)

    def test_controller_preserves_mv_limits(self):
        """Controller outputs respect MV limits (0-100)."""
        ctrl = ReactorTemperatureController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[8] = 200.0  # Extremely high temp
        xmv = np.ones(NUM_MANIPULATED_VARS) * 99.0  # Near max

        result = ctrl.calculate(xmeas, xmv, step=3)

        # Output should be clipped to 0-100
        assert 0.0 <= result[9] <= 100.0


# =============================================================================
# PERFORMANCE / TIMING TESTS
# =============================================================================

class TestControllerTiming:
    """Tests for controller timing behavior."""

    def test_fast_loop_timing(self):
        """Fast loops execute every 3 steps."""
        ctrl = ReactorTemperatureController()
        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[8] = 130.0  # High temp
        xmv_base = np.ones(NUM_MANIPULATED_VARS) * 50.0

        # On non-divisible-by-3 steps, output should equal input
        for step in [1, 2, 4, 5, 7, 8]:
            xmv = xmv_base.copy()
            result = ctrl.calculate(xmeas, xmv, step)
            # MV 10 should be unchanged since step % 3 != 0
            assert result[9] == 50.0, f"MV10 changed on step {step}"

        # On divisible-by-3 steps, output may change
        ctrl.reset()
        xmv = xmv_base.copy()
        result = ctrl.calculate(xmeas, xmv, step=3)
        # With high temp error, controller should respond
        assert result[9] != 50.0, "MV10 should change on step 3"

    def test_medium_loop_timing(self):
        """Medium loops (6 min) execute every 360 steps."""
        ctrl = ReactorCompositionController()

        xmeas = np.zeros(NUM_MEASUREMENTS)
        xmeas[22] = 40.0  # High comp A (setpoint is 32.188)
        xmv = np.ones(NUM_MANIPULATED_VARS) * 50.0

        # Record initial setpoint
        initial_sp = ctrl.a_feed_setpoint

        # Run at steps that are NOT 360
        # Fast loop runs at steps divisible by 3, but medium loop only at 360
        for step in [3, 6, 9, 300]:
            ctrl.calculate(xmeas, xmv, step)

        # After running non-360 steps, outer loop setpoint should not change
        # (only the fast loop executes)
        assert ctrl.a_feed_setpoint == initial_sp, \
            f"Setpoint changed before step 360: {ctrl.a_feed_setpoint} != {initial_sp}"

        # Now run at step 360
        ctrl.calculate(xmeas, xmv, step=360)

        # Setpoint should have changed due to medium loop
        assert ctrl.a_feed_setpoint != initial_sp, \
            f"Setpoint should have changed at step 360"
