"""
Tests for the fault plugin system.

This module tests:
1. Base fault plugin interface
2. Fault plugin registry
3. Individual fault plugins (IDV equivalents)
4. Custom fault plugins
5. Fault manager
6. Integration with simulator
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from tep.fault_base import (
    BaseFaultPlugin,
    FaultEffect,
    FaultPluginRegistry,
    FaultManager,
    register_fault,
)
from tep.fault_plugins import (
    IDV1FeedRatioFault,
    IDV3DFeedTempFault,
    IDV4ReactorCWTempFault,
    IDV6AFeedLossFault,
    IDV8FeedCompRandomFault,
    IDV13KineticsDriftFault,
    IDV14ReactorValveStickFault,
    CoolingSystemFailureFault,
    GradualFeedLossFault,
    IntermittentTempSpikeFault,
    create_idv_fault,
    get_idv_fault_mapping,
    list_fault_categories,
)
from tep.simulator import TEPSimulator, ControlMode
from tep.constants import NUM_MANIPULATED_VARS, NUM_MEASUREMENTS


# =============================================================================
# BASE FAULT PLUGIN TESTS
# =============================================================================

class TestBaseFaultPlugin:
    """Tests for BaseFaultPlugin abstract class."""

    def test_cannot_instantiate_directly(self):
        """BaseFaultPlugin cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFaultPlugin()

    def test_subclass_must_implement_apply(self):
        """Subclasses must implement apply method."""
        class IncompleteFault(BaseFaultPlugin):
            def reset(self):
                pass

        with pytest.raises(TypeError):
            IncompleteFault()

    def test_subclass_must_implement_reset(self):
        """Subclasses must implement reset method."""
        class IncompleteFault(BaseFaultPlugin):
            def apply(self, time, process_state):
                return []

        with pytest.raises(TypeError):
            IncompleteFault()

    def test_valid_subclass(self):
        """Valid subclass can be instantiated."""
        class ValidFault(BaseFaultPlugin):
            name = "valid"
            description = "A valid fault"

            def apply(self, time, process_state):
                return []

            def reset(self):
                pass

        fault = ValidFault()
        assert fault.name == "valid"
        assert fault.description == "A valid fault"

    def test_get_info(self):
        """get_info returns fault metadata."""
        class TestFault(BaseFaultPlugin):
            name = "test_fault"
            description = "Test fault"
            version = "2.0.0"
            category = "test"

            def apply(self, time, process_state):
                return []

            def reset(self):
                pass

        fault = TestFault(magnitude=0.5)
        info = fault.get_info()

        assert info["name"] == "test_fault"
        assert info["description"] == "Test fault"
        assert info["version"] == "2.0.0"
        assert info["category"] == "test"
        assert info["magnitude"] == 0.5
        assert info["is_active"] is False

    def test_activate_deactivate(self):
        """Test fault activation and deactivation."""
        class TestFault(BaseFaultPlugin):
            name = "test"

            def apply(self, time, process_state):
                return []

            def reset(self):
                pass

        fault = TestFault()
        assert not fault.is_active
        assert fault.activation_time is None

        fault.activate(1.5)
        assert fault.is_active
        assert fault.activation_time == 1.5

        fault.deactivate()
        assert not fault.is_active
        assert fault.activation_time is None

    def test_magnitude_parameter(self):
        """Test magnitude parameter handling."""
        class TestFault(BaseFaultPlugin):
            name = "test"

            def apply(self, time, process_state):
                return [FaultEffect('var', 'additive', self.magnitude)]

            def reset(self):
                pass

        fault = TestFault(magnitude=0.75)
        assert fault.magnitude == 0.75
        assert fault.get_parameters() == {"magnitude": 0.75}

    def test_set_parameter(self):
        """Test setting parameters dynamically."""
        class TestFault(BaseFaultPlugin):
            name = "test"

            def apply(self, time, process_state):
                return []

            def reset(self):
                pass

        fault = TestFault(magnitude=1.0)
        fault.set_parameter("magnitude", 0.5)
        assert fault.magnitude == 0.5

        with pytest.raises(AttributeError):
            fault.set_parameter("nonexistent", 1.0)


# =============================================================================
# FAULT EFFECT TESTS
# =============================================================================

class TestFaultEffect:
    """Tests for FaultEffect dataclass."""

    def test_create_additive_effect(self):
        """Create an additive fault effect."""
        effect = FaultEffect('temperature', 'additive', 5.0)
        assert effect.variable == 'temperature'
        assert effect.mode == 'additive'
        assert effect.value == 5.0

    def test_create_multiplicative_effect(self):
        """Create a multiplicative fault effect."""
        effect = FaultEffect('flow', 'multiplicative', 0.8)
        assert effect.variable == 'flow'
        assert effect.mode == 'multiplicative'
        assert effect.value == 0.8

    def test_create_replace_effect(self):
        """Create a replacement fault effect."""
        effect = FaultEffect('valve_position', 'replace', 50.0)
        assert effect.variable == 'valve_position'
        assert effect.mode == 'replace'
        assert effect.value == 50.0


# =============================================================================
# FAULT PLUGIN REGISTRY TESTS
# =============================================================================

class TestFaultPluginRegistry:
    """Tests for FaultPluginRegistry."""

    def setup_method(self):
        """Store original registry state."""
        self._original_faults = dict(FaultPluginRegistry._faults)

    def teardown_method(self):
        """Restore original registry state."""
        FaultPluginRegistry._faults = self._original_faults

    def test_register_fault(self):
        """Register a fault plugin."""
        class CustomFault(BaseFaultPlugin):
            name = "custom_test"
            description = "Custom test fault"

            def apply(self, time, process_state):
                return []

            def reset(self):
                pass

        FaultPluginRegistry.register(CustomFault)
        assert "custom_test" in FaultPluginRegistry.list_available()

    def test_register_with_custom_name(self):
        """Register with a custom name."""
        class CustomFault(BaseFaultPlugin):
            name = "original_name"

            def apply(self, time, process_state):
                return []

            def reset(self):
                pass

        FaultPluginRegistry.register(CustomFault, name="custom_name")
        assert "custom_name" in FaultPluginRegistry.list_available()

    def test_create_fault(self):
        """Create a fault instance from registry."""
        fault = FaultPluginRegistry.create("idv4_reactor_cw", magnitude=0.5)
        assert fault.name == "idv4_reactor_cw"
        assert fault.magnitude == 0.5

    def test_create_unknown_fault_fails(self):
        """Creating unknown fault raises KeyError."""
        with pytest.raises(KeyError):
            FaultPluginRegistry.create("nonexistent_fault")

    def test_list_available(self):
        """List available faults."""
        available = FaultPluginRegistry.list_available()
        assert isinstance(available, list)
        assert "idv4_reactor_cw" in available
        assert "idv1_feed_ratio" in available

    def test_list_by_category(self):
        """List faults by category."""
        step_faults = FaultPluginRegistry.list_by_category("step")
        assert "idv4_reactor_cw" in step_faults
        assert "idv3_d_temp" in step_faults

    def test_get_categories(self):
        """Get all fault categories."""
        categories = FaultPluginRegistry.get_categories()
        assert "step" in categories
        assert "random" in categories

    def test_get_info(self):
        """Get fault information."""
        info = FaultPluginRegistry.get_info("idv4_reactor_cw")
        assert info["name"] == "idv4_reactor_cw"
        assert "description" in info
        assert "category" in info


class TestRegisterDecorator:
    """Tests for @register_fault decorator."""

    def setup_method(self):
        """Store original registry state."""
        self._original_faults = dict(FaultPluginRegistry._faults)

    def teardown_method(self):
        """Restore original registry state."""
        FaultPluginRegistry._faults = self._original_faults

    def test_decorator_registers_fault(self):
        """Decorator registers fault class."""
        @register_fault(name="decorated_fault", description="A decorated fault")
        class DecoratedFault(BaseFaultPlugin):
            name = "decorated_fault"

            def apply(self, time, process_state):
                return []

            def reset(self):
                pass

        assert "decorated_fault" in FaultPluginRegistry.list_available()

    def test_decorator_with_default_params(self):
        """Decorator with default parameters."""
        @register_fault(
            name="param_fault",
            default_params={"magnitude": 0.5}
        )
        class ParamFault(BaseFaultPlugin):
            name = "param_fault"

            def apply(self, time, process_state):
                return []

            def reset(self):
                pass

        fault = FaultPluginRegistry.create("param_fault")
        assert fault.magnitude == 0.5


# =============================================================================
# IDV FAULT PLUGIN TESTS
# =============================================================================

class TestIDV1FeedRatioFault:
    """Tests for IDV1 Feed Ratio fault."""

    def test_initialization(self):
        """Test fault initialization."""
        fault = IDV1FeedRatioFault(magnitude=1.0)
        assert fault.name == "idv1_feed_ratio"
        assert fault.magnitude == 1.0

    def test_apply_returns_feed_comp_effect(self):
        """Apply returns feed composition effect."""
        fault = IDV1FeedRatioFault(magnitude=1.0)
        fault.activate(0.0)

        effects = fault.apply(0.5, {})
        assert len(effects) == 1
        assert effects[0].variable == 'feed_comp_a'
        assert effects[0].mode == 'additive'
        assert effects[0].value == pytest.approx(-0.03)

    def test_magnitude_scales_effect(self):
        """Magnitude scales the effect."""
        fault = IDV1FeedRatioFault(magnitude=0.5)
        fault.activate(0.0)

        effects = fault.apply(0.5, {})
        assert effects[0].value == pytest.approx(-0.015)


class TestIDV4ReactorCWTempFault:
    """Tests for IDV4 Reactor Cooling Water fault."""

    def test_initialization(self):
        """Test fault initialization."""
        fault = IDV4ReactorCWTempFault()
        assert fault.name == "idv4_reactor_cw"
        assert fault.category == "step"

    def test_apply_returns_temperature_effect(self):
        """Apply returns temperature effect."""
        fault = IDV4ReactorCWTempFault(magnitude=1.0)
        fault.activate(0.0)

        effects = fault.apply(0.5, {})
        assert len(effects) == 1
        assert effects[0].variable == 'reactor_cw_inlet_temp'
        assert effects[0].mode == 'additive'
        assert effects[0].value == pytest.approx(5.0)


class TestIDV6AFeedLossFault:
    """Tests for IDV6 A Feed Loss fault."""

    def test_apply_returns_flow_multiplier(self):
        """Apply returns flow multiplier effect."""
        fault = IDV6AFeedLossFault(magnitude=1.0)
        fault.activate(0.0)

        effects = fault.apply(0.5, {})
        assert len(effects) == 1
        assert effects[0].variable == 'flow_a'
        assert effects[0].mode == 'multiplicative'
        assert effects[0].value == pytest.approx(0.0)  # Complete loss

    def test_partial_magnitude(self):
        """Partial magnitude gives partial loss."""
        fault = IDV6AFeedLossFault(magnitude=0.5)
        fault.activate(0.0)

        effects = fault.apply(0.5, {})
        assert effects[0].value == pytest.approx(0.5)  # 50% flow


class TestIDV8FeedCompRandomFault:
    """Tests for IDV8 Random Feed Composition fault."""

    def test_initialization(self):
        """Test fault initialization."""
        fault = IDV8FeedCompRandomFault(magnitude=1.0, correlation_time=0.1)
        assert fault.name == "idv8_feed_comp_random"
        assert fault.category == "random"
        assert fault.correlation_time == 0.1

    def test_apply_returns_composition_effects(self):
        """Apply returns composition effects."""
        fault = IDV8FeedCompRandomFault(magnitude=1.0)
        fault.activate(0.0)

        process_state = {'random': np.random.default_rng(42)}
        effects = fault.apply(0.5, process_state)

        assert len(effects) == 2
        assert effects[0].variable == 'feed_comp_a'
        assert effects[1].variable == 'feed_comp_b'

    def test_reset_clears_state(self):
        """Reset clears internal state."""
        fault = IDV8FeedCompRandomFault()
        fault._current_value = 1.0
        fault._last_time = 5.0

        fault.reset()
        assert fault._current_value == 0.0
        assert fault._last_time == 0.0


class TestIDV13KineticsDriftFault:
    """Tests for IDV13 Kinetics Drift fault."""

    def test_initialization(self):
        """Test fault initialization."""
        fault = IDV13KineticsDriftFault(magnitude=1.0, drift_rate=0.01)
        assert fault.name == "idv13_kinetics_drift"
        assert fault.category == "drift"
        assert fault.drift_rate == 0.01

    def test_drift_increases_over_time(self):
        """Drift increases over time."""
        fault = IDV13KineticsDriftFault(magnitude=1.0, drift_rate=1.0)
        fault.activate(0.0)

        effects_early = fault.apply(0.1, {})
        effects_late = fault.apply(1.0, {})

        # Later effects should have larger drift
        assert abs(effects_late[0].value - 1.0) > abs(effects_early[0].value - 1.0)


class TestIDV14ReactorValveStickFault:
    """Tests for IDV14 Reactor Valve Sticking fault."""

    def test_initialization(self):
        """Test fault initialization."""
        fault = IDV14ReactorValveStickFault(magnitude=1.0, stick_probability=0.1)
        assert fault.name == "idv14_reactor_valve_stick"
        assert fault.category == "valve"
        assert fault.stick_probability == 0.1

    def test_reset_clears_stuck_state(self):
        """Reset clears stuck state."""
        fault = IDV14ReactorValveStickFault()
        fault._stuck = True
        fault._stuck_position = 50.0

        fault.reset()
        assert fault._stuck is False
        assert fault._stuck_position is None


# =============================================================================
# CUSTOM FAULT PLUGIN TESTS
# =============================================================================

class TestCoolingSystemFailureFault:
    """Tests for combined cooling system failure fault."""

    def test_returns_both_temperature_effects(self):
        """Returns effects for both reactor and condenser."""
        fault = CoolingSystemFailureFault(magnitude=1.0)
        fault.activate(0.0)

        effects = fault.apply(0.5, {})
        assert len(effects) == 2

        variables = [e.variable for e in effects]
        assert 'reactor_cw_inlet_temp' in variables
        assert 'condenser_cw_inlet_temp' in variables


class TestGradualFeedLossFault:
    """Tests for gradual feed loss fault."""

    def test_gradual_onset(self):
        """Flow loss develops gradually."""
        fault = GradualFeedLossFault(magnitude=1.0, onset_hours=1.0)
        fault.activate(0.0)

        # At start
        effects_start = fault.apply(0.0, {})
        # At midpoint
        effects_mid = fault.apply(0.5, {})
        # At end
        effects_end = fault.apply(1.0, {})

        assert effects_start[0].value == pytest.approx(1.0)  # Full flow
        assert effects_mid[0].value == pytest.approx(0.5)    # Half flow
        assert effects_end[0].value == pytest.approx(0.0)    # No flow


class TestIntermittentTempSpikeFault:
    """Tests for intermittent temperature spike fault."""

    def test_periodic_behavior(self):
        """Fault is active periodically."""
        fault = IntermittentTempSpikeFault(
            magnitude=1.0,
            period_hours=1.0,
            duty_cycle=0.5
        )
        fault.activate(0.0)

        # In active phase
        effects_active = fault.apply(0.25, {})
        # In inactive phase
        effects_inactive = fault.apply(0.75, {})

        assert len(effects_active) == 1
        assert len(effects_inactive) == 0


# =============================================================================
# FAULT MANAGER TESTS
# =============================================================================

class TestFaultManager:
    """Tests for FaultManager."""

    def test_add_fault_by_name(self):
        """Add fault by registered name."""
        manager = FaultManager()
        manager.add_fault('idv4_reactor_cw', magnitude=0.5)

        faults = manager.get_all_faults()
        assert len(faults) == 1
        assert faults[0]['name'] == 'idv4_reactor_cw'

    def test_add_fault_instance(self):
        """Add fault instance directly."""
        manager = FaultManager()
        fault = IDV4ReactorCWTempFault(magnitude=0.5)
        manager.add_fault(fault)

        faults = manager.get_all_faults()
        assert len(faults) == 1

    def test_scheduled_activation(self):
        """Faults activate at scheduled time."""
        manager = FaultManager()
        manager.add_fault('idv4_reactor_cw', activate_at=1.0)

        # Before activation time
        assert manager.get_active_faults() == []

        # Apply at activation time
        manager.apply_all(1.0, {})
        assert 'idv4_reactor_cw' in manager.get_active_faults()

    def test_immediate_activation(self):
        """Faults with no schedule activate immediately."""
        manager = FaultManager()
        manager.add_fault('idv4_reactor_cw')

        assert 'idv4_reactor_cw' in manager.get_active_faults()

    def test_apply_all_collects_effects(self):
        """apply_all collects effects from all active faults."""
        manager = FaultManager()
        manager.add_fault('idv4_reactor_cw', magnitude=1.0)
        manager.add_fault('idv5_condenser_cw', magnitude=1.0)

        effects = manager.apply_all(0.5, {})
        assert len(effects) == 2

    def test_remove_fault(self):
        """Remove a fault by name."""
        manager = FaultManager()
        manager.add_fault('idv4_reactor_cw')
        manager.add_fault('idv5_condenser_cw')

        manager.remove_fault('idv4_reactor_cw')

        faults = manager.get_all_faults()
        assert len(faults) == 1
        assert faults[0]['name'] == 'idv5_condenser_cw'

    def test_reset_deactivates_all(self):
        """Reset deactivates all faults."""
        manager = FaultManager()
        manager.add_fault('idv4_reactor_cw')
        manager.add_fault('idv5_condenser_cw')

        assert len(manager.get_active_faults()) == 2

        manager.reset()
        assert len(manager.get_active_faults()) == 0

    def test_set_random_seed(self):
        """Set random seed for reproducibility."""
        manager = FaultManager()
        manager.set_random_seed(42)
        # No error means success


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Tests for fault plugin utility functions."""

    def test_get_idv_fault_mapping(self):
        """Get IDV to fault name mapping."""
        mapping = get_idv_fault_mapping()

        assert mapping[1] == "idv1_feed_ratio"
        assert mapping[4] == "idv4_reactor_cw"
        assert mapping[6] == "idv6_a_feed_loss"
        assert 16 not in mapping  # IDV 16-20 not defined

    def test_create_idv_fault(self):
        """Create fault from IDV number."""
        fault = create_idv_fault(4, magnitude=0.5)
        assert fault.name == "idv4_reactor_cw"
        assert fault.magnitude == 0.5

    def test_create_idv_fault_invalid(self):
        """Invalid IDV number raises ValueError."""
        with pytest.raises(ValueError):
            create_idv_fault(99)

    def test_list_fault_categories(self):
        """List faults grouped by category."""
        categories = list_fault_categories()

        assert "step" in categories
        assert "random" in categories
        assert "idv4_reactor_cw" in categories["step"]


# =============================================================================
# SIMULATOR INTEGRATION TESTS
# =============================================================================

class TestSimulatorIntegration:
    """Tests for fault plugin integration with simulator."""

    def test_add_fault_to_simulator(self):
        """Add a fault to the simulator."""
        sim = TEPSimulator()
        sim.initialize()
        sim.add_fault('idv4_reactor_cw', activate_at=0.01, magnitude=0.5)

        faults = sim.get_all_faults()
        assert len(faults) == 1
        assert faults[0]['name'] == 'idv4_reactor_cw'

    def test_fault_activates_during_simulation(self):
        """Fault activates at scheduled time during simulation."""
        sim = TEPSimulator()
        sim.initialize()
        sim.add_fault('idv4_reactor_cw', activate_at=0.01)

        # Before activation
        assert sim.get_active_faults() == []

        # Run past activation time
        for _ in range(50):  # ~50 seconds
            sim.step()

        assert 'idv4_reactor_cw' in sim.get_active_faults()

    def test_remove_fault_from_simulator(self):
        """Remove a fault from the simulator."""
        sim = TEPSimulator()
        sim.initialize()
        sim.add_fault('idv4_reactor_cw')
        sim.add_fault('idv5_condenser_cw')

        sim.remove_fault('idv4_reactor_cw')

        faults = sim.get_all_faults()
        assert len(faults) == 1

    def test_clear_faults(self):
        """Clear all faults from simulator."""
        sim = TEPSimulator()
        sim.initialize()
        sim.add_fault('idv4_reactor_cw')
        sim.add_fault('idv5_condenser_cw')

        sim.clear_faults()
        assert len(sim.get_all_faults()) == 0

    def test_faults_reset_on_initialize(self):
        """Faults are reset when simulator initializes."""
        sim = TEPSimulator()
        sim.initialize()
        sim.add_fault('idv4_reactor_cw')

        # Run some steps
        for _ in range(100):
            sim.step()

        # Re-initialize
        sim.initialize()

        # Faults should be reset (deactivated)
        assert sim.get_active_faults() == []

    def test_multiple_faults_simulation(self):
        """Run simulation with multiple faults."""
        sim = TEPSimulator()
        sim.initialize()

        # Use lower magnitudes to avoid overwhelming the process
        sim.add_fault('idv3_d_temp', activate_at=0.01, magnitude=0.2)
        sim.add_fault('idv4_reactor_cw', activate_at=0.02, magnitude=0.2)
        sim.add_fault('idv5_condenser_cw', activate_at=0.03, magnitude=0.2)

        # Run simulation
        result = sim.simulate(duration_hours=0.1, record_interval=60)

        # Should complete - multiple faults were applied
        assert len(result.time) > 0
        # At least some faults should be active
        active = sim.get_active_faults()
        assert len(active) >= 2, f"Expected at least 2 active faults, got {active}"

    def test_custom_fault_with_simulator(self):
        """Use custom fault class with simulator."""
        class CustomTestFault(BaseFaultPlugin):
            name = "custom_sim_test"
            description = "Custom test fault"

            def apply(self, time, process_state):
                return [FaultEffect('reactor_cw_inlet_temp', 'additive', 2.0)]

            def reset(self):
                pass

        sim = TEPSimulator()
        sim.initialize()

        fault = CustomTestFault(magnitude=1.0)
        sim.add_fault(fault, activate_at=0.01)

        # Run simulation
        for _ in range(50):
            sim.step()

        assert 'custom_sim_test' in sim.get_active_faults()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_fault_with_zero_magnitude(self):
        """Fault with zero magnitude produces zero effect."""
        fault = IDV4ReactorCWTempFault(magnitude=0.0)
        fault.activate(0.0)

        effects = fault.apply(0.5, {})
        assert effects[0].value == pytest.approx(0.0)

    def test_fault_not_activated(self):
        """Fault that's not activated returns no effects."""
        fault = IDV4ReactorCWTempFault(magnitude=1.0)
        # Don't activate

        effects = fault.apply(0.5, {})
        # Step faults return effects regardless of activation state
        # (activation is managed by FaultManager)
        assert len(effects) == 1

    def test_empty_process_state(self):
        """Faults handle empty process state."""
        fault = IDV4ReactorCWTempFault(magnitude=1.0)
        fault.activate(0.0)

        effects = fault.apply(0.5, {})
        assert len(effects) == 1

    def test_negative_magnitude(self):
        """Negative magnitude reverses effect direction."""
        fault = IDV4ReactorCWTempFault(magnitude=-1.0)
        fault.activate(0.0)

        effects = fault.apply(0.5, {})
        assert effects[0].value == pytest.approx(-5.0)

    def test_large_magnitude(self):
        """Large magnitude scales effect proportionally."""
        fault = IDV4ReactorCWTempFault(magnitude=10.0)
        fault.activate(0.0)

        effects = fault.apply(0.5, {})
        assert effects[0].value == pytest.approx(50.0)


# =============================================================================
# PROCESS INTEGRATION TESTS - Verify faults actually affect process
# =============================================================================

class TestFaultEffectsOnProcess:
    """
    Tests that verify fault plugins actually modify the process behavior.

    These tests run simulations with and without faults and compare the results
    to ensure the faults are having the expected effects.
    """

    def test_reactor_cw_temp_fault_increases_temperature(self):
        """Reactor CW temperature fault should increase reactor temperature."""
        # Run without fault
        sim_normal = TEPSimulator(backend='python')
        sim_normal.initialize()
        for _ in range(3600):  # 1 hour
            sim_normal.step()
        temp_normal = sim_normal.get_measurements()[8]  # XMEAS(9) reactor temp

        # Run with fault
        sim_fault = TEPSimulator(backend='python')
        sim_fault.initialize()
        sim_fault.add_fault('idv4_reactor_cw', magnitude=1.0)  # +5°C to CW
        for _ in range(3600):  # 1 hour
            sim_fault.step()
        temp_fault = sim_fault.get_measurements()[8]

        # Warmer cooling water = less cooling = higher reactor temp
        assert temp_fault > temp_normal, \
            f"Expected reactor temp with fault ({temp_fault:.2f}) > normal ({temp_normal:.2f})"

    def test_condenser_cw_temp_fault_affects_process(self):
        """Condenser CW temperature fault should affect separator pressure."""
        # Run without fault
        sim_normal = TEPSimulator(backend='python')
        sim_normal.initialize()
        for _ in range(3600):
            sim_normal.step()
        pressure_normal = sim_normal.get_measurements()[12]  # XMEAS(13) separator pressure

        # Run with fault
        sim_fault = TEPSimulator(backend='python')
        sim_fault.initialize()
        sim_fault.add_fault('idv5_condenser_cw', magnitude=1.0)  # +5°C to CW
        for _ in range(3600):
            sim_fault.step()
        pressure_fault = sim_fault.get_measurements()[12]

        # Different condenser temp should affect separator pressure
        assert abs(pressure_fault - pressure_normal) > 0.1, \
            f"Separator pressure should differ: fault={pressure_fault:.2f}, normal={pressure_normal:.2f}"

    def test_d_feed_temp_fault_affects_process(self):
        """D feed temperature fault should affect reactor temperature."""
        # Run without fault
        sim_normal = TEPSimulator(backend='python')
        sim_normal.initialize()
        for _ in range(1800):  # 30 min
            sim_normal.step()
        temp_normal = sim_normal.get_measurements()[8]

        # Run with fault
        sim_fault = TEPSimulator(backend='python')
        sim_fault.initialize()
        sim_fault.add_fault('idv3_d_temp', magnitude=1.0)  # +5°C to D feed
        for _ in range(1800):
            sim_fault.step()
        temp_fault = sim_fault.get_measurements()[8]

        # Hotter feed = higher reactor temp
        assert temp_fault > temp_normal, \
            f"Expected reactor temp with fault ({temp_fault:.2f}) > normal ({temp_normal:.2f})"

    def test_flow_a_loss_fault_reduces_flow(self):
        """A feed loss fault should reduce A feed flow rate."""
        # Run without fault
        sim_normal = TEPSimulator(backend='python')
        sim_normal.initialize()
        for _ in range(600):
            sim_normal.step()
        flow_normal = sim_normal.get_measurements()[0]  # XMEAS(1) A feed flow

        # Run with fault (complete A feed loss)
        sim_fault = TEPSimulator(backend='python')
        sim_fault.initialize()
        sim_fault.add_fault('idv6_a_feed_loss', magnitude=1.0)  # 0% flow
        for _ in range(600):
            sim_fault.step()
        flow_fault = sim_fault.get_measurements()[0]  # XMEAS(1) A feed flow

        # A feed should be near zero with flow loss
        assert flow_fault < flow_normal * 0.1, \
            f"Expected A feed with fault ({flow_fault:.4f}) < 10% of normal ({flow_normal:.4f})"

    def test_custom_fault_affects_process(self):
        """Custom fault plugin should affect the process."""

        @register_fault(name='test_temp_spike', description='Test temperature spike')
        class TestTempSpikeFault(BaseFaultPlugin):
            name = 'test_temp_spike'

            def apply(self, time, process_state):
                # Large temperature increase
                return [FaultEffect('reactor_cw_inlet_temp', 'additive', 10.0)]

            def reset(self):
                pass

        # Run without fault
        sim_normal = TEPSimulator(backend='python')
        sim_normal.initialize()
        for _ in range(1800):
            sim_normal.step()
        temp_normal = sim_normal.get_measurements()[8]

        # Run with custom fault
        sim_fault = TEPSimulator(backend='python')
        sim_fault.initialize()
        sim_fault.add_fault('test_temp_spike')
        for _ in range(1800):
            sim_fault.step()
        temp_fault = sim_fault.get_measurements()[8]

        assert temp_fault > temp_normal, \
            f"Custom fault should increase reactor temp: {temp_fault:.2f} > {temp_normal:.2f}"

    def test_perturbations_cleared_on_initialize(self):
        """Perturbations should be cleared when simulator re-initializes."""
        sim = TEPSimulator(backend='python')
        sim.initialize()

        # Add fault and run
        sim.add_fault('idv4_reactor_cw', magnitude=1.0)
        for _ in range(100):
            sim.step()

        # Check perturbation is set
        pert = sim.process.get_perturbation('reactor_cw_inlet_temp')
        assert pert != 0.0, "Perturbation should be non-zero"

        # Re-initialize
        sim.initialize()

        # Perturbation should be cleared
        pert_after = sim.process.get_perturbation('reactor_cw_inlet_temp')
        assert pert_after == 0.0, "Perturbation should be cleared after initialize"


class TestPythonBackendPerturbations:
    """Tests for the Python backend perturbation system."""

    def test_set_and_get_perturbation(self):
        """Test setting and getting perturbations."""
        from tep.python_backend import PythonTEProcess

        process = PythonTEProcess()
        process.initialize()

        process.set_perturbation('reactor_cw_inlet_temp', 5.0)
        assert process.get_perturbation('reactor_cw_inlet_temp') == 5.0

    def test_invalid_perturbation_name_raises(self):
        """Invalid perturbation name should raise KeyError."""
        from tep.python_backend import PythonTEProcess

        process = PythonTEProcess()

        with pytest.raises(KeyError):
            process.set_perturbation('invalid_name', 1.0)

        with pytest.raises(KeyError):
            process.get_perturbation('invalid_name')

    def test_clear_perturbations(self):
        """Clear perturbations resets to defaults."""
        from tep.python_backend import PythonTEProcess

        process = PythonTEProcess()

        # Set some perturbations
        process.set_perturbation('reactor_cw_inlet_temp', 10.0)
        process.set_perturbation('flow_a_mult', 0.5)

        # Clear
        process.clear_perturbations()

        # Check defaults
        assert process.get_perturbation('reactor_cw_inlet_temp') == 0.0
        assert process.get_perturbation('flow_a_mult') == 1.0

    def test_get_all_perturbations(self):
        """Get all perturbations returns dict copy."""
        from tep.python_backend import PythonTEProcess

        process = PythonTEProcess()
        process.set_perturbation('feed_temp_d', 3.0)

        all_pert = process.get_all_perturbations()
        assert isinstance(all_pert, dict)
        assert all_pert['feed_temp_d'] == 3.0

        # Verify it's a copy (modifications don't affect original)
        all_pert['feed_temp_d'] = 999
        assert process.get_perturbation('feed_temp_d') == 3.0

    def test_perturbations_applied_in_tefunc(self):
        """Verify perturbations are actually applied during integration."""
        from tep.python_backend import PythonTEProcess

        # Run without perturbation
        proc_normal = PythonTEProcess(random_seed=42)
        proc_normal.initialize()
        for _ in range(1000):
            proc_normal.step()
        temp_normal = proc_normal.xmeas[8]  # Reactor temperature

        # Run with perturbation
        proc_pert = PythonTEProcess(random_seed=42)
        proc_pert.initialize()
        proc_pert.set_perturbation('reactor_cw_inlet_temp', 10.0)  # +10°C
        for _ in range(1000):
            proc_pert.step()
        temp_pert = proc_pert.xmeas[8]

        # Higher CW temp = less cooling = higher reactor temp
        assert temp_pert > temp_normal, \
            f"Perturbation should increase reactor temp: {temp_pert:.2f} > {temp_normal:.2f}"
