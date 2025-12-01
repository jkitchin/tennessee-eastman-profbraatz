"""
Built-in fault plugins for the Tennessee Eastman Process simulator.

This module provides fault plugins corresponding to the standard 20 IDV
disturbances, plus example custom faults demonstrating the plugin system.

Standard TEP Faults (IDV 1-20):
    - IDV 1-7: Step disturbances (deterministic)
    - IDV 8-12: Random variation disturbances
    - IDV 13: Slow drift (reaction kinetics)
    - IDV 14-15: Valve sticking faults
    - IDV 16-20: Unknown/undefined in original TEP

Custom Fault Examples:
    - Composite faults (multiple simultaneous effects)
    - Time-varying faults (gradual onset)
    - Intermittent faults (periodic)
"""

from typing import Dict, Any, List, Optional
import numpy as np

from .fault_base import (
    BaseFaultPlugin,
    FaultEffect,
    FaultPluginRegistry,
    register_fault
)


# =============================================================================
# Base Classes for Common Fault Patterns
# =============================================================================

class StepFault(BaseFaultPlugin):
    """
    Base class for step disturbances (IDV 1-7 type).

    Step faults apply a fixed perturbation when activated.
    """
    category = "step"

    def reset(self):
        """Reset fault state."""
        pass  # Step faults have no internal state


class RandomVariationFault(BaseFaultPlugin):
    """
    Base class for random variation disturbances (IDV 8-12 type).

    Random faults use stochastic processes to generate time-varying
    perturbations with realistic dynamics.
    """
    category = "random"

    def __init__(self, magnitude: float = 1.0, correlation_time: float = 0.1, **kwargs):
        """
        Args:
            magnitude: Scaling factor for variation amplitude
            correlation_time: Time constant for variation (hours)
        """
        super().__init__(magnitude=magnitude, **kwargs)
        self.correlation_time = correlation_time
        self._current_value = 0.0
        self._last_time = 0.0

    def reset(self):
        """Reset random walk state."""
        self._current_value = 0.0
        self._last_time = 0.0


class ValveStickingFault(BaseFaultPlugin):
    """
    Base class for valve sticking faults (IDV 14-15 type).

    Valve sticking causes the valve to temporarily lock at its
    current position, then suddenly release.
    """
    category = "valve"

    def __init__(self, magnitude: float = 1.0, stick_probability: float = 0.1, **kwargs):
        """
        Args:
            magnitude: Severity of sticking (affects duration)
            stick_probability: Probability of sticking per step
        """
        super().__init__(magnitude=magnitude, **kwargs)
        self.stick_probability = stick_probability
        self._stuck = False
        self._stuck_position = None
        self._stuck_time = 0.0

    def reset(self):
        """Reset valve state."""
        self._stuck = False
        self._stuck_position = None
        self._stuck_time = 0.0


# =============================================================================
# IDV 1: A/C Feed Ratio, B Composition Constant (Step)
# =============================================================================

@register_fault(
    name="idv1_feed_ratio",
    description="A/C Feed Ratio, B Composition Constant (Step) - IDV(1)",
    category="step",
    default_params={"magnitude": 1.0}
)
class IDV1FeedRatioFault(StepFault):
    """
    IDV(1): Step change in A/C feed ratio.

    Reduces A composition in feed by 0.03 (from ~0.485 to ~0.455).
    This affects the reactor chemistry and product distribution.
    """
    name = "idv1_feed_ratio"
    description = "A/C Feed Ratio, B Composition Constant (Step)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        return [
            FaultEffect('feed_comp_a', 'additive', -0.03 * self.magnitude)
        ]


# =============================================================================
# IDV 2: B Composition, A/C Ratio Constant (Step)
# =============================================================================

@register_fault(
    name="idv2_b_composition",
    description="B Composition, A/C Ratio Constant (Step) - IDV(2)",
    category="step",
    default_params={"magnitude": 1.0}
)
class IDV2BCompositionFault(StepFault):
    """
    IDV(2): Step change in B composition.

    Changes both A (slight decrease) and B (increase) compositions.
    This affects the impurity levels in the product.
    """
    name = "idv2_b_composition"
    description = "B Composition, A/C Ratio Constant (Step)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        return [
            FaultEffect('feed_comp_a', 'additive', -2.43719e-3 * self.magnitude),
            FaultEffect('feed_comp_b', 'additive', 0.005 * self.magnitude)
        ]


# =============================================================================
# IDV 3: D Feed Temperature (Step)
# =============================================================================

@register_fault(
    name="idv3_d_temp",
    description="D Feed Temperature (Step) - IDV(3)",
    category="step",
    default_params={"magnitude": 1.0}
)
class IDV3DFeedTempFault(StepFault):
    """
    IDV(3): Step change in D feed temperature.

    Increases D feed temperature by 5°C. This affects the reactor
    heat balance and reaction rates.
    """
    name = "idv3_d_temp"
    description = "D Feed Temperature (Step)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        return [
            FaultEffect('feed_temp_d', 'additive', 5.0 * self.magnitude)
        ]


# =============================================================================
# IDV 4: Reactor Cooling Water Inlet Temperature (Step)
# =============================================================================

@register_fault(
    name="idv4_reactor_cw",
    description="Reactor Cooling Water Inlet Temperature (Step) - IDV(4)",
    category="step",
    default_params={"magnitude": 1.0}
)
class IDV4ReactorCWTempFault(StepFault):
    """
    IDV(4): Step change in reactor cooling water inlet temperature.

    Increases cooling water temperature by 5°C. This reduces cooling
    capacity and can cause reactor temperature rise.
    """
    name = "idv4_reactor_cw"
    description = "Reactor Cooling Water Inlet Temperature (Step)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        return [
            FaultEffect('reactor_cw_inlet_temp', 'additive', 5.0 * self.magnitude)
        ]


# =============================================================================
# IDV 5: Condenser Cooling Water Inlet Temperature (Step)
# =============================================================================

@register_fault(
    name="idv5_condenser_cw",
    description="Condenser Cooling Water Inlet Temperature (Step) - IDV(5)",
    category="step",
    default_params={"magnitude": 1.0}
)
class IDV5CondenserCWTempFault(StepFault):
    """
    IDV(5): Step change in condenser cooling water inlet temperature.

    Increases cooling water temperature by 5°C. This affects
    condensation and can increase compressor load.
    """
    name = "idv5_condenser_cw"
    description = "Condenser Cooling Water Inlet Temperature (Step)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        return [
            FaultEffect('condenser_cw_inlet_temp', 'additive', 5.0 * self.magnitude)
        ]


# =============================================================================
# IDV 6: A Feed Loss (Step)
# =============================================================================

@register_fault(
    name="idv6_a_feed_loss",
    description="A Feed Loss (Step) - IDV(6)",
    category="step",
    default_params={"magnitude": 1.0}
)
class IDV6AFeedLossFault(StepFault):
    """
    IDV(6): Loss of A feed.

    Completely blocks the A feed (flow multiplier = 0). This is a
    severe fault that significantly impacts reactor operation.
    """
    name = "idv6_a_feed_loss"
    description = "A Feed Loss (Step)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        # flow_a multiplier: 1.0 = normal, 0.0 = complete loss
        return [
            FaultEffect('flow_a', 'multiplicative', 1.0 - self.magnitude)
        ]


# =============================================================================
# IDV 7: C Header Pressure Loss (Step)
# =============================================================================

@register_fault(
    name="idv7_c_pressure_loss",
    description="C Header Pressure Loss (Step) - IDV(7)",
    category="step",
    default_params={"magnitude": 1.0}
)
class IDV7CPressureLossFault(StepFault):
    """
    IDV(7): Loss of C header pressure.

    Reduces C feed flow by 20%. This affects the feed balance
    and reactor composition.
    """
    name = "idv7_c_pressure_loss"
    description = "C Header Pressure Loss (Step)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        # 20% flow reduction at full magnitude
        return [
            FaultEffect('flow_c', 'multiplicative', 1.0 - 0.2 * self.magnitude)
        ]


# =============================================================================
# IDV 8: A, B, C Feed Composition (Random Variation)
# =============================================================================

@register_fault(
    name="idv8_feed_comp_random",
    description="A, B, C Feed Composition (Random Variation) - IDV(8)",
    category="random",
    default_params={"magnitude": 1.0, "correlation_time": 0.1}
)
class IDV8FeedCompRandomFault(RandomVariationFault):
    """
    IDV(8): Random variation in feed compositions.

    Creates correlated random variations in A and B feed compositions
    simulating upstream process variability.
    """
    name = "idv8_feed_comp_random"
    description = "A, B, C Feed Composition (Random Variation)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        rng = process_state.get('random', np.random.default_rng())

        # Update random walk if enough time has passed
        dt = time - self._last_time
        if dt >= self.correlation_time or self._last_time == 0:
            # Simple exponential smoothing for correlated noise
            noise = rng.normal(0, 0.01 * self.magnitude)
            alpha = min(1.0, dt / self.correlation_time) if self.correlation_time > 0 else 1.0
            self._current_value = (1 - alpha) * self._current_value + alpha * noise
            self._last_time = time

        return [
            FaultEffect('feed_comp_a', 'additive', self._current_value),
            FaultEffect('feed_comp_b', 'additive', self._current_value * 0.1)
        ]


# =============================================================================
# IDV 9: D Feed Temperature (Random Variation)
# =============================================================================

@register_fault(
    name="idv9_d_temp_random",
    description="D Feed Temperature (Random Variation) - IDV(9)",
    category="random",
    default_params={"magnitude": 1.0, "correlation_time": 0.1}
)
class IDV9DTempRandomFault(RandomVariationFault):
    """
    IDV(9): Random variation in D feed temperature.

    Creates random temperature fluctuations in the D feed stream.
    """
    name = "idv9_d_temp_random"
    description = "D Feed Temperature (Random Variation)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        rng = process_state.get('random', np.random.default_rng())

        dt = time - self._last_time
        if dt >= self.correlation_time or self._last_time == 0:
            noise = rng.normal(0, 2.0 * self.magnitude)  # ~2°C std dev
            alpha = min(1.0, dt / self.correlation_time) if self.correlation_time > 0 else 1.0
            self._current_value = (1 - alpha) * self._current_value + alpha * noise
            self._last_time = time

        return [
            FaultEffect('feed_temp_d', 'additive', self._current_value)
        ]


# =============================================================================
# IDV 10: C Feed Temperature (Random Variation)
# =============================================================================

@register_fault(
    name="idv10_c_temp_random",
    description="C Feed Temperature (Random Variation) - IDV(10)",
    category="random",
    default_params={"magnitude": 1.0, "correlation_time": 0.1}
)
class IDV10CTempRandomFault(RandomVariationFault):
    """
    IDV(10): Random variation in C feed temperature.

    Creates random temperature fluctuations in the C feed stream.
    """
    name = "idv10_c_temp_random"
    description = "C Feed Temperature (Random Variation)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        rng = process_state.get('random', np.random.default_rng())

        dt = time - self._last_time
        if dt >= self.correlation_time or self._last_time == 0:
            noise = rng.normal(0, 2.0 * self.magnitude)
            alpha = min(1.0, dt / self.correlation_time) if self.correlation_time > 0 else 1.0
            self._current_value = (1 - alpha) * self._current_value + alpha * noise
            self._last_time = time

        return [
            FaultEffect('feed_temp_c', 'additive', self._current_value)
        ]


# =============================================================================
# IDV 11: Reactor Cooling Water Inlet Temp (Random Variation)
# =============================================================================

@register_fault(
    name="idv11_reactor_cw_random",
    description="Reactor Cooling Water Inlet Temp (Random Variation) - IDV(11)",
    category="random",
    default_params={"magnitude": 1.0, "correlation_time": 0.1}
)
class IDV11ReactorCWRandomFault(RandomVariationFault):
    """
    IDV(11): Random variation in reactor cooling water temperature.

    Creates random temperature fluctuations in the reactor cooling
    water supply, affecting heat removal.
    """
    name = "idv11_reactor_cw_random"
    description = "Reactor Cooling Water Inlet Temp (Random Variation)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        rng = process_state.get('random', np.random.default_rng())

        dt = time - self._last_time
        if dt >= self.correlation_time or self._last_time == 0:
            noise = rng.normal(0, 1.5 * self.magnitude)
            alpha = min(1.0, dt / self.correlation_time) if self.correlation_time > 0 else 1.0
            self._current_value = (1 - alpha) * self._current_value + alpha * noise
            self._last_time = time

        return [
            FaultEffect('reactor_cw_inlet_temp', 'additive', self._current_value)
        ]


# =============================================================================
# IDV 12: Condenser Cooling Water Inlet Temp (Random Variation)
# =============================================================================

@register_fault(
    name="idv12_condenser_cw_random",
    description="Condenser Cooling Water Inlet Temp (Random Variation) - IDV(12)",
    category="random",
    default_params={"magnitude": 1.0, "correlation_time": 0.1}
)
class IDV12CondenserCWRandomFault(RandomVariationFault):
    """
    IDV(12): Random variation in condenser cooling water temperature.

    Creates random temperature fluctuations in the condenser cooling
    water supply.
    """
    name = "idv12_condenser_cw_random"
    description = "Condenser Cooling Water Inlet Temp (Random Variation)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        rng = process_state.get('random', np.random.default_rng())

        dt = time - self._last_time
        if dt >= self.correlation_time or self._last_time == 0:
            noise = rng.normal(0, 1.5 * self.magnitude)
            alpha = min(1.0, dt / self.correlation_time) if self.correlation_time > 0 else 1.0
            self._current_value = (1 - alpha) * self._current_value + alpha * noise
            self._last_time = time

        return [
            FaultEffect('condenser_cw_inlet_temp', 'additive', self._current_value)
        ]


# =============================================================================
# IDV 13: Reaction Kinetics (Slow Drift)
# =============================================================================

@register_fault(
    name="idv13_kinetics_drift",
    description="Reaction Kinetics (Slow Drift) - IDV(13)",
    category="drift",
    default_params={"magnitude": 1.0, "drift_rate": 0.01}
)
class IDV13KineticsDriftFault(BaseFaultPlugin):
    """
    IDV(13): Slow drift in reaction kinetics.

    Gradually changes the reaction rate constants over time,
    simulating catalyst deactivation or fouling.
    """
    name = "idv13_kinetics_drift"
    description = "Reaction Kinetics (Slow Drift)"
    category = "drift"

    def __init__(self, magnitude: float = 1.0, drift_rate: float = 0.01, **kwargs):
        """
        Args:
            magnitude: Maximum drift factor
            drift_rate: Rate of drift per hour
        """
        super().__init__(magnitude=magnitude, **kwargs)
        self.drift_rate = drift_rate
        self._accumulated_drift = 0.0

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        # Calculate drift based on time since activation
        if self._activation_time is not None:
            elapsed = time - self._activation_time
            # Saturating drift: approaches magnitude asymptotically
            self._accumulated_drift = self.magnitude * (1 - np.exp(-self.drift_rate * elapsed))

        return [
            FaultEffect('reaction_1_factor', 'multiplicative', 1.0 - self._accumulated_drift),
            FaultEffect('reaction_2_factor', 'multiplicative', 1.0 - self._accumulated_drift)
        ]

    def reset(self):
        self._accumulated_drift = 0.0


# =============================================================================
# IDV 14: Reactor Cooling Water Valve (Sticking)
# =============================================================================

@register_fault(
    name="idv14_reactor_valve_stick",
    description="Reactor Cooling Water Valve (Sticking) - IDV(14)",
    category="valve",
    default_params={"magnitude": 1.0, "stick_probability": 0.1}
)
class IDV14ReactorValveStickFault(ValveStickingFault):
    """
    IDV(14): Reactor cooling water valve sticking.

    The valve periodically sticks at its current position, then
    releases. This affects reactor temperature control.
    """
    name = "idv14_reactor_valve_stick"
    description = "Reactor Cooling Water Valve (Sticking)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        rng = process_state.get('random', np.random.default_rng())
        xmv = process_state.get('xmv', np.zeros(12))

        effects = []

        # Check if valve should stick/unstick
        if not self._stuck:
            if rng.random() < self.stick_probability * self.magnitude:
                self._stuck = True
                self._stuck_position = xmv[9]  # XMV(10) - reactor CW flow
                self._stuck_time = time
        else:
            # Probability of unsticking increases with time
            stuck_duration = time - self._stuck_time
            unstick_prob = min(1.0, stuck_duration * 2.0)  # ~50% chance after 0.5 hours
            if rng.random() < unstick_prob:
                self._stuck = False
                self._stuck_position = None

        if self._stuck and self._stuck_position is not None:
            effects.append(
                FaultEffect('valve_reactor_cw', 'replace', self._stuck_position)
            )

        return effects


# =============================================================================
# IDV 15: Condenser Cooling Water Valve (Sticking)
# =============================================================================

@register_fault(
    name="idv15_condenser_valve_stick",
    description="Condenser Cooling Water Valve (Sticking) - IDV(15)",
    category="valve",
    default_params={"magnitude": 1.0, "stick_probability": 0.1}
)
class IDV15CondenserValveStickFault(ValveStickingFault):
    """
    IDV(15): Condenser cooling water valve sticking.

    The valve periodically sticks at its current position, then
    releases. This affects condensation and separator pressure.
    """
    name = "idv15_condenser_valve_stick"
    description = "Condenser Cooling Water Valve (Sticking)"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        rng = process_state.get('random', np.random.default_rng())
        xmv = process_state.get('xmv', np.zeros(12))

        effects = []

        if not self._stuck:
            if rng.random() < self.stick_probability * self.magnitude:
                self._stuck = True
                self._stuck_position = xmv[10]  # XMV(11) - condenser CW flow
                self._stuck_time = time
        else:
            stuck_duration = time - self._stuck_time
            unstick_prob = min(1.0, stuck_duration * 2.0)
            if rng.random() < unstick_prob:
                self._stuck = False
                self._stuck_position = None

        if self._stuck and self._stuck_position is not None:
            effects.append(
                FaultEffect('valve_condenser_cw', 'replace', self._stuck_position)
            )

        return effects


# =============================================================================
# CUSTOM FAULT EXAMPLES
# =============================================================================

@register_fault(
    name="cooling_system_failure",
    description="Combined cooling system failure (both reactor and condenser)",
    category="custom",
    default_params={"magnitude": 1.0}
)
class CoolingSystemFailureFault(StepFault):
    """
    Custom fault: Combined cooling system failure.

    Simulates a scenario where both reactor and condenser cooling
    water temperatures increase simultaneously (e.g., cooling tower problem).
    """
    name = "cooling_system_failure"
    description = "Combined cooling system failure"
    category = "custom"

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        return [
            FaultEffect('reactor_cw_inlet_temp', 'additive', 5.0 * self.magnitude),
            FaultEffect('condenser_cw_inlet_temp', 'additive', 5.0 * self.magnitude)
        ]


@register_fault(
    name="gradual_feed_loss",
    description="Gradual loss of A feed (develops over time)",
    category="custom",
    default_params={"magnitude": 1.0, "onset_hours": 1.0}
)
class GradualFeedLossFault(BaseFaultPlugin):
    """
    Custom fault: Gradual feed loss.

    Unlike the step change in IDV(6), this fault develops gradually
    over time, simulating a developing blockage or pump degradation.
    """
    name = "gradual_feed_loss"
    description = "Gradual loss of A feed"
    category = "custom"

    def __init__(self, magnitude: float = 1.0, onset_hours: float = 1.0, **kwargs):
        """
        Args:
            magnitude: Maximum flow reduction (1.0 = complete loss)
            onset_hours: Time to reach full severity
        """
        super().__init__(magnitude=magnitude, **kwargs)
        self.onset_hours = onset_hours

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        if self._activation_time is None:
            return []

        elapsed = time - self._activation_time
        # Sigmoid-like progression
        severity = self.magnitude * min(1.0, elapsed / self.onset_hours)

        return [
            FaultEffect('flow_a', 'multiplicative', 1.0 - severity)
        ]

    def reset(self):
        pass


@register_fault(
    name="intermittent_temp_spike",
    description="Periodic temperature spikes in reactor cooling water",
    category="custom",
    default_params={"magnitude": 1.0, "period_hours": 0.25, "duty_cycle": 0.2}
)
class IntermittentTempSpikeFault(BaseFaultPlugin):
    """
    Custom fault: Intermittent temperature spikes.

    Creates periodic temperature increases, simulating an
    intermittent upstream disturbance.
    """
    name = "intermittent_temp_spike"
    description = "Periodic temperature spikes"
    category = "custom"

    def __init__(
        self,
        magnitude: float = 1.0,
        period_hours: float = 0.25,
        duty_cycle: float = 0.2,
        **kwargs
    ):
        """
        Args:
            magnitude: Temperature spike magnitude (°C)
            period_hours: Period of the intermittent pattern
            duty_cycle: Fraction of period when fault is active (0-1)
        """
        super().__init__(magnitude=magnitude, **kwargs)
        self.period_hours = period_hours
        self.duty_cycle = duty_cycle

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        if self._activation_time is None:
            return []

        # Determine if we're in the "active" portion of the cycle
        elapsed = time - self._activation_time
        cycle_position = (elapsed % self.period_hours) / self.period_hours

        if cycle_position < self.duty_cycle:
            return [
                FaultEffect('reactor_cw_inlet_temp', 'additive', 10.0 * self.magnitude)
            ]

        return []

    def reset(self):
        pass


@register_fault(
    name="sensor_bias",
    description="Sensor measurement bias (does not affect actual process)",
    category="custom",
    default_params={"magnitude": 1.0, "sensor_index": 0, "bias_value": 5.0}
)
class SensorBiasFault(BaseFaultPlugin):
    """
    Custom fault: Sensor bias.

    Adds a bias to a sensor reading without affecting the actual
    process variable. Useful for testing fault detection algorithms.

    Note: This fault type requires special handling in the simulator
    to apply the bias to measurements rather than process variables.
    """
    name = "sensor_bias"
    description = "Sensor measurement bias"
    category = "custom"

    def __init__(
        self,
        magnitude: float = 1.0,
        sensor_index: int = 0,
        bias_value: float = 5.0,
        **kwargs
    ):
        """
        Args:
            magnitude: Scaling factor for bias
            sensor_index: XMEAS index (0-40)
            bias_value: Base bias value (scaled by magnitude)
        """
        super().__init__(magnitude=magnitude, **kwargs)
        self.sensor_index = sensor_index
        self.bias_value = bias_value

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        return [
            FaultEffect(
                f'sensor_{self.sensor_index}',
                'additive',
                self.bias_value * self.magnitude
            )
        ]

    def reset(self):
        pass


# =============================================================================
# Utility Functions
# =============================================================================

def get_idv_fault_mapping() -> Dict[int, str]:
    """
    Get mapping from IDV number to fault plugin name.

    Returns:
        Dict mapping IDV index (1-20) to fault plugin name
    """
    return {
        1: "idv1_feed_ratio",
        2: "idv2_b_composition",
        3: "idv3_d_temp",
        4: "idv4_reactor_cw",
        5: "idv5_condenser_cw",
        6: "idv6_a_feed_loss",
        7: "idv7_c_pressure_loss",
        8: "idv8_feed_comp_random",
        9: "idv9_d_temp_random",
        10: "idv10_c_temp_random",
        11: "idv11_reactor_cw_random",
        12: "idv12_condenser_cw_random",
        13: "idv13_kinetics_drift",
        14: "idv14_reactor_valve_stick",
        15: "idv15_condenser_valve_stick",
    }


def create_idv_fault(idv_number: int, **kwargs) -> BaseFaultPlugin:
    """
    Create a fault plugin for a specific IDV number.

    Args:
        idv_number: IDV index (1-15)
        **kwargs: Additional parameters for the fault

    Returns:
        Configured fault plugin instance

    Raises:
        ValueError: If IDV number is not supported
    """
    mapping = get_idv_fault_mapping()
    if idv_number not in mapping:
        raise ValueError(
            f"IDV({idv_number}) is not supported. "
            f"Available: {list(mapping.keys())}"
        )

    fault_name = mapping[idv_number]
    return FaultPluginRegistry.create(fault_name, **kwargs)


def list_fault_categories() -> Dict[str, List[str]]:
    """
    List all faults grouped by category.

    Returns:
        Dict mapping category name to list of fault names
    """
    categories = {}
    for name in FaultPluginRegistry.list_available():
        info = FaultPluginRegistry.get_info(name)
        cat = info['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(name)
    return categories
