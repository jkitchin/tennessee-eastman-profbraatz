#!/usr/bin/env python3
"""
Example: Creating and Using Custom Fault Plugins

This example demonstrates how to create custom process faults using
the fault plugin system. You can define faults as individual functions
with their own dynamics, then combine them or schedule them during
simulation.

The fault plugin system allows you to:
1. Use pre-defined IDV-equivalent faults
2. Create composite faults (multiple simultaneous effects)
3. Define time-varying faults (gradual onset, intermittent)
4. Create completely custom fault scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from tep import TEPSimulator, ControlMode
from tep.fault_base import (
    BaseFaultPlugin,
    FaultEffect,
    FaultPluginRegistry,
    register_fault,
)
from tep.fault_plugins import (
    create_idv_fault,
    list_fault_categories,
    get_idv_fault_mapping,
)


# =============================================================================
# Example 1: Using Pre-defined Fault Plugins
# =============================================================================

def example_predefined_faults():
    """
    Use the pre-defined fault plugins that correspond to the
    standard 20 IDV disturbances.
    """
    print("=" * 60)
    print("Example 1: Pre-defined Fault Plugins")
    print("=" * 60)

    # List available faults by category
    print("\nAvailable fault categories:")
    for category, faults in list_fault_categories().items():
        print(f"  {category}: {', '.join(faults)}")

    # Create simulator
    sim = TEPSimulator(control_mode=ControlMode.CLOSED_LOOP)
    sim.initialize()

    # Add IDV(4) equivalent fault (reactor cooling water temp)
    # Activates at 1.0 hours with 50% magnitude
    sim.add_fault('idv4_reactor_cw', activate_at=1.0, magnitude=0.5)

    print("\nSimulating with reactor CW temperature fault...")
    result = sim.simulate(duration_hours=4.0, record_interval=60)

    # Plot reactor temperature response
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    time_hr = result.time

    # Reactor temperature (XMEAS 9)
    ax1.plot(time_hr, result.measurements[:, 8], 'b-', label='Reactor Temp')
    ax1.axvline(x=1.0, color='r', linestyle='--', label='Fault onset')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Reactor Temperature Response to Cooling Water Fault')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cooling water flow (XMV 10)
    ax2.plot(time_hr, result.manipulated_vars[:, 9], 'g-', label='CW Flow')
    ax2.axvline(x=1.0, color='r', linestyle='--', label='Fault onset')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Flow (%)')
    ax2.set_title('Cooling Water Flow Response')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example1_predefined_fault.png', dpi=150)
    plt.close()

    print(f"  Shutdown: {result.shutdown}")
    print("  Plot saved to: example1_predefined_fault.png")


# =============================================================================
# Example 2: Creating a Custom Fault Plugin
# =============================================================================

@register_fault(
    name="oscillating_temperature",
    description="Periodic temperature oscillations in reactor cooling water",
    category="custom",
    default_params={"magnitude": 5.0, "period_hours": 0.5}
)
class OscillatingTemperatureFault(BaseFaultPlugin):
    """
    Custom fault: Creates sinusoidal temperature oscillations.

    This simulates upstream process variability that causes
    periodic temperature changes in the cooling water supply.
    """
    name = "oscillating_temperature"
    description = "Periodic temperature oscillations"
    category = "custom"

    def __init__(
        self,
        magnitude: float = 5.0,
        period_hours: float = 0.5,
        **kwargs
    ):
        """
        Args:
            magnitude: Temperature oscillation amplitude (°C)
            period_hours: Period of oscillation (hours)
        """
        super().__init__(magnitude=magnitude, **kwargs)
        self.period_hours = period_hours

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        if self._activation_time is None:
            return []

        elapsed = time - self._activation_time

        # Sinusoidal oscillation
        phase = 2.0 * np.pi * elapsed / self.period_hours
        delta_temp = self.magnitude * np.sin(phase)

        return [
            FaultEffect('reactor_cw_inlet_temp', 'additive', delta_temp)
        ]

    def reset(self):
        pass


def example_custom_fault():
    """
    Create and use a completely custom fault plugin.
    """
    print("\n" + "=" * 60)
    print("Example 2: Custom Fault Plugin")
    print("=" * 60)

    sim = TEPSimulator(control_mode=ControlMode.CLOSED_LOOP)
    sim.initialize()

    # Add our custom oscillating temperature fault
    sim.add_fault('oscillating_temperature',
                  activate_at=0.5,
                  magnitude=3.0,
                  period_hours=0.25)

    print("\nSimulating with oscillating temperature fault...")
    result = sim.simulate(duration_hours=3.0, record_interval=30)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    time_hr = result.time

    # Reactor temperature
    ax1.plot(time_hr, result.measurements[:, 8], 'b-')
    ax1.axvline(x=0.5, color='r', linestyle='--', label='Fault onset')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Reactor Temperature with Oscillating CW Disturbance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reactor pressure
    ax2.plot(time_hr, result.measurements[:, 6], 'g-')
    ax2.axvline(x=0.5, color='r', linestyle='--', label='Fault onset')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Pressure (kPa)')
    ax2.set_title('Reactor Pressure Response')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example2_custom_fault.png', dpi=150)
    plt.close()

    print(f"  Shutdown: {result.shutdown}")
    print("  Plot saved to: example2_custom_fault.png")


# =============================================================================
# Example 3: Inline Fault Definition (No Registration)
# =============================================================================

class RampingFault(BaseFaultPlugin):
    """
    A fault that linearly increases in severity over time.

    This demonstrates creating a fault class without using the
    @register_fault decorator. Useful for one-off scenarios.
    """
    name = "ramping_fault"
    description = "Linearly increasing fault severity"

    def __init__(
        self,
        magnitude: float = 1.0,
        ramp_hours: float = 1.0,
        target_variable: str = 'reactor_cw_inlet_temp',
        **kwargs
    ):
        super().__init__(magnitude=magnitude, **kwargs)
        self.ramp_hours = ramp_hours
        self.target_variable = target_variable

    def apply(self, time: float, process_state: Dict[str, Any]) -> List[FaultEffect]:
        if self._activation_time is None:
            return []

        elapsed = time - self._activation_time
        # Linear ramp from 0 to magnitude
        severity = min(1.0, elapsed / self.ramp_hours) * self.magnitude

        return [
            FaultEffect(self.target_variable, 'additive', severity)
        ]

    def reset(self):
        pass


def example_inline_fault():
    """
    Create a fault without registering it (useful for one-off scenarios).
    """
    print("\n" + "=" * 60)
    print("Example 3: Inline Fault Definition")
    print("=" * 60)

    sim = TEPSimulator(control_mode=ControlMode.CLOSED_LOOP)
    sim.initialize()

    # Create fault instance directly (not registered)
    ramping_fault = RampingFault(
        magnitude=8.0,  # Max +8°C
        ramp_hours=2.0,  # Reaches max after 2 hours
        target_variable='reactor_cw_inlet_temp'
    )

    # Add the instance directly
    sim.add_fault(ramping_fault, activate_at=1.0)

    print("\nSimulating with ramping temperature fault...")
    result = sim.simulate(duration_hours=5.0, record_interval=60)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    time_hr = result.time

    ax.plot(time_hr, result.measurements[:, 8], 'b-', linewidth=2)
    ax.axvline(x=1.0, color='r', linestyle='--', label='Fault starts')
    ax.axvline(x=3.0, color='orange', linestyle='--', label='Fault at max')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Reactor Temperature (°C)')
    ax.set_title('Reactor Temperature with Ramping CW Temperature Fault')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example3_ramping_fault.png', dpi=150)
    plt.close()

    print(f"  Shutdown: {result.shutdown}")
    print("  Plot saved to: example3_ramping_fault.png")


# =============================================================================
# Example 4: Multiple Simultaneous Faults
# =============================================================================

def example_multiple_faults():
    """
    Combine multiple faults in a single simulation.
    """
    print("\n" + "=" * 60)
    print("Example 4: Multiple Simultaneous Faults")
    print("=" * 60)

    sim = TEPSimulator(control_mode=ControlMode.CLOSED_LOOP)
    sim.initialize()

    # Add multiple faults with different activation times
    sim.add_fault('idv3_d_temp', activate_at=0.5, magnitude=0.5)
    sim.add_fault('idv4_reactor_cw', activate_at=1.0, magnitude=0.5)
    sim.add_fault('idv5_condenser_cw', activate_at=1.5, magnitude=0.5)

    print("\nSimulating with three sequential faults...")
    print("  - IDV3 (D feed temp) at 0.5 hours")
    print("  - IDV4 (reactor CW temp) at 1.0 hours")
    print("  - IDV5 (condenser CW temp) at 1.5 hours")

    result = sim.simulate(duration_hours=4.0, record_interval=60)

    # List active faults at different times
    print(f"\n  Active faults at end: {sim.get_active_faults()}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    time_hr = result.time

    # Reactor temperature
    axes[0].plot(time_hr, result.measurements[:, 8], 'b-')
    for t in [0.5, 1.0, 1.5]:
        axes[0].axvline(x=t, color='r', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Reactor Temp (°C)')
    axes[0].set_title('Multiple Fault Scenario')
    axes[0].grid(True, alpha=0.3)

    # Separator temperature
    axes[1].plot(time_hr, result.measurements[:, 10], 'g-')
    for t in [0.5, 1.0, 1.5]:
        axes[1].axvline(x=t, color='r', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Separator Temp (°C)')
    axes[1].grid(True, alpha=0.3)

    # Stripper temperature
    axes[2].plot(time_hr, result.measurements[:, 17], 'm-')
    for t in [0.5, 1.0, 1.5]:
        axes[2].axvline(x=t, color='r', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time (hours)')
    axes[2].set_ylabel('Stripper Temp (°C)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example4_multiple_faults.png', dpi=150)
    plt.close()

    print(f"  Shutdown: {result.shutdown}")
    print("  Plot saved to: example4_multiple_faults.png")


# =============================================================================
# Example 5: Listing Available Faults
# =============================================================================

def example_list_faults():
    """
    Display all available fault plugins.
    """
    print("\n" + "=" * 60)
    print("Example 5: Available Fault Plugins")
    print("=" * 60)

    print("\nIDV to Fault Plugin Mapping:")
    print("-" * 40)
    for idv, name in get_idv_fault_mapping().items():
        info = FaultPluginRegistry.get_info(name)
        print(f"  IDV({idv:2d}): {name}")
        print(f"           {info['description']}")

    print("\nAll Registered Faults by Category:")
    print("-" * 40)
    for category, faults in list_fault_categories().items():
        print(f"\n  [{category.upper()}]")
        for fault_name in faults:
            info = FaultPluginRegistry.get_info(fault_name)
            print(f"    - {fault_name}: {info['description']}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Custom Fault Plugin Examples")
    print("=" * 60)

    # Run all examples
    example_predefined_faults()
    example_custom_fault()
    example_inline_fault()
    example_multiple_faults()
    example_list_faults()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
