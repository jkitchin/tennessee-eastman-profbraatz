#!/usr/bin/env python3
"""
Custom Controller Example

This example demonstrates how to implement and use custom
control strategies with the TEP simulator.
"""

import numpy as np
from tep import TEPSimulator
from tep.simulator import ControlMode
from tep.controllers import PIController, ManualController


def main():
    print("TEP Custom Controller Example")
    print("=" * 50)

    # Example 1: Simple P-only controller for reactor temperature
    print("\nExample 1: Simple P-only reactor temperature control")
    print("-" * 50)

    class SimpleReactorTempController:
        """Simple P-only controller for reactor temperature."""

        def __init__(self, setpoint=120.0, gain=0.5):
            self.setpoint = setpoint
            self.gain = gain

        def calculate(self, xmeas, xmv, step):
            """Calculate new MV values."""
            new_xmv = xmv.copy()

            # Get reactor temperature (index 8)
            reactor_temp = xmeas[8]

            # Calculate error
            error = self.setpoint - reactor_temp

            # Adjust reactor cooling water (MV 10, index 9)
            # Increase cooling if temp too high
            adjustment = -self.gain * error
            new_xmv[9] = np.clip(new_xmv[9] + adjustment, 0, 100)

            return new_xmv

    sim = TEPSimulator(random_seed=12345)
    sim.initialize()

    controller = SimpleReactorTempController(setpoint=120.0, gain=0.3)

    result = sim.simulate_with_controller(
        duration_hours=2.0,
        controller=controller,
        disturbances={4: (0.5, 1)},  # Cooling water temp disturbance
        record_interval=60
    )

    print(f"Setpoint: 120.0 °C")
    print(f"Final reactor temp: {result.measurements[-1, 8]:.2f} °C")
    print(f"Temperature std: {np.std(result.measurements[:, 8]):.2f} °C")

    # Example 2: Using a function as controller
    print("\n" + "=" * 50)
    print("Example 2: Function-based controller")
    print("-" * 50)

    def my_control_function(xmeas, xmv, step):
        """
        Simple function-based controller.
        Maintains reactor level by adjusting A+C feed.
        """
        new_xmv = xmv.copy()

        # Reactor level control (MV 4 -> XMEAS 8)
        level_error = 75.0 - xmeas[7]  # Setpoint = 75%
        new_xmv[3] += 0.1 * level_error
        new_xmv[3] = np.clip(new_xmv[3], 0, 100)

        return new_xmv

    sim.initialize()
    result = sim.simulate_with_controller(
        duration_hours=1.0,
        controller=my_control_function,
        record_interval=60
    )

    print(f"Level setpoint: 75%")
    print(f"Final reactor level: {result.measurements[-1, 7]:.1f}%")

    # Example 3: Cascade control using PIController
    print("\n" + "=" * 50)
    print("Example 3: Cascade control structure")
    print("-" * 50)

    class CascadeController:
        """
        Cascade controller: Reactor temp -> Cooling water flow.
        Outer loop adjusts CW outlet temp setpoint,
        Inner loop controls CW valve.
        """

        def __init__(self):
            # Outer loop: Reactor temp -> CW outlet setpoint
            self.outer = PIController(
                setpoint=120.0,  # Reactor temp setpoint
                gain=2.0,
                taui=0.5,  # 30 minutes reset
                output_min=80,
                output_max=110,
                scale=1.0
            )

            # Inner loop: CW outlet temp -> CW valve
            self.inner = PIController(
                setpoint=95.0,  # Initial CW temp setpoint
                gain=-0.5,  # Negative: increase valve to decrease temp
                taui=0.05,
                scale=1.0
            )

            self.dt = 1.0 / 3600.0  # 1 second

        def calculate(self, xmeas, xmv, step):
            new_xmv = xmv.copy()

            # Every 10 seconds, run outer loop
            if step % 10 == 0:
                reactor_temp = xmeas[8]
                cw_setpoint = self.outer.calculate(reactor_temp, self.inner.setpoint, 10*self.dt)
                self.inner.setpoint = cw_setpoint

            # Every step, run inner loop
            cw_outlet_temp = xmeas[20]
            new_xmv[9] = self.inner.calculate(cw_outlet_temp, new_xmv[9], self.dt)

            return new_xmv

    sim.initialize()
    cascade = CascadeController()

    result = sim.simulate_with_controller(
        duration_hours=2.0,
        controller=cascade,
        disturbances={4: (0.5, 1)},
        record_interval=60
    )

    print(f"Reactor temp setpoint: 120.0 °C")
    print(f"Mean reactor temp: {np.mean(result.measurements[:, 8]):.2f} °C")
    print(f"Temp std: {np.std(result.measurements[:, 8]):.2f} °C")

    # Example 4: Manual control with step changes
    print("\n" + "=" * 50)
    print("Example 4: Manual control mode")
    print("-" * 50)

    sim = TEPSimulator(control_mode=ControlMode.MANUAL)
    sim.initialize()

    print("Running manual control simulation...")
    print("Step changes:")

    # Make step changes during simulation
    for step in range(3600):  # 1 hour
        # Step change at 10 minutes
        if step == 600:
            sim.set_mv(10, 55.0)  # Increase reactor CW
            print(f"  t=10min: MV10 -> 55%")

        # Another step at 30 minutes
        if step == 1800:
            sim.set_mv(10, 45.0)  # Decrease reactor CW
            print(f"  t=30min: MV10 -> 45%")

        if not sim.step():
            print("Process shutdown!")
            break

    print(f"\nFinal reactor temp: {sim.get_measurements()[8]:.1f} °C")
    print(f"(Manual control without feedback leads to drift)")

    print("\nAll examples complete!")


if __name__ == "__main__":
    main()
