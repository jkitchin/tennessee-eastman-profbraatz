#!/usr/bin/env python3
"""
Disturbance Simulation Example

This example demonstrates how to apply process disturbances
and observe the controller response.
"""

import numpy as np
from tep import TEPSimulator, DISTURBANCE_NAMES
from tep.simulator import ControlMode


def main():
    print("TEP Disturbance Simulation Example")
    print("=" * 50)

    # Create simulator
    sim = TEPSimulator(
        random_seed=12345,
        control_mode=ControlMode.CLOSED_LOOP
    )
    sim.initialize()

    # Show available disturbances
    print("\nAvailable disturbances:")
    for i, name in enumerate(DISTURBANCE_NAMES[:15], 1):
        print(f"  IDV({i:2d}): {name}")

    # Example 1: Step disturbance in A/C feed ratio
    print("\n" + "=" * 50)
    print("Example 1: IDV(1) - A/C Feed Ratio Step Change")
    print("=" * 50)

    result = sim.simulate(
        duration_hours=4.0,
        disturbances={1: (1.0, 1)},  # Apply IDV(1) at t=1 hour
        record_interval=60
    )

    print(f"\nDisturbance applied at t=1.0 hour")
    print(f"Simulation completed: {result.time[-1]:.1f} hours")

    # Show impact on key variables
    idx_before = np.argmin(np.abs(result.time - 0.9))
    idx_after = np.argmin(np.abs(result.time - 3.5))

    print(f"\nKey measurements (before -> after disturbance):")
    print(f"  Reactor temp:  {result.measurements[idx_before, 8]:.1f} -> {result.measurements[idx_after, 8]:.1f} °C")
    print(f"  Reactor level: {result.measurements[idx_before, 7]:.1f} -> {result.measurements[idx_after, 7]:.1f} %")
    print(f"  A+C feed flow: {result.measurements[idx_before, 3]:.2f} -> {result.measurements[idx_after, 3]:.2f} kscmh")

    # Example 2: Multiple disturbances
    print("\n" + "=" * 50)
    print("Example 2: Multiple disturbances")
    print("=" * 50)

    sim.initialize()  # Reset to steady state

    result = sim.simulate(
        duration_hours=6.0,
        disturbances={
            1: (1.0, 1),   # IDV(1) at 1 hour
            4: (3.0, 1),   # IDV(4) at 3 hours
        },
        record_interval=60
    )

    print(f"\nDisturbance schedule:")
    print(f"  IDV(1) applied at t=1.0 hour")
    print(f"  IDV(4) applied at t=3.0 hours")
    print(f"Shutdown: {result.shutdown}")

    # Example 3: Real-time disturbance application
    print("\n" + "=" * 50)
    print("Example 3: Real-time disturbance activation")
    print("=" * 50)

    sim.initialize()

    # Step simulation manually
    measurements_log = []

    for i in range(7200):  # 2 hours at 1 second/step
        # Apply disturbance after 30 minutes
        if i == 1800:  # 30 minutes = 1800 seconds
            print(f"  Applying IDV(1) at t={sim.time*60:.1f} minutes")
            sim.set_disturbance(1, 1)

        # Remove disturbance after 90 minutes
        if i == 5400:  # 90 minutes
            print(f"  Removing IDV(1) at t={sim.time*60:.1f} minutes")
            sim.set_disturbance(1, 0)

        running = sim.step()
        if not running:
            print("  Process shutdown!")
            break

        # Log every minute
        if i % 60 == 0:
            measurements_log.append(sim.get_measurements().copy())

    print(f"\nFinal time: {sim.time*60:.1f} minutes")
    print(f"Data points logged: {len(measurements_log)}")

    # Example 4: Random disturbances
    print("\n" + "=" * 50)
    print("Example 4: Random disturbances (IDV 8-12)")
    print("=" * 50)

    sim.initialize()

    # Apply random variations
    sim.set_disturbance(8, 1)   # Random feed composition
    sim.set_disturbance(11, 1)  # Random CW temp

    result = sim.simulate(duration_hours=2.0, record_interval=60)

    temps = result.measurements[:, 8]
    print(f"\nReactor temperature with random disturbances:")
    print(f"  Mean: {np.mean(temps):.2f} °C")
    print(f"  Std:  {np.std(temps):.2f} °C (increased due to disturbances)")

    print("\nExample complete!")


if __name__ == "__main__":
    main()
