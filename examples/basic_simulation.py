#!/usr/bin/env python3
"""
Basic TEP Simulation Example

This example demonstrates the fundamental usage of the TEPSimulator
for running batch simulations and accessing results.
"""

import numpy as np
from tep import TEPSimulator
from tep.simulator import ControlMode


def main():
    # Create simulator with default settings (closed-loop control)
    print("Creating TEP simulator...")
    sim = TEPSimulator(
        random_seed=12345,  # For reproducibility
        control_mode=ControlMode.CLOSED_LOOP
    )

    # Initialize to steady state
    sim.initialize()
    print(f"Initial reactor temperature: {sim.get_measurements()[8]:.1f} °C")
    print(f"Initial reactor pressure: {sim.get_measurements()[6]:.1f} kPa")

    # Run a 1-hour simulation
    print("\nRunning 1-hour simulation...")
    result = sim.simulate(
        duration_hours=1.0,
        record_interval=60  # Record every minute
    )

    # Check results
    print(f"\nSimulation complete!")
    print(f"  Duration: {result.time[-1]:.2f} hours ({result.time_minutes[-1]:.0f} minutes)")
    print(f"  Data points recorded: {len(result.time)}")
    print(f"  Shutdown occurred: {result.shutdown}")

    # Print final state
    print(f"\nFinal conditions:")
    print(f"  Reactor temperature: {result.measurements[-1, 8]:.1f} °C")
    print(f"  Reactor pressure: {result.measurements[-1, 6]:.1f} kPa")
    print(f"  Reactor level: {result.measurements[-1, 7]:.1f} %")
    print(f"  Separator level: {result.measurements[-1, 11]:.1f} %")
    print(f"  Stripper level: {result.measurements[-1, 14]:.1f} %")

    # Calculate statistics
    print(f"\nReactor temperature statistics:")
    temps = result.measurements[:, 8]
    print(f"  Mean: {np.mean(temps):.2f} °C")
    print(f"  Std:  {np.std(temps):.2f} °C")
    print(f"  Min:  {np.min(temps):.2f} °C")
    print(f"  Max:  {np.max(temps):.2f} °C")

    return result


if __name__ == "__main__":
    result = main()
