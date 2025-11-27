#!/usr/bin/env python3
"""
Data Generation Example

This example demonstrates how to generate datasets for
fault detection and diagnosis research, similar to the
original TEP benchmark datasets.
"""

import numpy as np
from tep import TEPSimulator, MEASUREMENT_NAMES, MANIPULATED_VAR_NAMES
from tep.simulator import ControlMode


def generate_normal_data(duration_hours=8.0, seed=12345):
    """Generate normal operating data."""
    print(f"Generating {duration_hours} hours of normal operating data...")

    sim = TEPSimulator(random_seed=seed, control_mode=ControlMode.CLOSED_LOOP)
    sim.initialize()

    # Run simulation
    result = sim.simulate(
        duration_hours=duration_hours,
        record_interval=180  # Record every 3 minutes (like original)
    )

    print(f"  Generated {len(result.time)} samples")
    return result


def generate_faulty_data(fault_id, duration_hours=8.0, fault_time=1.0, seed=12345):
    """Generate data with a specific fault."""
    print(f"Generating fault {fault_id} data...")

    sim = TEPSimulator(random_seed=seed, control_mode=ControlMode.CLOSED_LOOP)
    sim.initialize()

    result = sim.simulate(
        duration_hours=duration_hours,
        disturbances={fault_id: (fault_time, 1)},
        record_interval=180
    )

    print(f"  Generated {len(result.time)} samples")
    print(f"  Shutdown: {result.shutdown}")
    return result


def create_dataset_matrix(result):
    """
    Create combined measurement + MV matrix like original TEP datasets.

    Returns (n_samples, 52) array: [XMEAS(1-41), XMV(1-11)]
    """
    # Original datasets don't include XMV(12) (agitator)
    data = np.hstack([
        result.measurements,           # 41 columns
        result.manipulated_vars[:, :11]  # 11 columns
    ])
    return data


def main():
    print("TEP Data Generation Example")
    print("=" * 50)

    # Generate normal operating data
    print("\n1. Normal operating conditions (d00.dat equivalent)")
    print("-" * 50)
    normal_result = generate_normal_data(duration_hours=8.0, seed=12345)
    normal_data = create_dataset_matrix(normal_result)
    print(f"   Dataset shape: {normal_data.shape}")

    # Generate faulty data for multiple faults
    print("\n2. Generating fault datasets")
    print("-" * 50)

    fault_results = {}
    for fault_id in [1, 2, 4, 5, 6, 7]:
        fault_results[fault_id] = generate_faulty_data(
            fault_id=fault_id,
            duration_hours=8.0,
            fault_time=1.0,  # Fault introduced at 1 hour
            seed=12345
        )

    # Save data to files (numpy format)
    print("\n3. Saving datasets")
    print("-" * 50)

    # Save normal data
    np.save("normal_data.npy", normal_data)
    print("   Saved: normal_data.npy")

    # Save fault data
    for fault_id, result in fault_results.items():
        fault_data = create_dataset_matrix(result)
        filename = f"fault{fault_id}_data.npy"
        np.save(filename, fault_data)
        print(f"   Saved: {filename}")

    # Save as CSV for easy inspection
    print("\n4. Saving CSV sample (first 100 rows of normal data)")
    print("-" * 50)

    # Create header
    header = MEASUREMENT_NAMES + MANIPULATED_VAR_NAMES[:11]

    # Save first 100 rows as CSV
    np.savetxt(
        "normal_data_sample.csv",
        normal_data[:100],
        delimiter=",",
        header=",".join(header),
        comments=""
    )
    print("   Saved: normal_data_sample.csv")

    # Print statistics
    print("\n5. Data statistics")
    print("-" * 50)
    print("\nNormal data statistics (selected variables):")
    print(f"  {'Variable':<30} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*50}")

    key_vars = [8, 6, 7, 11, 14]  # Reactor temp, pressure, level; sep level; strip level
    for idx in key_vars:
        name = MEASUREMENT_NAMES[idx][:28]
        mean = np.mean(normal_data[:, idx])
        std = np.std(normal_data[:, idx])
        print(f"  {name:<30} {mean:>10.2f} {std:>10.2f}")

    # Compare normal vs fault data
    print("\n6. Normal vs Fault comparison (Reactor Temperature)")
    print("-" * 50)

    normal_temp = normal_result.measurements[:, 8]
    print(f"  Normal:  mean={np.mean(normal_temp):.2f}, std={np.std(normal_temp):.2f}")

    for fault_id, result in fault_results.items():
        fault_temp = result.measurements[:, 8]
        print(f"  Fault {fault_id}: mean={np.mean(fault_temp):.2f}, std={np.std(fault_temp):.2f}")

    # Generate multiple random seeds for training/testing split
    print("\n7. Generate train/test split data")
    print("-" * 50)

    print("   Training data (seed=12345):")
    train_result = generate_normal_data(duration_hours=8.0, seed=12345)
    train_data = create_dataset_matrix(train_result)
    print(f"   Shape: {train_data.shape}")

    print("   Testing data (seed=67890):")
    test_result = generate_normal_data(duration_hours=16.0, seed=67890)
    test_data = create_dataset_matrix(test_result)
    print(f"   Shape: {test_data.shape}")

    print("\nData generation complete!")
    print(f"Generated files:")
    print("  - normal_data.npy")
    print("  - normal_data_sample.csv")
    print("  - fault{1,2,4,5,6,7}_data.npy")


if __name__ == "__main__":
    main()
