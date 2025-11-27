#!/usr/bin/env python
"""Compare Python TEP simulation results with Fortran reference output.

This script:
1. Loads Fortran output from TE_data_*.dat files
2. Runs Python simulation with matching parameters
3. Compares results and reports statistics

Usage:
    # First run Fortran simulation:
    make fortran-run

    # Then run this comparison:
    python examples/compare_fortran_python.py
"""

import numpy as np
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tep import TEPSimulator
from tep.simulator import ControlMode


def load_fortran_data(base_path: Path = Path(".")):
    """Load Fortran output files into arrays.

    Returns:
        time: array of time points in hours
        xmeas: array of measurements (n_samples, 41)
        xmv: array of manipulated variables (n_samples, 12)
    """
    # Load time data (in seconds)
    time_file = base_path / "TE_data_inc.dat"
    if not time_file.exists():
        raise FileNotFoundError(
            f"Fortran output not found at {time_file}. "
            "Run 'make fortran-run' first."
        )

    time_sec = np.loadtxt(time_file)
    time_hr = time_sec / 3600.0
    n_samples = len(time_sec)

    # Load measurements (XMEAS 1-41)
    xmeas = np.zeros((n_samples, 41))

    # me01.dat has XMEAS 1-4, me02.dat has 5-8, etc.
    for i in range(10):
        filename = base_path / f"TE_data_me{i+1:02d}.dat"
        data = np.loadtxt(filename)
        start_idx = i * 4
        end_idx = min(start_idx + 4, 41)
        n_cols = end_idx - start_idx
        xmeas[:, start_idx:end_idx] = data[:, :n_cols]

    # me11.dat has XMEAS 41 only
    filename = base_path / "TE_data_me11.dat"
    data = np.loadtxt(filename)
    if data.ndim == 1:
        xmeas[:, 40] = data
    else:
        xmeas[:, 40] = data[:, 0]

    # Load manipulated variables (XMV 1-12)
    xmv = np.zeros((n_samples, 12))

    # mv1.dat has XMV 1-4, mv2.dat has 5-8, mv3.dat has 9-12
    for i in range(3):
        filename = base_path / f"TE_data_mv{i+1}.dat"
        data = np.loadtxt(filename)
        start_idx = i * 4
        xmv[:, start_idx:start_idx+4] = data

    return time_hr, xmeas, xmv


def run_python_simulation(duration_hours: float, save_interval_sec: float = 180.0):
    """Run Python simulation matching Fortran parameters.

    Args:
        duration_hours: Simulation duration in hours
        save_interval_sec: How often to save data (seconds)

    Returns:
        result: SimulationResult object
    """
    sim = TEPSimulator()
    sim.initialize()

    # Default dt is 1 second (1/3600 hours)
    # Fortran saves every 180 seconds, so record_interval = 180
    record_interval = int(save_interval_sec)

    # Run closed-loop simulation (IDV=0, no disturbance)
    result = sim.simulate(
        duration_hours=duration_hours,
        record_interval=record_interval
    )

    return result


def compare_results(fortran_time, fortran_xmeas, fortran_xmv,
                   python_result, tolerance_pct: float = 5.0):
    """Compare Fortran and Python results.

    Args:
        tolerance_pct: Tolerance for "close enough" as percentage

    Returns:
        dict with comparison statistics
    """
    py_time = python_result.time
    py_xmeas = python_result.measurements
    py_xmv = python_result.manipulated_vars

    # Align time points (both should have same sampling)
    n_compare = min(len(fortran_time), len(py_time))

    print(f"\n{'='*70}")
    print("COMPARISON: Fortran vs Python TEP Simulation")
    print(f"{'='*70}")
    print(f"Fortran samples: {len(fortran_time)}")
    print(f"Python samples:  {len(py_time)}")
    print(f"Comparing first: {n_compare} samples")
    print(f"Time range: 0 to {fortran_time[n_compare-1]:.2f} hours")

    # Compare measurements
    print(f"\n{'='*70}")
    print("MEASUREMENT COMPARISON (XMEAS 1-41)")
    print(f"{'='*70}")
    print(f"{'XMEAS':<8} {'Fortran Mean':>14} {'Python Mean':>14} {'Rel Diff %':>12} {'Status':<8}")
    print("-" * 70)

    stats = {'xmeas': [], 'xmv': []}

    for i in range(41):
        f_mean = np.mean(fortran_xmeas[:n_compare, i])
        p_mean = np.mean(py_xmeas[:n_compare, i])

        if abs(f_mean) > 1e-10:
            rel_diff = abs(f_mean - p_mean) / abs(f_mean) * 100
        else:
            rel_diff = abs(f_mean - p_mean) * 100

        status = "OK" if rel_diff < tolerance_pct else "DIFF"
        stats['xmeas'].append({
            'idx': i+1,
            'fortran_mean': f_mean,
            'python_mean': p_mean,
            'rel_diff_pct': rel_diff,
            'ok': rel_diff < tolerance_pct
        })

        # Only print if there's a notable difference
        if rel_diff > 1.0 or i < 10:  # Always show first 10
            print(f"XMEAS{i+1:<3} {f_mean:>14.4f} {p_mean:>14.4f} {rel_diff:>12.2f} {status:<8}")

    # Compare manipulated variables
    print(f"\n{'='*70}")
    print("MANIPULATED VARIABLE COMPARISON (XMV 1-12)")
    print(f"{'='*70}")
    print(f"{'XMV':<8} {'Fortran Mean':>14} {'Python Mean':>14} {'Rel Diff %':>12} {'Status':<8}")
    print("-" * 70)

    for i in range(12):
        f_mean = np.mean(fortran_xmv[:n_compare, i])
        p_mean = np.mean(py_xmv[:n_compare, i])

        if abs(f_mean) > 1e-10:
            rel_diff = abs(f_mean - p_mean) / abs(f_mean) * 100
        else:
            rel_diff = abs(f_mean - p_mean) * 100

        status = "OK" if rel_diff < tolerance_pct else "DIFF"
        stats['xmv'].append({
            'idx': i+1,
            'fortran_mean': f_mean,
            'python_mean': p_mean,
            'rel_diff_pct': rel_diff,
            'ok': rel_diff < tolerance_pct
        })

        print(f"XMV{i+1:<5} {f_mean:>14.4f} {p_mean:>14.4f} {rel_diff:>12.2f} {status:<8}")

    # Summary
    xmeas_ok = sum(1 for s in stats['xmeas'] if s['ok'])
    xmv_ok = sum(1 for s in stats['xmv'] if s['ok'])

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Measurements within {tolerance_pct}%: {xmeas_ok}/41")
    print(f"MVs within {tolerance_pct}%: {xmv_ok}/12")

    # Check for shutdown
    print(f"\nFortran shutdown: N/A (ran to completion)")
    print(f"Python shutdown:  {python_result.shutdown}")
    if python_result.shutdown:
        print(f"  Reason: {python_result.shutdown_reason}")

    return stats


def plot_comparison(fortran_time, fortran_xmeas, fortran_xmv,
                   python_result, variables=None):
    """Plot side-by-side comparison of selected variables.

    Args:
        variables: List of (type, idx) tuples, e.g. [('xmeas', 7), ('xmv', 3)]
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not available for plotting.")
        return

    if variables is None:
        # Default: key process variables
        variables = [
            ('xmeas', 7),   # Reactor pressure
            ('xmeas', 9),   # Reactor temperature
            ('xmeas', 17),  # Stripper underflow
            ('xmv', 3),     # A feed
            ('xmv', 10),    # Reactor cooling
        ]

    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3*n_vars), sharex=True)
    if n_vars == 1:
        axes = [axes]

    py_time = python_result.time
    n_compare = min(len(fortran_time), len(py_time))

    for ax, (var_type, idx) in zip(axes, variables):
        if var_type == 'xmeas':
            f_data = fortran_xmeas[:n_compare, idx-1]
            p_data = python_result.measurements[:n_compare, idx-1]
            label = f"XMEAS {idx}"
        else:
            f_data = fortran_xmv[:n_compare, idx-1]
            p_data = python_result.manipulated_vars[:n_compare, idx-1]
            label = f"XMV {idx}"

        ax.plot(fortran_time[:n_compare], f_data, 'b-', label='Fortran', alpha=0.7)
        ax.plot(py_time[:n_compare], p_data, 'r--', label='Python', alpha=0.7)
        ax.set_ylabel(label)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (hours)')
    plt.suptitle('Fortran vs Python TEP Simulation Comparison')
    plt.tight_layout()
    plt.savefig('fortran_python_comparison.png', dpi=150)
    print(f"\nPlot saved to: fortran_python_comparison.png")
    plt.show()


def main():
    """Main comparison routine."""
    print("Loading Fortran output data...")
    try:
        f_time, f_xmeas, f_xmv = load_fortran_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Fortran runs for ~48 hours
    duration = f_time[-1]
    print(f"Fortran simulation duration: {duration:.2f} hours")

    print(f"\nRunning Python simulation for {duration:.2f} hours...")
    print("(This may take a minute...)")

    py_result = run_python_simulation(duration)

    # Compare
    stats = compare_results(f_time, f_xmeas, f_xmv, py_result)

    # Plot if matplotlib available
    try:
        plot_comparison(f_time, f_xmeas, f_xmv, py_result)
    except Exception as e:
        print(f"\nCould not create plot: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
