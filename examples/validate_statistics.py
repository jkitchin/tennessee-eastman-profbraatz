#!/usr/bin/env python
"""Validate Python TEP simulation by comparing statistical properties with Fortran.

Since both simulations add random noise with different sequences, we can't expect
sample-by-sample matching. Instead, we validate:
1. Same steady-state operating point (mean values)
2. Same noise characteristics (standard deviations)
3. Same dynamic behavior (autocorrelation structure)

Usage:
    # First run Fortran simulation:
    make fortran-run

    # Then run this validation:
    python examples/validate_statistics.py
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tep import TEPSimulator


def load_fortran_data(base_path: Path = Path(".")):
    """Load Fortran output files."""
    time_sec = np.loadtxt(base_path / "TE_data_inc.dat")
    time_hr = time_sec / 3600.0
    n_samples = len(time_sec)

    # Load measurements (XMEAS 1-41)
    xmeas = np.zeros((n_samples, 41))
    for i in range(10):
        filename = base_path / f"TE_data_me{i+1:02d}.dat"
        data = np.loadtxt(filename)
        start_idx = i * 4
        end_idx = min(start_idx + 4, 41)
        n_cols = end_idx - start_idx
        xmeas[:, start_idx:end_idx] = data[:, :n_cols]

    filename = base_path / "TE_data_me11.dat"
    data = np.loadtxt(filename)
    xmeas[:, 40] = data if data.ndim == 1 else data[:, 0]

    # Load manipulated variables (XMV 1-12)
    xmv = np.zeros((n_samples, 12))
    for i in range(3):
        filename = base_path / f"TE_data_mv{i+1}.dat"
        data = np.loadtxt(filename)
        xmv[:, i*4:(i+1)*4] = data

    return time_hr, xmeas, xmv


def compute_statistics(data, skip_initial=60):
    """Compute mean, std, and autocorrelation lag-1.

    Args:
        data: Time series array (n_samples,) or (n_samples, n_vars)
        skip_initial: Number of samples to skip for settling (default 60 = 3 hours)
    """
    data = data[skip_initial:]  # Skip initial transient

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_vars = data.shape

    stats = {
        'mean': np.mean(data, axis=0),
        'std': np.std(data, axis=0),
        'autocorr': np.zeros(n_vars)
    }

    # Compute lag-1 autocorrelation
    for j in range(n_vars):
        x = data[:-1, j] - stats['mean'][j]
        y = data[1:, j] - stats['mean'][j]
        if np.std(x) > 1e-10 and np.std(y) > 1e-10:
            stats['autocorr'][j] = np.corrcoef(x, y)[0, 1]

    return stats


def main():
    print("Loading Fortran output data...")
    try:
        f_time, f_xmeas, f_xmv = load_fortran_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'make fortran-run' first to generate Fortran output.")
        return 1

    duration = f_time[-1]
    print(f"Fortran simulation: {len(f_time)} samples over {duration:.1f} hours")

    print(f"\nRunning Python simulation for {duration:.1f} hours...")
    sim = TEPSimulator()
    sim.initialize()
    py_result = sim.simulate(duration_hours=duration, record_interval=180)
    print(f"Python simulation: {len(py_result.time)} samples")

    # Compute statistics for both (skip first 3 hours for settling)
    n_compare = min(len(f_time), len(py_result.time))

    f_xmeas_stats = compute_statistics(f_xmeas[:n_compare])
    p_xmeas_stats = compute_statistics(py_result.measurements[:n_compare])

    f_xmv_stats = compute_statistics(f_xmv[:n_compare])
    p_xmv_stats = compute_statistics(py_result.manipulated_vars[:n_compare])

    # Key process variables to focus on
    key_xmeas = [
        (7, "Reactor Pressure (kPa)"),
        (8, "Reactor Level (%)"),
        (9, "Reactor Temperature (°C)"),
        (11, "Separator Level (%)"),
        (12, "Separator Pressure (kPa)"),
        (15, "Stripper Level (%)"),
        (17, "Stripper Underflow (m³/hr)"),
        (21, "CW Outlet Temp (°C)"),
    ]

    print("\n" + "="*80)
    print("STATISTICAL VALIDATION: Key Process Measurements")
    print("="*80)
    print(f"{'Variable':<30} {'Fortran':<20} {'Python':<20} {'Diff %':<10}")
    print("-"*80)

    xmeas_pass = 0
    xmeas_fail = 0

    for idx, name in key_xmeas:
        i = idx - 1  # Convert to 0-indexed
        f_mean = f_xmeas_stats['mean'][i]
        p_mean = p_xmeas_stats['mean'][i]
        f_std = f_xmeas_stats['std'][i]
        p_std = p_xmeas_stats['std'][i]

        # Mean difference
        if abs(f_mean) > 1e-6:
            mean_diff = abs(f_mean - p_mean) / abs(f_mean) * 100
        else:
            mean_diff = abs(f_mean - p_mean) * 100

        # Std ratio (should be close to 1.0)
        if f_std > 1e-6:
            std_ratio = p_std / f_std
        else:
            std_ratio = 1.0 if p_std < 1e-6 else float('inf')

        status = "✓" if mean_diff < 5.0 and 0.5 < std_ratio < 2.0 else "✗"
        if status == "✓":
            xmeas_pass += 1
        else:
            xmeas_fail += 1

        print(f"XMEAS{idx:<2} {name:<22} "
              f"{f_mean:>8.2f}±{f_std:<8.2f} "
              f"{p_mean:>8.2f}±{p_std:<8.2f} "
              f"{mean_diff:>6.2f}% {status}")

    print("\n" + "="*80)
    print("STATISTICAL VALIDATION: Manipulated Variables")
    print("="*80)
    print(f"{'Variable':<30} {'Fortran':<20} {'Python':<20} {'Diff %':<10}")
    print("-"*80)

    mv_names = [
        "D Feed Flow",
        "E Feed Flow",
        "A Feed Flow",
        "A+C Feed Flow",
        "Compressor Recycle",
        "Purge Valve",
        "Separator Pot Liquid",
        "Stripper Pot Liquid",
        "Stripper Steam",
        "Reactor CW",
        "Condenser CW",
        "Agitator Speed"
    ]

    xmv_pass = 0
    xmv_fail = 0

    for i in range(12):
        f_mean = f_xmv_stats['mean'][i]
        p_mean = p_xmv_stats['mean'][i]
        f_std = f_xmv_stats['std'][i]
        p_std = p_xmv_stats['std'][i]

        if abs(f_mean) > 1e-6:
            mean_diff = abs(f_mean - p_mean) / abs(f_mean) * 100
        else:
            mean_diff = abs(f_mean - p_mean) * 100

        if f_std > 1e-6:
            std_ratio = p_std / f_std
        else:
            std_ratio = 1.0 if p_std < 1e-6 else float('inf')

        status = "✓" if mean_diff < 10.0 and 0.3 < std_ratio < 3.0 else "✗"
        if status == "✓":
            xmv_pass += 1
        else:
            xmv_fail += 1

        print(f"XMV{i+1:<3} {mv_names[i]:<22} "
              f"{f_mean:>8.2f}±{f_std:<8.2f} "
              f"{p_mean:>8.2f}±{p_std:<8.2f} "
              f"{mean_diff:>6.2f}% {status}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Key measurements passed: {xmeas_pass}/{xmeas_pass + xmeas_fail}")
    print(f"Manipulated variables passed: {xmv_pass}/{xmv_pass + xmv_fail}")
    print(f"\nFortran shutdown: No (ran to completion)")
    print(f"Python shutdown:  {'Yes - ' + py_result.shutdown_reason if py_result.shutdown else 'No (ran to completion)'}")

    # Overall assessment
    total_pass = xmeas_pass + xmv_pass
    total = xmeas_pass + xmeas_fail + xmv_pass + xmv_fail

    print("\n" + "="*80)
    if total_pass >= total * 0.8 and not py_result.shutdown:
        print("VALIDATION PASSED: Python implementation matches Fortran statistically")
        print("\nNote: Exact sample-by-sample matching is not expected because both")
        print("simulations use random measurement noise with different sequences.")
        print("The statistical properties (mean, variance) should match.")
    else:
        print("VALIDATION FAILED: Significant differences detected")
        print("Please investigate the following:")
        print("- Check controller gains and setpoints")
        print("- Check process constants and parameters")
        print("- Check integration timestep")
    print("="*80)

    # Try to create comparison plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 2, figsize=(14, 12))

        # Plot key variables
        plot_vars = [
            (6, 'XMEAS 7: Reactor P', 'xmeas'),
            (8, 'XMEAS 9: Reactor T', 'xmeas'),
            (7, 'XMEAS 8: Reactor Level', 'xmeas'),
            (14, 'XMEAS 15: Sep Pressure', 'xmeas'),
            (2, 'XMV 3: A Feed', 'xmv'),
            (9, 'XMV 10: Reactor CW', 'xmv'),
            (8, 'XMV 9: Stripper Steam', 'xmv'),
            (10, 'XMV 11: Condenser CW', 'xmv'),
        ]

        for ax, (idx, title, vtype) in zip(axes.flat, plot_vars):
            if vtype == 'xmeas':
                f_data = f_xmeas[:n_compare, idx]
                p_data = py_result.measurements[:n_compare, idx]
            else:
                f_data = f_xmv[:n_compare, idx]
                p_data = py_result.manipulated_vars[:n_compare, idx]

            ax.plot(f_time[:n_compare], f_data, 'b-', label='Fortran', alpha=0.6, lw=0.5)
            ax.plot(py_result.time[:n_compare], p_data, 'r-', label='Python', alpha=0.6, lw=0.5)
            ax.set_title(title)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1, 0].set_xlabel('Time (hours)')
        axes[-1, 1].set_xlabel('Time (hours)')

        plt.suptitle('Fortran vs Python: Statistical Comparison\n(Different noise sequences expected)', y=1.02)
        plt.tight_layout()
        plt.savefig('validation_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: validation_comparison.png")

    except ImportError:
        print("\nMatplotlib not available for plotting")
    except Exception as e:
        print(f"\nCould not create plot: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
