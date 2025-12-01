"""
Command-line interface for running TEP batch simulations.

This module provides a CLI for running Tennessee Eastman Process simulations
with configurable faults, random seeds, and output options.

Output format matches the original Fortran code:
- Scientific notation (E13.5 format)
- 4 values per line, space-separated
- Data recorded every 180 seconds (3 minutes)
"""

import argparse
import sys
import os
import numpy as np
from typing import List, Tuple, Optional

from .simulator import TEPSimulator, ControlMode
from .constants import (
    DEFAULT_RANDOM_SEED,
    MEASUREMENT_NAMES,
    MANIPULATED_VAR_NAMES,
    DISTURBANCE_NAMES,
)
from .controller_base import ControllerRegistry, BaseController


def parse_faults(fault_str: str) -> List[int]:
    """
    Parse fault specification string into list of fault IDs.

    Supports:
    - Single fault: "1"
    - Multiple faults: "1,2,5"
    - Range: "1-5"
    - Combined: "1,3-5,7"

    Args:
        fault_str: Fault specification string

    Returns:
        List of fault IDs (1-20)
    """
    faults = []
    for part in fault_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            faults.extend(range(int(start), int(end) + 1))
        else:
            faults.append(int(part))

    # Validate fault IDs
    for f in faults:
        if f < 1 or f > 20:
            raise ValueError(f"Invalid fault ID {f}. Must be between 1 and 20.")

    return faults


def parse_fault_times(time_str: str, num_faults: int) -> List[float]:
    """
    Parse fault start time specification.

    Args:
        time_str: Comma-separated list of start times in hours, or single time
        num_faults: Number of faults to match

    Returns:
        List of start times (one per fault)
    """
    times = [float(t.strip()) for t in time_str.split(',')]

    if len(times) == 1:
        # Single time applies to all faults
        return times * num_faults
    elif len(times) != num_faults:
        raise ValueError(
            f"Number of start times ({len(times)}) must match number of faults ({num_faults}) "
            "or be a single value."
        )

    return times


def format_fortran_value(value: float) -> str:
    """
    Format a value in Fortran E13.5 format.

    Fortran FORMAT(1X,E13.5) produces: " 0.12345E+02"
    """
    # Python's scientific notation with adjustment for Fortran style
    return f"{value:13.5E}"


def write_fortran_format(
    filepath: str,
    data: np.ndarray,
    values_per_line: int = 4
):
    """
    Write data in original Fortran output format.

    Args:
        filepath: Output file path
        data: 2D array (n_samples, n_values)
        values_per_line: Number of values per line (default 4)
    """
    with open(filepath, 'w') as f:
        for row in data:
            n_values = len(row)
            for i in range(0, n_values, values_per_line):
                chunk = row[i:i + values_per_line]
                line = "  ".join(format_fortran_value(v) for v in chunk)
                f.write(f" {line}\n")


def write_single_file(
    filepath: str,
    result,
    include_header: bool = True
):
    """
    Write all data to a single file in Fortran-compatible format.

    Format: Each row contains time (seconds), 41 measurements, 12 MVs
    Values are in E13.5 scientific notation.

    Args:
        filepath: Output file path
        result: SimulationResult object
        include_header: Whether to include column headers
    """
    # Combine time, measurements, and MVs
    time_seconds = result.time_seconds.reshape(-1, 1)

    with open(filepath, 'w') as f:
        # Optional header
        if include_header:
            headers = ['Time(s)'] + MEASUREMENT_NAMES + MANIPULATED_VAR_NAMES
            f.write(' '.join(f'{h:>13}' for h in headers) + '\n')

        # Data rows
        for i in range(len(result.time)):
            values = [time_seconds[i, 0]] + list(result.measurements[i]) + list(result.manipulated_vars[i])
            line = ' '.join(format_fortran_value(v) for v in values)
            f.write(f'{line}\n')


def write_multi_file(
    output_dir: str,
    prefix: str,
    result
):
    """
    Write data to multiple files matching original Fortran output structure.

    Files created:
    - {prefix}_inc.dat: Time increments (seconds)
    - {prefix}_mv1.dat: MVs 1-4
    - {prefix}_mv2.dat: MVs 5-8
    - {prefix}_mv3.dat: MVs 9-12
    - {prefix}_me01.dat through {prefix}_me11.dat: Measurements

    Args:
        output_dir: Output directory
        prefix: File name prefix
        result: SimulationResult object
    """
    os.makedirs(output_dir, exist_ok=True)

    time_seconds = result.time_seconds
    meas = result.measurements
    mvs = result.manipulated_vars

    # Time file (single column)
    time_file = os.path.join(output_dir, f'{prefix}_inc.dat')
    with open(time_file, 'w') as f:
        for t in time_seconds:
            f.write(f' {int(t):6d}\n')

    # MV files (4 values per line each)
    mv_groups = [(1, 4), (5, 8), (9, 12)]
    for idx, (start, end) in enumerate(mv_groups, 1):
        mv_file = os.path.join(output_dir, f'{prefix}_mv{idx}.dat')
        mv_data = mvs[:, start-1:end]
        write_fortran_format(mv_file, mv_data, values_per_line=4)

    # Measurement files (4 values per line, except last which has 1)
    meas_groups = [
        (1, 4), (5, 8), (9, 12), (13, 16), (17, 20),
        (21, 24), (25, 28), (29, 32), (33, 36), (37, 40), (41, 41)
    ]
    for idx, (start, end) in enumerate(meas_groups, 1):
        meas_file = os.path.join(output_dir, f'{prefix}_me{idx:02d}.dat')
        meas_data = meas[:, start-1:end]
        write_fortran_format(meas_file, meas_data, values_per_line=4)

    print(f"Written {len(mv_groups) + len(meas_groups) + 1} files to {output_dir}/")


def list_faults():
    """Print list of available faults."""
    print("\nAvailable Disturbances (IDV 1-20):")
    print("-" * 60)
    for i, name in enumerate(DISTURBANCE_NAMES, 1):
        print(f"  IDV({i:2d}): {name}")
    print()


def list_controllers():
    """Print list of available controller plugins."""
    # Import plugins to ensure they're registered
    from . import controller_plugins  # noqa: F401

    print("\nAvailable Controller Plugins:")
    print("-" * 70)

    controllers = ControllerRegistry.list_all_info()
    if not controllers:
        print("  No controllers registered.")
    else:
        for info in controllers:
            print(f"  {info['name']:20s} - {info['description']}")
    print()
    print("Use --controller <name> to select a controller.")
    print("Default: decentralized (original TEP control scheme)")
    print()


def plot_results(result, faults: Optional[List[int]] = None, save_path: Optional[str] = None):
    """
    Display simulation results in a graphical form.

    Args:
        result: SimulationResult object
        faults: List of fault IDs that were active (for title)
        save_path: If provided, save plot to this path instead of displaying
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for plotting.", file=sys.stderr)
        print("Install with: pip install matplotlib", file=sys.stderr)
        return

    time_hours = result.time
    meas = result.measurements
    mvs = result.manipulated_vars

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        f'TEP Simulation Results' +
        (f' - Faults: {faults}' if faults else ' - Normal Operation'),
        fontsize=14
    )

    # Plot 1: Reactor conditions
    ax = axes[0, 0]
    ax.plot(time_hours, meas[:, 8], label='Reactor Temp (C)', color='red')
    ax.set_ylabel('Temperature (C)', color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_xlabel('Time (hours)')
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.plot(time_hours, meas[:, 6], label='Reactor Press (kPa)', color='blue', linestyle='--')
    ax2.set_ylabel('Pressure (kPa)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.legend(loc='upper right')
    ax.set_title('Reactor Conditions')
    ax.grid(True, alpha=0.3)

    # Plot 2: Levels
    ax = axes[0, 1]
    ax.plot(time_hours, meas[:, 7], label='Reactor Level (%)')
    ax.plot(time_hours, meas[:, 11], label='Separator Level (%)')
    ax.plot(time_hours, meas[:, 14], label='Stripper Level (%)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Level (%)')
    ax.set_title('Process Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Feed flows
    ax = axes[1, 0]
    ax.plot(time_hours, meas[:, 0], label='A Feed (kscmh)')
    ax.plot(time_hours, meas[:, 1] / 1000, label='D Feed (kkg/hr)')
    ax.plot(time_hours, meas[:, 2] / 1000, label='E Feed (kkg/hr)')
    ax.plot(time_hours, meas[:, 3], label='A+C Feed (kscmh)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Flow Rate')
    ax.set_title('Feed Flows')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Product quality
    ax = axes[1, 1]
    ax.plot(time_hours, meas[:, 16], label='Stripper Underflow (m3/hr)')
    ax.plot(time_hours, meas[:, 9], label='Product Sep Temp (C)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Value')
    ax.set_title('Product Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Key manipulated variables
    ax = axes[2, 0]
    ax.plot(time_hours, mvs[:, 0], label='D Feed (MV1)')
    ax.plot(time_hours, mvs[:, 1], label='E Feed (MV2)')
    ax.plot(time_hours, mvs[:, 2], label='A Feed (MV3)')
    ax.plot(time_hours, mvs[:, 3], label='A+C Feed (MV4)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Valve Position (%)')
    ax.set_title('Feed Valve Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Cooling and control valves
    ax = axes[2, 1]
    ax.plot(time_hours, mvs[:, 9], label='Reactor CW (MV10)')
    ax.plot(time_hours, mvs[:, 10], label='Condenser CW (MV11)')
    ax.plot(time_hours, mvs[:, 5], label='Purge (MV6)')
    ax.plot(time_hours, mvs[:, 6], label='Sep Underflow (MV7)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Valve Position (%)')
    ax.set_title('Cooling & Control Valves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def run_simulation(
    duration_hours: float,
    faults: Optional[List[int]] = None,
    fault_times: Optional[List[float]] = None,
    seed: int = DEFAULT_RANDOM_SEED,
    output: str = "tep_data.dat",
    multi_file: bool = False,
    record_interval: int = 180,
    no_header: bool = False,
    quiet: bool = False,
    plot: bool = False,
    plot_save: Optional[str] = None,
    controller: Optional[str] = None,
    mode: int = 1
) -> int:
    """
    Run a batch simulation and save results.

    Args:
        duration_hours: Simulation duration in hours
        faults: List of fault IDs to activate (1-20)
        fault_times: Start times for each fault (hours)
        seed: Random seed
        output: Output file path or directory (for multi-file)
        multi_file: Use original Fortran multi-file output format
        record_interval: Steps between recordings (default 180 = 3 min)
        no_header: Omit header in single-file output
        quiet: Suppress progress output
        plot: Display results graphically
        plot_save: Save plot to file path instead of displaying
        controller: Name of controller plugin to use (default: decentralized)
        mode: Operating mode 1-6 (default: 1 = 50/50 G/H, base rate)

    Returns:
        Exit code (0 = success)
    """
    # Build disturbance schedule
    disturbances = {}
    if faults:
        if fault_times is None:
            fault_times = [1.0] * len(faults)  # Default: start at 1 hour

        for fault_id, start_time in zip(faults, fault_times):
            disturbances[fault_id] = (start_time, 1)

    # Validate mode
    from .constants import OPERATING_MODES
    if mode not in OPERATING_MODES:
        print(f"Error: Invalid mode {mode}. Must be 1-6.", file=sys.stderr)
        return 1
    mode_info = OPERATING_MODES[mode]

    # Create controller if specified
    controller_instance = None
    controller_name = controller or "decentralized"
    if controller_name != "decentralized":
        # Import plugins to ensure they're registered
        from . import controller_plugins  # noqa: F401
        try:
            controller_instance = ControllerRegistry.create(controller_name)
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Create simulator
    sim = TEPSimulator(random_seed=seed, control_mode=ControlMode.CLOSED_LOOP)
    sim.initialize()

    # Replace controller if custom one specified
    if controller_instance is not None:
        sim.controller = controller_instance

    # Set operating mode on the controller
    if hasattr(sim.controller, 'set_mode'):
        sim.controller.set_mode(mode)

    if not quiet:
        print(f"Tennessee Eastman Process Simulation")
        print(f"=" * 40)
        print(f"Duration: {duration_hours} hours")
        print(f"Random seed: {seed}")
        print(f"Controller: {controller_name}")
        print(f"Operating mode: {mode} ({mode_info.g_h_ratio} G/H, {mode_info.production})")
        print(f"Record interval: {record_interval} steps ({record_interval} seconds)")
        if faults:
            print(f"Faults: {faults}")
            print(f"Fault start times: {fault_times} hours")
        else:
            print("Faults: None (normal operation)")
        print()

    # Progress callback
    def progress(p):
        if not quiet:
            print(f"\rProgress: {p*100:.1f}%", end='', flush=True)

    # Run simulation
    if not quiet:
        print("Running simulation...")

    result = sim.simulate(
        duration_hours=duration_hours,
        disturbances=disturbances if disturbances else None,
        record_interval=record_interval,
        progress_callback=progress if not quiet else None
    )

    if not quiet:
        print("\rProgress: 100.0%")
        print()

    # Check for shutdown
    if result.shutdown:
        if not quiet:
            print(f"WARNING: Simulation ended in safety shutdown at {result.shutdown_time:.4f} hours")
            print(f"         ({result.shutdown_time * 3600:.1f} seconds)")

    # Save results
    if not quiet:
        print(f"Saving results...")

    if multi_file:
        # Multi-file format (original Fortran style)
        output_dir = output if os.path.isdir(output) or not output.endswith('.dat') else os.path.dirname(output) or '.'
        prefix = 'TE_data'
        write_multi_file(output_dir, prefix, result)
    else:
        # Single file format
        write_single_file(output, result, include_header=not no_header)
        if not quiet:
            print(f"Written: {output}")

    if not quiet:
        print()
        print(f"Simulation complete:")
        print(f"  Samples recorded: {len(result.time)}")
        print(f"  Time range: 0 - {result.time[-1]*3600:.0f} seconds ({result.time[-1]:.2f} hours)")
        if result.shutdown:
            print(f"  Status: SHUTDOWN")
        else:
            print(f"  Status: Normal")

    # Plot results if requested
    if plot or plot_save:
        plot_results(result, faults=faults, save_path=plot_save)

    return 0


def main():
    """Entry point for the TEP batch simulation CLI."""
    parser = argparse.ArgumentParser(
        description='Tennessee Eastman Process Batch Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 8-hour normal operation simulation
  tep-sim --duration 8 --output normal.dat

  # Run with fault 1 starting at 1 hour
  tep-sim --duration 8 --faults 1 --fault-times 1.0 --output fault1.dat

  # Multiple faults with different start times
  tep-sim --duration 8 --faults 1,4,7 --fault-times 1.0,2.0,3.0 --output multi_fault.dat

  # Use specific random seed for reproducibility
  tep-sim --duration 8 --seed 12345 --output reproducible.dat

  # Output in original Fortran multi-file format
  tep-sim --duration 8 --faults 1 --multi-file --output ./data/

  # Display results graphically
  tep-sim --duration 2 --faults 1 --plot

  # Save plot to file
  tep-sim --duration 2 --faults 1 --plot-save results.png

  # List available faults
  tep-sim --list-faults
        """
    )

    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=8.0,
        help='Simulation duration in hours (default: 8.0)'
    )

    parser.add_argument(
        '--faults', '-f',
        type=str,
        default=None,
        help='Fault IDs to activate (e.g., "1", "1,2,5", "1-5", "1,3-5,7")'
    )

    parser.add_argument(
        '--fault-times', '-t',
        type=str,
        default='1.0',
        help='Fault start times in hours, comma-separated (default: 1.0)'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED})'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='tep_data.dat',
        help='Output file path (default: tep_data.dat)'
    )

    parser.add_argument(
        '--multi-file', '-m',
        action='store_true',
        help='Output in original Fortran multi-file format (15 separate .dat files)'
    )

    parser.add_argument(
        '--record-interval', '-r',
        type=int,
        default=180,
        help='Recording interval in steps/seconds (default: 180 = every 3 minutes)'
    )

    parser.add_argument(
        '--no-header',
        action='store_true',
        help='Omit column headers in single-file output'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    parser.add_argument(
        '--list-faults',
        action='store_true',
        help='List available faults and exit'
    )

    parser.add_argument(
        '--controller', '-c',
        type=str,
        default=None,
        metavar='NAME',
        help='Controller plugin to use (default: decentralized). Use --list-controllers to see available options.'
    )

    parser.add_argument(
        '--list-controllers',
        action='store_true',
        help='List available controller plugins and exit'
    )

    parser.add_argument(
        '--mode',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help='Operating mode 1-6 (default: 1). Modes: 1=50/50 base, 2=10/90 base, 3=90/10 base, 4=50/50 max, 5=10/90 max, 6=90/10 max'
    )

    parser.add_argument(
        '--plot', '-p',
        action='store_true',
        help='Display results graphically (requires matplotlib)'
    )

    parser.add_argument(
        '--plot-save',
        type=str,
        default=None,
        metavar='FILE',
        help='Save plot to file instead of displaying (e.g., results.png)'
    )

    args = parser.parse_args()

    # Handle --list-faults
    if args.list_faults:
        list_faults()
        return 0

    # Handle --list-controllers
    if args.list_controllers:
        list_controllers()
        return 0

    # Parse faults
    faults = None
    fault_times = None
    if args.faults:
        try:
            faults = parse_faults(args.faults)
            fault_times = parse_fault_times(args.fault_times, len(faults))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Run simulation
    try:
        return run_simulation(
            duration_hours=args.duration,
            faults=faults,
            fault_times=fault_times,
            seed=args.seed,
            output=args.output,
            multi_file=args.multi_file,
            record_interval=args.record_interval,
            no_header=args.no_header,
            quiet=args.quiet,
            plot=args.plot,
            plot_save=args.plot_save,
            controller=args.controller,
            mode=args.mode
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
