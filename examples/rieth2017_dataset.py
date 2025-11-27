#!/usr/bin/env python3
"""
Rieth et al. 2017 TEP Dataset Generator

This script reproduces the Tennessee Eastman Process dataset published by
Rieth et al. (2017) using the local TEP simulator.

The generated dataset matches the structure and parameters of the original:
- 500 simulations per fault type
- 25 hours training data / 48 hours testing data
- 3-minute sampling interval
- 20 fault types + normal operation

Original Dataset DOI: https://doi.org/10.7910/DVN/6C3JR1

Citation:
    Rieth, C.A., Amsel, B.D., Tran, R., Cook, M.B. (2018). Issues and Advances
    in Anomaly Detection Evaluation for Joint Human-Automated Systems.
    Advances in Intelligent Systems and Computing, vol 595, pp. 52-63. Springer.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Literal, List, Dict, Any
import json

import numpy as np

try:
    from tep import TEPSimulator
    from tep.simulator import ControlMode
except ImportError:
    print("Error: tep package not installed. Run: pip install -e .")
    sys.exit(1)


# Dataset parameters matching Rieth et al. 2017
RIETH_PARAMS = {
    "n_simulations": 500,           # Simulations per fault type
    "train_duration_hours": 25.0,   # Training simulation duration
    "test_duration_hours": 48.0,    # Testing simulation duration
    "sampling_interval_min": 3,     # 3-minute sampling interval
    "fault_onset_hours": 1.0,       # Fault introduced at 1 hour (test only)
    "n_faults": 20,                 # Number of fault types
}

# Fault descriptions from the TEP
FAULT_DESCRIPTIONS = {
    0: "Normal operation (no fault)",
    1: "A/C feed ratio, B composition constant (Stream 4) - Step",
    2: "B composition, A/C ratio constant (Stream 4) - Step",
    3: "D feed temperature (Stream 2) - Step",
    4: "Reactor cooling water inlet temperature - Step",
    5: "Condenser cooling water inlet temperature - Step",
    6: "A feed loss (Stream 1) - Step",
    7: "C header pressure loss (Stream 4) - Step",
    8: "A, B, C feed composition (Stream 4) - Random",
    9: "D feed temperature (Stream 2) - Random",
    10: "C feed temperature (Stream 4) - Random",
    11: "Reactor cooling water inlet temperature - Random",
    12: "Condenser cooling water inlet temperature - Random",
    13: "Reaction kinetics - Slow drift",
    14: "Reactor cooling water valve - Sticking",
    15: "Condenser cooling water valve - Sticking",
    16: "Unknown fault 16",
    17: "Unknown fault 17",
    18: "Unknown fault 18",
    19: "Unknown fault 19",
    20: "Unknown fault 20",
}


class Rieth2017DatasetGenerator:
    """
    Generate TEP dataset matching Rieth et al. 2017 specifications.

    This class generates datasets with 500 simulations per fault type,
    using independent random seeds for each simulation.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save generated data files.
    n_simulations : int
        Number of simulations per fault type (default: 500).
    seed_offset : int
        Base seed offset for reproducibility.

    Examples
    --------
    >>> generator = Rieth2017DatasetGenerator(output_dir="./data/rieth2017")
    >>> generator.generate_fault_free_training()
    >>> generator.generate_faulty_testing(fault_numbers=[1, 2, 3, 4])
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        n_simulations: int = 500,
        seed_offset: int = 1000000,
    ):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "data" / "rieth2017"
        self.output_dir = Path(output_dir)
        self.n_simulations = n_simulations
        self.seed_offset = seed_offset

        # Separate seed ranges for training and testing (non-overlapping)
        self.train_seed_base = seed_offset
        self.test_seed_base = seed_offset + 1000000

    def _get_seed(self, simulation_run: int, is_training: bool, fault_number: int) -> int:
        """Generate unique seed for a simulation run."""
        base = self.train_seed_base if is_training else self.test_seed_base
        # Unique seed: base + (fault * 1000) + simulation_run
        return base + (fault_number * 1000) + simulation_run

    def _run_simulation(
        self,
        seed: int,
        duration_hours: float,
        fault_number: int = 0,
        fault_onset_hours: float = 1.0,
    ) -> dict:
        """
        Run a single TEP simulation.

        Returns
        -------
        dict
            Simulation results with measurements and MVs
        """
        sim = TEPSimulator(random_seed=seed, control_mode=ControlMode.CLOSED_LOOP)
        sim.initialize()

        # Calculate record interval for 3-minute sampling
        # dt_hours = 1/3600 (1 second), so 3 minutes = 180 seconds = 180 steps
        record_interval = 180

        # Set up disturbance if fault > 0
        disturbances = None
        if fault_number > 0:
            disturbances = {fault_number: (fault_onset_hours, 1)}

        try:
            result = sim.simulate(
                duration_hours=duration_hours,
                record_interval=record_interval,
                disturbances=disturbances,
            )

            # Combine measurements (41) + MVs (11) = 52 columns
            data = np.hstack([
                result.measurements,
                result.manipulated_vars[:, :11]  # Exclude XMV(12) like original
            ])

            return {
                "data": data,
                "time": result.time,
                "shutdown": result.shutdown,
                "shutdown_time": result.shutdown_time,
            }

        except Exception as e:
            print(f"  Warning: Simulation failed with seed {seed}: {e}")
            return None

    def generate_fault_free_training(
        self,
        n_simulations: Optional[int] = None,
        save: bool = True,
    ) -> np.ndarray:
        """
        Generate fault-free training data.

        Parameters
        ----------
        n_simulations : int, optional
            Number of simulations (default: 500)
        save : bool
            Whether to save to file

        Returns
        -------
        np.ndarray
            Array of shape (n_simulations * n_samples, 55)
            Columns: faultNumber, simulationRun, sample, xmeas_1..41, xmv_1..11
        """
        n_sims = n_simulations or self.n_simulations
        duration = RIETH_PARAMS["train_duration_hours"]

        print(f"Generating fault-free training data ({n_sims} simulations)...")
        print(f"  Duration: {duration} hours")
        print(f"  Sampling: {RIETH_PARAMS['sampling_interval_min']} minutes")

        all_data = []

        for sim_run in range(1, n_sims + 1):
            seed = self._get_seed(sim_run, is_training=True, fault_number=0)

            if sim_run % 50 == 0 or sim_run == 1:
                print(f"  Simulation {sim_run}/{n_sims}...")

            result = self._run_simulation(seed, duration, fault_number=0)

            if result is None:
                continue

            n_samples = result["data"].shape[0]

            # Build full row: [faultNumber, simulationRun, sample, features...]
            for sample_idx in range(n_samples):
                row = np.zeros(55)
                row[0] = 0                          # faultNumber
                row[1] = sim_run                    # simulationRun
                row[2] = sample_idx + 1             # sample (1-indexed)
                row[3:] = result["data"][sample_idx]  # xmeas + xmv
                all_data.append(row)

        data_array = np.array(all_data)
        print(f"  Generated {len(data_array)} rows")

        if save:
            self._save_data(data_array, "fault_free_training.npy")

        return data_array

    def generate_fault_free_testing(
        self,
        n_simulations: Optional[int] = None,
        save: bool = True,
    ) -> np.ndarray:
        """
        Generate fault-free testing data.

        Parameters
        ----------
        n_simulations : int, optional
            Number of simulations (default: 500)
        save : bool
            Whether to save to file

        Returns
        -------
        np.ndarray
            Array of shape (n_simulations * n_samples, 55)
        """
        n_sims = n_simulations or self.n_simulations
        duration = RIETH_PARAMS["test_duration_hours"]

        print(f"Generating fault-free testing data ({n_sims} simulations)...")
        print(f"  Duration: {duration} hours")

        all_data = []

        for sim_run in range(1, n_sims + 1):
            seed = self._get_seed(sim_run, is_training=False, fault_number=0)

            if sim_run % 50 == 0 or sim_run == 1:
                print(f"  Simulation {sim_run}/{n_sims}...")

            result = self._run_simulation(seed, duration, fault_number=0)

            if result is None:
                continue

            n_samples = result["data"].shape[0]

            for sample_idx in range(n_samples):
                row = np.zeros(55)
                row[0] = 0
                row[1] = sim_run
                row[2] = sample_idx + 1
                row[3:] = result["data"][sample_idx]
                all_data.append(row)

        data_array = np.array(all_data)
        print(f"  Generated {len(data_array)} rows")

        if save:
            self._save_data(data_array, "fault_free_testing.npy")

        return data_array

    def generate_faulty_training(
        self,
        fault_numbers: Optional[List[int]] = None,
        n_simulations: Optional[int] = None,
        save: bool = True,
    ) -> np.ndarray:
        """
        Generate faulty training data.

        In the training set, faults are active from the beginning.

        Parameters
        ----------
        fault_numbers : list of int, optional
            Fault numbers to generate (default: 1-20)
        n_simulations : int, optional
            Simulations per fault (default: 500)
        save : bool
            Whether to save to file

        Returns
        -------
        np.ndarray
            Array of shape (n_faults * n_simulations * n_samples, 55)
        """
        fault_nums = fault_numbers or list(range(1, RIETH_PARAMS["n_faults"] + 1))
        n_sims = n_simulations or self.n_simulations
        duration = RIETH_PARAMS["train_duration_hours"]

        print(f"Generating faulty training data...")
        print(f"  Faults: {fault_nums}")
        print(f"  Simulations per fault: {n_sims}")
        print(f"  Duration: {duration} hours")

        all_data = []

        for fault_num in fault_nums:
            print(f"\nFault {fault_num}: {FAULT_DESCRIPTIONS.get(fault_num, 'Unknown')}")

            shutdown_count = 0

            for sim_run in range(1, n_sims + 1):
                seed = self._get_seed(sim_run, is_training=True, fault_number=fault_num)

                if sim_run % 100 == 0 or sim_run == 1:
                    print(f"  Simulation {sim_run}/{n_sims}...")

                # In training, fault is active from t=0
                result = self._run_simulation(
                    seed, duration, fault_number=fault_num, fault_onset_hours=0.0
                )

                if result is None:
                    continue

                if result["shutdown"]:
                    shutdown_count += 1

                n_samples = result["data"].shape[0]

                for sample_idx in range(n_samples):
                    row = np.zeros(55)
                    row[0] = fault_num
                    row[1] = sim_run
                    row[2] = sample_idx + 1
                    row[3:] = result["data"][sample_idx]
                    all_data.append(row)

            if shutdown_count > 0:
                print(f"  Shutdowns: {shutdown_count}/{n_sims}")

        data_array = np.array(all_data)
        print(f"\nTotal rows generated: {len(data_array)}")

        if save:
            self._save_data(data_array, "faulty_training.npy")

        return data_array

    def generate_faulty_testing(
        self,
        fault_numbers: Optional[List[int]] = None,
        n_simulations: Optional[int] = None,
        save: bool = True,
    ) -> np.ndarray:
        """
        Generate faulty testing data.

        In the testing set, faults are introduced at 1 hour.

        Parameters
        ----------
        fault_numbers : list of int, optional
            Fault numbers to generate (default: 1-20)
        n_simulations : int, optional
            Simulations per fault (default: 500)
        save : bool
            Whether to save to file

        Returns
        -------
        np.ndarray
            Array of shape (n_faults * n_simulations * n_samples, 55)
        """
        fault_nums = fault_numbers or list(range(1, RIETH_PARAMS["n_faults"] + 1))
        n_sims = n_simulations or self.n_simulations
        duration = RIETH_PARAMS["test_duration_hours"]
        fault_onset = RIETH_PARAMS["fault_onset_hours"]

        print(f"Generating faulty testing data...")
        print(f"  Faults: {fault_nums}")
        print(f"  Simulations per fault: {n_sims}")
        print(f"  Duration: {duration} hours")
        print(f"  Fault onset: {fault_onset} hour")

        all_data = []

        for fault_num in fault_nums:
            print(f"\nFault {fault_num}: {FAULT_DESCRIPTIONS.get(fault_num, 'Unknown')}")

            shutdown_count = 0

            for sim_run in range(1, n_sims + 1):
                seed = self._get_seed(sim_run, is_training=False, fault_number=fault_num)

                if sim_run % 100 == 0 or sim_run == 1:
                    print(f"  Simulation {sim_run}/{n_sims}...")

                result = self._run_simulation(
                    seed, duration, fault_number=fault_num, fault_onset_hours=fault_onset
                )

                if result is None:
                    continue

                if result["shutdown"]:
                    shutdown_count += 1

                n_samples = result["data"].shape[0]

                for sample_idx in range(n_samples):
                    row = np.zeros(55)
                    row[0] = fault_num
                    row[1] = sim_run
                    row[2] = sample_idx + 1
                    row[3:] = result["data"][sample_idx]
                    all_data.append(row)

            if shutdown_count > 0:
                print(f"  Shutdowns: {shutdown_count}/{n_sims}")

        data_array = np.array(all_data)
        print(f"\nTotal rows generated: {len(data_array)}")

        if save:
            self._save_data(data_array, "faulty_testing.npy")

        return data_array

    def _save_data(self, data: np.ndarray, filename: str) -> Path:
        """Save data to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.output_dir / filename
        np.save(filepath, data)
        print(f"  Saved: {filepath}")
        return filepath

    def generate_all(
        self,
        n_simulations: Optional[int] = None,
        fault_numbers: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate complete dataset (all 4 files).

        Parameters
        ----------
        n_simulations : int, optional
            Simulations per fault (default: 500)
        fault_numbers : list of int, optional
            Fault numbers to include (default: 1-20)

        Returns
        -------
        dict
            Dictionary with keys: fault_free_training, fault_free_testing,
            faulty_training, faulty_testing
        """
        print("=" * 60)
        print("Rieth 2017 TEP Dataset Generation")
        print("=" * 60)
        print()

        results = {}

        results["fault_free_training"] = self.generate_fault_free_training(n_simulations)
        print()

        results["fault_free_testing"] = self.generate_fault_free_testing(n_simulations)
        print()

        results["faulty_training"] = self.generate_faulty_training(fault_numbers, n_simulations)
        print()

        results["faulty_testing"] = self.generate_faulty_testing(fault_numbers, n_simulations)

        # Save metadata
        self._save_metadata(n_simulations, fault_numbers)

        print()
        print("=" * 60)
        print("Dataset generation complete!")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)

        return results

    def _save_metadata(
        self,
        n_simulations: Optional[int],
        fault_numbers: Optional[List[int]],
    ) -> None:
        """Save dataset metadata."""
        metadata = {
            "description": "TEP dataset matching Rieth et al. 2017 specifications",
            "reference": {
                "authors": "Rieth, C.A., Amsel, B.D., Tran, R., Cook, M.B.",
                "title": "Issues and Advances in Anomaly Detection Evaluation for Joint Human-Automated Systems",
                "year": 2017,
                "doi": "10.1007/978-3-319-60384-1_6",
            },
            "parameters": {
                "n_simulations": n_simulations or self.n_simulations,
                "train_duration_hours": RIETH_PARAMS["train_duration_hours"],
                "test_duration_hours": RIETH_PARAMS["test_duration_hours"],
                "sampling_interval_min": RIETH_PARAMS["sampling_interval_min"],
                "fault_onset_hours": RIETH_PARAMS["fault_onset_hours"],
                "fault_numbers": fault_numbers or list(range(1, 21)),
            },
            "columns": {
                "0": "faultNumber",
                "1": "simulationRun",
                "2": "sample",
                "3-43": "xmeas_1 to xmeas_41 (41 measured variables)",
                "44-54": "xmv_1 to xmv_11 (11 manipulated variables)",
            },
            "files": {
                "fault_free_training.npy": "Normal operation training data",
                "fault_free_testing.npy": "Normal operation testing data",
                "faulty_training.npy": "Faulty training data (fault active from t=0)",
                "faulty_testing.npy": "Faulty testing data (fault at t=1h)",
            },
        }

        filepath = self.output_dir / "metadata.json"
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: {filepath}")


def load_rieth2017_dataset(
    data_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Load previously generated Rieth 2017 dataset.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the data files

    Returns
    -------
    dict
        Dictionary with arrays for each data split
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "rieth2017"
    data_dir = Path(data_dir)

    files = {
        "fault_free_training": "fault_free_training.npy",
        "fault_free_testing": "fault_free_testing.npy",
        "faulty_training": "faulty_training.npy",
        "faulty_testing": "faulty_testing.npy",
    }

    data = {}
    for key, filename in files.items():
        filepath = data_dir / filename
        if filepath.exists():
            data[key] = np.load(filepath)
            print(f"Loaded {filename}: shape {data[key].shape}")
        else:
            print(f"File not found: {filepath}")

    return data


def get_fault_data(
    data: np.ndarray,
    fault_number: int,
    simulation_run: Optional[int] = None,
) -> np.ndarray:
    """
    Extract data for a specific fault from dataset array.

    Parameters
    ----------
    data : np.ndarray
        Full dataset array (n_rows, 55)
    fault_number : int
        Fault number (0-20)
    simulation_run : int, optional
        Specific simulation run (1-500)

    Returns
    -------
    np.ndarray
        Filtered data
    """
    mask = data[:, 0] == fault_number
    if simulation_run is not None:
        mask &= data[:, 1] == simulation_run
    return data[mask]


def get_features(data: np.ndarray) -> np.ndarray:
    """
    Extract feature columns (xmeas + xmv) from dataset.

    Parameters
    ----------
    data : np.ndarray
        Dataset array with shape (n_rows, 55)

    Returns
    -------
    np.ndarray
        Feature array with shape (n_rows, 52)
    """
    return data[:, 3:]  # Skip faultNumber, simulationRun, sample


# Example usage functions

def example_generate_small():
    """Example: Generate a small test dataset."""
    print("Example: Generate Small Test Dataset")
    print("=" * 60)

    generator = Rieth2017DatasetGenerator(
        output_dir="./data/rieth2017_small",
        n_simulations=5,  # Only 5 simulations for testing
    )

    # Generate only a few faults for quick testing
    generator.generate_fault_free_training(n_simulations=5)
    generator.generate_faulty_testing(fault_numbers=[1, 4, 6], n_simulations=5)


def example_generate_full():
    """Example: Generate full dataset (takes several hours)."""
    print("Example: Generate Full Rieth 2017 Dataset")
    print("=" * 60)
    print()
    print("WARNING: This will generate the full dataset with 500 simulations")
    print("per fault type. This may take several hours to complete.")
    print()

    generator = Rieth2017DatasetGenerator()
    generator.generate_all()


def example_load_and_analyze():
    """Example: Load and analyze generated dataset."""
    print("Example: Load and Analyze Dataset")
    print("=" * 60)

    # Load dataset
    data = load_rieth2017_dataset("./data/rieth2017_small")

    if "faulty_testing" not in data:
        print("Dataset not found. Run example_generate_small() first.")
        return

    faulty_test = data["faulty_testing"]
    print(f"\nFaulty testing data shape: {faulty_test.shape}")

    # Analyze fault 1
    fault1_data = get_fault_data(faulty_test, fault_number=1)
    fault1_features = get_features(fault1_data)

    print(f"\nFault 1 data:")
    print(f"  Rows: {len(fault1_data)}")
    print(f"  Simulations: {len(np.unique(fault1_data[:, 1]))}")
    print(f"  Samples per sim: {len(fault1_data) // len(np.unique(fault1_data[:, 1]))}")

    # Reactor temperature statistics
    reactor_temp_idx = 8  # XMEAS(9) is index 8 in features
    reactor_temp = fault1_features[:, reactor_temp_idx]
    print(f"\nReactor temperature (XMEAS 9):")
    print(f"  Mean: {reactor_temp.mean():.2f}")
    print(f"  Std:  {reactor_temp.std():.2f}")
    print(f"  Min:  {reactor_temp.min():.2f}")
    print(f"  Max:  {reactor_temp.max():.2f}")


def main():
    """Run examples."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate TEP dataset matching Rieth et al. 2017 specifications"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Generate full dataset (500 simulations per fault)",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Generate small test dataset (5 simulations)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Load and analyze existing dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=None,
        help="Number of simulations per fault type",
    )
    parser.add_argument(
        "--faults",
        type=str,
        default=None,
        help="Comma-separated fault numbers to generate (e.g., '1,2,4,6')",
    )

    args = parser.parse_args()

    if args.analyze:
        example_load_and_analyze()
    elif args.full:
        example_generate_full()
    elif args.small:
        example_generate_small()
    elif args.n_simulations or args.faults or args.output_dir:
        # Custom generation
        generator = Rieth2017DatasetGenerator(
            output_dir=args.output_dir,
            n_simulations=args.n_simulations or 500,
        )

        fault_numbers = None
        if args.faults:
            fault_numbers = [int(f.strip()) for f in args.faults.split(",")]

        generator.generate_all(
            n_simulations=args.n_simulations,
            fault_numbers=fault_numbers,
        )
    else:
        # Default: show help and run small example
        print("Rieth et al. 2017 TEP Dataset Generator")
        print("=" * 60)
        print()
        print("This script generates TEP datasets matching the specifications")
        print("of Rieth et al. (2017) using the local TEP simulator.")
        print()
        print("Usage:")
        print("  python rieth2017_dataset.py --small    # Quick test (5 sims)")
        print("  python rieth2017_dataset.py --full     # Full dataset (500 sims)")
        print("  python rieth2017_dataset.py --analyze  # Analyze existing data")
        print()
        print("Custom generation:")
        print("  python rieth2017_dataset.py --n-simulations 100 --faults 1,2,4,6")
        print()
        print("Running small example...")
        print()
        example_generate_small()


if __name__ == "__main__":
    main()
