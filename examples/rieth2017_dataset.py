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
    HAS_TEP = True
except ImportError:
    HAS_TEP = False

# Optional dependencies for downloading/comparing with Harvard Dataverse
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pyreadr
    HAS_PYREADR = True
except ImportError:
    HAS_PYREADR = False


# Dataset parameters matching Rieth et al. 2017
RIETH_PARAMS = {
    "n_simulations": 500,           # Simulations per fault type
    "train_duration_hours": 25.0,   # Training simulation duration
    "val_duration_hours": 48.0,     # Validation simulation duration
    "test_duration_hours": 48.0,    # Testing simulation duration
    "sampling_interval_min": 3,     # 3-minute sampling interval
    "fault_onset_hours": 1.0,       # Fault introduced at 1 hour (val/test only)
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

# Harvard Dataverse file IDs for the original dataset
HARVARD_DATAVERSE_FILES = {
    "fault_free_training": {
        "id": "3364637",
        "filename": "TEP_FaultFree_Training.RData",
        "var_name": "fault_free_training",
    },
    "fault_free_testing": {
        "id": "3364636",
        "filename": "TEP_FaultFree_Testing.RData",
        "var_name": "fault_free_testing",
    },
    "faulty_training": {
        "id": "3364635",
        "filename": "TEP_Faulty_Training.RData",
        "var_name": "faulty_training",
    },
    "faulty_testing": {
        "id": "3364634",
        "filename": "TEP_Faulty_Testing.RData",
        "var_name": "faulty_testing",
    },
}

# Variable names for comparison
VARIABLE_NAMES = (
    ["faultNumber", "simulationRun", "sample"]
    + [f"xmeas_{i}" for i in range(1, 42)]
    + [f"xmv_{i}" for i in range(1, 12)]
)

# Key variables for comparison (indices into feature columns, 0-indexed)
KEY_VARIABLES = {
    "Reactor Temperature": 8,      # XMEAS(9)
    "Reactor Pressure": 6,         # XMEAS(7)
    "Reactor Level": 7,            # XMEAS(8)
    "Separator Temperature": 10,   # XMEAS(11)
    "Separator Level": 11,         # XMEAS(12)
    "Stripper Level": 14,          # XMEAS(15)
    "Compressor Work": 19,         # XMEAS(20)
    "D Feed Flow (MV)": 41,        # XMV(1)
    "Reactor CW Flow (MV)": 50,    # XMV(10)
}


class HarvardDataverseDataset:
    """
    Download and load the original Rieth 2017 dataset from Harvard Dataverse.

    This class provides access to the original dataset for comparison with
    locally generated data.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory to store downloaded files.

    Examples
    --------
    >>> harvard = HarvardDataverseDataset()
    >>> harvard.download()
    >>> df = harvard.load("fault_free_training")
    """

    DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile"

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data" / "harvard_dataverse"
        self.data_dir = Path(data_dir)
        self._cache = {}

    def download(self, files: Optional[List[str]] = None, force: bool = False) -> None:
        """
        Download dataset files from Harvard Dataverse.

        Parameters
        ----------
        files : list of str, optional
            Which files to download. Options: fault_free_training,
            fault_free_testing, faulty_training, faulty_testing.
            Default: all files.
        force : bool
            Re-download even if files exist.
        """
        if not HAS_REQUESTS:
            raise ImportError(
                "requests library required for download. "
                "Install with: pip install requests"
            )

        self.data_dir.mkdir(parents=True, exist_ok=True)
        files = files or list(HARVARD_DATAVERSE_FILES.keys())

        for name in files:
            if name not in HARVARD_DATAVERSE_FILES:
                print(f"Unknown file: {name}, skipping...")
                continue

            info = HARVARD_DATAVERSE_FILES[name]
            filepath = self.data_dir / info["filename"]

            if filepath.exists() and not force:
                print(f"  {info['filename']} already exists, skipping...")
                continue

            print(f"Downloading {info['filename']} from Harvard Dataverse...")
            url = f"{self.DATAVERSE_URL}/{info['id']}"

            try:
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

                print(f"\n  Saved: {filepath}")

            except requests.RequestException as e:
                print(f"  Error downloading {name}: {e}")

    def load(self, name: str) -> np.ndarray:
        """
        Load a dataset file as numpy array.

        Parameters
        ----------
        name : str
            Dataset name: fault_free_training, fault_free_testing,
            faulty_training, or faulty_testing.

        Returns
        -------
        np.ndarray
            Dataset array with shape (n_rows, 55)
        """
        if not HAS_PYREADR:
            raise ImportError(
                "pyreadr library required to load RData files. "
                "Install with: pip install pyreadr"
            )

        if name in self._cache:
            return self._cache[name]

        if name not in HARVARD_DATAVERSE_FILES:
            raise ValueError(f"Unknown dataset: {name}")

        info = HARVARD_DATAVERSE_FILES[name]
        filepath = self.data_dir / info["filename"]

        if not filepath.exists():
            raise FileNotFoundError(
                f"File not found: {filepath}\n"
                "Run harvard.download() first."
            )

        print(f"Loading {info['filename']}...")
        result = pyreadr.read_r(str(filepath))

        # Get the dataframe from the RData file
        df = list(result.values())[0]

        # Convert to numpy array
        data = df.values
        self._cache[name] = data

        print(f"  Loaded shape: {data.shape}")
        return data

    def load_all(self) -> Dict[str, np.ndarray]:
        """Load all available dataset files."""
        data = {}
        for name in HARVARD_DATAVERSE_FILES:
            try:
                data[name] = self.load(name)
            except FileNotFoundError:
                print(f"  {name}: not downloaded")
        return data


def compare_datasets(
    generated: np.ndarray,
    original: np.ndarray,
    name: str = "Dataset",
    fault_numbers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compare generated dataset with original Harvard Dataverse dataset.

    Parameters
    ----------
    generated : np.ndarray
        Locally generated dataset (n_rows, 55)
    original : np.ndarray
        Original Harvard Dataverse dataset (n_rows, 55)
    name : str
        Name for the comparison report
    fault_numbers : list of int, optional
        Specific faults to compare (default: all available)

    Returns
    -------
    dict
        Comparison results with statistics for each variable
    """
    print(f"\n{'='*70}")
    print(f"Dataset Comparison: {name}")
    print(f"{'='*70}")

    print(f"\nShape comparison:")
    print(f"  Generated: {generated.shape}")
    print(f"  Original:  {original.shape}")

    # Get unique fault numbers
    gen_faults = np.unique(generated[:, 0]).astype(int)
    orig_faults = np.unique(original[:, 0]).astype(int)

    if fault_numbers is None:
        fault_numbers = list(set(gen_faults) & set(orig_faults))

    print(f"\nFaults in generated: {list(gen_faults)}")
    print(f"Faults in original:  {list(orig_faults)}")
    print(f"Comparing faults:    {fault_numbers}")

    results = {"name": name, "faults": {}}

    for fault_num in sorted(fault_numbers):
        gen_mask = generated[:, 0] == fault_num
        orig_mask = original[:, 0] == fault_num

        gen_data = generated[gen_mask]
        orig_data = original[orig_mask]

        if len(gen_data) == 0 or len(orig_data) == 0:
            print(f"\n  Fault {fault_num}: insufficient data, skipping")
            continue

        print(f"\n  Fault {fault_num}: {FAULT_DESCRIPTIONS.get(fault_num, 'Unknown')}")
        print(f"    Generated samples: {len(gen_data)}")
        print(f"    Original samples:  {len(orig_data)}")

        fault_results = {"n_generated": len(gen_data), "n_original": len(orig_data), "variables": {}}

        # Compare key variables
        print(f"\n    {'Variable':<25} {'Gen Mean':>10} {'Orig Mean':>10} {'Diff %':>10} {'Gen Std':>10} {'Orig Std':>10}")
        print(f"    {'-'*75}")

        for var_name, var_idx in KEY_VARIABLES.items():
            col_idx = var_idx + 3  # Offset for faultNumber, simulationRun, sample

            gen_vals = gen_data[:, col_idx]
            orig_vals = orig_data[:, col_idx]

            gen_mean = np.mean(gen_vals)
            orig_mean = np.mean(orig_vals)
            gen_std = np.std(gen_vals)
            orig_std = np.std(orig_vals)

            if abs(orig_mean) > 1e-6:
                diff_pct = ((gen_mean - orig_mean) / orig_mean) * 100
            else:
                diff_pct = 0.0 if abs(gen_mean) < 1e-6 else float('inf')

            print(f"    {var_name:<25} {gen_mean:>10.2f} {orig_mean:>10.2f} {diff_pct:>+10.1f}% {gen_std:>10.2f} {orig_std:>10.2f}")

            fault_results["variables"][var_name] = {
                "gen_mean": float(gen_mean),
                "orig_mean": float(orig_mean),
                "diff_pct": float(diff_pct),
                "gen_std": float(gen_std),
                "orig_std": float(orig_std),
            }

        results["faults"][fault_num] = fault_results

    # Overall summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    all_gen_features = generated[:, 3:]
    all_orig_features = original[:, 3:]

    # Compute correlation between means
    gen_means = np.mean(all_gen_features, axis=0)
    orig_means = np.mean(all_orig_features, axis=0)

    correlation = np.corrcoef(gen_means, orig_means)[0, 1]
    print(f"\nOverall mean correlation: {correlation:.4f}")

    # Mean absolute percentage error
    valid_mask = np.abs(orig_means) > 1e-6
    mape = np.mean(np.abs((gen_means[valid_mask] - orig_means[valid_mask]) / orig_means[valid_mask])) * 100
    print(f"Mean absolute % error:    {mape:.2f}%")

    results["summary"] = {
        "mean_correlation": float(correlation),
        "mape": float(mape),
    }

    return results


def compare_with_harvard(
    local_dir: Optional[str] = None,
    harvard_dir: Optional[str] = None,
    datasets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare locally generated dataset with Harvard Dataverse original.

    Parameters
    ----------
    local_dir : str, optional
        Directory containing generated data
    harvard_dir : str, optional
        Directory containing Harvard Dataverse data
    datasets : list of str, optional
        Which datasets to compare (default: all available)

    Returns
    -------
    dict
        Full comparison results
    """
    if local_dir is None:
        local_dir = Path(__file__).parent.parent / "data" / "rieth2017"
    local_dir = Path(local_dir)

    # Load local data
    print("Loading locally generated data...")
    local_data = load_rieth2017_dataset(str(local_dir))

    if not local_data:
        print("No local data found. Generate data first with --small or --full")
        return {}

    # Download and load Harvard data
    print("\nLoading Harvard Dataverse data...")
    harvard = HarvardDataverseDataset(data_dir=harvard_dir)

    # Download files that correspond to local data
    available_local = list(local_data.keys())
    if datasets:
        to_download = [d for d in datasets if d in available_local]
    else:
        to_download = available_local

    harvard.download(files=to_download)

    # Run comparisons
    all_results = {}

    for dataset_name in to_download:
        if dataset_name not in local_data:
            continue

        try:
            harvard_data = harvard.load(dataset_name)
            results = compare_datasets(
                local_data[dataset_name],
                harvard_data,
                name=dataset_name,
            )
            all_results[dataset_name] = results
        except (FileNotFoundError, ImportError) as e:
            print(f"Could not compare {dataset_name}: {e}")

    return all_results


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

        # Separate seed ranges for training, validation, and testing (non-overlapping)
        self.train_seed_base = seed_offset
        self.val_seed_base = seed_offset + 1000000
        self.test_seed_base = seed_offset + 2000000

    def _get_seed(
        self,
        simulation_run: int,
        split: Literal["train", "val", "test"],
        fault_number: int,
    ) -> int:
        """Generate unique seed for a simulation run."""
        if split == "train":
            base = self.train_seed_base
        elif split == "val":
            base = self.val_seed_base
        else:
            base = self.test_seed_base
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
        if not HAS_TEP:
            raise ImportError("TEP simulator not available. Install with: pip install -e .")

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
            seed = self._get_seed(sim_run, split="train", fault_number=0)

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
            seed = self._get_seed(sim_run, split="test", fault_number=0)

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

    def generate_fault_free_validation(
        self,
        n_simulations: Optional[int] = None,
        save: bool = True,
    ) -> np.ndarray:
        """
        Generate fault-free validation data.

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
        duration = RIETH_PARAMS["val_duration_hours"]

        print(f"Generating fault-free validation data ({n_sims} simulations)...")
        print(f"  Duration: {duration} hours")

        all_data = []

        for sim_run in range(1, n_sims + 1):
            seed = self._get_seed(sim_run, split="val", fault_number=0)

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
            self._save_data(data_array, "fault_free_validation.npy")

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
                seed = self._get_seed(sim_run, split="train", fault_number=fault_num)

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
                seed = self._get_seed(sim_run, split="test", fault_number=fault_num)

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

    def generate_faulty_validation(
        self,
        fault_numbers: Optional[List[int]] = None,
        n_simulations: Optional[int] = None,
        save: bool = True,
    ) -> np.ndarray:
        """
        Generate faulty validation data.

        In the validation set, faults are introduced at 1 hour (same as testing).

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
        duration = RIETH_PARAMS["val_duration_hours"]
        fault_onset = RIETH_PARAMS["fault_onset_hours"]

        print(f"Generating faulty validation data...")
        print(f"  Faults: {fault_nums}")
        print(f"  Simulations per fault: {n_sims}")
        print(f"  Duration: {duration} hours")
        print(f"  Fault onset: {fault_onset} hour")

        all_data = []

        for fault_num in fault_nums:
            print(f"\nFault {fault_num}: {FAULT_DESCRIPTIONS.get(fault_num, 'Unknown')}")

            shutdown_count = 0

            for sim_run in range(1, n_sims + 1):
                seed = self._get_seed(sim_run, split="val", fault_number=fault_num)

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
            self._save_data(data_array, "faulty_validation.npy")

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
        include_validation: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate complete dataset (6 files with validation, or 4 without).

        Parameters
        ----------
        n_simulations : int, optional
            Simulations per fault (default: 500)
        fault_numbers : list of int, optional
            Fault numbers to include (default: 1-20)
        include_validation : bool
            Whether to generate validation sets (default: True)

        Returns
        -------
        dict
            Dictionary with keys: fault_free_training, fault_free_validation,
            fault_free_testing, faulty_training, faulty_validation, faulty_testing
        """
        print("=" * 60)
        print("Rieth 2017 TEP Dataset Generation")
        print("=" * 60)
        print()

        results = {}

        results["fault_free_training"] = self.generate_fault_free_training(n_simulations)
        print()

        if include_validation:
            results["fault_free_validation"] = self.generate_fault_free_validation(n_simulations)
            print()

        results["fault_free_testing"] = self.generate_fault_free_testing(n_simulations)
        print()

        results["faulty_training"] = self.generate_faulty_training(fault_numbers, n_simulations)
        print()

        if include_validation:
            results["faulty_validation"] = self.generate_faulty_validation(fault_numbers, n_simulations)
            print()

        results["faulty_testing"] = self.generate_faulty_testing(fault_numbers, n_simulations)

        # Save metadata
        self._save_metadata(n_simulations, fault_numbers, include_validation)

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
        include_validation: bool = True,
    ) -> None:
        """Save dataset metadata."""
        files = {
            "fault_free_training.npy": "Normal operation training data",
            "fault_free_testing.npy": "Normal operation testing data",
            "faulty_training.npy": "Faulty training data (fault active from t=0)",
            "faulty_testing.npy": "Faulty testing data (fault at t=1h)",
        }

        if include_validation:
            files["fault_free_validation.npy"] = "Normal operation validation data"
            files["faulty_validation.npy"] = "Faulty validation data (fault at t=1h)"

        metadata = {
            "description": "TEP dataset matching Rieth et al. 2017 specifications (extended with validation set)",
            "reference": {
                "authors": "Rieth, C.A., Amsel, B.D., Tran, R., Cook, M.B.",
                "title": "Issues and Advances in Anomaly Detection Evaluation for Joint Human-Automated Systems",
                "year": 2017,
                "doi": "10.1007/978-3-319-60384-1_6",
            },
            "parameters": {
                "n_simulations": n_simulations or self.n_simulations,
                "train_duration_hours": RIETH_PARAMS["train_duration_hours"],
                "val_duration_hours": RIETH_PARAMS["val_duration_hours"],
                "test_duration_hours": RIETH_PARAMS["test_duration_hours"],
                "sampling_interval_min": RIETH_PARAMS["sampling_interval_min"],
                "fault_onset_hours": RIETH_PARAMS["fault_onset_hours"],
                "fault_numbers": fault_numbers or list(range(1, 21)),
                "include_validation": include_validation,
            },
            "columns": {
                "0": "faultNumber",
                "1": "simulationRun",
                "2": "sample",
                "3-43": "xmeas_1 to xmeas_41 (41 measured variables)",
                "44-54": "xmv_1 to xmv_11 (11 manipulated variables)",
            },
            "files": files,
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
        "fault_free_validation": "fault_free_validation.npy",
        "fault_free_testing": "fault_free_testing.npy",
        "faulty_training": "faulty_training.npy",
        "faulty_validation": "faulty_validation.npy",
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


def example_compare_with_harvard(local_dir: Optional[str] = None):
    """Example: Compare generated dataset with Harvard Dataverse original."""
    print("Example: Compare with Harvard Dataverse Dataset")
    print("=" * 60)
    print()
    print("This compares locally generated data with the original Rieth et al.")
    print("2017 dataset from Harvard Dataverse (https://doi.org/10.7910/DVN/6C3JR1)")
    print()

    # Check dependencies
    if not HAS_REQUESTS:
        print("ERROR: 'requests' library required for download.")
        print("Install with: pip install requests")
        return

    if not HAS_PYREADR:
        print("ERROR: 'pyreadr' library required to load RData files.")
        print("Install with: pip install pyreadr")
        return

    results = compare_with_harvard(local_dir=local_dir)

    if results:
        # Save comparison results
        output_file = Path(local_dir or "./data/rieth2017") / "comparison_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def example_download_harvard_only():
    """Example: Just download the Harvard Dataverse dataset without comparison."""
    print("Example: Download Harvard Dataverse Dataset")
    print("=" * 60)
    print()
    print("Downloading original Rieth et al. 2017 dataset from Harvard Dataverse...")
    print("DOI: https://doi.org/10.7910/DVN/6C3JR1")
    print()

    if not HAS_REQUESTS:
        print("ERROR: 'requests' library required for download.")
        print("Install with: pip install requests")
        return

    harvard = HarvardDataverseDataset()
    harvard.download()

    print()
    print(f"Files downloaded to: {harvard.data_dir}")


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
        "--compare",
        action="store_true",
        help="Compare generated data with Harvard Dataverse original",
    )
    parser.add_argument(
        "--download-harvard",
        action="store_true",
        help="Download original dataset from Harvard Dataverse",
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
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip generating validation sets (only train/test)",
    )

    args = parser.parse_args()

    if args.compare:
        example_compare_with_harvard(local_dir=args.output_dir)
    elif args.download_harvard:
        example_download_harvard_only()
    elif args.analyze:
        example_load_and_analyze()
    elif args.full:
        example_generate_full()
    elif args.small:
        example_generate_small()
    elif args.n_simulations or args.faults or args.output_dir or args.no_validation:
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
            include_validation=not args.no_validation,
        )
    else:
        # Default: show help
        print("Rieth et al. 2017 TEP Dataset Generator")
        print("=" * 60)
        print()
        print("This script generates TEP datasets matching the specifications")
        print("of Rieth et al. (2017) using the local TEP simulator.")
        print()
        print("Output files (6 total with validation):")
        print("  - fault_free_training.npy    (25h, normal, fault from t=0)")
        print("  - fault_free_validation.npy  (48h, normal, fault at t=1h)")
        print("  - fault_free_testing.npy     (48h, normal, fault at t=1h)")
        print("  - faulty_training.npy        (25h, faulty, fault from t=0)")
        print("  - faulty_validation.npy      (48h, faulty, fault at t=1h)")
        print("  - faulty_testing.npy         (48h, faulty, fault at t=1h)")
        print()
        print("Generate data:")
        print("  python rieth2017_dataset.py --small    # Quick test (5 sims)")
        print("  python rieth2017_dataset.py --full     # Full dataset (500 sims)")
        print()
        print("Custom generation:")
        print("  python rieth2017_dataset.py --n-simulations 100 --faults 1,2,4,6")
        print("  python rieth2017_dataset.py --no-validation  # Skip validation sets")
        print()
        print("Compare with original:")
        print("  python rieth2017_dataset.py --download-harvard  # Download original")
        print("  python rieth2017_dataset.py --compare           # Compare datasets")
        print()
        print("Analyze:")
        print("  python rieth2017_dataset.py --analyze  # Analyze existing data")
        print()
        print("Requirements for comparison:")
        print("  pip install requests pyreadr")


if __name__ == "__main__":
    main()
