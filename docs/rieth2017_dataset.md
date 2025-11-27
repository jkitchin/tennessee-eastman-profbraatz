# Rieth et al. 2017 TEP Dataset

This document describes the Tennessee Eastman Process dataset published by Rieth et al. (2017) for anomaly detection research.

## Overview

The Rieth et al. 2017 dataset addresses a critical limitation of previous TEP datasets: **they contained only a single simulation per fault type**, which can lead to biased evaluation results. This dataset provides **500 independent simulations per fault type** using non-overlapping random number generator seeds.

## Citation

```bibtex
@inproceedings{rieth2017issues,
  title={Issues and Advances in Anomaly Detection Evaluation for Joint Human-Automated Systems},
  author={Rieth, Christoph A. and Amsel, Ben D. and Tran, Randy and Cook, Maia B.},
  booktitle={Advances in Human Factors in Robots and Unmanned Systems},
  series={Advances in Intelligent Systems and Computing},
  volume={595},
  pages={52--63},
  year={2018},
  publisher={Springer},
  doi={10.1007/978-3-319-60384-1_6}
}
```

**Presented at:** AHFE 2017 (Applied Human Factors and Ergonomics Conference), July 17-21, 2017, Los Angeles, CA

**Sponsor:** Office of Naval Research (contract N00014-15-C-5003)

## Dataset Access

**Harvard Dataverse:** https://doi.org/10.7910/DVN/6C3JR1

**License:** Public Domain (CC0)

## Dataset Structure

### Files

The dataset consists of 4 RData files:

| File | Description |
|------|-------------|
| `fault_free_training.RData` | Normal operation training data |
| `fault_free_testing.RData` | Normal operation testing data |
| `faulty_training.RData` | Fault scenario training data |
| `faulty_testing.RData` | Fault scenario testing data |

### Column Structure (55 columns)

| Column | Name | Values | Description |
|--------|------|--------|-------------|
| 1 | `faultNumber` | 0-20 | Fault type (0 = normal) |
| 2 | `simulationRun` | 1-500 | Simulation run with unique seed |
| 3 | `sample` | 1-500 or 1-960 | Time sample index |
| 4-44 | `xmeas_1` to `xmeas_41` | float | 41 measured variables |
| 45-55 | `xmv_1` to `xmv_11` | float | 11 manipulated variables |

### Simulation Parameters

| Parameter | Training | Validation | Testing |
|-----------|----------|------------|---------|
| Duration | 25 hours | 48 hours | 48 hours |
| Sampling rate | 3 minutes | 3 minutes | 3 minutes |
| Samples per run | 500 | 960 | 960 |
| Simulations per fault | 500 | 500 | 500 |
| Operating mode | Mode 1 | Mode 1 | Mode 1 |
| Fault introduction | t=0 | t=1 hour | t=1 hour |

**Note:** The validation set is an extension to the original Rieth 2017 dataset for machine learning workflows requiring train/validation/test splits. It uses independent random seeds that do not overlap with training or testing data.

## Fault Types (IDV 1-20)

### Step Disturbances (IDV 1-7)

| IDV | Description |
|-----|-------------|
| 1 | A/C feed ratio, B composition constant (Stream 4) |
| 2 | B composition, A/C ratio constant (Stream 4) |
| 3 | D feed temperature (Stream 2) |
| 4 | Reactor cooling water inlet temperature |
| 5 | Condenser cooling water inlet temperature |
| 6 | A feed loss (Stream 1) |
| 7 | C header pressure loss - Loss of reactor feed (Stream 4) |

### Random Variation Disturbances (IDV 8-12)

| IDV | Description |
|-----|-------------|
| 8 | A, B, C feed composition (Stream 4) |
| 9 | D feed temperature (Stream 2) |
| 10 | C feed temperature (Stream 4) |
| 11 | Reactor cooling water inlet temperature |
| 12 | Condenser cooling water inlet temperature |

### Special Disturbances (IDV 13-20)

| IDV | Type | Description |
|-----|------|-------------|
| 13 | Slow drift | Reaction kinetics |
| 14 | Sticking | Reactor cooling water valve |
| 15 | Sticking | Condenser cooling water valve |
| 16-20 | Unknown | Intentionally undisclosed |

## Why This Dataset Matters

1. **Statistical rigor:** 500 simulations per fault allow proper ROC curve analysis and statistical significance testing.

2. **Reproducibility:** Non-overlapping random seeds ensure independent simulation runs.

3. **Fair benchmarking:** Enables meaningful comparison of anomaly detection methods.

4. **Addresses bias:** Previous single-simulation datasets gave inconsistent, potentially misleading results.

## Process Variables

### Measured Variables (XMEAS 1-41)

| Index | Variable | Description | Units |
|-------|----------|-------------|-------|
| 1 | XMEAS(1) | A Feed | kscmh |
| 2 | XMEAS(2) | D Feed | kg/hr |
| 3 | XMEAS(3) | E Feed | kg/hr |
| 4 | XMEAS(4) | A and C Feed | kscmh |
| 5 | XMEAS(5) | Recycle Flow | kscmh |
| 6 | XMEAS(6) | Reactor Feed Rate | kscmh |
| 7 | XMEAS(7) | Reactor Pressure | kPa gauge |
| 8 | XMEAS(8) | Reactor Level | % |
| 9 | XMEAS(9) | Reactor Temperature | deg C |
| 10 | XMEAS(10) | Purge Rate | kscmh |
| 11 | XMEAS(11) | Separator Temperature | deg C |
| 12 | XMEAS(12) | Separator Level | % |
| 13 | XMEAS(13) | Separator Pressure | kPa gauge |
| 14 | XMEAS(14) | Separator Underflow | m3/hr |
| 15 | XMEAS(15) | Stripper Level | % |
| 16 | XMEAS(16) | Stripper Pressure | kPa gauge |
| 17 | XMEAS(17) | Stripper Underflow | m3/hr |
| 18 | XMEAS(18) | Stripper Temperature | deg C |
| 19 | XMEAS(19) | Stripper Steam Flow | kg/hr |
| 20 | XMEAS(20) | Compressor Work | kW |
| 21 | XMEAS(21) | Reactor CW Outlet Temp | deg C |
| 22 | XMEAS(22) | Separator CW Outlet Temp | deg C |
| 23-28 | XMEAS(23-28) | Reactor Feed Composition | mol% A-F |
| 29-36 | XMEAS(29-36) | Purge Gas Composition | mol% A-H |
| 37-41 | XMEAS(37-41) | Product Composition | mol% D-H |

### Manipulated Variables (XMV 1-11)

| Index | Variable | Description |
|-------|----------|-------------|
| 1 | XMV(1) | D Feed Flow |
| 2 | XMV(2) | E Feed Flow |
| 3 | XMV(3) | A Feed Flow |
| 4 | XMV(4) | A and C Feed Flow |
| 5 | XMV(5) | Compressor Recycle Valve |
| 6 | XMV(6) | Purge Valve |
| 7 | XMV(7) | Separator Pot Liquid Flow |
| 8 | XMV(8) | Stripper Liquid Product Flow |
| 9 | XMV(9) | Stripper Steam Valve |
| 10 | XMV(10) | Reactor Cooling Water Flow |
| 11 | XMV(11) | Condenser Cooling Water Flow |

## Generating the Dataset

This repository includes a script to reproduce the Rieth 2017 dataset using the local TEP simulator. All parameters are configurable, with defaults matching the original Rieth 2017 specifications. See `examples/rieth2017_dataset.py` for the full implementation.

### Quick Start

```bash
# Generate a small test dataset (5 simulations per fault)
python examples/rieth2017_dataset.py --small

# Generate the full dataset (500 simulations per fault, takes several hours)
python examples/rieth2017_dataset.py --full

# Generate a custom dataset
python examples/rieth2017_dataset.py --n-simulations 100 --faults 1,2,4,6

# Use a preset configuration
python examples/rieth2017_dataset.py --preset quick

# Generate in parallel with 4 workers
python examples/rieth2017_dataset.py --preset quick --workers 4

# Output as CSV instead of NumPy
python examples/rieth2017_dataset.py --preset quick --format csv
```

### Presets

Named presets provide convenient configurations for common use cases:

| Preset | Simulations | Train | Test | Sampling | Description |
|--------|-------------|-------|------|----------|-------------|
| `rieth2017` | 500 | 25h | 48h | 3 min | Original paper specifications |
| `quick` | 5 | 2h | 4h | 3 min | Fast testing and development |
| `high-res` | 500 | 25h | 48h | 1 min | Higher temporal resolution |
| `minimal` | 2 | 0.5h | 1h | 3 min | Minimal for unit tests |

```bash
# List available presets
python examples/rieth2017_dataset.py --list-presets

# Use a preset with overrides
python examples/rieth2017_dataset.py --preset quick --n-simulations 20
```

### Output Formats

Data can be saved in multiple formats:

| Format | Extension | Description |
|--------|-----------|-------------|
| `npy` | `.npy` | NumPy binary format (default, fastest) |
| `csv` | `.csv` | Comma-separated values with headers |
| `hdf5` | `.h5` | HDF5 with gzip compression (requires h5py) |

```bash
# Single format
python examples/rieth2017_dataset.py --preset quick --format csv

# Multiple formats
python examples/rieth2017_dataset.py --preset quick --format npy,csv,hdf5
```

### Parallel Generation

Use multiple CPU cores to speed up dataset generation:

```bash
# Use 4 parallel workers
python examples/rieth2017_dataset.py --preset quick --workers 4

# Use all available CPU cores
python examples/rieth2017_dataset.py --preset quick --workers -1
```

### Column Selection

Select subsets of process variables to reduce dataset size:

| Group | Variables | Count | Description |
|-------|-----------|-------|-------------|
| `all` | xmeas_1-41, xmv_1-11 | 52 | All variables (default) |
| `xmeas` | xmeas_1-41 | 41 | Measured variables only |
| `xmv` | xmv_1-11 | 11 | Manipulated variables only |
| `key` | xmeas_7-9,11,12,15,20, xmv_1,10 | 9 | Key process variables |
| `flows` | xmeas_1-6,10,14,17,19 | 10 | Flow measurements |
| `temperatures` | xmeas_9,11,18,21,22 | 5 | Temperature measurements |
| `pressures` | xmeas_7,13,16 | 3 | Pressure measurements |
| `levels` | xmeas_8,12,15 | 3 | Level measurements |
| `compositions` | xmeas_23-41 | 19 | Composition measurements |

```bash
# List available column groups
python examples/rieth2017_dataset.py --list-columns

# Use a column group
python examples/rieth2017_dataset.py --preset quick --columns xmeas

# Select specific columns
python examples/rieth2017_dataset.py --preset quick --columns xmeas_1,xmeas_9,xmv_1
```

### Configurable Parameters

All simulation parameters can be customized via CLI or Python API:

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `n_simulations` | `--n-simulations` | 500 | Simulations per fault type |
| `train_duration_hours` | `--train-duration` | 25.0 | Training simulation duration (hours) |
| `val_duration_hours` | `--val-duration` | 48.0 | Validation simulation duration (hours) |
| `test_duration_hours` | `--test-duration` | 48.0 | Testing simulation duration (hours) |
| `sampling_interval_min` | `--sampling-interval` | 3.0 | Sampling interval (minutes) |
| `fault_onset_hours` | `--fault-onset` | 1.0 | Fault onset time for val/test (hours) |
| `n_faults` | `--faults` | 20 | Number of fault types (or specific list) |
| `output_formats` | `--format` | npy | Output format(s): npy, csv, hdf5 |
| `n_workers` | `--workers` | 1 | Number of parallel workers (-1 for all CPUs) |
| `columns` | `--columns` | all | Column subset or group name |

**CLI example with custom parameters:**

```bash
python examples/rieth2017_dataset.py \
    --n-simulations 50 \
    --train-duration 10 \
    --test-duration 20 \
    --sampling-interval 1 \
    --fault-onset 0.5 \
    --faults 1,4,6 \
    --format npy,csv \
    --workers 4 \
    --columns key
```

### Python API

```python
from examples.rieth2017_dataset import Rieth2017DatasetGenerator

# Default Rieth 2017 parameters
generator = Rieth2017DatasetGenerator(output_dir="./data/rieth2017")
generator.generate_all()

# Using presets
generator = Rieth2017DatasetGenerator.from_preset("quick", output_dir="./data/quick")
generator.generate_all()

# Preset with overrides
generator = Rieth2017DatasetGenerator.from_preset(
    "quick",
    output_dir="./data/custom",
    n_simulations=20,
    output_formats=["npy", "csv"],
)

# Custom parameters with all new features
generator = Rieth2017DatasetGenerator(
    output_dir="./data/custom",
    n_simulations=100,
    train_duration_hours=10.0,
    test_duration_hours=20.0,
    sampling_interval_min=1.0,
    fault_onset_hours=0.5,
    output_formats=["npy", "csv"],  # Multiple formats
    n_workers=4,                     # Parallel generation
    columns="key",                   # Column subset
)
generator.generate_all(fault_numbers=[1, 4, 6])

# List available presets and column groups
print(Rieth2017DatasetGenerator.list_presets())
print(Rieth2017DatasetGenerator.list_column_groups())

# Or generate specific files
generator.generate_fault_free_training(n_simulations=500)
generator.generate_faulty_testing(fault_numbers=[1, 4, 6], n_simulations=100)
```

### Loading Generated Data

```python
from examples.rieth2017_dataset import load_rieth2017_dataset, get_fault_data, get_features

# Load all data files
data = load_rieth2017_dataset("./data/rieth2017")

# Access fault-free testing data
normal_test = data["fault_free_testing"]

# Extract data for a specific fault
fault1_data = get_fault_data(data["faulty_testing"], fault_number=1)

# Get feature columns only (52 columns: 41 xmeas + 11 xmv)
features = get_features(fault1_data)
```

### Comparing with Harvard Dataverse Original

The script can download the original dataset from Harvard Dataverse and compare it with locally generated data:

```bash
# Download original dataset from Harvard Dataverse
python examples/rieth2017_dataset.py --download-harvard

# Compare generated data with original
python examples/rieth2017_dataset.py --compare

# Requirements for comparison
pip install requests pyreadr
```

**Python API:**

```python
from examples.rieth2017_dataset import (
    HarvardDataverseDataset,
    compare_datasets,
    compare_with_harvard,
)

# Download and load original dataset
harvard = HarvardDataverseDataset()
harvard.download()
original_data = harvard.load("fault_free_training")

# Compare with generated data
results = compare_with_harvard(local_dir="./data/rieth2017")

# Or compare specific arrays
from examples.rieth2017_dataset import load_rieth2017_dataset
local_data = load_rieth2017_dataset("./data/rieth2017")
comparison = compare_datasets(
    local_data["fault_free_training"],
    original_data,
    name="fault_free_training"
)
```

The comparison reports:
- Shape differences
- Per-fault statistics for key variables
- Mean correlation between datasets
- Mean absolute percentage error (MAPE)

## Related Datasets

### Original Braatz Dataset
- Single simulation per fault
- Available at: https://github.com/camaramm/tennessee-eastman-profBraatz

### Reinartz et al. 2021 Extended Dataset
- 28 fault types (including 8 additional random variation faults)
- 6 operating modes
- Mode transitions
- Available at: https://data.dtu.dk/articles/dataset/Tennessee_Eastman_Reference_Data_for_Fault-Detection_and_Decision_Support_Systems/13385936

## References

1. Rieth, C.A., Amsel, B.D., Tran, R., Cook, M.B. (2018). Issues and Advances in Anomaly Detection Evaluation for Joint Human-Automated Systems. In: Advances in Human Factors in Robots and Unmanned Systems. AHFE 2017. Advances in Intelligent Systems and Computing, vol 595. Springer, Cham.

2. Downs, J.J., Vogel, E.F. (1993). A plant-wide industrial process control problem. Computers & Chemical Engineering, 17(3), 245-255.

3. Russell, E.L., Chiang, L.H., Braatz, R.D. (2000). Data-driven Methods for Fault Detection and Diagnosis in Chemical Processes. Springer-Verlag, London.
