# TEP Simulator Examples

This directory contains example scripts demonstrating various uses of the Tennessee Eastman Process simulator.

## Examples

### basic_simulation.py

Basic usage of the TEP simulator for running batch simulations.

**Topics covered:**
- Creating and initializing the simulator
- Running simulations with `simulate()`
- Accessing results (measurements, states, MVs)
- Computing statistics

**Run:**
```bash
python examples/basic_simulation.py
```

### disturbance_simulation.py

Demonstrates how to apply process disturbances and observe controller response.

**Topics covered:**
- Applying step disturbances (IDV 1-7)
- Scheduling multiple disturbances
- Real-time disturbance activation/deactivation
- Random disturbances (IDV 8-12)

**Run:**
```bash
python examples/disturbance_simulation.py
```

### custom_controller.py

Shows how to implement and use custom control strategies.

**Topics covered:**
- Creating custom controller classes
- Using functions as controllers
- Cascade control structures
- PIController configuration
- Manual control mode

**Run:**
```bash
python examples/custom_controller.py
```

### data_generation.py

Generates datasets for fault detection and diagnosis research.

**Topics covered:**
- Generating normal operating data
- Generating faulty data with specific faults
- Creating combined measurement/MV matrices
- Saving data in NumPy and CSV formats
- Train/test data splitting with different seeds

**Run:**
```bash
python examples/data_generation.py
```

**Output files:**
- `normal_data.npy` - Normal operating data
- `fault{N}_data.npy` - Fault scenario data
- `normal_data_sample.csv` - CSV sample for inspection

### rieth2017_dataset.py

Generates TEP datasets with configurable parameters. Defaults match Rieth et al. (2017) specifications for anomaly detection research.

**Topics covered:**
- Generating datasets with configurable simulations per fault type
- Matching the original dataset structure (55 columns)
- Configurable duration, sampling interval, and fault onset time
- Non-overlapping random seeds for independent simulations
- Named presets for common configurations
- Multiple output formats (npy, csv, hdf5)
- Parallel generation with multiprocessing
- Column/variable selection

**Run:**
```bash
# Quick test with 5 simulations
python examples/rieth2017_dataset.py --small

# Full dataset (500 simulations, takes several hours)
python examples/rieth2017_dataset.py --full

# Use a preset configuration
python examples/rieth2017_dataset.py --preset quick

# Parallel generation with 4 workers
python examples/rieth2017_dataset.py --preset quick --workers 4

# Output as CSV format
python examples/rieth2017_dataset.py --preset quick --format csv

# Select only key variables
python examples/rieth2017_dataset.py --preset quick --columns key

# Custom: 100 simulations for faults 1, 4, 6 only
python examples/rieth2017_dataset.py --n-simulations 100 --faults 1,4,6

# Custom timing parameters
python examples/rieth2017_dataset.py --n-simulations 50 \
    --train-duration 10 --test-duration 20 \
    --sampling-interval 1 --fault-onset 0.5
```

**Presets:**
| Preset | Simulations | Train | Test | Description |
|--------|-------------|-------|------|-------------|
| `rieth2017` | 500 | 25h | 48h | Original paper specs (default) |
| `quick` | 5 | 2h | 4h | Fast testing |
| `high-res` | 500 | 25h | 48h | 1-minute sampling |
| `minimal` | 2 | 0.5h | 1h | Unit tests |

**Configurable parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--preset` | - | Use named preset (rieth2017, quick, high-res, minimal) |
| `--n-simulations` | 500 | Simulations per fault type |
| `--train-duration` | 25.0 | Training duration (hours) |
| `--val-duration` | 48.0 | Validation duration (hours) |
| `--test-duration` | 48.0 | Testing duration (hours) |
| `--sampling-interval` | 3.0 | Sampling interval (minutes) |
| `--fault-onset` | 1.0 | Fault onset time for val/test (hours) |
| `--faults` | 1-20 | Comma-separated fault numbers |
| `--format` | npy | Output format(s): npy, csv, hdf5 |
| `--workers` | 1 | Parallel workers (-1 for all CPUs) |
| `--columns` | all | Column group or list (all, xmeas, xmv, key) |
| `--no-validation` | - | Skip validation sets |
| `--list-presets` | - | Show available presets |
| `--list-columns` | - | Show available column groups |

**Output files (6 files with validation):**
- `fault_free_training.npy` - Normal operation training data
- `fault_free_validation.npy` - Normal operation validation data
- `fault_free_testing.npy` - Normal operation testing data
- `faulty_training.npy` - Faulty training data (fault from t=0)
- `faulty_validation.npy` - Faulty validation data (fault at onset time)
- `faulty_testing.npy` - Faulty testing data (fault at onset time)
- `metadata.json` - Dataset metadata with all parameters

Use `--no-validation` to generate only train/test splits (4 files).

**Intermittent fault mode** (faults turn on/off throughout trajectory):
```bash
# Generate 10 trajectories with faults cycling on/off
python examples/rieth2017_dataset.py --intermittent --n-simulations 10

# Custom timing: 3h fault active, 1.5h normal between faults
python examples/rieth2017_dataset.py --intermittent \
    --faults 1,4,6,11 --fault-duration 3 --normal-duration 1.5
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--intermittent` | - | Enable intermittent fault mode |
| `--fault-duration` | 4.0 | Avg hours each fault is active |
| `--normal-duration` | 2.0 | Avg hours between faults |
| `--duration-variance` | 0.5 | Randomness factor (±50%) |
| `--initial-normal` | 1.0 | Normal period before first fault |
| `--no-randomize-order` | - | Keep faults in numerical order |

**Compare with Harvard Dataverse original:**
```bash
# Download original Rieth 2017 dataset
python examples/rieth2017_dataset.py --download-harvard

# Compare generated vs original
python examples/rieth2017_dataset.py --compare

# Requirements: pip install requests pyreadr
```

**Reference:** Rieth, C.A., et al. (2017). "Issues and Advances in Anomaly Detection Evaluation for Joint Human-Automated Systems." AHFE 2017.

## Quick Start

```bash
# Install package first
pip install -e .

# Run any example
python examples/basic_simulation.py
```

## Using Examples as Templates

These examples can be used as starting points for your own scripts. Common patterns:

### Basic Simulation

```python
from tep import TEPSimulator

sim = TEPSimulator(random_seed=12345)
sim.initialize()
result = sim.simulate(duration_hours=1.0)
```

### With Disturbances

```python
result = sim.simulate(
    duration_hours=4.0,
    disturbances={1: (1.0, 1)}  # IDV(1) at t=1h
)
```

### Custom Controller

```python
def my_controller(xmeas, xmv, step):
    new_xmv = xmv.copy()
    # Your control logic
    return new_xmv

result = sim.simulate_with_controller(
    duration_hours=2.0,
    controller=my_controller
)
```

### Step-by-Step Simulation

```python
sim.initialize()
while sim.time < 1.0:  # 1 hour
    if not sim.step():
        print("Shutdown!")
        break
    measurements = sim.get_measurements()
    # Process data...
```
