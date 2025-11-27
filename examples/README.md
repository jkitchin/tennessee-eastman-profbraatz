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
