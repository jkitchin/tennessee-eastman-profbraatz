# CLI Reference

## tep-sim - Batch Simulation

Run Tennessee Eastman Process simulations from the command line.

### Basic Usage

```bash
# Normal 8-hour simulation
tep-sim --duration 8 --output normal.dat

# With fault injection
tep-sim --duration 8 --faults 4 --fault-times 1.0 --output fault4.dat

# List available faults
tep-sim --list-faults

# List available controllers
tep-sim --list-controllers
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--duration` | `-d` | 8.0 | Simulation duration in hours |
| `--faults` | `-f` | None | Fault IDs to activate |
| `--fault-times` | `-t` | 1.0 | Fault start times in hours |
| `--seed` | `-s` | 4651207995 | Random seed |
| `--output` | `-o` | tep_data.dat | Output file path |
| `--multi-file` | `-m` | False | Original Fortran multi-file format |
| `--record-interval` | `-r` | 180 | Steps between recordings |
| `--no-header` | | False | Omit column headers |
| `--quiet` | `-q` | False | Suppress progress output |
| `--plot` | `-p` | False | Display results graphically |
| `--plot-save` | | None | Save plot to file |
| `--controller` | `-c` | decentralized | Controller plugin name |
| `--list-faults` | | | List available faults and exit |
| `--list-controllers` | | | List controllers and exit |

### Fault Specification

Faults can be specified in several formats:

```bash
# Single fault
tep-sim -f 1

# Multiple faults (comma-separated)
tep-sim -f 1,4,7

# Range of faults
tep-sim -f 1-5

# Combined
tep-sim -f 1,3-5,7

# Multiple faults with different start times
tep-sim -f 1,4,7 -t 1.0,2.0,3.0
```

### Output Formats

**Single file (default):**
- Space-separated values with E13.5 scientific notation
- Columns: Time(s), XMEAS(1-41), XMV(1-12)
- Optional header row

```bash
tep-sim -d 8 -o data.dat
```

**Multi-file (original Fortran format):**
- 15 separate .dat files
- `TE_data_inc.dat`: Time increments
- `TE_data_mv1.dat` through `TE_data_mv3.dat`: MVs 1-4, 5-8, 9-12
- `TE_data_me01.dat` through `TE_data_me11.dat`: Measurements

```bash
tep-sim -d 8 -m -o ./data/
```

### Examples

```bash
# Quick 2-hour test with plot
tep-sim -d 2 -f 4 -p

# Save plot to file
tep-sim -d 2 -f 4 --plot-save results.png

# Generate training data (normal operation)
tep-sim -d 48 -s 12345 -o train_normal.dat

# Generate test data with fault
tep-sim -d 48 -f 1 -t 8.0 -s 67890 -o test_fault1.dat

# Reproducible simulation
tep-sim -d 8 -s 42 -o reproducible.dat

# Fast recording (every second)
tep-sim -d 1 -r 1 -o fine_resolution.dat

# 3-minute intervals (default, matches original)
tep-sim -d 8 -r 180 -o standard.dat
```

---

## tep-web - Interactive Dashboard

Launch a real-time web dashboard for process visualization and control.

### Basic Usage

```bash
# Launch dashboard (opens browser automatically)
tep-web

# Specify port
tep-web --port 8080

# Don't open browser automatically
tep-web --no-browser

# Specify host (for remote access)
tep-web --host 0.0.0.0 --port 8080
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | 8050 | Port to run the server on |
| `--host` | 127.0.0.1 | Host address |
| `--no-browser` | False | Don't auto-open browser |
| `--debug` | False | Enable Dash debug mode |

### Dashboard Features

1. **Real-time Plots**
   - Reactor conditions (temperature, pressure)
   - Process levels (reactor, separator, stripper)
   - Feed flows and compositions

2. **Process Controls**
   - Start/Stop/Reset simulation
   - Simulation speed adjustment
   - Disturbance injection buttons

3. **Manual Control**
   - Individual valve position sliders
   - XMV(1-12) adjustments

4. **Data Export**
   - Download current data as CSV

### Programmatic Access

```python
from tep.dashboard_dash import run_dashboard

# Launch with defaults
run_dashboard()

# Or with options
run_dashboard(port=8080, debug=True)
```

---

## Python API Alternative

For programmatic use, prefer the Python API:

```python
from tep import TEPSimulator

sim = TEPSimulator(random_seed=12345)
sim.initialize()

result = sim.simulate(
    duration_hours=8.0,
    disturbances={4: (1.0, 1)},
    record_interval=180
)

# Save as numpy arrays
import numpy as np
np.savez('data.npz',
    time=result.time,
    measurements=result.measurements,
    mvs=result.manipulated_vars
)

# Or as pandas DataFrame
import pandas as pd
df = pd.DataFrame(
    result.measurements,
    columns=[f'XMEAS{i+1}' for i in range(41)]
)
df['time_hours'] = result.time
df.to_csv('data.csv', index=False)
```
