# TEP Batch Simulation CLI

The `tep-sim` command provides a batch simulation interface for generating Tennessee Eastman Process data with configurable faults, duration, and output format.

## Quick Start

```bash
# Run an 8-hour normal operation simulation
tep-sim --duration 8 --output normal.dat

# Run with a fault
tep-sim --duration 8 --faults 1 --output fault1.dat

# Display results graphically
tep-sim --duration 2 --faults 1 --plot
```

## Command Line Options

### Simulation Parameters

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--duration` | `-d` | 8.0 | Simulation duration in hours |
| `--faults` | `-f` | None | Fault IDs to activate |
| `--fault-times` | `-t` | 1.0 | Fault start times in hours |
| `--seed` | `-s` | 4651207995 | Random seed for reproducibility |
| `--record-interval` | `-r` | 180 | Recording interval in seconds |

### Output Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | tep_data.dat | Output file path |
| `--multi-file` | `-m` | False | Use original Fortran multi-file format |
| `--no-header` | | False | Omit column headers in output |
| `--plot` | `-p` | False | Display results graphically |
| `--plot-save` | | None | Save plot to file instead of displaying |
| `--quiet` | `-q` | False | Suppress progress output |

### Information

| Option | Description |
|--------|-------------|
| `--list-faults` | List all 20 available faults and exit |
| `--help` | Show help message and exit |

## Fault Specification

The `--faults` option supports flexible fault specification:

```bash
# Single fault
tep-sim --faults 1

# Multiple faults (comma-separated)
tep-sim --faults 1,4,7

# Range of faults
tep-sim --faults 1-5

# Combined specification
tep-sim --faults 1,3-5,7
```

### Available Faults (IDV 1-20)

| IDV | Type | Description |
|-----|------|-------------|
| 1 | Step | A/C Feed Ratio, B Composition Constant |
| 2 | Step | B Composition, A/C Ratio Constant |
| 3 | Step | D Feed Temperature |
| 4 | Step | Reactor Cooling Water Inlet Temperature |
| 5 | Step | Condenser Cooling Water Inlet Temperature |
| 6 | Step | A Feed Loss |
| 7 | Step | C Header Pressure Loss |
| 8 | Random | A, B, C Feed Composition |
| 9 | Random | D Feed Temperature |
| 10 | Random | C Feed Temperature |
| 11 | Random | Reactor Cooling Water Inlet Temperature |
| 12 | Random | Condenser Cooling Water Inlet Temperature |
| 13 | Drift | Reaction Kinetics |
| 14 | Sticking | Reactor Cooling Water Valve |
| 15 | Sticking | Condenser Cooling Water Valve |
| 16-20 | Unknown | Reserved |

Run `tep-sim --list-faults` for a complete list.

## Fault Start Times

The `--fault-times` option specifies when faults are activated:

```bash
# Single time applies to all faults
tep-sim --faults 1,4,7 --fault-times 1.0

# Different times for each fault
tep-sim --faults 1,4,7 --fault-times 1.0,2.0,3.0
```

**Note:** The number of times must either be 1 (applied to all faults) or match the number of faults exactly.

## Output Formats

### Single File (Default)

The default output is a single `.dat` file with all data:

```bash
tep-sim --duration 8 --output simulation.dat
```

**Format:**
- First row: Column headers (unless `--no-header` is used)
- Each subsequent row: Time (seconds), 41 measurements, 12 manipulated variables
- Values in Fortran E13.5 scientific notation
- Space-separated columns

**Example output:**
```
        Time(s)       XMEAS(1)       XMEAS(2)  ...
  0.00000E+00  2.50520E-01  3.66400E+03  ...
  1.80000E+02  2.50510E-01  3.66420E+03  ...
```

### Multi-File Format (Original Fortran)

The `--multi-file` option generates 15 separate files matching the original Fortran output:

```bash
tep-sim --duration 8 --faults 1 --multi-file --output ./data/
```

**Files generated:**

| File | Content |
|------|---------|
| TE_data_inc.dat | Time (seconds) |
| TE_data_mv1.dat | MV 1-4 |
| TE_data_mv2.dat | MV 5-8 |
| TE_data_mv3.dat | MV 9-12 |
| TE_data_me01.dat | XMEAS 1-4 |
| TE_data_me02.dat | XMEAS 5-8 |
| TE_data_me03.dat | XMEAS 9-12 |
| TE_data_me04.dat | XMEAS 13-16 |
| TE_data_me05.dat | XMEAS 17-20 |
| TE_data_me06.dat | XMEAS 21-24 |
| TE_data_me07.dat | XMEAS 25-28 |
| TE_data_me08.dat | XMEAS 29-32 |
| TE_data_me09.dat | XMEAS 33-36 |
| TE_data_me10.dat | XMEAS 37-40 |
| TE_data_me11.dat | XMEAS 41 |

## Graphical Output

### Display Plot

```bash
tep-sim --duration 2 --faults 1 --plot
```

This opens an interactive matplotlib window showing:
- Reactor temperature and pressure
- Process levels (reactor, separator, stripper)
- Feed flows
- Product quality metrics
- Key manipulated variables
- Cooling and control valves

**Note:** Requires matplotlib (`pip install matplotlib`).

### Save Plot to File

```bash
tep-sim --duration 2 --faults 1 --plot-save results.png
```

Supported formats: PNG, PDF, SVG, JPG (based on file extension).

## Reproducibility

Use the `--seed` option for reproducible simulations:

```bash
# These will produce identical results
tep-sim --duration 8 --seed 12345 --output run1.dat
tep-sim --duration 8 --seed 12345 --output run2.dat
```

The random seed affects:
- Measurement noise
- Random variation disturbances (IDV 8-12)

## Examples

### Generate Training Dataset

```bash
# Normal operation (8 hours, record every 3 min)
tep-sim --duration 8 --seed 42 --output d00_train.dat

# Fault 1 training data
tep-sim --duration 8 --faults 1 --fault-times 1.0 --seed 42 --output d01_train.dat
```

### Generate Multiple Fault Datasets

```bash
for fault in 1 2 3 4 5 6 7; do
    tep-sim --duration 8 --faults $fault --seed 12345 --output fault${fault}.dat
done
```

### Compare Normal vs Fault Operation

```bash
# Run both simulations with plots
tep-sim --duration 4 --plot-save normal.png
tep-sim --duration 4 --faults 6 --fault-times 1.0 --plot-save fault6.png
```

### High-Resolution Data

```bash
# Record every second instead of every 3 minutes
tep-sim --duration 1 --record-interval 1 --output highres.dat
```

**Note:** This generates much more data (3600 samples/hour vs 20 samples/hour).

### Quick Test Simulation

```bash
# Short simulation with progress suppressed
tep-sim --duration 0.1 --quiet --output test.dat
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid arguments, simulation failure, etc.) |

## Troubleshooting

### "Fortran extension not available"

The TEP simulator requires the compiled Fortran extension. Install with:

```bash
pip install tep  # Requires gfortran
```

On macOS: `brew install gcc`
On Linux: `apt install gfortran`

### "matplotlib is required for plotting"

Install matplotlib for graphical output:

```bash
pip install matplotlib
# or
pip install tep[gui]
```

### Simulation ends in shutdown

Some faults cause safety shutdowns. Check the output message for shutdown time:

```
WARNING: Simulation ended in safety shutdown at 2.3456 hours
```

Try:
- Using a different fault
- Reducing simulation duration
- Starting fault later (`--fault-times`)
