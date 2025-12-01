# Tennessee Eastman Process Reference Data

This directory contains reference data files from the original Fortran TEP simulation by E.L. Russell, L.H. Chiang, and R.D. Braatz.

## File Format

Each file contains simulation data in a space-delimited text format:
- **Rows**: Observation samples (one per sampling interval)
- **Columns**: 52 variables per sample

The observation vector at each time step is:
```
x = [XMEAS(1), XMEAS(2), ..., XMEAS(41), XMV(1), ..., XMV(11)]^T
```

Where:
- `XMEAS(1-41)`: 41 measured variables (columns 0-40)
- `XMV(1-11)`: 11 manipulated variables (columns 41-51)

**Sampling interval**: 180 seconds (3 minutes)

## File Naming Convention

| Pattern | Description | Samples |
|---------|-------------|---------|
| `d00.dat` | Normal operation (training) | 480 |
| `d00_te.dat` | Normal operation (testing) | 960 |
| `d01.dat` - `d21.dat` | Fault scenarios (training) | 480 |
| `d01_te.dat` - `d21_te.dat` | Fault scenarios (testing) | 960 |

## Fault Scenarios

| File | Fault ID | Description | Type |
|------|----------|-------------|------|
| d00 | - | Normal operation | - |
| d01 | IDV(1) | A/C Feed Ratio, B Composition Constant (Stream 4) | Step |
| d02 | IDV(2) | B Composition, A/C Ratio Constant (Stream 4) | Step |
| d03 | IDV(3) | D Feed Temperature (Stream 2) | Step |
| d04 | IDV(4) | Reactor Cooling Water Inlet Temperature | Step |
| d05 | IDV(5) | Condenser Cooling Water Inlet Temperature | Step |
| d06 | IDV(6) | A Feed Loss (Stream 1) | Step |
| d07 | IDV(7) | C Header Pressure Loss (Stream 4) | Step |
| d08 | IDV(8) | A, B, C Feed Composition (Stream 4) | Random |
| d09 | IDV(9) | D Feed Temperature (Stream 2) | Random |
| d10 | IDV(10) | C Feed Temperature (Stream 4) | Random |
| d11 | IDV(11) | Reactor Cooling Water Inlet Temperature | Random |
| d12 | IDV(12) | Condenser Cooling Water Inlet Temperature | Random |
| d13 | IDV(13) | Reaction Kinetics | Slow Drift |
| d14 | IDV(14) | Reactor Cooling Water Valve | Sticking |
| d15 | IDV(15) | Condenser Cooling Water Valve | Sticking |
| d16 | IDV(16) | Unknown | - |
| d17 | IDV(17) | Unknown | - |
| d18 | IDV(18) | Unknown | - |
| d19 | IDV(19) | Unknown | - |
| d20 | IDV(20) | Unknown | - |
| d21 | IDV(21) | Valve position constant (Stream 4) | - |

## Usage

These files are used for:
1. **Validation**: Comparing Python implementation against original Fortran outputs
2. **Training**: Building fault detection models (PCA, PLS, FDA, CVA)
3. **Testing**: Evaluating fault detection performance

### Loading Data in Python

```python
import numpy as np
from pathlib import Path

# Load normal operation test data
data = np.loadtxt(Path("data/d00_te.dat"))

# Split into measurements and manipulated variables
xmeas = data[:, :41]  # XMEAS(1-41)
xmv = data[:, 41:]    # XMV(1-11)
```

## References

- J.J. Downs and E.F. Vogel, "A plant-wide industrial process control problem," *Computers and Chemical Engineering*, 17:245-255 (1993)
- E.L. Russell, L.H. Chiang, and R.D. Braatz, *Data-driven Techniques for Fault Detection and Diagnosis in Chemical Processes*, Springer-Verlag, London, 2000
