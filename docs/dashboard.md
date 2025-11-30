# TEP Web Dashboard Guide

The TEP package provides an interactive web dashboard for running and monitoring Tennessee Eastman Process simulations using Dash and Plotly.

## Quick Start

### Installation

Install web dashboard dependencies:

```bash
pip install -e ".[web]"
```

### Launch Dashboard

From the command line:

```bash
tep-web
```

With options:

```bash
tep-web --port 8080 --no-browser --host 0.0.0.0
```

Or from Python:

```python
from tep import run_dashboard
run_dashboard()
```

Or directly:

```python
from tep.dashboard_dash import run_dashboard
run_dashboard(port=8050, open_browser=True)
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--port` | Port number (default: 8050) |
| `--host` | Host address (default: 127.0.0.1) |
| `--no-browser` | Don't open browser automatically |
| `--debug` | Enable Dash debug mode |

## Interface Overview

The dashboard is organized into a header, control sidebar, and main plotting area:

```
+------------------------------------------------------------------+
|  Tennessee Eastman Process Simulator    [Author: ...]            |
+------------------------------------------------------------------+
|                    |                                              |
|  Simulation        |   Real-Time Plots (2x3 grid)                |
|  [Start] [Stop]    |                                              |
|  [Reset]           |   +------------------+------------------+    |
|                    |   | Reactor          | Separator        |    |
|  Speed: [slider]   |   +------------------+------------------+    |
|                    |   | Stripper         | Flows            |    |
|  Record: [toggle]  |   +------------------+------------------+    |
|  [Download CSV]    |   | Compositions     | Utilities        |    |
|                    |   +------------------+------------------+    |
|  Disturbances      |                                              |
|  [checkboxes 1-20] |                                              |
|  [Apply] [Clear]   |                                              |
|  Active: IDV(6)    |                                              |
|                    |                                              |
|  Time: 1.5 hours   |                                              |
|                    |                                              |
+------------------------------------------------------------------+
```

## Control Panel

### Simulation Controls

| Button | Action |
|--------|--------|
| **Start** | Begin or resume simulation |
| **Stop** | Pause simulation |
| **Reset** | Reset to initial steady-state conditions |

### Speed Control

The speed slider adjusts how fast the simulation runs:
- **1x**: Slower, detailed observation
- **10x**: Normal operation
- **100x**: Fast-forward for long simulations

### Data Recording

| Control | Description |
|---------|-------------|
| **Record toggle** | Enable/disable data recording |
| **Download CSV** | Download recorded data as CSV file |

The CSV includes all 41 measurements, 12 manipulated variables, time, and active disturbances.

## Disturbances Panel

### Applying Disturbances

1. Check the boxes for desired disturbances (IDV 1-20)
2. Click **Apply Disturbances** to activate them
3. The "Active" display shows which disturbances are currently enabled in the Fortran simulation

### Clearing Disturbances

Click **Clear All** to disable all disturbances and return to normal operation.

### Available Disturbances

#### Step Disturbances (IDV 1-7)
| IDV | Description |
|-----|-------------|
| 1 | A/C feed ratio, B composition constant |
| 2 | B composition, A/C ratio constant |
| 3 | D feed temperature step |
| 4 | Reactor cooling water inlet temperature step |
| 5 | Condenser cooling water inlet temperature step |
| 6 | A feed loss (stream 1) |
| 7 | C header pressure loss |

#### Random Disturbances (IDV 8-12)
| IDV | Description |
|-----|-------------|
| 8 | Random A, B, C feed composition |
| 9 | Random D feed temperature |
| 10 | Random C feed temperature |
| 11 | Random reactor CW inlet temperature |
| 12 | Random condenser CW inlet temperature |

#### Drift and Sticking (IDV 13-15)
| IDV | Description |
|-----|-------------|
| 13 | Slow drift in reaction kinetics |
| 14 | Reactor cooling water valve sticking |
| 15 | Condenser cooling water valve sticking |

#### Unknown Faults (IDV 16-20)
Reserved for testing fault detection algorithms.

## Real-Time Plots

The dashboard displays 6 plot panels in a 2x3 grid:

### Reactor Panel
- **Pressure** (XMEAS 7) - kPa gauge
- **Level** (XMEAS 8) - %
- **Temperature** (XMEAS 9) - °C

### Separator Panel
- **Temperature** (XMEAS 11) - °C
- **Level** (XMEAS 12) - %
- **Pressure** (XMEAS 13) - kPa gauge

### Stripper Panel
- **Level** (XMEAS 15) - %
- **Pressure** (XMEAS 16) - kPa gauge
- **Temperature** (XMEAS 18) - °C

### Flows Panel
- **A Feed** (XMEAS 1) - kscmh
- **D Feed** (XMEAS 2) - kg/hr
- **E Feed** (XMEAS 3) - kg/hr
- **Recycle** (XMEAS 5) - kscmh

### Compositions Panel
- **Reactor Feed A** (XMEAS 23) - mol%
- **Product G** (XMEAS 40) - mol%
- **Product H** (XMEAS 41) - mol%

### Utilities Panel
- **Compressor Work** (XMEAS 20) - kW
- **Reactor CW Temp** (XMEAS 21) - °C
- **Separator CW Temp** (XMEAS 22) - °C

## Status Display

The sidebar shows:
- **Simulation Time**: Current time in hours
- **Active Disturbances**: Currently enabled IDVs from Fortran state

## Tips for Effective Use

### Observing Normal Operation
1. Start the simulation without any disturbances
2. Watch the system reach steady state (~5-10 minutes simulated time)
3. Note nominal values for key variables

### Testing Fault Signatures
1. Run at steady state for 1+ hour
2. Check a disturbance (e.g., IDV 6)
3. Click **Apply Disturbances**
4. Observe the characteristic fault signature
5. Enable recording to capture data for analysis

### Generating Training Data
1. Start recording
2. Run normal operation for desired duration
3. Apply a fault at a known time
4. Continue simulation
5. Download CSV for offline analysis

### Comparing Fault Types
1. Reset and run with IDV(4) - cooling water fault
2. Note pressure rise characteristic
3. Reset and run with IDV(6) - A feed loss
4. Compare the different fault signatures

## Troubleshooting

### Dashboard Won't Start

**Error: No module named 'dash'**

Install web dependencies:
```bash
pip install -e ".[web]"
```

### Port Already in Use

Use a different port:
```bash
tep-web --port 8051
```

### Browser Doesn't Open

Open manually at `http://127.0.0.1:8050` or use:
```bash
tep-web --no-browser
```
Then open the URL in your browser.

### Plots Not Updating

- Ensure simulation is running (click Start)
- Check browser console for errors
- Try refreshing the page

### Process Shuts Down

The simulation may shut down if safety limits are exceeded:
- Reactor pressure > 3000 kPa or < 2500 kPa
- Reactor level > 100% or < 2%
- Reactor temperature > 175°C

Try:
1. Click Reset to return to steady state
2. Apply disturbances one at a time
3. Some disturbances (like IDV 6) will cause shutdown - this is expected behavior

## Data Export Format

The CSV export includes columns:

| Column | Description |
|--------|-------------|
| Time_hours | Simulation time |
| XMEAS_1 through XMEAS_41 | All measurements |
| XMV_1 through XMV_12 | Manipulated variables |
| Active_IDVs | Comma-separated list of active disturbances |

Example:
```csv
Time_hours,XMEAS_1,XMEAS_2,...,XMV_1,...,Active_IDVs
0.0,0.251,3664.0,...,63.05,...,
0.5,0.250,3662.1,...,63.10,...,
1.0,0.002,3660.5,...,63.15,...,6
```
