# TEP Interactive Dashboard Guide

The TEP Dashboard provides a real-time graphical interface for running and monitoring Tennessee Eastman Process simulations.

## Quick Start

### Installation

Install the GUI dependencies:

```bash
pip install -e ".[gui]"
```

### Launch

From the command line:

```bash
tep-dashboard
```

Or from Python:

```python
from tep import run_dashboard
run_dashboard()
```

Or directly:

```python
from tep.dashboard import TEPDashboard
app = TEPDashboard()
app.run()
```

## Interface Overview

The dashboard window is divided into several sections:

```
+------------------+------------------------------------------+
|                  |                                          |
|   Control Panel  |           Real-Time Plots                |
|                  |                                          |
|  - Start/Stop    |   +----------------+------------------+  |
|  - Reset         |   | Reactor        | Separator        |  |
|  - Speed         |   +----------------+------------------+  |
|  - Control Mode  |   | Stripper       | Flows            |  |
|                  |   +----------------+------------------+  |
|   MV Sliders     |   | Compositions   | Utilities        |  |
|                  |   +----------------+------------------+  |
|   Disturbance    |                                          |
|   Checkboxes     |                                          |
|                  |                                          |
+------------------+------------------------------------------+
```

## Control Panel

### Simulation Controls

| Button | Action |
|--------|--------|
| **Start** | Begin simulation (or resume if paused) |
| **Stop** | Pause simulation |
| **Reset** | Reset to initial steady-state conditions |

### Speed Control

The speed slider adjusts simulation speed:
- **1x**: Real-time (1 simulated second = 1 real second)
- **10x**: 10x faster
- **100x**: 100x faster (useful for observing long-term dynamics)

### Control Mode

| Mode | Description |
|------|-------------|
| **Closed-loop** | Automatic PI control active. MVs are adjusted automatically to maintain setpoints. |
| **Manual** | User controls MVs directly via sliders. Automatic control is disabled. |

## Manipulated Variable Sliders

When in **Manual** control mode, use the 12 MV sliders to adjust valve positions:

| Slider | Variable | Description |
|--------|----------|-------------|
| MV 1 | D Feed Flow | Controls stream 2 flow |
| MV 2 | E Feed Flow | Controls stream 3 flow |
| MV 3 | A Feed Flow | Controls stream 1 flow |
| MV 4 | A+C Feed Flow | Controls stream 4 flow |
| MV 5 | Compressor Recycle | Recycle valve position |
| MV 6 | Purge Valve | Controls stream 9 purge |
| MV 7 | Separator Liquid | Controls stream 10 flow |
| MV 8 | Stripper Product | Controls stream 11 flow |
| MV 9 | Stripper Steam | Steam valve position |
| MV 10 | Reactor CW | Reactor cooling water |
| MV 11 | Condenser CW | Condenser cooling water |
| MV 12 | Agitator Speed | Reactor agitator |

All sliders range from 0% to 100%.

**Note:** In closed-loop mode, sliders show current MV values but adjustments are overridden by the controller.

## Disturbance Checkboxes

Enable process disturbances by checking the corresponding boxes:

### Step Disturbances (IDV 1-7)
These cause immediate step changes in process conditions:

| Checkbox | Effect |
|----------|--------|
| IDV 1 | A/C feed ratio change |
| IDV 2 | B composition change |
| IDV 3 | D feed temperature step |
| IDV 4 | Reactor cooling water temp step |
| IDV 5 | Condenser cooling water temp step |
| IDV 6 | A feed loss |
| IDV 7 | C header pressure loss |

### Random Disturbances (IDV 8-12)
These introduce random variations:

| Checkbox | Effect |
|----------|--------|
| IDV 8 | Random A,B,C feed composition |
| IDV 9 | Random D feed temperature |
| IDV 10 | Random C feed temperature |
| IDV 11 | Random reactor CW inlet temp |
| IDV 12 | Random condenser CW inlet temp |

### Drift and Sticking (IDV 13-15)
| Checkbox | Effect |
|----------|--------|
| IDV 13 | Slow drift in reaction kinetics |
| IDV 14 | Reactor cooling water valve sticking |
| IDV 15 | Condenser cooling water valve sticking |

### Unknown Faults (IDV 16-20)
Reserved for testing fault detection algorithms.

## Real-Time Plots

The dashboard displays 6 plot panels showing key process variables:

### Reactor Panel
- **Temperature** (XMEAS 9) - Reactor temperature in °C
- **Pressure** (XMEAS 7) - Reactor pressure in kPa gauge
- **Level** (XMEAS 8) - Reactor level in %

### Separator Panel
- **Temperature** (XMEAS 11) - Separator temperature in °C
- **Pressure** (XMEAS 13) - Separator pressure in kPa gauge
- **Level** (XMEAS 12) - Separator level in %

### Stripper Panel
- **Temperature** (XMEAS 18) - Stripper temperature in °C
- **Pressure** (XMEAS 16) - Stripper pressure in kPa gauge
- **Level** (XMEAS 15) - Stripper level in %

### Flows Panel
- **Feed Rates** - A, D, E feed flows
- **Recycle** - Recycle flow rate
- **Product** - Stripper underflow rate

### Compositions Panel
- **Reactor Feed** - Components in reactor feed stream
- **Product** - Product stream composition

### Utilities Panel
- **Compressor Work** (XMEAS 20) - in kW
- **Cooling Water Temps** - Reactor and separator CW outlet temperatures

## Status Bar

The bottom of the window shows:
- **Simulation Time**: Current time in hours and minutes
- **Status**: Running, Stopped, or SHUTDOWN
- **Step Count**: Number of simulation steps completed

## Safety Shutdown

The process will shut down automatically if safety limits are exceeded:
- Reactor pressure too high/low
- Reactor level too high/low
- Reactor temperature too high
- Separator level too high/low
- Stripper level too high

When shutdown occurs:
1. The status shows "SHUTDOWN" in red
2. Simulation pauses
3. Click **Reset** to restart from steady state

## Tips for Effective Use

### Observing Normal Operation
1. Start with closed-loop control
2. Watch the system reach steady state (~5-10 minutes)
3. Note nominal values for key variables

### Testing Controller Response
1. Use closed-loop mode
2. Apply a step disturbance (IDV 1-7)
3. Observe how the controllers respond
4. Watch MVs adjust to maintain setpoints

### Exploring Manual Control
1. Switch to Manual mode
2. Start from steady state
3. Make small changes to one MV at a time
4. Observe the process response
5. Try to maintain stable operation manually

### Fault Detection Studies
1. Run in closed-loop at steady state
2. Enable a fault (e.g., IDV 13 for slow drift)
3. Watch for subtle changes in measurements
4. Test if the fault is detectable

### Comparing Control Strategies
1. Run simulation and note performance
2. Reset and try different manual settings
3. Compare closed-loop vs manual response to disturbances

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Start/Stop toggle |
| R | Reset simulation |
| Q | Quit application |

## Troubleshooting

### Dashboard Won't Start

**Error: No module named 'tkinter'**

Install tkinter:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter

# macOS (usually included)
# Windows (usually included)
```

**Error: No module named 'matplotlib'**

Install with GUI extras:
```bash
pip install -e ".[gui]"
```

### Plots Not Updating

- Ensure simulation is running (click Start)
- Check that speed is > 0
- Try resetting the simulation

### Process Keeps Shutting Down

The process may be unstable due to:
- Extreme MV settings in manual mode
- Multiple simultaneous disturbances
- Aggressive control tuning

Try:
1. Reset to steady state
2. Use closed-loop control
3. Apply disturbances one at a time

## Programmatic Dashboard Control

For advanced users, you can programmatically control the dashboard:

```python
from tep.dashboard import TEPDashboard

app = TEPDashboard()

# Pre-configure settings before running
app.simulator.set_disturbance(1, 1)  # Enable IDV(1)

# Custom initialization
app.run()
```

## Screenshots

### Normal Operation
The dashboard at steady state with closed-loop control shows stable reactor temperature around 120°C, separator and stripper levels near 50%, and stable flow rates.

### Disturbance Response
When IDV(1) is applied, you'll see the controllers adjust feed flows and other MVs to maintain setpoints while the process reaches a new steady state.

### Manual Control
In manual mode, the operator must actively adjust MVs to maintain process stability. Without automatic control, the process will drift from setpoints.
