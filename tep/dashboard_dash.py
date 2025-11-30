"""
Web-based Dashboard for the Tennessee Eastman Process Simulator using Dash.

This module provides an interactive web-based interface for:
- Controlling manipulated variables
- Enabling/disabling process disturbances
- Real-time visualization of process measurements
- Simulation control (start, stop, reset)

Requirements:
    - dash
    - plotly

Usage:
    python -m tep.dashboard_dash

    Or from Python:
        from tep.dashboard_dash import run_dashboard
        run_dashboard()
"""

import numpy as np
import webbrowser
import threading
import logging
import sys

# Configure logging for TEP dashboard
logging.basicConfig(
    level=logging.INFO,
    format='[TEP] %(asctime)s %(levelname)s: %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('tep.dashboard')
logger.setLevel(logging.INFO)

# Suppress Flask/Werkzeug logging early (before app is created)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('flask.app').setLevel(logging.WARNING)

from dash import Dash, html, dcc, Output, Input, State, ctx, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .simulator import TEPSimulator, ControlMode
from .constants import (
    NUM_MEASUREMENTS, NUM_MANIPULATED_VARS, NUM_DISTURBANCES, INITIAL_STATES,
    SAFETY_LIMITS, MEASUREMENT_NAMES, MANIPULATED_VAR_NAMES
)
from . import get_available_backends, get_default_backend, __version__, is_fortran_available

# Log startup info
logger.info(f"TEP Dashboard v{__version__} starting...")
logger.info(f"Available backends: {get_available_backends()}")
logger.info(f"Fortran available: {is_fortran_available()}")
logger.info(f"Default backend: {get_default_backend()}")


# Global simulator instance and data storage
simulator = None
sim_data = {
    'time': [],
    'measurements': {i: [] for i in range(NUM_MEASUREMENTS)},
    'mvs': {i: [] for i in range(NUM_MANIPULATED_VARS)},
    'idv': [],  # Track active disturbances at each time point (list of active IDV numbers)
    'running': False,
    'max_display_points': 2000,  # Max points to display (decimated from full data)
    'output_interval': 180,  # Seconds between data recordings (default: 3 min like Fortran)
    'last_output_time': 0,  # Last time data was recorded (in seconds)
}

# MV short names
MV_SHORT_NAMES = [
    "D Feed Flow", "E Feed Flow", "A Feed Flow", "A+C Feed Flow",
    "Recycle Valve", "Purge Valve", "Sep Liq Flow", "Strip Liq Flow",
    "Steam Valve", "React CW Flow", "Cond CW Flow", "Agitator Speed"
]

# Disturbance names and descriptions - IDV(1-20)
IDV_INFO = [
    ("IDV(1) A/C Ratio", "Step change in A/C feed ratio (stream 4)"),
    ("IDV(2) B Comp", "Step change in B composition (stream 4)"),
    ("IDV(3) D Feed Temp", "Step change in D feed temperature (stream 2)"),
    ("IDV(4) Reactor CW", "Step change in reactor cooling water inlet temp"),
    ("IDV(5) Condenser CW", "Step change in condenser cooling water inlet temp"),
    ("IDV(6) A Feed Loss", "Loss of A feed (stream 1) - major disruption!"),
    ("IDV(7) C Header", "C header pressure loss (stream 4)"),
    ("IDV(8) A,B,C Comp", "Random variation in A,B,C feed composition"),
    ("IDV(9) D Temp Rand", "Random variation in D feed temperature"),
    ("IDV(10) C Temp Rand", "Random variation in C feed temperature"),
    ("IDV(11) React CW Rand", "Random reactor cooling water inlet temp"),
    ("IDV(12) Cond CW Rand", "Random condenser cooling water inlet temp"),
    ("IDV(13) Kinetics", "Slow drift in reaction kinetics"),
    ("IDV(14) React Valve", "Reactor cooling water valve sticking"),
    ("IDV(15) Cond Valve", "Condenser cooling water valve sticking"),
    ("IDV(16)", "Unknown disturbance"),
    ("IDV(17)", "Unknown disturbance"),
    ("IDV(18)", "Unknown disturbance"),
    ("IDV(19)", "Unknown disturbance"),
    ("IDV(20)", "Unknown disturbance"),
]

# Plot configurations: (title, [(label, measurement_index, secondary_y), ...], use_dual_y)
# Organized by unit operation to show all relevant process measurements
# XMEAS indices are 0-based (XMEAS(n) -> index n-1)
# secondary_y=True means use right y-axis (for different magnitude signals)
PLOT_CONFIGS = [
    # Row 1: Feeds - D,E (kg/hr ~3600-4500) vs A,A+C (kscmh ~0.25-9.35)
    ("Feed Flows", [
        ("XMEAS(1) A Feed", 0, False),      # kscmh ~0.25
        ("XMEAS(4) A+C Feed", 3, False),    # kscmh ~9.35
        ("XMEAS(2) D Feed", 1, True),       # kg/hr ~3600
        ("XMEAS(3) E Feed", 2, True),       # kg/hr ~4500
    ], True),
    ("Feed & Recycle", [
        ("XMEAS(5) Recycle", 4, False),     # kscmh
        ("XMEAS(6) React Feed", 5, False),  # kscmh
    ], False),
    # Row 2: Reactor - Pressure (kPa ~2705) vs Level/Temp (~65%/~122°C)
    ("Reactor", [
        ("XMEAS(8) Level", 7, False),       # % ~65
        ("XMEAS(9) Temp", 8, False),        # °C ~122
        ("XMEAS(7) Pressure", 6, True),     # kPa ~2705
    ], True),
    ("Reactor CW & Work", [
        ("XMEAS(21) CW Out", 20, False),    # °C ~92
        ("XMEAS(20) Comp Work", 19, True),  # kW ~341
    ], True),
    # Row 3: Separator - Pressure (kPa ~2633) vs Temp/Level (~80°C/~50%)
    ("Separator", [
        ("XMEAS(11) Temp", 10, False),      # °C ~80
        ("XMEAS(12) Level", 11, False),     # % ~50
        ("XMEAS(13) Pressure", 12, True),   # kPa ~2633
    ], True),
    ("Separator Flows", [
        ("XMEAS(10) Purge", 9, False),      # kscmh
        ("XMEAS(14) Underflow", 13, False), # m3/hr
        ("XMEAS(22) CW Out", 21, True),     # °C (different scale)
    ], True),
    # Row 4: Stripper - Pressure (kPa ~3102) vs Level/Temp (~50%/~65°C)
    ("Stripper", [
        ("XMEAS(15) Level", 14, False),     # % ~50
        ("XMEAS(18) Temp", 17, False),      # °C ~65
        ("XMEAS(16) Pressure", 15, True),   # kPa ~3102
    ], True),
    ("Stripper Flows", [
        ("XMEAS(17) Product", 16, False),   # m3/hr ~22
        ("XMEAS(19) Steam", 18, True),      # kg/hr ~230
    ], True),
    # Row 5: Compositions - all in mol%, similar scales
    ("Reactor Feed Comp", [
        ("XMEAS(23) A%", 22, False),
        ("XMEAS(24) B%", 23, False),
        ("XMEAS(25) C%", 24, False),
        ("XMEAS(26) D%", 25, False),
    ], False),
    ("Product Comp", [
        ("XMEAS(37) D%", 36, False),
        ("XMEAS(38) E%", 37, False),
        ("XMEAS(39) F%", 38, False),
        ("XMEAS(40) G%", 39, False),
    ], False),
]


def init_simulator(backend=None):
    """Initialize or reset the simulator."""
    global simulator, sim_data
    if backend is None:
        backend = get_default_backend()
    logger.info(f"Initializing simulator with backend: {backend}")
    try:
        simulator = TEPSimulator(control_mode=ControlMode.CLOSED_LOOP, backend=backend)
        simulator.initialize()
        logger.info(f"Simulator initialized successfully, actual backend: {simulator.backend}")
    except Exception as e:
        logger.error(f"Failed to initialize simulator with {backend}: {e}")
        # Fall back to Python if Fortran fails
        if backend == 'fortran':
            logger.info("Falling back to Python backend")
            backend = 'python'
            simulator = TEPSimulator(control_mode=ControlMode.CLOSED_LOOP, backend=backend)
            simulator.initialize()
        else:
            raise
    sim_data['time'] = []
    sim_data['measurements'] = {i: [] for i in range(NUM_MEASUREMENTS)}
    sim_data['mvs'] = {i: [] for i in range(NUM_MANIPULATED_VARS)}
    sim_data['idv'] = []
    sim_data['running'] = False
    sim_data['shutdown_reason'] = None
    sim_data['last_output_time'] = 0
    sim_data['backend'] = backend


def get_shutdown_reason(meas):
    """Determine the reason for process shutdown based on measurements."""
    reasons = []
    limits = SAFETY_LIMITS

    # Check for NaN/Inf (numerical instability)
    if not np.all(np.isfinite(meas)):
        return "Numerical instability detected"

    # Check each safety limit
    if meas[6] > limits.reactor_pressure_max:
        reasons.append(f"Reactor pressure ({meas[6]:.0f} kPa) exceeded {limits.reactor_pressure_max:.0f} kPa limit")

    if meas[8] > limits.reactor_temp_max:
        reasons.append(f"Reactor temperature ({meas[8]:.1f}°C) exceeded {limits.reactor_temp_max:.1f}°C limit")

    # Level checks - reactor level (XMEAS 8, index 7)
    # The simulator checks volume: vlr/35.3145 > 24 or < 2 (in m³)
    # XMEAS(8) is reactor level in % (0-100 scale)
    reactor_level = meas[7]
    if reactor_level > 100:
        reasons.append(f"Reactor level ({reactor_level:.1f}%) too high")
    if reactor_level < 10:
        reasons.append(f"Reactor level ({reactor_level:.1f}%) too low")

    # Separator level (XMEAS 12, index 11)
    sep_level = meas[11]
    if sep_level > 100:
        reasons.append(f"Separator level ({sep_level:.1f}%) too high")
    if sep_level < 10:
        reasons.append(f"Separator level ({sep_level:.1f}%) too low")

    # Stripper level (XMEAS 15, index 14)
    stripper_level = meas[14]
    if stripper_level > 100:
        reasons.append(f"Stripper level ({stripper_level:.1f}%) too high")
    if stripper_level < 10:
        reasons.append(f"Stripper level ({stripper_level:.1f}%) too low")

    if reasons:
        return " | ".join(reasons)

    # If we get here, shutdown was triggered but we couldn't identify why
    # This can happen due to timing between when shutdown is detected and when we read measurements
    return "Process unstable - safety shutdown triggered"


def create_layout():
    """Create the Dash layout."""
    initial_mvs = INITIAL_STATES[38:50]

    return html.Div([
        # Header
        html.Div([
            html.H1("Tennessee Eastman Process Simulator",
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '5px'}),
            html.P("Interactive Process Control Dashboard",
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '0', 'marginBottom': '5px'}),
            html.P([
                f"v{__version__} | ",
                "By John Kitchin | ",
                html.A("GitHub Repository",
                      href="https://github.com/jkitchin/tennessee-eastman-profbraatz",
                      target="_blank",
                      style={'color': '#3498db', 'textDecoration': 'none'})
            ], style={'textAlign': 'center', 'color': '#95a5a6', 'marginTop': '0', 'fontSize': '12px'})
        ], style={'backgroundColor': '#ecf0f1', 'padding': '10px', 'marginBottom': '10px'}),

        # Shutdown Alert - initially hidden
        html.Div(
            id='shutdown-alert',
            children=[
                html.Div([
                    html.H2("PROCESS SHUTDOWN", style={
                        'margin': '0', 'color': 'white', 'textAlign': 'center'
                    }),
                    html.P(id='shutdown-reason', children="", style={
                        'margin': '5px 0 0 0', 'color': 'white', 'textAlign': 'center',
                        'fontSize': '14px'
                    }),
                    html.P("Click 'Reset' to restart the simulation", style={
                        'margin': '5px 0 0 0', 'color': '#ffcccc', 'textAlign': 'center',
                        'fontSize': '12px'
                    })
                ])
            ],
            style={
                'display': 'none',  # Hidden by default
                'backgroundColor': '#c0392b',
                'padding': '20px',
                'marginBottom': '10px',
                'marginLeft': '15px',
                'marginRight': '15px',
                'borderRadius': '5px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.3)',
                'animation': 'pulse 1s infinite'
            }
        ),

        # Main content
        html.Div([
            # Left panel - Controls
            html.Div([
                # Simulation controls
                html.Div([
                    html.H3("Simulation Control", style={'marginTop': '0'}),

                    # Backend selection
                    html.Label("Backend:"),
                    dcc.Dropdown(
                        id='backend-dropdown',
                        options=[{'label': b.capitalize(), 'value': b} for b in get_available_backends()],
                        value=get_default_backend(),
                        clearable=False,
                        style={'marginBottom': '10px'}
                    ),

                    # Control mode
                    html.Label("Control Mode:"),
                    dcc.RadioItems(
                        id='control-mode',
                        options=[
                            {'label': ' Closed Loop', 'value': 'closed_loop'},
                            {'label': ' Manual', 'value': 'manual'}
                        ],
                        value='closed_loop',
                        inline=True,
                        style={'marginBottom': '10px'}
                    ),

                    # Speed control
                    html.Label("Simulation Speed (steps/update):"),
                    dcc.Slider(
                        id='speed-slider',
                        min=1,
                        max=50,
                        value=50,
                        marks={1: '1', 10: '10', 25: '25', 50: '50'},
                        step=1
                    ),

                    # Output interval control
                    html.Label("Data Output Interval (sec):"),
                    dcc.Slider(
                        id='output-interval-slider',
                        min=1,
                        max=300,
                        value=180,
                        marks={1: '1s', 60: '1m', 180: '3m', 300: '5m'},
                        step=1,
                        tooltip={'placement': 'bottom', 'always_visible': False}
                    ),

                    # Buttons
                    html.Div([
                        html.Button('Start', id='start-btn', n_clicks=0,
                                   style={'marginRight': '5px', 'backgroundColor': '#27ae60', 'color': 'white',
                                         'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer'}),
                        html.Button('Stop', id='stop-btn', n_clicks=0,
                                   style={'marginRight': '5px', 'backgroundColor': '#e74c3c', 'color': 'white',
                                         'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer'}),
                        html.Button('Reset', id='reset-btn', n_clicks=0,
                                   style={'backgroundColor': '#3498db', 'color': 'white',
                                         'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer'}),
                    ], style={'marginTop': '15px', 'marginBottom': '10px'}),

                    # Download button
                    html.Div([
                        html.Button('Download Data (CSV)', id='download-btn', n_clicks=0,
                                   style={'backgroundColor': '#9b59b6', 'color': 'white',
                                         'border': 'none', 'padding': '8px 15px', 'cursor': 'pointer',
                                         'width': '100%', 'fontSize': '12px'}),
                        dcc.Download(id='download-data'),
                    ], style={'marginBottom': '15px'}),

                    # Status
                    html.Div([
                        html.Span("Status: ", style={'fontWeight': 'bold'}),
                        html.Span(id='status-text', children="Ready",
                                 style={'color': '#27ae60'})
                    ]),
                    html.Div([
                        html.Span("Time: ", style={'fontWeight': 'bold'}),
                        html.Span(id='time-text', children="0.00 hr")
                    ]),
                    html.Div([
                        html.Span("Active Faults (Fortran): ", style={'fontWeight': 'bold'}),
                        html.Span(id='active-faults-text', children="None",
                                 style={'color': '#27ae60'})
                    ], style={'marginTop': '5px'}),
                ], style={'backgroundColor': '#fff', 'padding': '15px', 'borderRadius': '5px',
                         'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '15px'}),

                # Manipulated Variables
                html.Div([
                    html.H3("Manipulated Variables", style={'marginTop': '0'}),
                    html.Div([
                        html.Div([
                            html.Label(f"{i+1}. {MV_SHORT_NAMES[i]}:",
                                      style={'fontSize': '12px', 'marginBottom': '2px'}),
                            dcc.Slider(
                                id=f'mv-slider-{i}',
                                min=0,
                                max=100,
                                value=initial_mvs[i],
                                marks={0: '0', 50: '50', 100: '100'},
                                step=0.1,
                                tooltip={'placement': 'right', 'always_visible': True}
                            )
                        ], style={'marginBottom': '10px'})
                        for i in range(NUM_MANIPULATED_VARS)
                    ], style={'maxHeight': '400px', 'overflowY': 'auto'})
                ], style={'backgroundColor': '#fff', 'padding': '15px', 'borderRadius': '5px',
                         'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'})
            ], style={'width': '300px', 'flexShrink': '0', 'marginRight': '15px'}),

            # Center - Main plots and All Variables (no tabs)
            html.Div([
                # Welcome message shown before simulation starts
                html.Div(id='welcome-message', children=[
                    html.Div([
                        html.H2("Welcome to the TEP Simulator",
                               style={'color': '#2c3e50', 'marginBottom': '20px'}),
                        html.P("Press the Start button to begin the simulation.",
                              style={'fontSize': '18px', 'color': '#7f8c8d', 'marginBottom': '30px'}),
                        html.Div([
                            html.H4("Quick Start Guide:", style={'color': '#34495e', 'marginBottom': '15px'}),
                            html.Ul([
                                html.Li("Adjust simulation speed with the slider"),
                                html.Li("Set data output interval (how often points are recorded)"),
                                html.Li("Enable disturbances on the right panel to test fault scenarios"),
                                html.Li("Switch to Manual mode to control valves directly"),
                                html.Li("View measurements by unit: Feeds, Reactor, Separator, Stripper, Compositions"),
                            ], style={'textAlign': 'left', 'color': '#555', 'lineHeight': '1.8'})
                        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px',
                                 'maxWidth': '600px', 'margin': '0 auto', 'marginBottom': '20px'}),
                        # References section
                        html.Div([
                            html.H4("References:", style={'color': '#34495e', 'marginBottom': '10px'}),
                            html.Ul([
                                html.Li([
                                    "Downs, J.J. and Vogel, E.F. (1993). ",
                                    html.I("A plant-wide industrial process control problem"),
                                    ". Computers & Chemical Engineering, 17(3), 245-255. ",
                                    html.A("DOI:10.1016/0098-1354(93)80018-I",
                                          href="https://doi.org/10.1016/0098-1354(93)80018-I",
                                          target="_blank", style={'color': '#3498db'})
                                ]),
                                html.Li([
                                    "Ricker, N.L. (1996). ",
                                    html.I("Decentralized control of the Tennessee Eastman Challenge Process"),
                                    ". Journal of Process Control, 6(4), 205-221. ",
                                    html.A("DOI:10.1016/0959-1524(96)00031-5",
                                          href="https://doi.org/10.1016/0959-1524(96)00031-5",
                                          target="_blank", style={'color': '#3498db'})
                                ]),
                                html.Li([
                                    "Bathelt, A., Ricker, N.L., and Jelali, M. (2015). ",
                                    html.I("Revision of the Tennessee Eastman Process Model"),
                                    ". IFAC-PapersOnLine, 48(8), 309-314. ",
                                    html.A("DOI:10.1016/j.ifacol.2015.08.199",
                                          href="https://doi.org/10.1016/j.ifacol.2015.08.199",
                                          target="_blank", style={'color': '#3498db'})
                                ]),
                            ], style={'textAlign': 'left', 'color': '#555', 'lineHeight': '1.6',
                                     'fontSize': '13px', 'paddingLeft': '20px'})
                        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px',
                                 'maxWidth': '600px', 'margin': '0 auto'})
                    ], style={'textAlign': 'center', 'padding': '50px 20px'})
                ], style={'display': 'block'}),

                # Main process plots (5 rows x 2 cols = 10 subplots)
                dcc.Graph(id='main-plots', style={'height': '1100px', 'display': 'none'})
            ], style={'flexGrow': '1', 'backgroundColor': '#fff', 'borderRadius': '5px',
                     'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'padding': '10px',
                     'overflowY': 'auto', 'maxHeight': 'calc(100vh - 150px)'}),

            # Right panel - Disturbances
            html.Div([
                html.H3("Disturbances", style={'marginTop': '0'}),
                # Active disturbances display
                html.Div([
                    html.Span("Active: ", style={'fontWeight': 'bold', 'fontSize': '12px'}),
                    html.Span(id='active-idv-display', children="None",
                             style={'fontSize': '12px', 'color': '#27ae60'})
                ], style={'marginBottom': '10px', 'padding': '8px',
                         'backgroundColor': '#f8f9fa', 'borderRadius': '4px'}),
                html.P("Select faults, then click Apply:",
                      style={'fontSize': '12px', 'color': '#7f8c8d', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id=f'idv-{i}',
                            options=[{'label': f" {IDV_INFO[i][0]}", 'value': i + 1}],
                            value=[],
                            style={'display': 'inline-block'}
                        ),
                        html.Div(
                            IDV_INFO[i][1],
                            style={
                                'fontSize': '10px',
                                'color': '#95a5a6',
                                'marginLeft': '22px',
                                'marginBottom': '8px',
                                'lineHeight': '1.3'
                            }
                        )
                    ]) for i in range(NUM_DISTURBANCES)
                ], style={'maxHeight': '550px', 'overflowY': 'auto'}),
                html.Div([
                    html.Button('Apply Disturbances', id='apply-disturbances-btn', n_clicks=0,
                               style={'marginTop': '10px', 'backgroundColor': '#e74c3c', 'color': 'white',
                                     'border': 'none', 'padding': '8px 15px', 'cursor': 'pointer', 'width': '100%',
                                     'fontWeight': 'bold'}),
                    html.Button('Clear All', id='clear-disturbances-btn', n_clicks=0,
                               style={'marginTop': '5px', 'backgroundColor': '#95a5a6', 'color': 'white',
                                     'border': 'none', 'padding': '8px 15px', 'cursor': 'pointer', 'width': '100%'})
                ])
            ], style={'width': '280px', 'flexShrink': '0', 'marginLeft': '15px',
                     'backgroundColor': '#fff', 'padding': '15px', 'borderRadius': '5px',
                     'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'})
        ], style={'display': 'flex', 'padding': '0 15px'}),

        # Interval for updates (200ms = 5 updates/sec for smooth but responsive UI)
        dcc.Interval(id='interval-component', interval=200, n_intervals=0, disabled=True),

        # Store for simulation state
        dcc.Store(id='sim-state', data={'running': False, 'speed': 50, 'output_interval': 180}),

        # Dummy store for apply disturbances callback
        dcc.Store(id='apply-disturbances-dummy', data={})
    ], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ecf0f1',
              'minHeight': '100vh', 'paddingBottom': '20px'})


def decimate_data(data, max_points):
    """Decimate data to at most max_points while preserving shape.

    Uses simple striding to reduce data size for display.
    Always includes first and last points.
    """
    if not data or len(data) <= max_points:
        return data

    n = len(data)
    step = max(1, n // max_points)

    # Use numpy for efficient slicing if data is large
    if isinstance(data, np.ndarray):
        indices = np.arange(0, n, step)
        # Always include last point
        if indices[-1] != n - 1:
            indices = np.append(indices, n - 1)
        return data[indices].tolist()
    else:
        # List version
        result = data[::step]
        if len(data) > 1 and (len(data) - 1) % step != 0:
            result = list(result) + [data[-1]]
        return result


def create_empty_figure():
    """Create an empty figure with subplots (with secondary y-axes where needed)."""
    n_rows = 5
    n_cols = 2

    # Build specs for secondary_y based on PLOT_CONFIGS
    specs = []
    for row_idx in range(n_rows):
        row_specs = []
        for col_idx in range(n_cols):
            cfg_idx = row_idx * n_cols + col_idx
            if cfg_idx < len(PLOT_CONFIGS):
                use_dual_y = PLOT_CONFIGS[cfg_idx][2]
                row_specs.append({"secondary_y": use_dual_y})
            else:
                row_specs.append({})
        specs.append(row_specs)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[cfg[0] for cfg in PLOT_CONFIGS],
        vertical_spacing=0.06,
        horizontal_spacing=0.10,
        specs=specs
    )

    # Use distinct colors and line styles for better differentiation
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']
    dashes = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

    # Subplot positions for legends (x, y in paper coordinates) - 5 rows x 2 cols
    legend_positions = [
        (0.02, 0.99), (0.52, 0.99),  # row 1
        (0.02, 0.79), (0.52, 0.79),  # row 2
        (0.02, 0.59), (0.52, 0.59),  # row 3
        (0.02, 0.39), (0.52, 0.39),  # row 4
        (0.02, 0.19), (0.52, 0.19),  # row 5
    ]

    for idx, (title, signals, use_dual_y) in enumerate(PLOT_CONFIGS):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        legend_name = f'legend{idx + 1}' if idx > 0 else 'legend'

        for sig_idx, (label, _, secondary_y) in enumerate(signals):
            fig.add_trace(
                go.Scatter(x=[], y=[], name=label, mode='lines',
                          line=dict(color=colors[sig_idx % len(colors)],
                                   width=2,
                                   dash=dashes[sig_idx % len(dashes)]),
                          showlegend=True, legend=legend_name),
                row=row, col=col,
                secondary_y=secondary_y if use_dual_y else False
            )

        fig.update_xaxes(title_text="Time (min)", row=row, col=col)

    # Create separate legends for each subplot
    legend_configs = {}
    for idx in range(len(PLOT_CONFIGS)):
        legend_name = f'legend{idx + 1}' if idx > 0 else 'legend'
        x_pos, y_pos = legend_positions[idx]
        legend_configs[legend_name] = dict(
            x=x_pos,
            y=y_pos,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1,
            font=dict(size=8)
        )

    fig.update_layout(
        height=1100,
        margin=dict(l=60, r=60, t=40, b=30),
        template='plotly_white',
        **legend_configs
    )

    return fig


# Initialize the app
app = Dash(__name__)
app.title = "TEP Simulator Dashboard"

# Expose the Flask server for production WSGI servers (gunicorn, etc.)
server = app.server
init_simulator()
app.layout = create_layout()


@app.callback(
    Output('interval-component', 'disabled'),
    Output('sim-state', 'data'),
    Output('status-text', 'children'),
    Output('status-text', 'style'),
    Input('start-btn', 'n_clicks'),
    Input('stop-btn', 'n_clicks'),
    Input('reset-btn', 'n_clicks'),
    Input('backend-dropdown', 'value'),
    State('sim-state', 'data'),
    State('speed-slider', 'value'),
    State('output-interval-slider', 'value'),
    prevent_initial_call=True
)
def control_simulation(start_clicks, stop_clicks, reset_clicks, backend, state, speed, output_interval):
    """Handle start/stop/reset button clicks and backend changes."""
    global sim_data

    triggered = ctx.triggered_id

    if triggered == 'start-btn':
        sim_data['running'] = True
        sim_data['output_interval'] = output_interval
        state['running'] = True
        state['speed'] = speed
        state['output_interval'] = output_interval
        return False, state, "Running", {'color': '#27ae60', 'fontWeight': 'bold'}

    elif triggered == 'stop-btn':
        sim_data['running'] = False
        state['running'] = False
        return True, state, "Stopped", {'color': '#e74c3c'}

    elif triggered == 'reset-btn':
        init_simulator(backend)
        state['running'] = False
        return True, state, "Ready", {'color': '#27ae60'}

    elif triggered == 'backend-dropdown':
        # Backend changed - reinitialize simulator with new backend
        init_simulator(backend)
        state['running'] = False
        return True, state, f"Ready ({backend})", {'color': '#27ae60'}

    return True, state, "Ready", {'color': '#27ae60'}


@app.callback(
    Output('main-plots', 'style'),
    Output('welcome-message', 'style'),
    Input('start-btn', 'n_clicks'),
    Input('reset-btn', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_welcome_message(start_clicks, reset_clicks):
    """Show/hide welcome message and plots based on simulation state."""
    triggered = ctx.triggered_id

    graph_visible = {'height': '1100px', 'display': 'block'}
    graph_hidden = {'height': '1100px', 'display': 'none'}
    welcome_visible = {'display': 'block'}
    welcome_hidden = {'display': 'none'}

    if triggered == 'start-btn':
        # Hide welcome, show graphs
        return graph_visible, welcome_hidden
    elif triggered == 'reset-btn':
        # Show welcome, hide graphs (data is cleared in init_simulator)
        return graph_hidden, welcome_visible

    # Default: show welcome
    return graph_hidden, welcome_visible


@app.callback(
    Output('apply-disturbances-dummy', 'data'),
    Output('active-idv-display', 'children', allow_duplicate=True),
    Output('active-idv-display', 'style', allow_duplicate=True),
    Input('apply-disturbances-btn', 'n_clicks'),
    *[State(f'idv-{i}', 'value') for i in range(NUM_DISTURBANCES)],
    prevent_initial_call=True
)
def apply_disturbances(n_clicks, *idv_values):
    """Apply selected disturbances to simulator when button is clicked."""
    active_idvs = []
    if simulator:
        # First clear all, then set the checked ones
        simulator.clear_disturbances()
        for i in range(NUM_DISTURBANCES):
            if idv_values[i] and (i + 1) in idv_values[i]:
                simulator.set_disturbance(i + 1, 1)
                active_idvs.append(i + 1)

    if active_idvs:
        display_text = f"IDV({', '.join(map(str, active_idvs))})"
        display_style = {'fontSize': '12px', 'color': '#e74c3c', 'fontWeight': 'bold'}
    else:
        display_text = "None"
        display_style = {'fontSize': '12px', 'color': '#27ae60'}

    return {'applied': True}, display_text, display_style


@app.callback(
    [Output(f'idv-{i}', 'value') for i in range(NUM_DISTURBANCES)],
    Output('active-idv-display', 'children', allow_duplicate=True),
    Output('active-idv-display', 'style', allow_duplicate=True),
    Input('clear-disturbances-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_disturbances(n_clicks):
    """Clear all disturbances."""
    if simulator:
        simulator.clear_disturbances()
    # Return empty checkboxes + reset display (must be flat tuple of 22 values)
    empty_checkboxes = [[] for _ in range(NUM_DISTURBANCES)]
    return (*empty_checkboxes, "None", {'fontSize': '12px', 'color': '#27ae60'})


@app.callback(
    Output('main-plots', 'figure'),
    Output('time-text', 'children'),
    Output('shutdown-alert', 'style'),
    Output('shutdown-reason', 'children'),
    Output('active-faults-text', 'children'),
    Output('active-faults-text', 'style'),
    Input('interval-component', 'n_intervals'),
    State('sim-state', 'data'),
    State('control-mode', 'value'),
    *[State(f'mv-slider-{i}', 'value') for i in range(NUM_MANIPULATED_VARS)],
    prevent_initial_call=True
)
def update_simulation(n_intervals, state, control_mode, *mv_values):
    """Run simulation step and update plots."""
    global simulator, sim_data

    # Default styles
    hidden_style = {'display': 'none'}
    shutdown_style = {
        'display': 'block',
        'backgroundColor': '#c0392b',
        'padding': '20px',
        'marginBottom': '10px',
        'marginLeft': '15px',
        'marginRight': '15px',
        'borderRadius': '5px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'
    }

    # Helper to get active faults display
    def get_faults_display():
        if simulator:
            active = simulator.get_active_disturbances()
            if active:
                return f"IDV({', '.join(map(str, active))})", {'color': '#e74c3c', 'fontWeight': 'bold'}
        return "None", {'color': '#27ae60'}

    # Check if already shutdown
    if simulator and simulator.is_shutdown():
        reason = sim_data.get('shutdown_reason', 'Safety limit violation')
        faults_text, faults_style = get_faults_display()
        return (create_figure_with_data(),
                f"{simulator.time:.2f} hr ({simulator.time*60:.1f} min)",
                shutdown_style, reason, faults_text, faults_style)

    if not state.get('running', False) or simulator is None:
        # Return current state without updating
        time_str = f"{simulator.time:.2f} hr" if simulator else "0.00 hr"
        faults_text, faults_style = get_faults_display()
        return create_figure_with_data(), time_str, hidden_style, "", faults_text, faults_style

    # Update control mode
    if control_mode == 'closed_loop':
        if simulator.control_mode != ControlMode.CLOSED_LOOP:
            simulator.control_mode = ControlMode.CLOSED_LOOP
            simulator._init_controller()
    else:
        if simulator.control_mode != ControlMode.MANUAL:
            simulator.control_mode = ControlMode.MANUAL
            simulator._init_controller()
        # Set MV values in manual mode
        for i, val in enumerate(mv_values):
            if val is not None:
                simulator.set_mv(i + 1, val)

    # Disturbances are now only modified via Apply/Clear buttons, not here.
    # This prevents any callback timing issues from affecting the simulation.

    speed = state.get('speed', 10)
    output_interval = sim_data.get('output_interval', 180)  # Default 180 sec (3 min)
    shutdown_occurred = False

    # Run simulation steps and record data at specified output interval
    for _ in range(speed):
        if not simulator.step():
            sim_data['running'] = False
            # Get measurements to determine shutdown reason
            meas = simulator.get_measurements()
            reason = get_shutdown_reason(meas)
            sim_data['shutdown_reason'] = reason
            logger.warning(f"Process shutdown at t={simulator.time:.4f}h: {reason}")
            logger.info(f"Measurements at shutdown: pressure={meas[6]:.1f}, temp={meas[8]:.1f}, reactor_level={meas[7]:.1f}")
            shutdown_occurred = True
            break

        # Record data only at specified output interval (like Fortran's 180 sec default)
        current_time_sec = simulator.time * 3600  # Convert hours to seconds
        if current_time_sec - sim_data['last_output_time'] >= output_interval:
            sim_data['last_output_time'] = current_time_sec
            sim_data['time'].append(simulator.time * 60)  # Convert to minutes for display

            meas = simulator.get_measurements()
            for i in range(NUM_MEASUREMENTS):
                sim_data['measurements'][i].append(meas[i])

            mvs = simulator.get_manipulated_vars()
            for i in range(NUM_MANIPULATED_VARS):
                sim_data['mvs'][i].append(mvs[i])

            # Record active disturbances directly from Fortran IDV array
            # This is the true source of truth for what the simulation is using
            sim_data['idv'].append(simulator.get_active_disturbances())

    # Limit total data stored to prevent memory issues
    # Keep max 20,000 points - at speed=10, this is ~3 min of real time before decimation
    # After decimation cycles, resolution decreases but full time range is preserved
    max_stored_points = 20000
    if len(sim_data['time']) > max_stored_points:
        # Decimate stored data by factor of 2 to free memory
        # This preserves the full time range but with fewer points
        step = 2
        sim_data['time'] = sim_data['time'][::step]
        for i in range(NUM_MEASUREMENTS):
            sim_data['measurements'][i] = sim_data['measurements'][i][::step]
        for i in range(NUM_MANIPULATED_VARS):
            sim_data['mvs'][i] = sim_data['mvs'][i][::step]
        sim_data['idv'] = sim_data['idv'][::step]

    time_str = f"{simulator.time:.2f} hr ({simulator.time*60:.1f} min)"

    faults_text, faults_style = get_faults_display()

    if shutdown_occurred:
        return (create_figure_with_data(), time_str,
                shutdown_style, sim_data.get('shutdown_reason', 'Safety limit violation'),
                faults_text, faults_style)

    return create_figure_with_data(), time_str, hidden_style, "", faults_text, faults_style


def create_figure_with_data():
    """Create figure with current simulation data (with secondary y-axes where needed)."""
    n_rows = 5
    n_cols = 2

    # Build specs for secondary_y based on PLOT_CONFIGS
    specs = []
    for row_idx in range(n_rows):
        row_specs = []
        for col_idx in range(n_cols):
            cfg_idx = row_idx * n_cols + col_idx
            if cfg_idx < len(PLOT_CONFIGS):
                use_dual_y = PLOT_CONFIGS[cfg_idx][2]
                row_specs.append({"secondary_y": use_dual_y})
            else:
                row_specs.append({})
        specs.append(row_specs)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[cfg[0] for cfg in PLOT_CONFIGS],
        vertical_spacing=0.06,
        horizontal_spacing=0.10,
        specs=specs
    )

    # Use distinct colors and line styles for better differentiation
    # Colors: blue, orange, green, purple, brown, pink
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']
    # Line dashes: solid, dash, dot, dashdot
    dashes = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
    max_points = sim_data.get('max_display_points', 2000)

    # Decimate time data for display
    time_data_full = sim_data['time']
    time_data = decimate_data(time_data_full, max_points)

    # Subplot positions for legends (x, y in paper coordinates) - 5 rows x 2 cols
    legend_positions = [
        (0.02, 0.99), (0.52, 0.99),  # row 1
        (0.02, 0.79), (0.52, 0.79),  # row 2
        (0.02, 0.59), (0.52, 0.59),  # row 3
        (0.02, 0.39), (0.52, 0.39),  # row 4
        (0.02, 0.19), (0.52, 0.19),  # row 5
    ]

    # Global trace counter for unique colors/dashes across all subplots
    trace_idx = 0

    for idx, (title, signals, use_dual_y) in enumerate(PLOT_CONFIGS):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Track min/max for primary and secondary y-axes separately
        y_min_primary = float('inf')
        y_max_primary = float('-inf')
        y_min_secondary = float('inf')
        y_max_secondary = float('-inf')

        # Determine legend name for this subplot
        legend_name = f'legend{idx + 1}' if idx > 0 else 'legend'

        for sig_idx, (label, meas_idx, secondary_y) in enumerate(signals):
            y_data_full = sim_data['measurements'].get(meas_idx, [])

            # Compute min/max for appropriate axis
            if y_data_full:
                if secondary_y and use_dual_y:
                    y_min_secondary = min(y_min_secondary, min(y_data_full))
                    y_max_secondary = max(y_max_secondary, max(y_data_full))
                else:
                    y_min_primary = min(y_min_primary, min(y_data_full))
                    y_max_primary = max(y_max_primary, max(y_data_full))

            # Decimate for display
            y_data = decimate_data(y_data_full, max_points)

            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=y_data,
                    name=label,
                    mode='lines',
                    line=dict(color=colors[sig_idx % len(colors)],
                             width=2,
                             dash=dashes[sig_idx % len(dashes)]),
                    legend=legend_name,
                    showlegend=True,
                ),
                row=row, col=col,
                secondary_y=secondary_y if use_dual_y else False
            )
            trace_idx += 1

        # Set y-axis range for primary axis
        if y_min_primary != float('inf'):
            y_range = y_max_primary - y_min_primary
            if y_range < 1e-6:
                padding = max(abs(y_min_primary) * 0.1, 1.0)
                y_min_plot = y_min_primary - padding
                y_max_plot = y_max_primary + padding
            else:
                margin = y_range * 0.15
                y_min_plot = y_min_primary - margin
                y_max_plot = y_max_primary + margin

            fig.update_yaxes(
                range=[y_min_plot, y_max_plot],
                autorange=False,
                row=row, col=col,
                secondary_y=False
            )

        # Set y-axis range for secondary axis (if used)
        if use_dual_y and y_min_secondary != float('inf'):
            y_range = y_max_secondary - y_min_secondary
            if y_range < 1e-6:
                padding = max(abs(y_min_secondary) * 0.1, 1.0)
                y_min_plot = y_min_secondary - padding
                y_max_plot = y_max_secondary + padding
            else:
                margin = y_range * 0.15
                y_min_plot = y_min_secondary - margin
                y_max_plot = y_max_secondary + margin

            fig.update_yaxes(
                range=[y_min_plot, y_max_plot],
                autorange=False,
                row=row, col=col,
                secondary_y=True
            )

        # Set x-axis to start from 0 and extend to current time (no auto-rescaling)
        if time_data_full:
            x_max = max(time_data_full)  # Use full data for axis range
            # Add small margin to the right
            fig.update_xaxes(title_text="Time (min)", range=[0, x_max * 1.02], row=row, col=col)
        else:
            fig.update_xaxes(title_text="Time (min)", range=[0, 1], row=row, col=col)

    # Create separate legends for each subplot, positioned in upper left corner
    legend_configs = {}
    for idx in range(len(PLOT_CONFIGS)):
        legend_name = f'legend{idx + 1}' if idx > 0 else 'legend'
        x_pos, y_pos = legend_positions[idx]
        legend_configs[legend_name] = dict(
            x=x_pos,
            y=y_pos,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1,
            font=dict(size=9)
        )

    fig.update_layout(
        height=1100,
        margin=dict(l=60, r=60, t=40, b=30),
        template='plotly_white',
        **legend_configs
    )

    return fig


@app.callback(
    Output('download-data', 'data'),
    Input('download-btn', 'n_clicks'),
    prevent_initial_call=True
)
def download_data(n_clicks):
    """Generate CSV data for download."""
    if not sim_data['time']:
        return None

    # Build CSV content
    lines = []

    # Header line with column names
    header = ['Time_hr', 'Time_min']
    # Measurements XMEAS(1-41)
    for i in range(NUM_MEASUREMENTS):
        header.append(f'XMEAS_{i+1}')
    # Manipulated variables XMV(1-12)
    for i in range(NUM_MANIPULATED_VARS):
        header.append(f'XMV_{i+1}')
    # IDV column for active faults
    header.append('Active_IDVs')
    lines.append(','.join(header))

    # Data rows
    n_points = len(sim_data['time'])
    for idx in range(n_points):
        row = []
        time_min = sim_data['time'][idx]
        time_hr = time_min / 60.0
        row.append(f'{time_hr:.6f}')
        row.append(f'{time_min:.4f}')

        # Measurements
        for i in range(NUM_MEASUREMENTS):
            val = sim_data['measurements'][i][idx] if idx < len(sim_data['measurements'][i]) else 0
            row.append(f'{val:.6f}')

        # Manipulated variables
        for i in range(NUM_MANIPULATED_VARS):
            val = sim_data['mvs'][i][idx] if idx < len(sim_data['mvs'][i]) else 0
            row.append(f'{val:.6f}')

        # Active IDVs (as semicolon-separated list or "0" if none)
        if idx < len(sim_data['idv']):
            idv_list = sim_data['idv'][idx]
            if idv_list and len(idv_list) > 0:
                idv_str = ';'.join(str(x) for x in idv_list)
            else:
                idv_str = '0'
        else:
            idv_str = '0'
        row.append(idv_str)

        lines.append(','.join(row))

    csv_content = '\n'.join(lines)

    return dict(
        content=csv_content,
        filename='tep_simulation_data.csv',
        type='text/csv'
    )


def run_dashboard(host='127.0.0.1', port=8050, debug=False, open_browser=True):
    """Run the dashboard application.

    Args:
        host: Host address to bind to (default: 127.0.0.1)
        port: Port number (default: 8050)
        debug: Enable debug mode (default: False)
        open_browser: Automatically open browser (default: True)
    """
    url = f"http://{host}:{port}"

    print(f"\n{'='*60}")
    print("Tennessee Eastman Process Simulator - Web Dashboard")
    print(f"{'='*60}")
    print(f"\nDashboard URL: {url}")
    print("Press Ctrl+C to stop the server\n")

    # Open browser after a short delay to allow server to start
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port, debug=debug)


def main():
    """Entry point for the dashboard."""
    import argparse
    parser = argparse.ArgumentParser(description='Tennessee Eastman Process Web Dashboard')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--port', type=int, default=8050, help='Port number (default: 8050)')
    parser.add_argument('--host', default='127.0.0.1', help='Host address (default: 127.0.0.1)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    run_dashboard(host=args.host, port=args.port, debug=args.debug, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
