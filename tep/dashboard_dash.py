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

# Suppress Flask/Werkzeug logging early (before app is created)
logging.getLogger('werkzeug').setLevel(logging.CRITICAL)
logging.getLogger('flask.app').setLevel(logging.CRITICAL)

from dash import Dash, html, dcc, Output, Input, State, ctx, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .simulator import TEPSimulator, ControlMode
from .constants import (
    NUM_MEASUREMENTS, NUM_MANIPULATED_VARS, NUM_DISTURBANCES, INITIAL_STATES,
    SAFETY_LIMITS, MEASUREMENT_NAMES, MANIPULATED_VAR_NAMES
)


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

# Plot configurations: (title, [(label, measurement_index), ...])
# XMEAS(1-41) measurements - index = XMEAS number - 1
#  XMEAS(1): A Feed             XMEAS(10): Purge Rate
#  XMEAS(2): D Feed             XMEAS(11): Product Sep Temp
#  XMEAS(3): E Feed             XMEAS(12): Product Sep Level
#  XMEAS(4): A+C Feed           XMEAS(13): Prod Sep Pressure
#  XMEAS(5): Recycle Flow       XMEAS(14): Prod Sep Underflow
#  XMEAS(6): Reactor Feed       XMEAS(15): Stripper Level
#  XMEAS(7): Reactor Pressure   XMEAS(16): Stripper Pressure
#  XMEAS(8): Reactor Level      XMEAS(17): Stripper Underflow (Product)
#  XMEAS(9): Reactor Temp       XMEAS(18): Stripper Temp
# XMEAS(23-28): Reactor Feed Comp  XMEAS(29-36): Purge Gas Comp
# XMEAS(37-41): Product Comp
PLOT_CONFIGS = [
    ("Reactor", [("XMEAS(9) Temp", 8), ("XMEAS(8) Level", 7)]),
    ("Reactor Pressure", [("XMEAS(7) Pressure", 6)]),
    ("Separator", [("XMEAS(11) Temp", 10), ("XMEAS(12) Level", 11)]),
    ("A Feed & Purge", [("XMEAS(1) A Feed", 0), ("XMEAS(10) Purge", 9)]),
    ("Product", [("XMEAS(17) Flow", 16), ("XMEAS(18) Temp", 17)]),
    ("Reactor Feed Comp", [("XMEAS(23) A%", 22), ("XMEAS(25) C%", 24), ("XMEAS(26) D%", 25)]),
]


def init_simulator():
    """Initialize or reset the simulator."""
    global simulator, sim_data
    simulator = TEPSimulator(control_mode=ControlMode.CLOSED_LOOP)
    simulator.initialize()
    sim_data['time'] = []
    sim_data['measurements'] = {i: [] for i in range(NUM_MEASUREMENTS)}
    sim_data['mvs'] = {i: [] for i in range(NUM_MANIPULATED_VARS)}
    sim_data['idv'] = []
    sim_data['running'] = False
    sim_data['shutdown_reason'] = None
    sim_data['last_output_time'] = 0


def get_shutdown_reason(meas):
    """Determine the reason for process shutdown based on measurements."""
    reasons = []
    limits = SAFETY_LIMITS

    # Check each safety limit
    if meas[6] > limits.reactor_pressure_max:
        reasons.append(f"Reactor pressure ({meas[6]:.0f} kPa) exceeded {limits.reactor_pressure_max:.0f} kPa limit")

    if meas[8] > limits.reactor_temp_max:
        reasons.append(f"Reactor temperature ({meas[8]:.1f}°C) exceeded {limits.reactor_temp_max:.1f}°C limit")

    # Note: Level measurements are in % but internal check uses volume
    # These are approximate checks based on typical operating ranges
    reactor_level = meas[7]
    if reactor_level > 95:  # High reactor level warning
        reasons.append(f"Reactor level ({reactor_level:.1f}%) too high")
    if reactor_level < 5:  # Low reactor level warning
        reasons.append(f"Reactor level ({reactor_level:.1f}%) too low")

    sep_level = meas[11]
    if sep_level > 95:
        reasons.append(f"Separator level ({sep_level:.1f}%) too high")
    if sep_level < 5:
        reasons.append(f"Separator level ({sep_level:.1f}%) too low")

    stripper_level = meas[14]
    if stripper_level > 95:
        reasons.append(f"Stripper level ({stripper_level:.1f}%) too high")
    if stripper_level < 5:
        reasons.append(f"Stripper level ({stripper_level:.1f}%) too low")

    if reasons:
        return " | ".join(reasons)
    return "Safety limit violation (unknown)"


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

            # Center - Tabs with Plots and All Variables
            html.Div([
                dcc.Tabs(id='main-tabs', value='plots-tab', children=[
                    dcc.Tab(label='Process Plots', value='plots-tab', children=[
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
                                        html.Li("Use the All Variables tab to see all 41 measurements"),
                                    ], style={'textAlign': 'left', 'color': '#555', 'lineHeight': '1.8'})
                                ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px',
                                         'maxWidth': '500px', 'margin': '0 auto'})
                            ], style={'textAlign': 'center', 'padding': '100px 20px'})
                        ], style={'display': 'block'}),
                        # Graph hidden initially, shown when simulation runs
                        dcc.Graph(id='main-plots', style={'height': '850px', 'display': 'none'})
                    ]),
                    dcc.Tab(label='All Variables', value='variables-tab', children=[
                        html.Div([
                            # Measurements plots
                            html.H4("Process Measurements (XMEAS 1-41)",
                                   style={'marginTop': '10px', 'marginBottom': '10px', 'color': '#2c3e50'}),
                            dcc.Graph(id='measurements-grid', style={'height': '600px'}),

                            # MVs plots
                            html.H4("Manipulated Variables (XMV 1-12)",
                                   style={'marginTop': '10px', 'marginBottom': '10px', 'color': '#2c3e50'}),
                            dcc.Graph(id='mvs-grid', style={'height': '350px'}),
                        ], style={'padding': '10px', 'maxHeight': '1000px', 'overflowY': 'auto'})
                    ]),
                ], style={'height': '100%'})
            ], style={'flexGrow': '1', 'backgroundColor': '#fff', 'borderRadius': '5px',
                     'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'padding': '10px'}),

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
    """Create an empty figure with subplots."""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[cfg[0] for cfg in PLOT_CONFIGS],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    # Use distinct colors and line styles for better differentiation
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']
    dashes = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

    for idx, (title, signals) in enumerate(PLOT_CONFIGS):
        row = idx // 2 + 1
        col = idx % 2 + 1

        for color_idx, (label, _) in enumerate(signals):
            fig.add_trace(
                go.Scatter(x=[], y=[], name=label, mode='lines',
                          line=dict(color=colors[color_idx % len(colors)],
                                   width=2,
                                   dash=dashes[color_idx % len(dashes)]),
                          showlegend=True, legendgroup=f'group{idx}'),
                row=row, col=col
            )

        fig.update_xaxes(title_text="Time (min)", row=row, col=col)

    fig.update_layout(
        height=800,
        margin=dict(l=60, r=30, t=40, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white'
    )

    return fig


# Initialize the app
app = Dash(__name__)
app.title = "TEP Simulator Dashboard"
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
    State('sim-state', 'data'),
    State('speed-slider', 'value'),
    State('output-interval-slider', 'value'),
    prevent_initial_call=True
)
def control_simulation(start_clicks, stop_clicks, reset_clicks, state, speed, output_interval):
    """Handle start/stop/reset button clicks."""
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
        init_simulator()
        state['running'] = False
        return True, state, "Ready", {'color': '#27ae60'}

    return True, state, "Ready", {'color': '#27ae60'}


@app.callback(
    Output('main-plots', 'style'),
    Output('welcome-message', 'style'),
    Input('start-btn', 'n_clicks'),
    Input('reset-btn', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_welcome_message(start_clicks, reset_clicks):
    """Show/hide welcome message based on simulation state."""
    triggered = ctx.triggered_id

    graph_visible = {'height': '850px', 'display': 'block'}
    graph_hidden = {'height': '850px', 'display': 'none'}
    welcome_visible = {'display': 'block'}
    welcome_hidden = {'display': 'none'}

    if triggered == 'start-btn':
        # Hide welcome, show graph
        return graph_visible, welcome_hidden
    elif triggered == 'reset-btn':
        # Show welcome, hide graph
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
            sim_data['shutdown_reason'] = get_shutdown_reason(meas)
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
    """Create figure with current simulation data."""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[cfg[0] for cfg in PLOT_CONFIGS],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
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

    for idx, (title, signals) in enumerate(PLOT_CONFIGS):
        row = idx // 2 + 1
        col = idx % 2 + 1

        # Track min/max efficiently without copying all data
        y_min_all = float('inf')
        y_max_all = float('-inf')

        for color_idx, (label, meas_idx) in enumerate(signals):
            y_data_full = sim_data['measurements'].get(meas_idx, [])

            # Compute min/max directly without creating intermediate list
            if y_data_full:
                y_min_all = min(y_min_all, min(y_data_full))
                y_max_all = max(y_max_all, max(y_data_full))

            # Decimate for display
            y_data = decimate_data(y_data_full, max_points)

            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=y_data,
                    name=label,
                    mode='lines',
                    line=dict(color=colors[color_idx % len(colors)],
                             width=2,
                             dash=dashes[color_idx % len(dashes)]),
                    legendgroup=f'group{idx}',
                    showlegend=True,
                ),
                row=row, col=col
            )

        # Set y-axis range with margin for autoscaling
        if y_min_all != float('inf'):
            y_min = y_min_all
            y_max = y_max_all
            y_range = y_max - y_min

            # Ensure minimum range to avoid flat lines at edges
            if y_range < 1e-6:
                # If data is essentially constant, add symmetric padding
                padding = max(abs(y_min) * 0.1, 1.0)
                y_min_plot = y_min - padding
                y_max_plot = y_max + padding
            else:
                # Add 15% margin on each side so data doesn't hit edges
                margin = y_range * 0.15
                y_min_plot = y_min - margin
                y_max_plot = y_max + margin

            fig.update_yaxes(
                range=[y_min_plot, y_max_plot],
                autorange=False,
                row=row, col=col
            )

        # Set x-axis to start from 0 and extend to current time (no auto-rescaling)
        if time_data_full:
            x_max = max(time_data_full)  # Use full data for axis range
            # Add small margin to the right
            fig.update_xaxes(title_text="Time (min)", range=[0, x_max * 1.02], row=row, col=col)
        else:
            fig.update_xaxes(title_text="Time (min)", range=[0, 1], row=row, col=col)

    fig.update_layout(
        height=850,
        margin=dict(l=60, r=30, t=50, b=80),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.08,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        )
    )

    return fig


@app.callback(
    Output('measurements-grid', 'figure'),
    Output('mvs-grid', 'figure'),
    Input('interval-component', 'n_intervals'),
    State('main-tabs', 'value'),
    prevent_initial_call=True
)
def update_variables_grid(n_intervals, active_tab):
    """Update the All Variables grid with time series plots."""
    global simulator, sim_data
    from dash import no_update

    # Only update if the Variables tab is active to reduce browser load
    if active_tab != 'variables-tab':
        return no_update, no_update

    max_points = sim_data.get('max_display_points', 2000)
    time_data_full = sim_data['time']
    time_data = decimate_data(time_data_full, max_points)

    # Create measurements figure (7 cols x 6 rows = 42 subplots, using 41)
    n_meas_rows = 6
    n_meas_cols = 7
    meas_titles = [f"XMEAS({i+1})" for i in range(NUM_MEASUREMENTS)]

    meas_fig = make_subplots(
        rows=n_meas_rows, cols=n_meas_cols,
        subplot_titles=meas_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.03
    )

    for i in range(NUM_MEASUREMENTS):
        row = i // n_meas_cols + 1
        col = i % n_meas_cols + 1

        y_data_full = sim_data['measurements'].get(i, [])
        y_data = decimate_data(y_data_full, max_points)

        meas_fig.add_trace(
            go.Scatter(
                x=time_data,
                y=y_data,
                mode='lines',
                line=dict(color='#3498db', width=1),
                showlegend=False,
            ),
            row=row, col=col
        )

    meas_fig.update_layout(
        height=600,
        margin=dict(l=30, r=10, t=30, b=30),
        template='plotly_white',
        showlegend=False,
    )
    # Update all axes to be minimal
    meas_fig.update_xaxes(showticklabels=False, showgrid=True, gridcolor='#f0f0f0')
    meas_fig.update_yaxes(showticklabels=True, tickfont=dict(size=8), showgrid=True, gridcolor='#f0f0f0')

    # Create MVs figure (4 cols x 3 rows = 12 subplots)
    n_mv_rows = 3
    n_mv_cols = 4
    mv_titles = [f"XMV({i+1})" for i in range(NUM_MANIPULATED_VARS)]

    mvs_fig = make_subplots(
        rows=n_mv_rows, cols=n_mv_cols,
        subplot_titles=mv_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.05
    )

    for i in range(NUM_MANIPULATED_VARS):
        row = i // n_mv_cols + 1
        col = i % n_mv_cols + 1

        y_data_full = sim_data['mvs'].get(i, [])
        y_data = decimate_data(y_data_full, max_points)

        mvs_fig.add_trace(
            go.Scatter(
                x=time_data,
                y=y_data,
                mode='lines',
                line=dict(color='#27ae60', width=1),
                showlegend=False,
            ),
            row=row, col=col
        )

    mvs_fig.update_layout(
        height=350,
        margin=dict(l=30, r=10, t=30, b=30),
        template='plotly_white',
        showlegend=False,
    )
    mvs_fig.update_xaxes(showticklabels=False, showgrid=True, gridcolor='#f0f0f0')
    mvs_fig.update_yaxes(showticklabels=True, tickfont=dict(size=8), showgrid=True, gridcolor='#f0f0f0')

    return meas_fig, mvs_fig


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
