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
from dash import Dash, html, dcc, callback, Output, Input, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .simulator import TEPSimulator, ControlMode
from .constants import (
    NUM_MEASUREMENTS, NUM_MANIPULATED_VARS, NUM_DISTURBANCES, INITIAL_STATES,
    SAFETY_LIMITS
)


# Global simulator instance and data storage
simulator = None
sim_data = {
    'time': [],
    'measurements': {i: [] for i in range(NUM_MEASUREMENTS)},
    'mvs': {i: [] for i in range(NUM_MANIPULATED_VARS)},
    'running': False,
    'history_length': 500,
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
    sim_data['running'] = False
    sim_data['shutdown_reason'] = None


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
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
            html.P("Interactive Process Control Dashboard",
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '0'})
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
                    html.Label("Simulation Speed:"),
                    dcc.Slider(
                        id='speed-slider',
                        min=1,
                        max=50,
                        value=10,
                        marks={1: '1', 10: '10', 25: '25', 50: '50'},
                        step=1
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
                    ], style={'marginTop': '15px', 'marginBottom': '15px'}),

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

            # Center - Plots
            html.Div([
                dcc.Graph(id='main-plots', style={'height': '800px'})
            ], style={'flexGrow': '1', 'backgroundColor': '#fff', 'borderRadius': '5px',
                     'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'padding': '10px'}),

            # Right panel - Disturbances
            html.Div([
                html.H3("Disturbances", style={'marginTop': '0'}),
                html.P("Check to enable process upsets:",
                      style={'fontSize': '12px', 'color': '#7f8c8d', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id=f'idv-{i}',
                            options=[{'label': f" {IDV_INFO[i][0]}", 'value': i}],
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
                ], style={'maxHeight': '650px', 'overflowY': 'auto'}),
                html.Button('Clear All', id='clear-disturbances-btn', n_clicks=0,
                           style={'marginTop': '10px', 'backgroundColor': '#95a5a6', 'color': 'white',
                                 'border': 'none', 'padding': '8px 15px', 'cursor': 'pointer', 'width': '100%'})
            ], style={'width': '280px', 'flexShrink': '0', 'marginLeft': '15px',
                     'backgroundColor': '#fff', 'padding': '15px', 'borderRadius': '5px',
                     'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'})
        ], style={'display': 'flex', 'padding': '0 15px'}),

        # Interval for updates
        dcc.Interval(id='interval-component', interval=100, n_intervals=0, disabled=True),

        # Store for simulation state
        dcc.Store(id='sim-state', data={'running': False, 'speed': 10})
    ], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ecf0f1',
              'minHeight': '100vh', 'paddingBottom': '20px'})


def create_empty_figure():
    """Create an empty figure with subplots."""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[cfg[0] for cfg in PLOT_CONFIGS],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for idx, (title, signals) in enumerate(PLOT_CONFIGS):
        row = idx // 2 + 1
        col = idx % 2 + 1

        for color_idx, (label, _) in enumerate(signals):
            fig.add_trace(
                go.Scatter(x=[], y=[], name=label, mode='lines',
                          line=dict(color=colors[color_idx], width=2),
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


@callback(
    Output('interval-component', 'disabled'),
    Output('sim-state', 'data'),
    Output('status-text', 'children'),
    Output('status-text', 'style'),
    Output('shutdown-alert', 'style', allow_duplicate=True),
    Input('start-btn', 'n_clicks'),
    Input('stop-btn', 'n_clicks'),
    Input('reset-btn', 'n_clicks'),
    State('sim-state', 'data'),
    State('speed-slider', 'value'),
    prevent_initial_call=True
)
def control_simulation(start_clicks, stop_clicks, reset_clicks, state, speed):
    """Handle start/stop/reset button clicks."""
    global sim_data

    triggered = ctx.triggered_id

    # Hidden style for shutdown alert
    hidden_style = {'display': 'none'}

    if triggered == 'start-btn':
        sim_data['running'] = True
        state['running'] = True
        state['speed'] = speed
        return False, state, "Running", {'color': '#27ae60', 'fontWeight': 'bold'}, hidden_style

    elif triggered == 'stop-btn':
        sim_data['running'] = False
        state['running'] = False
        return True, state, "Stopped", {'color': '#e74c3c'}, hidden_style

    elif triggered == 'reset-btn':
        init_simulator()
        state['running'] = False
        return True, state, "Ready", {'color': '#27ae60'}, hidden_style

    return True, state, "Ready", {'color': '#27ae60'}, hidden_style


@callback(
    [Output(f'idv-{i}', 'value') for i in range(NUM_DISTURBANCES)],
    Input('clear-disturbances-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_disturbances(n_clicks):
    """Clear all disturbances."""
    if simulator:
        simulator.clear_disturbances()
    return [[] for _ in range(NUM_DISTURBANCES)]


@callback(
    Output('main-plots', 'figure'),
    Output('time-text', 'children'),
    Output('shutdown-alert', 'style'),
    Output('shutdown-reason', 'children'),
    Input('interval-component', 'n_intervals'),
    State('sim-state', 'data'),
    State('control-mode', 'value'),
    *[State(f'idv-{i}', 'value') for i in range(NUM_DISTURBANCES)],
    *[State(f'mv-slider-{i}', 'value') for i in range(NUM_MANIPULATED_VARS)]
)
def update_simulation(n_intervals, state, control_mode, *args):
    """Run simulation step and update plots."""
    global simulator, sim_data

    # Default hidden style for shutdown alert
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

    # Split args into disturbances and mv_values
    idv_values = args[:NUM_DISTURBANCES]
    mv_values = args[NUM_DISTURBANCES:]

    # Check if already shutdown
    if simulator and simulator.is_shutdown():
        reason = sim_data.get('shutdown_reason', 'Safety limit violation')
        return (create_figure_with_data(),
                f"{simulator.time:.2f} hr ({simulator.time*60:.1f} min)",
                shutdown_style, reason)

    if not state.get('running', False) or simulator is None:
        # Return current state without updating
        time_str = f"{simulator.time:.2f} hr" if simulator else "0.00 hr"
        return create_figure_with_data(), time_str, hidden_style, ""

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

    # Update disturbances - each idv_values[i] is a list with [i] if checked or [] if not
    for i in range(NUM_DISTURBANCES):
        is_enabled = len(idv_values[i]) > 0 if idv_values[i] else False
        simulator.set_disturbance(i + 1, 1 if is_enabled else 0)

    # Run simulation steps and record data at each step
    speed = state.get('speed', 10)
    shutdown_occurred = False
    for _ in range(speed):
        if not simulator.step():
            sim_data['running'] = False
            # Get measurements to determine shutdown reason
            meas = simulator.get_measurements()
            sim_data['shutdown_reason'] = get_shutdown_reason(meas)
            shutdown_occurred = True
            break

        # Record data at every step to capture noise
        sim_data['time'].append(simulator.time * 60)  # Convert to minutes

        meas = simulator.get_measurements()
        for i in range(NUM_MEASUREMENTS):
            sim_data['measurements'][i].append(meas[i])

        mvs = simulator.get_manipulated_vars()
        for i in range(NUM_MANIPULATED_VARS):
            sim_data['mvs'][i].append(mvs[i])

    # Note: No longer trimming history - keep all data to show full trajectory
    # The x-axis will show from t=0 to current time

    time_str = f"{simulator.time:.2f} hr ({simulator.time*60:.1f} min)"

    if shutdown_occurred:
        return (create_figure_with_data(), time_str,
                shutdown_style, sim_data.get('shutdown_reason', 'Safety limit violation'))

    return create_figure_with_data(), time_str, hidden_style, ""


def create_figure_with_data():
    """Create figure with current simulation data."""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[cfg[0] for cfg in PLOT_CONFIGS],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    time_data = sim_data['time']

    for idx, (title, signals) in enumerate(PLOT_CONFIGS):
        row = idx // 2 + 1
        col = idx % 2 + 1

        # Collect y values for this subplot to compute axis range
        all_y_values = []

        for color_idx, (label, meas_idx) in enumerate(signals):
            y_data = sim_data['measurements'].get(meas_idx, [])
            if y_data:
                all_y_values.extend(y_data)

            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=y_data,
                    name=label,
                    mode='lines',
                    line=dict(color=colors[color_idx], width=2),
                    legendgroup=f'group{idx}',
                    showlegend=True,
                ),
                row=row, col=col
            )

        # Set y-axis range with margin for autoscaling
        # Use autorange with rangemode='tozero' or compute explicitly
        if all_y_values:
            y_min = min(all_y_values)
            y_max = max(all_y_values)
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
        if time_data:
            x_max = max(time_data)
            # Add small margin to the right
            fig.update_xaxes(title_text="Time (min)", range=[0, x_max * 1.02], row=row, col=col)
        else:
            fig.update_xaxes(title_text="Time (min)", range=[0, 1], row=row, col=col)

    fig.update_layout(
        height=800,
        margin=dict(l=60, r=30, t=50, b=40),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.05,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        )
    )

    return fig


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
    run_dashboard()


if __name__ == "__main__":
    main()
