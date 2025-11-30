"""
Web-based Dashboard for the Tennessee Eastman Process Simulator using Streamlit.

This module provides an interactive web-based interface for:
- Controlling manipulated variables
- Enabling/disabling process disturbances
- Real-time visualization of process measurements
- Simulation control (start, stop, reset)

Requirements:
    - streamlit
    - plotly

Usage:
    streamlit run tep/dashboard_streamlit.py

    Or via CLI:
        tep-streamlit
"""

import numpy as np
import time
import io
import logging
from datetime import datetime

import streamlit as st

# Set up logging that works on Streamlit Cloud
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tep.simulator import TEPSimulator, ControlMode
from tep.constants import (
    NUM_MEASUREMENTS, NUM_MANIPULATED_VARS, NUM_DISTURBANCES, INITIAL_STATES,
    SAFETY_LIMITS
)


# MV short names
MV_SHORT_NAMES = [
    "D Feed Flow", "E Feed Flow", "A Feed Flow", "A+C Feed Flow",
    "Recycle Valve", "Purge Valve", "Sep Liq Flow", "Strip Liq Flow",
    "Steam Valve", "React CW Flow", "Cond CW Flow", "Agitator Speed"
]

# Disturbance names and descriptions - IDV(1-20)
IDV_INFO = [
    ("IDV(1) A/C Ratio", "Step change in A/C feed ratio"),
    ("IDV(2) B Comp", "Step change in B composition"),
    ("IDV(3) D Feed Temp", "Step change in D feed temperature"),
    ("IDV(4) Reactor CW", "Reactor cooling water inlet temp"),
    ("IDV(5) Condenser CW", "Condenser cooling water inlet temp"),
    ("IDV(6) A Feed Loss", "Loss of A feed - major disruption!"),
    ("IDV(7) C Header", "C header pressure loss"),
    ("IDV(8) A,B,C Comp", "Random A,B,C feed composition"),
    ("IDV(9) D Temp Rand", "Random D feed temperature"),
    ("IDV(10) C Temp Rand", "Random C feed temperature"),
    ("IDV(11) React CW Rand", "Random reactor CW inlet temp"),
    ("IDV(12) Cond CW Rand", "Random condenser CW inlet temp"),
    ("IDV(13) Kinetics", "Slow drift in reaction kinetics"),
    ("IDV(14) React Valve", "Reactor CW valve sticking"),
    ("IDV(15) Cond Valve", "Condenser CW valve sticking"),
    ("IDV(16)", "Unknown disturbance"),
    ("IDV(17)", "Unknown disturbance"),
    ("IDV(18)", "Unknown disturbance"),
    ("IDV(19)", "Unknown disturbance"),
    ("IDV(20)", "Unknown disturbance"),
]

# Plot configurations
PLOT_CONFIGS = [
    ("Reactor Temp & Level", [("XMEAS(9) Temp", 8), ("XMEAS(8) Level", 7)]),
    ("Reactor Pressure", [("XMEAS(7) Pressure", 6)]),
    ("Separator Temp & Level", [("XMEAS(11) Temp", 10), ("XMEAS(12) Level", 11)]),
    ("A Feed & Purge", [("XMEAS(1) A Feed", 0), ("XMEAS(10) Purge", 9)]),
    ("Product Flow & Temp", [("XMEAS(17) Flow", 16), ("XMEAS(18) Temp", 17)]),
    ("Reactor Feed Comp", [("XMEAS(23) A%", 22), ("XMEAS(25) C%", 24), ("XMEAS(26) D%", 25)]),
]


def get_shutdown_reason(meas):
    """Determine the reason for process shutdown based on measurements."""
    reasons = []
    limits = SAFETY_LIMITS

    if meas[6] > limits.reactor_pressure_max:
        reasons.append(f"Reactor pressure ({meas[6]:.0f} kPa) exceeded {limits.reactor_pressure_max:.0f} kPa")
    if meas[8] > limits.reactor_temp_max:
        reasons.append(f"Reactor temp ({meas[8]:.1f}°C) exceeded {limits.reactor_temp_max:.1f}°C")

    reactor_level = meas[7]
    if reactor_level > 95:
        reasons.append(f"Reactor level ({reactor_level:.1f}%) too high")
    if reactor_level < 5:
        reasons.append(f"Reactor level ({reactor_level:.1f}%) too low")

    sep_level = meas[11]
    if sep_level > 95:
        reasons.append(f"Separator level ({sep_level:.1f}%) too high")
    if sep_level < 5:
        reasons.append(f"Separator level ({sep_level:.1f}%) too low")

    if reasons:
        return " | ".join(reasons)
    return "Safety limit violation"


def init_session_state():
    """Initialize Streamlit session state."""
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'sim_data' not in st.session_state:
        st.session_state.sim_data = {
            'time': [],
            'measurements': {i: [] for i in range(NUM_MEASUREMENTS)},
            'mvs': {i: [] for i in range(NUM_MANIPULATED_VARS)},
            'idv': [],
        }
    if 'last_output_time' not in st.session_state:
        st.session_state.last_output_time = 0
    if 'shutdown' not in st.session_state:
        st.session_state.shutdown = False
    if 'shutdown_reason' not in st.session_state:
        st.session_state.shutdown_reason = ""
    if 'backend' not in st.session_state:
        from tep import get_default_backend
        st.session_state.backend = get_default_backend()


def reset_simulator():
    """Reset the simulator to initial state."""
    logger.info(f"Creating simulator with backend: {st.session_state.backend}")
    st.session_state.simulator = TEPSimulator(
        control_mode=ControlMode.CLOSED_LOOP,
        backend=st.session_state.backend
    )
    logger.info(f"Simulator created, process type: {type(st.session_state.simulator.process).__name__}")
    st.session_state.simulator.initialize()
    st.session_state.sim_data = {
        'time': [],
        'measurements': {i: [] for i in range(NUM_MEASUREMENTS)},
        'mvs': {i: [] for i in range(NUM_MANIPULATED_VARS)},
        'idv': [],
    }
    st.session_state.running = False
    st.session_state.shutdown = False
    st.session_state.shutdown_reason = ""
    st.session_state.last_output_time = 0


def create_main_figure():
    """Create the main process plots figure."""
    sim_data = st.session_state.sim_data

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[cfg[0] for cfg in PLOT_CONFIGS],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']
    dashes = ['solid', 'dash', 'dot', 'dashdot']

    time_data = sim_data['time']
    trace_idx = 0

    for idx, (title, signals) in enumerate(PLOT_CONFIGS):
        row = idx // 2 + 1
        col = idx % 2 + 1

        for label, meas_idx in signals:
            y_data = sim_data['measurements'].get(meas_idx, [])

            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=y_data,
                    name=label,
                    mode='lines',
                    line=dict(
                        color=colors[trace_idx % len(colors)],
                        width=2,
                        dash=dashes[trace_idx % len(dashes)]
                    ),
                    legendgroup=f'group{idx}',
                    showlegend=True,
                ),
                row=row, col=col
            )
            trace_idx += 1

        fig.update_xaxes(title_text="Time (min)", row=row, col=col)

    fig.update_layout(
        height=800,
        margin=dict(l=50, r=30, t=60, b=50),
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.08,
            xanchor='center',
            x=0.5,
            font=dict(size=9)
        )
    )

    return fig


def create_variables_figures():
    """Create figures for all variables tab."""
    sim_data = st.session_state.sim_data
    time_data = sim_data['time']

    # Measurements figure (7 cols x 6 rows)
    n_meas_rows, n_meas_cols = 6, 7
    meas_titles = [f"XMEAS({i+1})" for i in range(NUM_MEASUREMENTS)]

    meas_fig = make_subplots(
        rows=n_meas_rows, cols=n_meas_cols,
        subplot_titles=meas_titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.03
    )

    for i in range(NUM_MEASUREMENTS):
        row = i // n_meas_cols + 1
        col = i % n_meas_cols + 1
        y_data = sim_data['measurements'].get(i, [])

        meas_fig.add_trace(
            go.Scatter(x=time_data, y=y_data, mode='lines',
                      line=dict(color='#3498db', width=1), showlegend=False),
            row=row, col=col
        )

    meas_fig.update_layout(
        height=500,
        margin=dict(l=30, r=10, t=30, b=20),
        template='plotly_white'
    )
    meas_fig.update_xaxes(showticklabels=False, showgrid=True, gridcolor='#f0f0f0')
    meas_fig.update_yaxes(showticklabels=True, tickfont=dict(size=7), showgrid=True, gridcolor='#f0f0f0')

    # MVs figure (4 cols x 3 rows)
    n_mv_rows, n_mv_cols = 3, 4
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
        y_data = sim_data['mvs'].get(i, [])

        mvs_fig.add_trace(
            go.Scatter(x=time_data, y=y_data, mode='lines',
                      line=dict(color='#27ae60', width=1), showlegend=False),
            row=row, col=col
        )

    mvs_fig.update_layout(
        height=300,
        margin=dict(l=30, r=10, t=30, b=20),
        template='plotly_white'
    )
    mvs_fig.update_xaxes(showticklabels=False, showgrid=True, gridcolor='#f0f0f0')
    mvs_fig.update_yaxes(showticklabels=True, tickfont=dict(size=8), showgrid=True, gridcolor='#f0f0f0')

    return meas_fig, mvs_fig


def generate_csv():
    """Generate CSV data for download."""
    sim_data = st.session_state.sim_data
    if not sim_data['time']:
        return ""

    lines = []
    header = ['Time_hr', 'Time_min']
    for i in range(NUM_MEASUREMENTS):
        header.append(f'XMEAS_{i+1}')
    for i in range(NUM_MANIPULATED_VARS):
        header.append(f'XMV_{i+1}')
    header.append('Active_IDVs')
    lines.append(','.join(header))

    n_points = len(sim_data['time'])
    for idx in range(n_points):
        row = []
        time_min = sim_data['time'][idx]
        time_hr = time_min / 60.0
        row.append(f'{time_hr:.6f}')
        row.append(f'{time_min:.4f}')

        for i in range(NUM_MEASUREMENTS):
            val = sim_data['measurements'][i][idx] if idx < len(sim_data['measurements'][i]) else 0
            row.append(f'{val:.6f}')

        for i in range(NUM_MANIPULATED_VARS):
            val = sim_data['mvs'][i][idx] if idx < len(sim_data['mvs'][i]) else 0
            row.append(f'{val:.6f}')

        if idx < len(sim_data['idv']):
            idv_list = sim_data['idv'][idx]
            idv_str = ';'.join(str(x) for x in idv_list) if idv_list else '0'
        else:
            idv_str = '0'
        row.append(idv_str)

        lines.append(','.join(row))

    return '\n'.join(lines)


def run_simulation_step():
    """Run simulation steps and record data.

    Reads parameters from session state to work with @st.fragment.
    """
    simulator = st.session_state.simulator
    sim_data = st.session_state.sim_data

    if simulator is None:
        return

    # Get parameters from session state
    # Note: Widget values are stored with their key names in session_state
    speed = st.session_state.get('speed', 50)
    output_interval = st.session_state.get('output_interval', 60)
    control_mode = st.session_state.get('control_mode', 'Closed Loop')
    mv_values = st.session_state.get('mv_values', [])

    # Read disturbances directly from checkbox keys (more reliable in fragments)
    disturbances = []
    for i in range(NUM_DISTURBANCES):
        if st.session_state.get(f'idv_{i}', False):
            disturbances.append(i + 1)

    # Only update disturbances when they change (avoid clearing/resetting every step)
    prev_disturbances = getattr(simulator, '_prev_disturbances', None)
    if prev_disturbances is None or set(disturbances) != set(prev_disturbances):
        logger.info(f"Disturbances changed: {prev_disturbances} -> {disturbances}")
        simulator.clear_disturbances()
        for idv in disturbances:
            simulator.set_disturbance(idv, 1)
        simulator._prev_disturbances = disturbances.copy()

    # Update control mode
    if control_mode == 'Closed Loop':
        if simulator.control_mode != ControlMode.CLOSED_LOOP:
            simulator.control_mode = ControlMode.CLOSED_LOOP
            simulator._init_controller()
    else:
        if simulator.control_mode != ControlMode.MANUAL:
            simulator.control_mode = ControlMode.MANUAL
            simulator._init_controller()
        for i, val in enumerate(mv_values):
            simulator.set_mv(i + 1, val)

    # Log active disturbances and pressure on EVERY fragment run
    active = simulator.get_active_disturbances()
    meas = simulator.get_measurements()
    # Also check internal IDV state directly
    idv_state = list(simulator.process.idv) if hasattr(simulator.process, 'idv') else 'N/A'
    logger.info(f"t={simulator.time:.3f}hr, step={simulator.step_count}, P={meas[6]:.1f}kPa, IDVs: {active}, idv_raw: {idv_state[:8]}")

    # Run more simulation steps per update to reduce rerun frequency
    steps_per_update = speed * 10  # Run 10x more steps before updating UI

    # Log pressure at start and end of batch if IDV(7) is active
    if 7 in disturbances:
        meas_start = simulator.get_measurements()
        logger.info(f"  Batch start: step={simulator.step_count}, P={meas_start[6]:.1f}kPa")

    for i in range(steps_per_update):
        if not simulator.step():
            st.session_state.running = False
            st.session_state.shutdown = True
            meas = simulator.get_measurements()
            reason = get_shutdown_reason(meas)
            st.session_state.shutdown_reason = reason
            logger.warning(f"SHUTDOWN at t={simulator.time:.3f}hr: {reason}")
            logger.warning(f"  Reactor pressure: {meas[6]:.1f} kPa")
            logger.warning(f"  Reactor temp: {meas[8]:.1f} C")
            logger.warning(f"  Reactor level: {meas[7]:.1f}%")
            logger.warning(f"  Active IDVs: {simulator.get_active_disturbances()}")
            return

        # Log every 100 steps if IDV(7) active to track pressure
        if 7 in disturbances and i > 0 and i % 100 == 0:
            meas_mid = simulator.get_measurements()
            logger.info(f"    step {simulator.step_count}: P={meas_mid[6]:.1f}kPa")

    if 7 in disturbances:
        meas_end = simulator.get_measurements()
        logger.info(f"  Batch end: step={simulator.step_count}, P={meas_end[6]:.1f}kPa")

    # Record data at specified interval
    current_time_sec = simulator.time * 3600
    if current_time_sec - st.session_state.last_output_time >= output_interval:
        st.session_state.last_output_time = current_time_sec
        sim_data['time'].append(simulator.time * 60)  # minutes

        meas = simulator.get_measurements()
        for i in range(NUM_MEASUREMENTS):
            sim_data['measurements'][i].append(meas[i])

        mvs = simulator.get_manipulated_vars()
        for i in range(NUM_MANIPULATED_VARS):
            sim_data['mvs'][i].append(mvs[i])

        sim_data['idv'].append(simulator.get_active_disturbances())

    # Limit stored data
    max_points = 10000
    if len(sim_data['time']) > max_points:
        step = 2
        sim_data['time'] = sim_data['time'][::step]
        for i in range(NUM_MEASUREMENTS):
            sim_data['measurements'][i] = sim_data['measurements'][i][::step]
        for i in range(NUM_MANIPULATED_VARS):
            sim_data['mvs'][i] = sim_data['mvs'][i][::step]
        sim_data['idv'] = sim_data['idv'][::step]


@st.fragment(run_every=0.5)
def simulation_fragment():
    """Fragment that runs the simulation and updates plots.

    Using @st.fragment with run_every allows this to update independently
    of the main app, reducing full-page reruns and blinking.
    """
    # Debug: log every fragment run
    sim = st.session_state.get('simulator')
    running = st.session_state.get('running', False)
    if sim and running:
        logger.info(f"Fragment running: t={sim.time:.3f}hr, running={running}")

    # Run simulation step if running
    if st.session_state.running and not st.session_state.shutdown:
        try:
            run_simulation_step()
        except Exception as e:
            logger.exception(f"ERROR in run_simulation_step: {type(e).__name__}: {e}")
            st.error(f"Simulation error: {e}")

    # Display status
    sim = st.session_state.simulator
    sim_time = sim.time if sim else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        status = "🟢 Running" if st.session_state.running else ("🔴 Shutdown" if st.session_state.shutdown else "⏸️ Stopped")
        st.metric("Status", status)
    with col2:
        st.metric("Simulation Time", f"{sim_time:.2f} hr ({sim_time*60:.1f} min)")
    with col3:
        # Show active disturbances for debugging
        active_idvs = sim.get_active_disturbances() if sim else []
        idv_str = ", ".join(map(str, active_idvs)) if active_idvs else "None"
        st.metric("Active IDVs", idv_str)

    # Display shutdown alert within fragment
    if st.session_state.shutdown:
        st.error(f"⚠️ **PROCESS SHUTDOWN**: {st.session_state.shutdown_reason}")
        st.info("Click 'Reset' in the sidebar to restart the simulation.")

    # Tabs for plots
    tab1, tab2 = st.tabs(["📊 Process Plots", "📈 All Variables"])

    with tab1:
        if st.session_state.sim_data['time']:
            fig = create_main_figure()
            st.plotly_chart(fig, width='stretch', key=f'main_plot_{len(st.session_state.sim_data["time"])}')
        else:
            st.info("👆 Click **Start** in the sidebar to begin the simulation.")
            st.markdown("""
            ### Quick Start Guide
            1. Select the simulation **backend** (Fortran for speed, Python for portability)
            2. Adjust **simulation speed** with the slider
            3. Enable **disturbances** to test fault scenarios
            4. Switch to **Manual** mode to control valves directly
            5. Use the **All Variables** tab to see all 41 measurements
            """)

    with tab2:
        if st.session_state.sim_data['time']:
            st.subheader("Process Measurements (XMEAS 1-41)")
            meas_fig, mvs_fig = create_variables_figures()
            st.plotly_chart(meas_fig, width='stretch', key=f'meas_plot_{len(st.session_state.sim_data["time"])}')

            st.subheader("Manipulated Variables (XMV 1-12)")
            st.plotly_chart(mvs_fig, width='stretch', key=f'mvs_plot_{len(st.session_state.sim_data["time"])}')
        else:
            st.info("Start the simulation to see variable plots.")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="TEP Simulator",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    # Initialize simulator if needed
    if st.session_state.simulator is None:
        reset_simulator()

    # Header
    from tep import __version__
    st.title("🏭 Tennessee Eastman Process Simulator")
    st.caption(f"v{__version__} | Interactive Process Control Dashboard | [GitHub](https://github.com/jkitchin/tennessee-eastman-profbraatz)")

    # Sidebar - Controls
    with st.sidebar:
        st.header("⚙️ Simulation Control")

        # Backend selection
        from tep import get_available_backends
        backends = get_available_backends()
        backend = st.selectbox(
            "Backend",
            backends,
            index=backends.index(st.session_state.backend) if st.session_state.backend in backends else 0,
            help="Select simulation backend"
        )
        if backend != st.session_state.backend:
            st.session_state.backend = backend
            reset_simulator()
            st.rerun()

        # Control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Start", width='stretch'):
                st.session_state.running = True
        with col2:
            if st.button("⏹️ Stop", width='stretch'):
                st.session_state.running = False
        with col3:
            if st.button("🔄 Reset", width='stretch'):
                reset_simulator()
                st.rerun()

        st.divider()

        # Simulation settings
        control_mode = st.radio(
            "Control Mode",
            ["Closed Loop", "Manual"],
            horizontal=True,
            key='control_mode'
        )

        speed = st.slider("Speed (steps/update)", 1, 100, 50, key='speed')
        output_interval = st.slider("Output Interval (sec)", 1, 300, 60, key='output_interval')

        st.divider()

        # Manipulated Variables
        st.header("🎛️ Manipulated Variables")
        initial_mvs = INITIAL_STATES[38:50]
        mv_values = []

        for i in range(NUM_MANIPULATED_VARS):
            val = st.slider(
                f"{i+1}. {MV_SHORT_NAMES[i]}",
                0.0, 100.0, float(initial_mvs[i]),
                key=f"mv_{i}",
                disabled=(control_mode == "Closed Loop")
            )
            mv_values.append(val)

        # Store mv_values in session state for fragment access
        st.session_state.mv_values = mv_values

        st.divider()

        # Disturbances
        st.header("⚡ Disturbances")
        active_disturbances = []

        for i in range(NUM_DISTURBANCES):
            if st.checkbox(IDV_INFO[i][0], key=f"idv_{i}", help=IDV_INFO[i][1]):
                active_disturbances.append(i + 1)

        if active_disturbances:
            st.warning(f"Active: IDV({', '.join(map(str, active_disturbances))})")

        st.divider()

        # Download button - only show when simulation is stopped to avoid cache issues
        if not st.session_state.running:
            csv_data = generate_csv()
            if csv_data:
                st.download_button(
                    "📥 Download Data (CSV)",
                    csv_data,
                    "tep_simulation_data.csv",
                    "text/csv",
                    width='stretch',
                    key=f"download_{len(st.session_state.sim_data['time'])}"
                )

    # Main content - use fragment for smooth updates
    simulation_fragment()


def run_dashboard(host='localhost', port=8501, open_browser=True):
    """Run the Streamlit dashboard.

    This function is primarily for documentation. To run the Streamlit app,
    use: streamlit run tep/dashboard_streamlit.py

    Args:
        host: Host address (default: localhost)
        port: Port number (default: 8501)
        open_browser: Open browser automatically (default: True)
    """
    import sys
    import subprocess

    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        __file__,
        '--server.address', host,
        '--server.port', str(port),
    ]

    if not open_browser:
        cmd.extend(['--server.headless', 'true'])

    print(f"\n{'='*60}")
    print("Tennessee Eastman Process Simulator - Streamlit Dashboard")
    print(f"{'='*60}")
    print(f"\nStarting Streamlit server...")
    print(f"URL: http://{host}:{port}")
    print("Press Ctrl+C to stop the server\n")

    subprocess.run(cmd)


def cli_main():
    """Entry point for CLI."""
    import argparse
    parser = argparse.ArgumentParser(description='Tennessee Eastman Process Streamlit Dashboard')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--port', type=int, default=8501, help='Port number (default: 8501)')
    parser.add_argument('--host', default='localhost', help='Host address (default: localhost)')
    args = parser.parse_args()

    run_dashboard(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
