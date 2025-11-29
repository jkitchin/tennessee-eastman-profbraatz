"""Tests for the Streamlit dashboard module.

This module tests the utility functions and components of the Streamlit dashboard.
Note: We cannot easily test the actual Streamlit app without running a Streamlit
server, so we focus on testing the utility functions and module structure.
"""

import pytest
import numpy as np

# Test imports work
def test_module_imports():
    """Test that dashboard_streamlit module can be imported."""
    from tep import dashboard_streamlit
    assert hasattr(dashboard_streamlit, 'get_shutdown_reason')
    assert hasattr(dashboard_streamlit, 'PLOT_CONFIGS')
    assert hasattr(dashboard_streamlit, 'MV_SHORT_NAMES')
    assert hasattr(dashboard_streamlit, 'IDV_INFO')


def test_mv_short_names():
    """Test MV short names are properly defined."""
    from tep.dashboard_streamlit import MV_SHORT_NAMES
    assert len(MV_SHORT_NAMES) == 12
    assert "D Feed Flow" in MV_SHORT_NAMES
    assert "Agitator Speed" in MV_SHORT_NAMES


def test_idv_info():
    """Test disturbance info is properly defined."""
    from tep.dashboard_streamlit import IDV_INFO
    assert len(IDV_INFO) == 20
    # Each entry should be a tuple of (name, description)
    for name, desc in IDV_INFO:
        assert isinstance(name, str)
        assert isinstance(desc, str)
        assert name.startswith("IDV(")


def test_plot_configs():
    """Test plot configurations are properly defined."""
    from tep.dashboard_streamlit import PLOT_CONFIGS
    assert len(PLOT_CONFIGS) > 0
    for title, traces in PLOT_CONFIGS:
        assert isinstance(title, str)
        assert isinstance(traces, list)
        for trace_name, xmeas_idx in traces:
            assert isinstance(trace_name, str)
            assert isinstance(xmeas_idx, int)
            assert 0 <= xmeas_idx < 41  # Valid XMEAS index


class TestShutdownReason:
    """Tests for the get_shutdown_reason function.

    Note: get_shutdown_reason returns a string. When there are violations,
    it returns them joined by " | ". When there are none, it returns
    "Safety limit violation" as a default message.
    """

    def test_normal_operation(self):
        """Test that normal measurements return default message."""
        from tep.dashboard_streamlit import get_shutdown_reason

        # Create normal measurement array
        meas = np.zeros(41)
        meas[6] = 2700  # Normal reactor pressure (kPa)
        meas[8] = 122   # Normal reactor temp (C)
        meas[7] = 65    # Normal reactor level (%)
        meas[11] = 50   # Normal separator level (%)
        meas[14] = 50   # Normal stripper level (%)

        reason = get_shutdown_reason(meas)
        # When all is normal, returns default message (no specific violation)
        assert reason == "Safety limit violation"

    def test_high_reactor_pressure(self):
        """Test detection of high reactor pressure."""
        from tep.dashboard_streamlit import get_shutdown_reason

        meas = np.zeros(41)
        meas[6] = 3100  # High reactor pressure
        meas[8] = 122
        meas[7] = 65
        meas[11] = 50
        meas[14] = 50

        reason = get_shutdown_reason(meas)
        assert "pressure" in reason.lower()

    def test_high_reactor_temp(self):
        """Test detection of high reactor temperature."""
        from tep.dashboard_streamlit import get_shutdown_reason

        meas = np.zeros(41)
        meas[6] = 2700
        meas[8] = 180   # High reactor temp
        meas[7] = 65
        meas[11] = 50
        meas[14] = 50

        reason = get_shutdown_reason(meas)
        assert "temp" in reason.lower()

    def test_high_reactor_level(self):
        """Test detection of high reactor level."""
        from tep.dashboard_streamlit import get_shutdown_reason

        meas = np.zeros(41)
        meas[6] = 2700
        meas[8] = 122
        meas[7] = 98    # High reactor level
        meas[11] = 50
        meas[14] = 50

        reason = get_shutdown_reason(meas)
        assert "reactor level" in reason.lower()

    def test_low_reactor_level(self):
        """Test detection of low reactor level."""
        from tep.dashboard_streamlit import get_shutdown_reason

        meas = np.zeros(41)
        meas[6] = 2700
        meas[8] = 122
        meas[7] = 2     # Low reactor level
        meas[11] = 50
        meas[14] = 50

        reason = get_shutdown_reason(meas)
        assert "reactor level" in reason.lower()

    def test_multiple_reasons(self):
        """Test multiple shutdown reasons detected."""
        from tep.dashboard_streamlit import get_shutdown_reason

        meas = np.zeros(41)
        meas[6] = 3100  # High pressure
        meas[8] = 180   # High temp
        meas[7] = 65
        meas[11] = 50
        meas[14] = 50

        reason = get_shutdown_reason(meas)
        # Multiple reasons are joined by " | "
        assert "|" in reason


class TestCLIEntry:
    """Test CLI entry point exists."""

    def test_cli_main_exists(self):
        """Test that cli_main function exists."""
        from tep.dashboard_streamlit import cli_main
        assert callable(cli_main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
