"""
Tests for the TEP batch simulation CLI.

These tests verify that:
1. The CLI produces output in the correct Fortran format (E13.5 scientific notation)
2. The output values match direct Python API calls (both use the same Fortran backend)
3. The CLI handles various configurations correctly

The Python CLI and direct Fortran code use the same underlying physics module (teprob.f),
so identical inputs should produce identical outputs.
"""

import pytest
import numpy as np
import os
import tempfile
import subprocess
import sys
import re
from pathlib import Path

from tep import TEPSimulator
from tep.simulator import ControlMode
from tep.cli import (
    parse_faults,
    parse_fault_times,
    format_fortran_value,
    write_single_file,
    write_multi_file,
    run_simulation,
)
from tep.constants import DEFAULT_RANDOM_SEED


class TestFaultParsing:
    """Test fault specification parsing."""

    def test_single_fault(self):
        """Parse single fault ID."""
        assert parse_faults("1") == [1]
        assert parse_faults("5") == [5]
        assert parse_faults("20") == [20]

    def test_comma_separated_faults(self):
        """Parse comma-separated fault IDs."""
        assert parse_faults("1,2,5") == [1, 2, 5]
        assert parse_faults("1, 2, 5") == [1, 2, 5]

    def test_range_faults(self):
        """Parse fault ID ranges."""
        assert parse_faults("1-5") == [1, 2, 3, 4, 5]
        assert parse_faults("3-5") == [3, 4, 5]

    def test_combined_faults(self):
        """Parse combined fault specifications."""
        assert parse_faults("1,3-5,7") == [1, 3, 4, 5, 7]

    def test_invalid_fault_id(self):
        """Invalid fault IDs should raise ValueError."""
        with pytest.raises(ValueError):
            parse_faults("0")
        with pytest.raises(ValueError):
            parse_faults("21")


class TestFaultTimeParsing:
    """Test fault start time parsing."""

    def test_single_time(self):
        """Single time applied to all faults."""
        assert parse_fault_times("1.0", 3) == [1.0, 1.0, 1.0]

    def test_multiple_times(self):
        """Multiple times for multiple faults."""
        assert parse_fault_times("1.0, 2.0, 3.0", 3) == [1.0, 2.0, 3.0]

    def test_time_count_mismatch(self):
        """Mismatched count should raise ValueError."""
        with pytest.raises(ValueError):
            parse_fault_times("1.0,2.0", 3)


class TestFortranFormat:
    """Test Fortran E13.5 format output."""

    def test_format_positive_value(self):
        """Format positive values correctly."""
        formatted = format_fortran_value(123.456)
        # Should be 13 chars wide, 5 decimal places in exponent notation
        assert len(formatted) == 13
        assert 'E' in formatted

    def test_format_small_value(self):
        """Format small values correctly."""
        formatted = format_fortran_value(0.00012345)
        assert len(formatted) == 13
        assert 'E' in formatted

    def test_format_negative_value(self):
        """Format negative values correctly."""
        formatted = format_fortran_value(-123.456)
        assert 'E' in formatted
        assert '-' in formatted

    def test_format_matches_fortran_spec(self):
        """Output should match Fortran E13.5 format specification."""
        # Fortran E13.5 produces values like " 0.12345E+02"
        val = 123.456
        formatted = format_fortran_value(val)

        # Should be parseable as float and round-trip correctly
        parsed = float(formatted)
        assert abs(parsed - val) / val < 1e-4  # 5 significant digits


class TestCLIOutputFormat:
    """Test that CLI output matches Fortran file format."""

    @pytest.fixture
    def short_simulation_result(self):
        """Run a short simulation for testing output format."""
        sim = TEPSimulator(random_seed=12345, control_mode=ControlMode.CLOSED_LOOP)
        sim.initialize()
        return sim.simulate(duration_hours=0.1, record_interval=180)

    def test_single_file_format(self, short_simulation_result):
        """Single file output should have correct format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            output_path = f.name

        try:
            write_single_file(output_path, short_simulation_result, include_header=True)

            # Read and verify format
            with open(output_path, 'r') as f:
                lines = f.readlines()

            # Should have header line
            assert len(lines) > 1

            # Data lines should have correct format
            data_line = lines[1].strip()  # First data line
            values = data_line.split()

            # Should have Time + 41 measurements + 12 MVs = 54 values
            assert len(values) == 54

            # Each value should be in scientific notation
            for v in values:
                assert 'E' in v or 'e' in v, f"Value {v} not in scientific notation"

        finally:
            os.unlink(output_path)

    def test_multi_file_format(self, short_simulation_result):
        """Multi-file output should match original Fortran structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            write_multi_file(tmpdir, 'TE_data', short_simulation_result)

            # Check all expected files exist
            expected_files = [
                'TE_data_inc.dat',
                'TE_data_mv1.dat', 'TE_data_mv2.dat', 'TE_data_mv3.dat',
                'TE_data_me01.dat', 'TE_data_me02.dat', 'TE_data_me03.dat',
                'TE_data_me04.dat', 'TE_data_me05.dat', 'TE_data_me06.dat',
                'TE_data_me07.dat', 'TE_data_me08.dat', 'TE_data_me09.dat',
                'TE_data_me10.dat', 'TE_data_me11.dat',
            ]

            for filename in expected_files:
                filepath = os.path.join(tmpdir, filename)
                assert os.path.exists(filepath), f"Missing file: {filename}"

            # Check MV file format (4 values per line)
            mv1_path = os.path.join(tmpdir, 'TE_data_mv1.dat')
            with open(mv1_path, 'r') as f:
                line = f.readline().strip()
                values = line.split()
                assert len(values) == 4, f"MV file should have 4 values per line, got {len(values)}"

            # Check measurement file format
            me01_path = os.path.join(tmpdir, 'TE_data_me01.dat')
            with open(me01_path, 'r') as f:
                line = f.readline().strip()
                values = line.split()
                assert len(values) == 4, f"Measurement file should have 4 values per line"


class TestCLINormalOperation:
    """Test Case 1: Normal operation (no faults)."""

    def test_cli_matches_python_api_normal(self):
        """CLI output should match direct Python API call for normal operation."""
        seed = 12345
        duration = 0.5  # 30 minutes
        record_interval = 180  # Every 3 minutes like Fortran

        # Run via Python API
        sim = TEPSimulator(random_seed=seed, control_mode=ControlMode.CLOSED_LOOP)
        sim.initialize()
        api_result = sim.simulate(
            duration_hours=duration,
            record_interval=record_interval
        )

        # Run via CLI function
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            output_path = f.name

        try:
            run_simulation(
                duration_hours=duration,
                faults=None,
                fault_times=None,
                seed=seed,
                output=output_path,
                record_interval=record_interval,
                no_header=True,
                quiet=True
            )

            # Load CLI output
            cli_data = np.loadtxt(output_path)

            # Compare measurements (columns 1-41, after time column)
            # Note: cli_data has Time, 41 meas, 12 MVs = 54 columns
            cli_measurements = cli_data[:, 1:42]
            cli_mvs = cli_data[:, 42:54]

            # API result arrays
            api_measurements = api_result.measurements
            api_mvs = api_result.manipulated_vars

            # Ensure same number of samples
            min_samples = min(len(cli_measurements), len(api_measurements))
            assert min_samples > 0, "Should have data samples"

            # Values should match within E13.5 format precision (5 significant digits)
            # When data is written in Fortran format and read back, there's some
            # precision loss, so we use relative tolerance instead of decimal places
            np.testing.assert_allclose(
                cli_measurements[:min_samples],
                api_measurements[:min_samples],
                rtol=1e-4,  # 0.01% relative tolerance for E13.5 format
                atol=1e-10,
                err_msg="CLI measurements should match API measurements"
            )

            np.testing.assert_allclose(
                cli_mvs[:min_samples],
                api_mvs[:min_samples],
                rtol=1e-4,
                atol=1e-10,
                err_msg="CLI MVs should match API MVs"
            )

        finally:
            os.unlink(output_path)


class TestCLIFault1:
    """Test Case 2: IDV(1) - A/C Feed Ratio step change."""

    def test_cli_matches_python_api_fault1(self):
        """CLI output should match direct Python API call for Fault 1."""
        seed = 54321
        duration = 0.5  # 30 minutes
        record_interval = 180
        fault_id = 1
        fault_start = 0.1  # Start fault at 6 minutes

        # Run via Python API
        sim = TEPSimulator(random_seed=seed, control_mode=ControlMode.CLOSED_LOOP)
        sim.initialize()
        api_result = sim.simulate(
            duration_hours=duration,
            disturbances={fault_id: (fault_start, 1)},
            record_interval=record_interval
        )

        # Run via CLI function
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            output_path = f.name

        try:
            run_simulation(
                duration_hours=duration,
                faults=[fault_id],
                fault_times=[fault_start],
                seed=seed,
                output=output_path,
                record_interval=record_interval,
                no_header=True,
                quiet=True
            )

            # Load CLI output
            cli_data = np.loadtxt(output_path)
            cli_measurements = cli_data[:, 1:42]
            cli_mvs = cli_data[:, 42:54]

            # Compare
            min_samples = min(len(cli_measurements), len(api_result.measurements))
            assert min_samples > 0

            np.testing.assert_allclose(
                cli_measurements[:min_samples],
                api_result.measurements[:min_samples],
                rtol=1e-4,
                atol=1e-10,
                err_msg="CLI measurements should match API for Fault 1"
            )

            np.testing.assert_allclose(
                cli_mvs[:min_samples],
                api_result.manipulated_vars[:min_samples],
                rtol=1e-4,
                atol=1e-10,
                err_msg="CLI MVs should match API for Fault 1"
            )

        finally:
            os.unlink(output_path)


class TestCLIFault4:
    """Test Case 3: IDV(4) - Reactor Cooling Water Inlet Temperature step."""

    def test_cli_matches_python_api_fault4(self):
        """CLI output should match direct Python API call for Fault 4."""
        seed = 99999
        duration = 0.5
        record_interval = 180
        fault_id = 4
        fault_start = 0.15  # Start at 9 minutes

        # Run via Python API
        sim = TEPSimulator(random_seed=seed, control_mode=ControlMode.CLOSED_LOOP)
        sim.initialize()
        api_result = sim.simulate(
            duration_hours=duration,
            disturbances={fault_id: (fault_start, 1)},
            record_interval=record_interval
        )

        # Run via CLI function
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            output_path = f.name

        try:
            run_simulation(
                duration_hours=duration,
                faults=[fault_id],
                fault_times=[fault_start],
                seed=seed,
                output=output_path,
                record_interval=record_interval,
                no_header=True,
                quiet=True
            )

            # Load CLI output
            cli_data = np.loadtxt(output_path)
            cli_measurements = cli_data[:, 1:42]
            cli_mvs = cli_data[:, 42:54]

            # Compare
            min_samples = min(len(cli_measurements), len(api_result.measurements))
            assert min_samples > 0

            np.testing.assert_allclose(
                cli_measurements[:min_samples],
                api_result.measurements[:min_samples],
                rtol=1e-4,
                atol=1e-10,
                err_msg="CLI measurements should match API for Fault 4"
            )

            np.testing.assert_allclose(
                cli_mvs[:min_samples],
                api_result.manipulated_vars[:min_samples],
                rtol=1e-4,
                atol=1e-10,
                err_msg="CLI MVs should match API for Fault 4"
            )

        finally:
            os.unlink(output_path)


class TestCLIReproducibility:
    """Test that CLI produces reproducible results with same seed."""

    def test_same_seed_same_output(self):
        """Same seed should produce identical output files."""
        seed = 11111
        duration = 0.25
        record_interval = 180

        with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as f1:
            output1 = f1.name
        with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as f2:
            output2 = f2.name

        try:
            # Run twice with same seed
            run_simulation(
                duration_hours=duration,
                seed=seed,
                output=output1,
                record_interval=record_interval,
                no_header=True,
                quiet=True
            )

            run_simulation(
                duration_hours=duration,
                seed=seed,
                output=output2,
                record_interval=record_interval,
                no_header=True,
                quiet=True
            )

            # Load and compare
            data1 = np.loadtxt(output1)
            data2 = np.loadtxt(output2)

            np.testing.assert_array_almost_equal(
                data1, data2,
                decimal=10,
                err_msg="Same seed should produce identical results"
            )

        finally:
            os.unlink(output1)
            os.unlink(output2)


class TestCLIMultipleFaults:
    """Test CLI with multiple simultaneous faults."""

    def test_multiple_faults_different_start_times(self):
        """Multiple faults with different start times should work."""
        seed = 77777
        duration = 0.5
        record_interval = 180
        faults = [1, 4]
        fault_times = [0.1, 0.2]

        # Run via Python API
        sim = TEPSimulator(random_seed=seed, control_mode=ControlMode.CLOSED_LOOP)
        sim.initialize()
        api_result = sim.simulate(
            duration_hours=duration,
            disturbances={1: (0.1, 1), 4: (0.2, 1)},
            record_interval=record_interval
        )

        # Run via CLI
        with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as f:
            output_path = f.name

        try:
            run_simulation(
                duration_hours=duration,
                faults=faults,
                fault_times=fault_times,
                seed=seed,
                output=output_path,
                record_interval=record_interval,
                no_header=True,
                quiet=True
            )

            cli_data = np.loadtxt(output_path)
            cli_measurements = cli_data[:, 1:42]

            min_samples = min(len(cli_measurements), len(api_result.measurements))

            np.testing.assert_allclose(
                cli_measurements[:min_samples],
                api_result.measurements[:min_samples],
                rtol=1e-4,
                atol=1e-10
            )

        finally:
            os.unlink(output_path)


class TestCLIEntryPoint:
    """Test CLI as a command-line entry point."""

    def test_cli_help(self):
        """CLI should display help without errors."""
        result = subprocess.run(
            [sys.executable, '-m', 'tep.cli', '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'Tennessee Eastman Process' in result.stdout
        assert '--duration' in result.stdout
        assert '--faults' in result.stdout
        assert '--seed' in result.stdout

    def test_cli_list_faults(self):
        """CLI should list available faults."""
        result = subprocess.run(
            [sys.executable, '-m', 'tep.cli', '--list-faults'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'IDV' in result.stdout
        assert 'Feed' in result.stdout or 'Step' in result.stdout

    def test_cli_short_simulation(self):
        """CLI should run a short simulation successfully."""
        with tempfile.NamedTemporaryFile(suffix='.dat', delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [
                    sys.executable, '-m', 'tep.cli',
                    '--duration', '0.01',
                    '--output', output_path,
                    '--quiet'
                ],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(output_path)

            # Verify file has content
            data = np.loadtxt(output_path, skiprows=1)  # Skip header
            assert len(data) > 0

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
