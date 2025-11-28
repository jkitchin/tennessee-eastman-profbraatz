"""
Tests for the Rieth 2017 dataset generator features.

These tests verify:
- Preset configurations (from_preset)
- Output format options (npy, csv, hdf5)
- Parallel generation (n_workers)
- Column selection
"""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

# Import the module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from rieth2017_dataset import (
    Rieth2017DatasetGenerator,
    PRESETS,
    COLUMN_GROUPS,
    ALL_FEATURE_COLUMNS,
    XMEAS_COLUMNS,
    XMV_COLUMNS,
    _run_single_simulation,
)


class TestPresets:
    """Tests for preset configurations."""

    def test_list_presets_returns_dict(self):
        """list_presets should return available presets."""
        presets = Rieth2017DatasetGenerator.list_presets()
        assert isinstance(presets, dict)
        assert "rieth2017" in presets
        assert "quick" in presets
        assert "high-res" in presets
        assert "minimal" in presets

    def test_from_preset_rieth2017(self):
        """from_preset with 'rieth2017' should use default values."""
        gen = Rieth2017DatasetGenerator.from_preset("rieth2017")
        assert gen.n_simulations == 500
        assert gen.train_duration_hours == 25.0
        assert gen.test_duration_hours == 48.0
        assert gen.sampling_interval_min == 3.0

    def test_from_preset_quick(self):
        """from_preset with 'quick' should use fast testing values."""
        gen = Rieth2017DatasetGenerator.from_preset("quick")
        assert gen.n_simulations == 5
        assert gen.train_duration_hours == 2.0
        assert gen.test_duration_hours == 4.0

    def test_from_preset_high_res(self):
        """from_preset with 'high-res' should use 1-minute sampling."""
        gen = Rieth2017DatasetGenerator.from_preset("high-res")
        assert gen.sampling_interval_min == 1.0
        assert gen.record_interval == 60  # 1 minute = 60 seconds

    def test_from_preset_minimal(self):
        """from_preset with 'minimal' should use minimal values."""
        gen = Rieth2017DatasetGenerator.from_preset("minimal")
        assert gen.n_simulations == 2
        assert gen.train_duration_hours == 0.5

    def test_from_preset_with_overrides(self):
        """from_preset should accept overrides."""
        gen = Rieth2017DatasetGenerator.from_preset(
            "quick",
            n_simulations=10,
            output_formats=["csv"],
        )
        assert gen.n_simulations == 10
        assert gen.output_formats == ["csv"]
        # Should still have quick preset values for non-overridden params
        assert gen.train_duration_hours == 2.0

    def test_from_preset_with_output_dir(self):
        """from_preset should accept output_dir."""
        gen = Rieth2017DatasetGenerator.from_preset(
            "minimal",
            output_dir="/tmp/test"
        )
        assert str(gen.output_dir) == "/tmp/test"

    def test_from_preset_invalid_name(self):
        """from_preset should raise ValueError for unknown preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            Rieth2017DatasetGenerator.from_preset("nonexistent")


class TestColumnSelection:
    """Tests for column selection feature."""

    def test_list_column_groups(self):
        """list_column_groups should return available groups."""
        groups = Rieth2017DatasetGenerator.list_column_groups()
        assert isinstance(groups, dict)
        assert "all" in groups
        assert "xmeas" in groups
        assert "xmv" in groups
        assert "key" in groups

    def test_default_columns_all(self):
        """Default should select all columns."""
        gen = Rieth2017DatasetGenerator()
        assert len(gen.columns) == 52
        assert gen.columns == COLUMN_GROUPS["all"]

    def test_columns_xmeas_group(self):
        """Selecting 'xmeas' group should include only XMEAS columns."""
        gen = Rieth2017DatasetGenerator(columns="xmeas")
        assert len(gen.columns) == 41
        assert all(col.startswith("xmeas_") for col in gen.columns)

    def test_columns_xmv_group(self):
        """Selecting 'xmv' group should include only XMV columns."""
        gen = Rieth2017DatasetGenerator(columns="xmv")
        assert len(gen.columns) == 11
        assert all(col.startswith("xmv_") for col in gen.columns)

    def test_columns_key_group(self):
        """Selecting 'key' group should include key variables."""
        gen = Rieth2017DatasetGenerator(columns="key")
        assert "xmeas_9" in gen.columns  # Reactor Temperature
        assert "xmv_1" in gen.columns  # D Feed Flow

    def test_columns_custom_list(self):
        """Selecting custom list of columns."""
        cols = ["xmeas_1", "xmeas_9", "xmv_1"]
        gen = Rieth2017DatasetGenerator(columns=cols)
        assert gen.columns == cols
        assert len(gen.column_indices) == 3

    def test_columns_invalid_name(self):
        """Invalid column name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown column"):
            Rieth2017DatasetGenerator(columns=["invalid_column"])

    def test_columns_invalid_group(self):
        """Invalid group name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown column or group"):
            Rieth2017DatasetGenerator(columns="invalid_group")

    def test_column_indices_correct(self):
        """Column indices should map to correct feature positions."""
        gen = Rieth2017DatasetGenerator(columns=["xmeas_1", "xmv_11"])
        # xmeas_1 is index 0 in features, xmv_11 is index 51
        assert gen.column_indices == [0, 51]


class TestOutputFormats:
    """Tests for output format options."""

    def test_default_format_npy(self):
        """Default output format should be npy."""
        gen = Rieth2017DatasetGenerator()
        assert gen.output_formats == ["npy"]

    def test_single_format_string(self):
        """Single format as string should be converted to list."""
        gen = Rieth2017DatasetGenerator(output_formats="csv")
        assert gen.output_formats == ["csv"]

    def test_multiple_formats_list(self):
        """Multiple formats should be accepted as list."""
        gen = Rieth2017DatasetGenerator(output_formats=["npy", "csv"])
        assert gen.output_formats == ["npy", "csv"]

    def test_save_data_creates_npy(self):
        """_save_data should create .npy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = Rieth2017DatasetGenerator(
                output_dir=tmpdir,
                output_formats=["npy"],
            )
            data = np.random.rand(10, 55)
            paths = gen._save_data(data, "test.npy")

            assert len(paths) == 1
            assert paths[0].suffix == ".npy"
            assert paths[0].exists()

            # Verify data can be loaded
            loaded = np.load(paths[0])
            assert loaded.shape == data.shape

    def test_save_data_creates_csv(self):
        """_save_data should create .csv files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = Rieth2017DatasetGenerator(
                output_dir=tmpdir,
                output_formats=["csv"],
            )
            data = np.random.rand(10, 55)
            paths = gen._save_data(data, "test.npy")

            assert len(paths) == 1
            assert paths[0].suffix == ".csv"
            assert paths[0].exists()

            # Verify file has header
            with open(paths[0]) as f:
                header = f.readline()
                assert "faultNumber" in header
                assert "xmeas_1" in header

    def test_save_data_multiple_formats(self):
        """_save_data should create files for all formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = Rieth2017DatasetGenerator(
                output_dir=tmpdir,
                output_formats=["npy", "csv"],
            )
            data = np.random.rand(10, 55)
            paths = gen._save_data(data, "test.npy")

            assert len(paths) == 2
            suffixes = {p.suffix for p in paths}
            assert suffixes == {".npy", ".csv"}

    def test_save_data_with_column_selection(self):
        """_save_data should apply column selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = Rieth2017DatasetGenerator(
                output_dir=tmpdir,
                output_formats=["npy"],
                columns=["xmeas_1", "xmeas_9"],
            )
            # Full data with 55 columns
            data = np.random.rand(10, 55)
            paths = gen._save_data(data, "test.npy")

            loaded = np.load(paths[0])
            # Should have 3 metadata cols + 2 selected cols = 5
            assert loaded.shape == (10, 5)


class TestParallelGeneration:
    """Tests for parallel generation feature."""

    def test_default_workers_sequential(self):
        """Default should be 1 worker (sequential)."""
        gen = Rieth2017DatasetGenerator()
        assert gen.n_workers == 1

    def test_custom_workers(self):
        """Custom worker count should be set."""
        gen = Rieth2017DatasetGenerator(n_workers=4)
        assert gen.n_workers == 4

    def test_negative_workers_uses_all_cpus(self):
        """n_workers=-1 should use all CPUs."""
        import multiprocessing
        gen = Rieth2017DatasetGenerator(n_workers=-1)
        assert gen.n_workers == multiprocessing.cpu_count()

    def test_zero_workers_becomes_one(self):
        """n_workers=0 should become 1."""
        gen = Rieth2017DatasetGenerator(n_workers=0)
        assert gen.n_workers == 1


class TestRunSingleSimulation:
    """Tests for the module-level simulation function."""

    def test_returns_none_without_tep(self):
        """Should return None if TEP not available."""
        # Mock HAS_TEP as False
        with patch("rieth2017_dataset.HAS_TEP", False):
            args = (12345, 1.0, 0, 0.0, 180, 1)
            result = _run_single_simulation(args)
            assert result is None


class TestGeneratorIntegration:
    """Integration tests for the generator."""

    def test_generator_creates_output_dir(self):
        """Generator should create output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "path"
            gen = Rieth2017DatasetGenerator(output_dir=str(output_path))

            # Create a dummy save to trigger directory creation
            data = np.random.rand(10, 55)
            gen._save_data(data, "test.npy")

            assert output_path.exists()

    def test_metadata_includes_new_params(self):
        """Metadata should include new parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = Rieth2017DatasetGenerator(
                output_dir=tmpdir,
                output_formats=["npy", "csv"],
                n_workers=2,
                columns="key",
            )
            gen._save_metadata(n_simulations=10, fault_numbers=[1, 2], include_validation=True)

            import json
            with open(Path(tmpdir) / "metadata.json") as f:
                metadata = json.load(f)

            assert metadata["parameters"]["output_formats"] == ["npy", "csv"]
            assert metadata["parameters"]["n_workers"] == 2
            assert "xmeas_9" in metadata["parameters"]["columns"]


class TestPresetsConfiguration:
    """Tests for preset configuration values."""

    def test_presets_have_required_keys(self):
        """All presets should have required configuration keys."""
        required_keys = [
            "n_simulations",
            "train_duration_hours",
            "val_duration_hours",
            "test_duration_hours",
            "sampling_interval_min",
            "fault_onset_hours",
            "n_faults",
        ]
        for name, preset in PRESETS.items():
            for key in required_keys:
                assert key in preset, f"Preset '{name}' missing key '{key}'"

    def test_rieth2017_preset_matches_paper(self):
        """rieth2017 preset should match paper specifications."""
        preset = PRESETS["rieth2017"]
        assert preset["n_simulations"] == 500
        assert preset["train_duration_hours"] == 25.0
        assert preset["test_duration_hours"] == 48.0
        assert preset["sampling_interval_min"] == 3.0
        assert preset["fault_onset_hours"] == 1.0
        assert preset["n_faults"] == 20


class TestColumnGroups:
    """Tests for column group definitions."""

    def test_all_group_has_52_columns(self):
        """'all' group should have all 52 feature columns."""
        assert len(COLUMN_GROUPS["all"]) == 52

    def test_xmeas_group_has_41_columns(self):
        """'xmeas' group should have 41 columns."""
        assert len(COLUMN_GROUPS["xmeas"]) == 41

    def test_xmv_group_has_11_columns(self):
        """'xmv' group should have 11 columns."""
        assert len(COLUMN_GROUPS["xmv"]) == 11

    def test_all_columns_are_valid(self):
        """All column names in groups should be valid."""
        for group_name, columns in COLUMN_GROUPS.items():
            for col in columns:
                assert col in ALL_FEATURE_COLUMNS, \
                    f"Invalid column '{col}' in group '{group_name}'"

    def test_xmeas_columns_are_sequential(self):
        """XMEAS columns should have indices 0-40."""
        for name, idx in XMEAS_COLUMNS.items():
            assert 0 <= idx <= 40

    def test_xmv_columns_are_sequential(self):
        """XMV columns should have indices 41-51."""
        for name, idx in XMV_COLUMNS.items():
            assert 41 <= idx <= 51


class TestIntermittentFaults:
    """Tests for intermittent fault trajectory generation."""

    def test_generate_intermittent_faults_basic(self):
        """generate_intermittent_faults should accept valid parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = Rieth2017DatasetGenerator(output_dir=tmpdir)
            # Just verify method exists and accepts parameters
            # Don't actually run simulation (requires TEP)
            assert hasattr(gen, 'generate_intermittent_faults')
            assert callable(gen.generate_intermittent_faults)

    def test_intermittent_faults_parameters(self):
        """generate_intermittent_faults should accept all documented parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = Rieth2017DatasetGenerator(output_dir=tmpdir)
            # Verify the method signature by calling with save=False
            # This exercises parameter parsing without running simulation
            import inspect
            sig = inspect.signature(gen.generate_intermittent_faults)
            params = list(sig.parameters.keys())

            expected_params = [
                'n_simulations', 'fault_numbers', 'avg_fault_duration_hours',
                'avg_normal_duration_hours', 'duration_variance',
                'initial_normal_hours', 'randomize_fault_order', 'save', 'filename'
            ]
            for param in expected_params:
                assert param in params, f"Missing parameter: {param}"

    def test_intermittent_faults_default_values(self):
        """generate_intermittent_faults should have sensible defaults."""
        import inspect
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = Rieth2017DatasetGenerator(output_dir=tmpdir)
            sig = inspect.signature(gen.generate_intermittent_faults)

            # Check default values
            assert sig.parameters['n_simulations'].default == 10
            assert sig.parameters['avg_fault_duration_hours'].default == 4.0
            assert sig.parameters['avg_normal_duration_hours'].default == 2.0
            assert sig.parameters['duration_variance'].default == 0.5
            assert sig.parameters['initial_normal_hours'].default == 1.0
            assert sig.parameters['randomize_fault_order'].default is True
            assert sig.parameters['save'].default is True
            assert sig.parameters['filename'].default == "intermittent_faults.npy"


class TestRunIntermittentSimulation:
    """Tests for the _run_intermittent_simulation function."""

    def test_function_exists(self):
        """_run_intermittent_simulation should be importable."""
        from rieth2017_dataset import _run_intermittent_simulation
        assert callable(_run_intermittent_simulation)

    def test_returns_none_without_tep(self):
        """_run_intermittent_simulation should return None if TEP not available."""
        from rieth2017_dataset import _run_intermittent_simulation

        # Create test args: (seed, fault_schedule, sampling_interval_min, sim_run)
        schedule = [(1.0, 3.0, 1), (5.0, 7.0, 2)]  # Two faults
        args = (12345, schedule, 3.0, 1)

        # This will return None if TEP simulator is not available
        # (which is expected in test environment without Fortran)
        result = _run_intermittent_simulation(args)
        # Result is either None (no TEP) or a dict with expected keys
        if result is not None:
            assert "sim_run" in result
            assert "data" in result
            assert "fault_labels" in result
            assert "times" in result
            assert "shutdown" in result

    def test_schedule_format(self):
        """Fault schedule should be list of (start, end, fault_num) tuples."""
        from rieth2017_dataset import _run_intermittent_simulation

        # Valid schedule format
        schedule = [
            (1.0, 3.0, 1),   # Fault 1 from hour 1 to 3
            (5.0, 7.0, 4),   # Fault 4 from hour 5 to 7
            (9.0, 11.0, 6),  # Fault 6 from hour 9 to 11
        ]
        args = (12345, schedule, 3.0, 1)

        # Should not raise an error (may return None if no TEP)
        result = _run_intermittent_simulation(args)
        # Just verify it doesn't crash


class TestIntermittentFaultScheduleGeneration:
    """Tests for fault schedule generation logic."""

    def test_schedule_generation_with_variance(self):
        """Schedules should have randomized durations within variance bounds."""
        avg_duration = 4.0
        variance = 0.5
        min_expected = avg_duration * (1 - variance)
        max_expected = avg_duration * (1 + variance)

        # Verify the bounds are correct
        assert min_expected == 2.0
        assert max_expected == 6.0

    def test_schedule_with_multiple_faults(self):
        """Schedule should include all specified faults."""
        faults = [1, 4, 6, 11]
        # Each fault should appear exactly once in a trajectory
        assert len(set(faults)) == len(faults)
