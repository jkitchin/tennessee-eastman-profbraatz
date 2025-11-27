"""
Tests for the disturbances module.
"""

import pytest
import numpy as np
from tep.disturbances import RandomGenerator, DisturbanceManager
from tep.constants import DEFAULT_RANDOM_SEED, NUM_DISTURBANCES


class TestRandomGenerator:
    """Test the random number generator."""

    def test_reproducibility(self):
        """Same seed should produce same sequence."""
        rng1 = RandomGenerator(12345)
        rng2 = RandomGenerator(12345)

        for _ in range(100):
            assert rng1.random() == rng2.random()

    def test_range_unsigned(self):
        """Unsigned random should be in [0, 1]."""
        rng = RandomGenerator(DEFAULT_RANDOM_SEED)
        for _ in range(1000):
            r = rng.random(signed=False)
            assert 0 <= r <= 1

    def test_range_signed(self):
        """Signed random should be in [-1, 1]."""
        rng = RandomGenerator(DEFAULT_RANDOM_SEED)
        for _ in range(1000):
            r = rng.random(signed=True)
            assert -1 <= r <= 1

    def test_gaussian_mean(self):
        """Gaussian samples should have approximately zero mean."""
        rng = RandomGenerator(DEFAULT_RANDOM_SEED)
        samples = [rng.gaussian(std=1.0) for _ in range(10000)]
        mean = np.mean(samples)
        assert abs(mean) < 0.1

    def test_gaussian_std(self):
        """Gaussian samples should have approximately correct std."""
        rng = RandomGenerator(DEFAULT_RANDOM_SEED)
        std_target = 2.0
        samples = [rng.gaussian(std=std_target) for _ in range(10000)]
        std = np.std(samples)
        assert abs(std - std_target) < 0.2


class TestDisturbanceManager:
    """Test the disturbance manager."""

    @pytest.fixture
    def manager(self):
        return DisturbanceManager(DEFAULT_RANDOM_SEED)

    def test_initial_idv_all_off(self, manager):
        """All disturbances should be off initially."""
        assert all(manager.idv == 0)

    def test_set_idv(self, manager):
        """Setting IDV should work correctly."""
        manager.set_idv(1, 1)
        assert manager.idv[0] == 1

        manager.set_idv(1, 0)
        assert manager.idv[0] == 0

    def test_clear_disturbances(self, manager):
        """Clear should turn off all disturbances."""
        for i in range(1, NUM_DISTURBANCES + 1):
            manager.set_idv(i, 1)

        manager.clear_all_disturbances()
        assert all(manager.idv == 0)

    def test_walk_value_at_zero(self, manager):
        """Walk value at t=0 should be the initial value."""
        manager.update_walks(0.0)
        val = manager.get_walk_value(0, 0.0)
        assert np.isfinite(val)

    def test_walk_value_changes_with_time(self, manager):
        """Walk value should change over time when disturbance is on."""
        manager.set_idv(8, 1)  # Enable random variation

        manager.update_walks(0.0)
        val1 = manager.get_walk_value(0, 0.0)

        # Advance time significantly
        for t in np.linspace(0.01, 1.0, 100):
            manager.update_walks(t)

        val2 = manager.get_walk_value(0, 1.0)

        # Values should be different
        assert val1 != val2

    def test_xst_composition_base(self, manager):
        """Base composition should sum to approximately 1."""
        comp = manager.get_xst_composition(0.0)
        assert abs(sum(comp) - 1.0) < 0.01

    def test_xst_composition_with_idv1(self, manager):
        """IDV(1) should affect composition."""
        comp_base = manager.get_xst_composition(0.0)

        manager.set_idv(1, 1)
        comp_disturbed = manager.get_xst_composition(0.0)

        # A composition should change
        assert comp_base[0] != comp_disturbed[0]

    def test_feed_temperature_base(self, manager):
        """Base feed temperature should be around 45 C."""
        temp = manager.get_feed_temperature(1, 0.0)
        assert 40 < temp < 50

    def test_cooling_water_temp_base(self, manager):
        """Base cooling water temp should be reasonable."""
        temp = manager.get_cooling_water_temp('reactor', 0.0)
        assert 30 < temp < 45

    def test_valve_stuck_default(self, manager):
        """Valves should not be stuck by default."""
        assert not manager.is_valve_stuck(10)
        assert not manager.is_valve_stuck(11)

    def test_valve_stuck_with_idv(self, manager):
        """IDV(14) should cause reactor CW valve to stick."""
        manager.set_idv(14, 1)
        assert manager.is_valve_stuck(10)

    def test_feed_loss_factor(self, manager):
        """Feed loss factor should be 1 normally, 0 with IDV(6)."""
        assert manager.get_feed_loss_factor() == 1.0

        manager.set_idv(6, 1)
        assert manager.get_feed_loss_factor() == 0.0

    def test_header_pressure_factor(self, manager):
        """Header pressure factor should be 1 normally, 0.8 with IDV(7)."""
        assert manager.get_header_pressure_factor() == 1.0

        manager.set_idv(7, 1)
        assert manager.get_header_pressure_factor() == 0.8
