"""Tests for the JAX backend implementation.

This module tests the JAX backend scaffold including:
- State structure creation and pytree compatibility
- API compatibility with PythonTEProcess
- Backend registration and selection
- Basic JAX operations (when JAX is available)
"""

import pytest
import numpy as np

# Check if JAX is available for conditional tests
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# Skip all tests in this module if JAX is not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


class TestJaxBackendImports:
    """Test that JAX backend modules can be imported."""

    def test_import_jax_backend_module(self):
        """Test importing the jax_backend module."""
        from tep import jax_backend
        assert jax_backend is not None

    def test_import_jax_te_process(self):
        """Test importing JaxTEProcess class."""
        from tep.jax_backend import JaxTEProcess
        assert JaxTEProcess is not None

    def test_import_jax_te_process_wrapper(self):
        """Test importing JaxTEProcessWrapper class."""
        from tep.jax_backend import JaxTEProcessWrapper
        assert JaxTEProcessWrapper is not None

    def test_import_state_structures(self):
        """Test importing state structure classes."""
        from tep.jax_backend import (
            TEPState, ConstBlock, ReactorState, SeparatorState,
            StripperState, CompressorState, ValveState, StreamState,
            WalkState, MeasurementState
        )
        assert TEPState is not None
        assert ConstBlock is not None
        assert ReactorState is not None

    def test_is_jax_available_function(self):
        """Test the is_jax_available utility function."""
        from tep.jax_backend import is_jax_available
        assert is_jax_available() == True

    def test_get_jax_backend_info(self):
        """Test the backend info function."""
        from tep.jax_backend import get_jax_backend_info
        info = get_jax_backend_info()
        assert info["available"] == True
        assert "version" in info
        assert "devices" in info


class TestStateStructures:
    """Test JAX-compatible state structures."""

    def test_const_block_creation(self):
        """Test creating ConstBlock with default values."""
        from tep.jax_backend import create_const_block
        const = create_const_block()

        assert const.xmw.shape == (8,)
        assert const.avp.shape == (8,)
        assert const.bvp.shape == (8,)

        # Check some known values
        assert float(const.xmw[0]) == 2.0  # Molecular weight of component A
        assert float(const.xmw[7]) == 76.0  # Molecular weight of component H

    def test_reactor_state_creation(self):
        """Test creating ReactorState."""
        from tep.jax_backend import create_initial_reactor_state
        reactor = create_initial_reactor_state()

        assert reactor.uclr.shape == (8,)
        assert reactor.ucvr.shape == (8,)
        assert reactor.ppr.shape == (8,)
        assert reactor.vtr == 1300.0  # Total reactor volume

    def test_separator_state_creation(self):
        """Test creating SeparatorState."""
        from tep.jax_backend import create_initial_separator_state
        sep = create_initial_separator_state()

        assert sep.ucls.shape == (8,)
        assert sep.vts == 3500.0  # Total separator volume

    def test_valve_state_creation(self):
        """Test creating ValveState."""
        from tep.jax_backend import create_initial_valve_state
        valves = create_initial_valve_state()

        assert valves.vcv.shape == (12,)
        assert valves.vrng.shape == (12,)
        assert valves.vtau.shape == (12,)

    def test_stream_state_creation(self):
        """Test creating StreamState with initial compositions."""
        from tep.jax_backend import create_initial_stream_state
        streams = create_initial_stream_state()

        assert streams.ftm.shape == (13,)
        assert streams.fcm.shape == (8, 13)
        assert streams.xst.shape == (8, 13)

        # Check initial feed composition for stream 4 (A and C)
        assert abs(float(streams.xst[0, 3]) - 0.4850) < 1e-6  # Component A
        assert abs(float(streams.xst[2, 3]) - 0.5100) < 1e-6  # Component C

    def test_measurement_state_creation(self):
        """Test creating MeasurementState."""
        from tep.jax_backend import create_initial_measurement_state
        meas = create_initial_measurement_state()

        assert meas.xmeas.shape == (41,)
        assert meas.xdel.shape == (41,)
        assert meas.xns.shape == (41,)
        assert meas.tgas == 0.1
        assert meas.tprod == 0.25

    def test_initial_yy_values(self):
        """Test initial state vector values."""
        from tep.jax_backend import INITIAL_YY

        assert INITIAL_YY.shape == (50,)
        # Check some known initial values
        assert abs(float(INITIAL_YY[0]) - 10.40491389) < 1e-6
        assert abs(float(INITIAL_YY[38]) - 63.05263039) < 1e-6  # XMV(1)

    def test_namedtuple_immutability(self):
        """Test that state structures are immutable (NamedTuple behavior)."""
        from tep.jax_backend import create_initial_reactor_state
        reactor = create_initial_reactor_state()

        # NamedTuples don't allow item assignment
        with pytest.raises((TypeError, AttributeError)):
            reactor.vtr = 2000.0

    def test_namedtuple_replace(self):
        """Test that _replace works for creating modified states."""
        from tep.jax_backend import create_initial_reactor_state
        reactor = create_initial_reactor_state()

        # _replace creates a new instance with modified fields
        new_reactor = reactor._replace(vtr=2000.0)

        assert reactor.vtr == 1300.0  # Original unchanged
        assert new_reactor.vtr == 2000.0  # New has updated value


class TestPytreeCompatibility:
    """Test that state structures are JAX pytree compatible."""

    def test_const_block_is_pytree(self):
        """Test ConstBlock is recognized as a pytree."""
        from tep.jax_backend import create_const_block
        const = create_const_block()

        # Should be able to flatten and unflatten
        leaves, treedef = jax.tree_util.tree_flatten(const)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Check reconstruction
        np.testing.assert_array_equal(const.xmw, reconstructed.xmw)

    def test_tep_state_is_pytree(self):
        """Test full TEPState is pytree compatible."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, _ = process.initialize(key)

        # Should be able to flatten
        leaves, treedef = jax.tree_util.tree_flatten(state)
        assert len(leaves) > 0

        # Should be able to unflatten
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_array_equal(state.yy, reconstructed.yy)

    def test_tree_map_on_state(self):
        """Test that jax.tree_map works on state."""
        from tep.jax_backend import create_initial_reactor_state

        reactor = create_initial_reactor_state()

        # Apply tree_map (e.g., double all arrays)
        def double_if_array(x):
            if hasattr(x, 'shape'):
                return x * 2
            return x

        doubled = jax.tree_util.tree_map(double_if_array, reactor)

        # Check that arrays were doubled
        np.testing.assert_array_almost_equal(
            doubled.uclr, reactor.uclr * 2
        )


class TestJaxTEProcessBasic:
    """Test basic JaxTEProcess functionality."""

    def test_process_creation(self):
        """Test creating JaxTEProcess instance."""
        from tep.jax_backend import JaxTEProcess
        process = JaxTEProcess()
        assert process is not None
        assert process._nn == 50

    def test_const_property(self):
        """Test accessing physical constants."""
        from tep.jax_backend import JaxTEProcess
        process = JaxTEProcess()
        const = process.const

        assert const is not None
        assert const.xmw.shape == (8,)

    def test_initialize_returns_state_and_key(self):
        """Test that initialize returns state and updated key."""
        from tep.jax_backend import JaxTEProcess, TEPState

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)

        state, new_key = process.initialize(key)

        assert isinstance(state, TEPState)
        assert state.yy.shape == (50,)
        assert state.yp.shape == (50,)
        assert state.time == 0.0
        # Key should be different after split
        assert not np.array_equal(key, new_key)

    def test_initial_state_values(self):
        """Test initial state values match expected."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, _ = process.initialize(key)

        # Check initial values
        assert abs(float(state.yy[0]) - 10.40491389) < 1e-6
        assert abs(float(state.yy[38]) - 63.05263039) < 1e-6

    def test_step_advances_time(self):
        """Test that step advances simulation time."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, key = process.initialize(key)

        initial_time = state.time
        new_state, _ = process.step(state, key)

        assert new_state.time > initial_time
        assert abs(new_state.time - 1.0/3600.0) < 1e-10

    def test_get_xmeas(self):
        """Test getting measurements from state."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, _ = process.initialize(key)

        xmeas = process.get_xmeas(state)
        assert xmeas.shape == (41,)

    def test_get_xmv(self):
        """Test getting manipulated variables from state."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, _ = process.initialize(key)

        xmv = process.get_xmv(state)
        assert xmv.shape == (12,)

    def test_set_xmv(self):
        """Test setting manipulated variables."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, _ = process.initialize(key)

        new_state = process.set_xmv(state, 1, 75.0)
        assert abs(float(new_state.xmv[0]) - 75.0) < 1e-10

        # Test clipping to 100
        new_state = process.set_xmv(state, 1, 150.0)
        assert abs(float(new_state.xmv[0]) - 100.0) < 1e-10

        # Test clipping to 0
        new_state = process.set_xmv(state, 1, -10.0)
        assert abs(float(new_state.xmv[0]) - 0.0) < 1e-10

    def test_set_idv(self):
        """Test setting disturbance variables."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, _ = process.initialize(key)

        new_state = process.set_idv(state, 1, 1)
        assert int(new_state.idv[0]) == 1

    def test_clear_disturbances(self):
        """Test clearing all disturbances."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, _ = process.initialize(key)

        # Set some disturbances
        state = process.set_idv(state, 1, 1)
        state = process.set_idv(state, 5, 1)

        # Clear all
        state = process.clear_disturbances(state)
        assert int(jnp.sum(state.idv)) == 0


class TestJaxTEProcessWrapperBasic:
    """Test JaxTEProcessWrapper (stateful interface)."""

    def test_wrapper_creation(self):
        """Test creating wrapper instance."""
        from tep.jax_backend import JaxTEProcessWrapper
        wrapper = JaxTEProcessWrapper()
        assert wrapper is not None
        assert wrapper._nn == 50

    def test_wrapper_with_seed(self):
        """Test creating wrapper with specific seed."""
        from tep.jax_backend import JaxTEProcessWrapper
        wrapper = JaxTEProcessWrapper(random_seed=42)
        assert wrapper is not None

    def test_wrapper_initialize(self):
        """Test wrapper initialization."""
        from tep.jax_backend import JaxTEProcessWrapper
        wrapper = JaxTEProcessWrapper()
        wrapper.initialize()

        assert wrapper._initialized
        assert wrapper.time == 0.0
        assert wrapper._state is not None

    def test_wrapper_step(self):
        """Test wrapper step function."""
        from tep.jax_backend import JaxTEProcessWrapper
        wrapper = JaxTEProcessWrapper()
        wrapper.initialize()

        initial_time = wrapper.time
        wrapper.step()

        assert wrapper.time > initial_time

    def test_wrapper_properties(self):
        """Test wrapper property accessors."""
        from tep.jax_backend import JaxTEProcessWrapper
        wrapper = JaxTEProcessWrapper()
        wrapper.initialize()

        # Test numpy array properties
        assert wrapper.yy.shape == (50,)
        assert wrapper.yp.shape == (50,)
        assert wrapper.xmeas.shape == (41,)
        assert wrapper.xmv.shape == (12,)
        assert wrapper.idv.shape == (20,)

        # Should return numpy arrays, not jax arrays
        assert isinstance(wrapper.yy, np.ndarray)
        assert isinstance(wrapper.xmeas, np.ndarray)

    def test_wrapper_set_xmv(self):
        """Test setting MV through wrapper."""
        from tep.jax_backend import JaxTEProcessWrapper
        wrapper = JaxTEProcessWrapper()
        wrapper.initialize()

        wrapper.set_xmv(1, 80.0)
        assert abs(wrapper.xmv[0] - 80.0) < 1e-10

    def test_wrapper_set_idv(self):
        """Test setting IDV through wrapper."""
        from tep.jax_backend import JaxTEProcessWrapper
        wrapper = JaxTEProcessWrapper()
        wrapper.initialize()

        wrapper.set_idv(1, 1)
        assert wrapper.idv[0] == 1

    def test_wrapper_clear_disturbances(self):
        """Test clearing disturbances through wrapper."""
        from tep.jax_backend import JaxTEProcessWrapper
        wrapper = JaxTEProcessWrapper()
        wrapper.initialize()

        wrapper.set_idv(1, 1)
        wrapper.set_idv(5, 1)
        wrapper.clear_disturbances()

        assert np.sum(wrapper.idv) == 0

    def test_wrapper_evaluate(self):
        """Test wrapper evaluate function."""
        from tep.jax_backend import JaxTEProcessWrapper
        wrapper = JaxTEProcessWrapper()
        wrapper.initialize()

        yp = wrapper.evaluate(0.0, wrapper.yy)
        assert yp.shape == (50,)
        assert isinstance(yp, np.ndarray)

    def test_wrapper_get_state(self):
        """Test getting state vector."""
        from tep.jax_backend import JaxTEProcessWrapper
        wrapper = JaxTEProcessWrapper()
        wrapper.initialize()

        state = wrapper.get_state()
        assert state.shape == (50,)
        assert isinstance(state, np.ndarray)

    def test_wrapper_set_state(self):
        """Test setting state vector."""
        from tep.jax_backend import JaxTEProcessWrapper
        wrapper = JaxTEProcessWrapper()
        wrapper.initialize()

        # Modify state
        new_state = wrapper.get_state()
        new_state[0] = 15.0

        wrapper.set_state(new_state)
        assert abs(wrapper.yy[0] - 15.0) < 1e-10

    def test_wrapper_requires_init(self):
        """Test that step raises error if not initialized."""
        from tep.jax_backend import JaxTEProcessWrapper
        wrapper = JaxTEProcessWrapper()

        with pytest.raises(RuntimeError, match="not initialized"):
            wrapper.step()


class TestBackendRegistration:
    """Test JAX backend registration with TEPSimulator."""

    def test_jax_in_available_backends(self):
        """Test that JAX appears in available backends list."""
        from tep import get_available_backends
        backends = get_available_backends()
        assert "jax" in backends

    def test_is_jax_available_function(self):
        """Test the is_jax_available package function."""
        from tep import is_jax_available
        assert is_jax_available() == True

    def test_jax_backend_exported(self):
        """Test JaxTEProcess is exported from tep."""
        from tep import JaxTEProcess, JaxTEProcessWrapper
        assert JaxTEProcess is not None
        assert JaxTEProcessWrapper is not None

    def test_simulator_with_jax_backend(self):
        """Test creating TEPSimulator with JAX backend."""
        from tep import TEPSimulator
        sim = TEPSimulator(backend="jax")
        assert sim.backend == "jax"

    def test_simulator_jax_initializes(self):
        """Test that simulator with JAX backend initializes."""
        from tep import TEPSimulator
        sim = TEPSimulator(backend="jax")
        sim.initialize()
        assert sim.initialized


class TestRandomNumberGeneration:
    """Test JAX-based random number generation."""

    def test_tesub6_produces_noise(self):
        """Test noise generation function."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)

        noise, new_key = process._tesub6(key, 1.0)

        # Noise should be a scalar
        assert noise.shape == ()
        # Key should be updated
        assert not np.array_equal(key, new_key)

    def test_tesub6_different_keys_different_noise(self):
        """Test that different keys produce different noise."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key1 = jax.random.PRNGKey(1234)
        key2 = jax.random.PRNGKey(5678)

        noise1, _ = process._tesub6(key1, 1.0)
        noise2, _ = process._tesub6(key2, 1.0)

        assert float(noise1) != float(noise2)

    def test_tesub6_same_key_same_noise(self):
        """Test that same key produces same noise."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)

        noise1, _ = process._tesub6(key, 1.0)
        noise2, _ = process._tesub6(key, 1.0)

        assert float(noise1) == float(noise2)

    def test_tesub6_std_scaling(self):
        """Test that std parameter scales the noise."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)

        # Generate many samples to check variance
        noises_std1 = []
        noises_std2 = []

        for i in range(100):
            subkey = jax.random.fold_in(key, i)
            n1, _ = process._tesub6(subkey, 1.0)
            n2, _ = process._tesub6(subkey, 2.0)
            noises_std1.append(float(n1))
            noises_std2.append(float(n2))

        # Variance should scale with std^2
        var1 = np.var(noises_std1)
        var2 = np.var(noises_std2)

        # var2 should be approximately 4x var1
        ratio = var2 / var1
        assert 2.0 < ratio < 8.0  # Allow some statistical variation


class TestDisturbanceWalk:
    """Test disturbance walk evaluation."""

    def test_tesub8_returns_value(self):
        """Test cubic spline evaluation."""
        from tep.jax_backend import JaxTEProcess, create_initial_walk_state

        process = JaxTEProcess()
        walks = create_initial_walk_state()

        # Evaluate at time 0
        value = process._tesub8(walks, 1, 0.0)
        assert value.shape == ()

        # Should return szero at time 0 (since tlast=0, h=0)
        assert abs(float(value) - walks.szero[0]) < 1e-6


class TestJITCompatibility:
    """Test that operations can be JIT compiled."""

    def test_step_can_be_jitted(self):
        """Test that step function can be JIT compiled."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()

        # JIT compile the step function
        jit_step = jax.jit(process.step)

        key = jax.random.PRNGKey(1234)
        state, key = process.initialize(key)

        # Should be able to call JIT'd function
        new_state, new_key = jit_step(state, key)

        assert new_state.time > state.time

    def test_initialize_can_be_jitted(self):
        """Test that initialize can be JIT compiled."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        jit_init = jax.jit(process.initialize)

        key = jax.random.PRNGKey(1234)
        state, new_key = jit_init(key)

        assert state.yy.shape == (50,)

    def test_multiple_jit_steps(self):
        """Test running multiple JIT'd steps."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        jit_step = jax.jit(process.step)

        key = jax.random.PRNGKey(1234)
        state, key = process.initialize(key)

        # Run multiple steps
        for _ in range(10):
            state, key = jit_step(state, key)

        # Time should have advanced
        assert state.time > 0


class TestThermodynamicFunctions:
    """Test JAX implementations of thermodynamic functions."""

    def test_tesub1_liquid_enthalpy(self):
        """Test liquid enthalpy calculation."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()

        # Test with pure component D at 100 deg C
        z = jnp.zeros(8)
        z = z.at[3].set(1.0)  # Pure component D

        h = process._tesub1(z, 100.0, 0)  # ity=0 for liquid
        assert h > 0  # Enthalpy should be positive at 100 C

    def test_tesub1_gas_enthalpy(self):
        """Test gas enthalpy calculation."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()

        # Test with pure component D at 100 deg C
        z = jnp.zeros(8)
        z = z.at[3].set(1.0)  # Pure component D

        h_liquid = process._tesub1(z, 100.0, 0)  # ity=0 for liquid
        h_gas = process._tesub1(z, 100.0, 1)     # ity=1 for gas

        # Both should be computed and finite
        assert jnp.isfinite(h_liquid)
        assert jnp.isfinite(h_gas)
        # They should be different (liquid vs gas have different coefficients)
        assert h_liquid != h_gas

    def test_tesub1_pressure_correction(self):
        """Test pressure correction for compressor (ity=2)."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()

        z = jnp.zeros(8)
        z = z.at[3].set(1.0)

        h_gas = process._tesub1(z, 100.0, 1)      # Gas without correction
        h_comp = process._tesub1(z, 100.0, 2)     # Gas with pressure correction

        # Pressure correction should reduce enthalpy
        assert h_comp < h_gas

    def test_tesub1_mixture(self):
        """Test enthalpy calculation for mixture."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()

        # Typical liquid composition
        z = jnp.array([0.0, 0.0, 0.0, 0.3, 0.25, 0.25, 0.1, 0.1])

        h = process._tesub1(z, 80.0, 0)
        assert jnp.isfinite(h)
        assert h > 0

    def test_tesub3_derivative(self):
        """Test enthalpy derivative dH/dT."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()

        z = jnp.zeros(8)
        z = z.at[3].set(1.0)

        dh = process._tesub3(z, 100.0, 0)
        assert dh > 0  # Heat capacity should be positive

    def test_tesub3_numerical_derivative(self):
        """Test that _tesub3 matches numerical derivative of _tesub1."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()

        z = jnp.array([0.0, 0.0, 0.0, 0.3, 0.25, 0.25, 0.1, 0.1])
        t = 80.0
        dt = 0.001

        # Analytical derivative
        dh_analytical = process._tesub3(z, t, 0)

        # Numerical derivative
        h1 = process._tesub1(z, t - dt, 0)
        h2 = process._tesub1(z, t + dt, 0)
        dh_numerical = (h2 - h1) / (2 * dt)

        # Should match within tolerance
        rel_error = abs(float(dh_analytical - dh_numerical)) / abs(float(dh_analytical))
        assert rel_error < 0.01

    def test_tesub4_density(self):
        """Test liquid density calculation."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()

        # Typical liquid composition
        x = jnp.array([0.0, 0.0, 0.0, 0.3, 0.25, 0.25, 0.1, 0.1])

        rho = process._tesub4(x, 80.0)

        # Density should be positive and reasonable for organic liquids
        # Typical range: 0.1 - 5 kmol/m^3
        assert 0.1 < float(rho) < 5.0

    def test_tesub4_temperature_effect(self):
        """Test that density decreases with temperature."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()

        x = jnp.array([0.0, 0.0, 0.0, 0.3, 0.25, 0.25, 0.1, 0.1])

        rho_low = process._tesub4(x, 20.0)
        rho_high = process._tesub4(x, 120.0)

        # Density should decrease with temperature
        assert rho_low > rho_high

    def test_tesub2_temperature_iteration(self):
        """Test temperature from enthalpy iteration."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()

        # Test round-trip: T -> H -> T
        z = jnp.array([0.0, 0.0, 0.0, 0.3, 0.25, 0.25, 0.1, 0.1])
        t_original = 80.0

        # Calculate enthalpy at known temperature
        h = process._tesub1(z, t_original, 0)

        # Recover temperature from enthalpy
        t_recovered = process._tesub2(z, 50.0, h, 0)  # Start with different initial guess

        # Should recover original temperature
        assert abs(float(t_recovered) - t_original) < 0.1

    def test_tesub2_different_phases(self):
        """Test temperature iteration for different phases."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()

        z = jnp.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.05, 0.05])

        # Test liquid (ity=0)
        t_original = 70.0
        h_liquid = process._tesub1(z, t_original, 0)
        t_recovered = process._tesub2(z, 50.0, h_liquid, 0)
        assert abs(float(t_recovered) - t_original) < 0.1

        # Test gas (ity=1)
        h_gas = process._tesub1(z, t_original, 1)
        t_recovered = process._tesub2(z, 50.0, h_gas, 1)
        assert abs(float(t_recovered) - t_original) < 0.1

    def test_tesub1_jit_compatible(self):
        """Test that _tesub1 can be JIT compiled."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()

        @jax.jit
        def compute_enthalpy(z, t):
            return process._tesub1(z, t, 0)

        z = jnp.array([0.0, 0.0, 0.0, 0.3, 0.25, 0.25, 0.1, 0.1])
        h = compute_enthalpy(z, 80.0)
        assert jnp.isfinite(h)

    def test_tesub2_jit_compatible(self):
        """Test that _tesub2 can be JIT compiled."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()

        @jax.jit
        def compute_temperature(z, h):
            return process._tesub2(z, 80.0, h, 0)

        z = jnp.array([0.0, 0.0, 0.0, 0.3, 0.25, 0.25, 0.1, 0.1])
        h = process._tesub1(z, 80.0, 0)
        t = compute_temperature(z, h)
        assert jnp.isfinite(t)
        assert abs(float(t) - 80.0) < 0.1


class TestThermodynamicComparisonWithPython:
    """Compare JAX thermodynamic functions with Python backend."""

    def test_tesub1_matches_python(self):
        """Test that JAX _tesub1 matches Python backend."""
        from tep.jax_backend import JaxTEProcess
        from tep.python_backend import PythonTEProcess

        jax_proc = JaxTEProcess()
        py_proc = PythonTEProcess()

        # Test various compositions and temperatures
        test_cases = [
            (np.array([0, 0, 0, 1, 0, 0, 0, 0]), 100.0, 0),
            (np.array([0, 0, 0, 0.3, 0.25, 0.25, 0.1, 0.1]), 80.0, 0),
            (np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.05, 0.05]), 120.0, 1),
            (np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.05, 0.05]), 120.0, 2),
        ]

        for z_np, t, ity in test_cases:
            z_jax = jnp.array(z_np)

            h_jax = float(jax_proc._tesub1(z_jax, t, ity))
            h_py = py_proc._tesub1(z_np, t, ity)

            rel_error = abs(h_jax - h_py) / abs(h_py) if abs(h_py) > 1e-10 else abs(h_jax - h_py)
            assert rel_error < 0.01, f"Mismatch at z={z_np}, t={t}, ity={ity}: JAX={h_jax}, Python={h_py}"

    def test_tesub3_matches_python(self):
        """Test that JAX _tesub3 matches Python backend."""
        from tep.jax_backend import JaxTEProcess
        from tep.python_backend import PythonTEProcess

        jax_proc = JaxTEProcess()
        py_proc = PythonTEProcess()

        z_np = np.array([0, 0, 0, 0.3, 0.25, 0.25, 0.1, 0.1])
        z_jax = jnp.array(z_np)
        t = 80.0

        for ity in [0, 1, 2]:
            dh_jax = float(jax_proc._tesub3(z_jax, t, ity))
            dh_py = py_proc._tesub3(z_np, t, ity)

            rel_error = abs(dh_jax - dh_py) / abs(dh_py) if abs(dh_py) > 1e-10 else abs(dh_jax - dh_py)
            assert rel_error < 0.01, f"Mismatch at ity={ity}: JAX={dh_jax}, Python={dh_py}"

    def test_tesub4_matches_python(self):
        """Test that JAX _tesub4 matches Python backend."""
        from tep.jax_backend import JaxTEProcess
        from tep.python_backend import PythonTEProcess

        jax_proc = JaxTEProcess()
        py_proc = PythonTEProcess()

        x_np = np.array([0, 0, 0, 0.3, 0.25, 0.25, 0.1, 0.1])
        x_jax = jnp.array(x_np)

        for t in [20.0, 80.0, 120.0]:
            rho_jax = float(jax_proc._tesub4(x_jax, t))
            rho_py = py_proc._tesub4(x_np, t)

            rel_error = abs(rho_jax - rho_py) / abs(rho_py)
            assert rel_error < 0.01, f"Mismatch at t={t}: JAX={rho_jax}, Python={rho_py}"

    def test_tesub2_matches_python(self):
        """Test that JAX _tesub2 matches Python backend."""
        from tep.jax_backend import JaxTEProcess
        from tep.python_backend import PythonTEProcess

        jax_proc = JaxTEProcess()
        py_proc = PythonTEProcess()

        z_np = np.array([0, 0, 0, 0.3, 0.25, 0.25, 0.1, 0.1])
        z_jax = jnp.array(z_np)

        # Compute enthalpy at known temperature
        t_original = 80.0
        h = py_proc._tesub1(z_np, t_original, 0)

        # Recover temperature using both backends
        t_jax = float(jax_proc._tesub2(z_jax, 50.0, h, 0))
        t_py = py_proc._tesub2(z_np, 50.0, h, 0)

        assert abs(t_jax - t_py) < 0.1, f"Mismatch: JAX={t_jax}, Python={t_py}"


class TestAPICompatibility:
    """Test API compatibility with PythonTEProcess."""

    def test_wrapper_has_same_methods(self):
        """Test that wrapper has same methods as PythonTEProcess."""
        from tep.jax_backend import JaxTEProcessWrapper
        from tep.python_backend import PythonTEProcess

        wrapper = JaxTEProcessWrapper()
        python_proc = PythonTEProcess()

        # Check key methods exist
        assert hasattr(wrapper, 'initialize')
        assert hasattr(wrapper, 'step')
        assert hasattr(wrapper, 'get_xmeas')
        assert hasattr(wrapper, 'get_xmv')
        assert hasattr(wrapper, 'set_xmv')
        assert hasattr(wrapper, 'set_idv')
        assert hasattr(wrapper, 'clear_disturbances')
        assert hasattr(wrapper, 'evaluate')
        assert hasattr(wrapper, 'is_shutdown')
        assert hasattr(wrapper, 'get_state')
        assert hasattr(wrapper, 'set_state')

    def test_wrapper_has_same_properties(self):
        """Test that wrapper has same properties as PythonTEProcess."""
        from tep.jax_backend import JaxTEProcessWrapper

        wrapper = JaxTEProcessWrapper()
        wrapper.initialize()

        # Check properties
        assert hasattr(wrapper, 'yy')
        assert hasattr(wrapper, 'yp')
        assert hasattr(wrapper, 'xmeas')
        assert hasattr(wrapper, 'xmv')
        assert hasattr(wrapper, 'idv')
        assert hasattr(wrapper, 'time')


class TestTefuncImplementation:
    """Test the _tefunc derivative calculation function."""

    def test_tefunc_produces_derivatives(self):
        """Test that _tefunc produces non-zero derivatives."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, key = process.initialize(key)

        # Check that derivatives are computed
        yp = state.yp
        assert yp.shape == (50,)
        assert jnp.any(yp != 0.0), "Derivatives should not all be zero"

    def test_tefunc_derivative_magnitudes(self):
        """Test that derivatives have reasonable magnitudes."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, key = process.initialize(key)

        yp = state.yp

        # Derivatives should be finite
        assert jnp.all(jnp.isfinite(yp)), "Derivatives should be finite"

        # At steady state, derivatives should be relatively small
        # (allowing for initialization differences)
        max_derivative = jnp.max(jnp.abs(yp))
        assert max_derivative < 1e6, f"Derivatives too large: {max_derivative}"

    def test_tefunc_measurements_computed(self):
        """Test that measurements are computed by _tefunc."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, key = process.initialize(key)

        xmeas = state.measurements.xmeas
        assert xmeas.shape == (41,)

        # Key measurements should be non-zero
        assert xmeas[6] != 0.0, "Reactor pressure should be non-zero"
        assert xmeas[8] != 0.0, "Reactor temperature should be non-zero"

    def test_tefunc_updates_reactor_state(self):
        """Test that _tefunc updates reactor state variables."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, key = process.initialize(key)

        reactor = state.reactor

        # Check key reactor variables
        assert reactor.tcr > 0, "Reactor temperature should be positive"
        assert reactor.ptr > 0, "Reactor pressure should be positive"
        assert reactor.vlr > 0, "Reactor liquid volume should be positive"

    def test_tefunc_simulation_step(self):
        """Test that simulation step updates state correctly."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, key = process.initialize(key)

        # Take one step
        initial_time = state.time
        initial_yy = state.yy.copy()

        state, key = process.step(state, key)

        # Time should advance
        assert state.time > initial_time

        # State should change (due to non-zero derivatives)
        assert not jnp.allclose(state.yy, initial_yy)

    def test_tefunc_multiple_steps(self):
        """Test simulation runs for multiple steps without issues."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, key = process.initialize(key)

        # Run for 100 steps (100 seconds)
        for _ in range(100):
            state, key = process.step(state, key)

        # Check state is still valid
        assert jnp.all(jnp.isfinite(state.yy)), "State should remain finite"
        assert jnp.all(jnp.isfinite(state.yp)), "Derivatives should remain finite"
        assert state.time > 0

    def test_tefunc_jit_compiled_simulation(self):
        """Test that JIT-compiled simulation runs correctly."""
        from tep.jax_backend import JaxTEProcess

        process = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        state, key = process.initialize(key)

        # JIT compile the step function
        step_jit = jax.jit(process.step)

        # Run for 50 steps with JIT
        for _ in range(50):
            state, key = step_jit(state, key)

        # Check state is still valid
        assert jnp.all(jnp.isfinite(state.yy)), "State should remain finite"
        assert state.time > 0


class TestTefuncComparisonWithPython:
    """Compare JAX _tefunc results with Python backend."""

    def test_initial_derivatives_comparable(self):
        """Test that initial derivatives are finite and simulation converges."""
        from tep.jax_backend import JaxTEProcess
        from tep.python_backend import PythonTEProcess

        # Initialize both backends
        jax_proc = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        jax_state, key = jax_proc.initialize(key)

        py_proc = PythonTEProcess()
        py_proc.initialize()

        # Get derivatives
        jax_yp = np.array(jax_state.yp)
        py_yp = py_proc.yp

        # Both should produce finite derivatives
        assert np.all(np.isfinite(jax_yp)), "JAX derivatives should be finite"
        assert np.all(np.isfinite(py_yp)), "Python derivatives should be finite"

        # Run both for 100 steps and compare - they should converge to similar behavior
        for _ in range(100):
            jax_state, key = jax_proc.step(jax_state, key)
            py_proc.step()

        # After simulation, states should be in same order of magnitude
        jax_yy = np.array(jax_state.yy)
        py_yy = py_proc.yy

        # Check key states (first 8 reactor states)
        for i in range(8):
            if abs(py_yy[i]) > 0.1:
                rel_error = abs(jax_yy[i] - py_yy[i]) / abs(py_yy[i])
                assert rel_error < 0.5, f"State {i} diverged after 100 steps: JAX={jax_yy[i]}, Python={py_yy[i]}"

    def test_measurements_comparable(self):
        """Test that measurements are comparable between backends."""
        from tep.jax_backend import JaxTEProcess
        from tep.python_backend import PythonTEProcess

        # Initialize both backends
        jax_proc = JaxTEProcess()
        key = jax.random.PRNGKey(1234)
        jax_state, key = jax_proc.initialize(key)

        py_proc = PythonTEProcess()
        py_proc.initialize()

        # Get measurements
        jax_xmeas = np.array(jax_state.measurements.xmeas)
        py_xmeas = py_proc.xmeas

        # Key measurements should be similar (first 22 are real-time, rest are sampled)
        # Check a few key ones
        key_indices = [6, 7, 8, 10, 11, 12, 17]  # Pressures, levels, temperatures
        for i in key_indices:
            if abs(py_xmeas[i]) > 0.01:
                rel_error = abs(jax_xmeas[i] - py_xmeas[i]) / abs(py_xmeas[i])
                assert rel_error < 0.5, f"Measurement {i}: JAX={jax_xmeas[i]}, Python={py_xmeas[i]}"

    def test_wrapper_produces_similar_simulation(self):
        """Test that wrapper produces similar simulation to Python backend."""
        from tep.jax_backend import JaxTEProcessWrapper
        from tep.python_backend import PythonTEProcess

        # Initialize both backends with same seed
        jax_wrapper = JaxTEProcessWrapper(random_seed=12345)
        jax_wrapper.initialize()

        py_proc = PythonTEProcess(random_seed=12345)
        py_proc.initialize()

        # Run for 10 steps
        for _ in range(10):
            jax_wrapper.step()
            py_proc.step()

        # Compare states - should be similar but not identical due to random differences
        jax_yy = jax_wrapper.yy
        py_yy = py_proc.yy

        # All state values should be finite
        assert np.all(np.isfinite(jax_yy))
        assert np.all(np.isfinite(py_yy))

        # States should be within same order of magnitude
        for i in range(50):
            if abs(py_yy[i]) > 0.1:
                rel_error = abs(jax_yy[i] - py_yy[i]) / abs(py_yy[i])
                assert rel_error < 1.0, f"State {i} diverged: JAX={jax_yy[i]}, Python={py_yy[i]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
