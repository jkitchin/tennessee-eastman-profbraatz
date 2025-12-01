"""
Base classes and plugin registry for TEP fault injection.

This module provides the foundation for a plugin-based fault system,
allowing users to easily create and register custom process faults
beyond the standard 20 IDV disturbances.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any, Tuple
import numpy as np


@dataclass
class FaultEffect:
    """
    Describes the effect of a fault on process variables.

    A fault can modify process variables through:
    - Additive perturbations: value = base_value + delta
    - Multiplicative perturbations: value = base_value * factor
    - Direct replacement: value = new_value

    Attributes:
        variable: Name of the variable to modify (e.g., 'feed_temp_d', 'flow_a')
        mode: 'additive', 'multiplicative', or 'replace'
        value: The perturbation value (delta, factor, or replacement)
    """
    variable: str
    mode: str  # 'additive', 'multiplicative', 'replace'
    value: float


class BaseFaultPlugin(ABC):
    """
    Abstract base class for all TEP fault plugins.

    All fault plugins must inherit from this class and implement
    the required methods. Faults receive the current process state
    and return perturbations to apply.

    Example:
        >>> class MyFault(BaseFaultPlugin):
        ...     name = "my_fault"
        ...     description = "My custom fault"
        ...
        ...     def apply(self, time, process_state):
        ...         # Return perturbations based on time and state
        ...         if time > 1.0:  # Activate after 1 hour
        ...             return [FaultEffect('feed_temp_d', 'additive', 10.0)]
        ...         return []
        ...
        ...     def reset(self):
        ...         pass

    Available process variables that can be modified:
        Feed compositions:
            - 'feed_comp_a': A component in feed stream 4 (base ~0.485)
            - 'feed_comp_b': B component in feed stream 4 (base ~0.005)
            - 'feed_comp_c': C component in feed stream 4 (computed)

        Temperatures:
            - 'feed_temp_d': D feed temperature (°C, base ~45)
            - 'feed_temp_c': C feed temperature (°C)
            - 'reactor_cw_inlet_temp': Reactor cooling water inlet (°C, base ~35)
            - 'condenser_cw_inlet_temp': Condenser cooling water inlet (°C, base ~18)

        Flow multipliers (multiplicative effects):
            - 'flow_a': A feed flow multiplier (1.0 = normal)
            - 'flow_c': C feed flow multiplier (1.0 = normal)

        Reaction kinetics:
            - 'reaction_1_factor': Reaction 1 rate factor
            - 'reaction_2_factor': Reaction 2 rate factor

        Valve positions (for sticking faults):
            - 'valve_reactor_cw': Reactor cooling water valve
            - 'valve_condenser_cw': Condenser cooling water valve
    """

    # Class attributes - override in subclasses
    name: str = "base"
    description: str = "Base fault class"
    version: str = "1.0.0"

    # Fault category for grouping in UI
    category: str = "general"

    # Whether this fault is active
    _active: bool = False

    # Activation time (hours)
    _activation_time: Optional[float] = None

    def __init__(self, magnitude: float = 1.0, **kwargs):
        """
        Initialize fault plugin.

        Args:
            magnitude: Scaling factor for fault severity (0.0-1.0+ typical)
            **kwargs: Additional plugin-specific parameters
        """
        self.magnitude = magnitude
        self._active = False
        self._activation_time = None

    @abstractmethod
    def apply(
        self,
        time: float,
        process_state: Dict[str, Any]
    ) -> List[FaultEffect]:
        """
        Calculate fault effects at the current time.

        This method is called at each simulation step when the fault
        is active. Return an empty list if no perturbations should
        be applied at this time.

        Args:
            time: Current simulation time in hours
            process_state: Dict containing current process variables:
                - 'xmeas': Current measurements (41 elements)
                - 'xmv': Current manipulated variables (12 elements)
                - 'step': Current simulation step
                - 'random': Random number generator for stochastic faults

        Returns:
            List of FaultEffect objects describing perturbations to apply
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset fault state to initial conditions.

        Called when simulation is reinitialized. Should reset any
        internal state (timers, accumulated values, etc.).
        """
        pass

    def activate(self, time: float = 0.0):
        """
        Activate the fault.

        Args:
            time: Simulation time when fault becomes active
        """
        self._active = True
        self._activation_time = time

    def deactivate(self):
        """Deactivate the fault."""
        self._active = False
        self._activation_time = None

    @property
    def is_active(self) -> bool:
        """Check if fault is currently active."""
        return self._active

    @property
    def activation_time(self) -> Optional[float]:
        """Get the time when fault was activated."""
        return self._activation_time

    def get_info(self) -> Dict[str, Any]:
        """
        Get fault information.

        Returns:
            Dict with name, description, version, category, and parameters
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "magnitude": self.magnitude,
            "is_active": self._active,
            "activation_time": self._activation_time,
        }

    def set_parameter(self, name: str, value: Any):
        """
        Set a fault parameter.

        Args:
            name: Parameter name
            value: Parameter value
        """
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            raise AttributeError(f"Unknown parameter: {name}")

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get fault parameters.

        Returns:
            Dict of parameter names and values
        """
        return {"magnitude": self.magnitude}


@dataclass
class FaultPluginConfig:
    """Configuration for a fault plugin instance."""
    fault_class: Type[BaseFaultPlugin]
    name: str
    description: str
    category: str = "general"
    default_params: Dict[str, Any] = field(default_factory=dict)


class FaultPluginRegistry:
    """
    Registry for fault plugins.

    Provides discovery, registration, and instantiation of fault plugins.

    Example:
        >>> # Register a fault
        >>> FaultPluginRegistry.register(MyFault)
        >>>
        >>> # List available faults
        >>> print(FaultPluginRegistry.list_available())
        ['feed_ratio', 'reactor_temp', 'my_fault']
        >>>
        >>> # Get a fault instance
        >>> fault = FaultPluginRegistry.create('my_fault', magnitude=0.5)
    """

    _faults: Dict[str, FaultPluginConfig] = {}

    @classmethod
    def register(
        cls,
        fault_class: Type[BaseFaultPlugin],
        name: str = None,
        description: str = None,
        category: str = None,
        default_params: Dict[str, Any] = None
    ):
        """
        Register a fault plugin.

        Args:
            fault_class: The fault class to register
            name: Override name (defaults to class.name)
            description: Override description (defaults to class.description)
            category: Override category (defaults to class.category)
            default_params: Default parameters for instantiation
        """
        fault_name = name or fault_class.name
        fault_desc = description or fault_class.description
        fault_cat = category or getattr(fault_class, 'category', 'general')

        config = FaultPluginConfig(
            fault_class=fault_class,
            name=fault_name,
            description=fault_desc,
            category=fault_cat,
            default_params=default_params or {}
        )
        cls._faults[fault_name] = config

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseFaultPlugin:
        """
        Create a fault plugin instance.

        Args:
            name: Name of the registered fault
            **kwargs: Override default parameters

        Returns:
            Configured fault instance

        Raises:
            KeyError: If fault name is not registered
        """
        if name not in cls._faults:
            available = ", ".join(cls._faults.keys())
            raise KeyError(
                f"Unknown fault: '{name}'. Available: {available}"
            )

        config = cls._faults[name]
        params = {**config.default_params, **kwargs}
        return config.fault_class(**params)

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all registered fault names.

        Returns:
            List of fault names
        """
        return list(cls._faults.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """
        List faults in a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of fault names in the category
        """
        return [
            name for name, config in cls._faults.items()
            if config.category == category
        ]

    @classmethod
    def get_categories(cls) -> List[str]:
        """
        Get all unique fault categories.

        Returns:
            List of category names
        """
        return list(set(config.category for config in cls._faults.values()))

    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about a registered fault.

        Args:
            name: Name of the fault

        Returns:
            Dict with fault configuration info
        """
        if name not in cls._faults:
            raise KeyError(f"Unknown fault: '{name}'")

        config = cls._faults[name]
        return {
            "name": config.name,
            "description": config.description,
            "category": config.category,
            "default_params": config.default_params,
            "class": config.fault_class.__name__
        }

    @classmethod
    def list_all_info(cls) -> List[Dict[str, Any]]:
        """
        List information for all registered faults.

        Returns:
            List of info dicts for all faults
        """
        return [cls.get_info(name) for name in cls._faults]

    @classmethod
    def clear(cls):
        """Clear all registered faults (mainly for testing)."""
        cls._faults.clear()


def register_fault(
    name: str = None,
    description: str = None,
    category: str = None,
    default_params: Dict[str, Any] = None
):
    """
    Decorator to register a fault plugin class.

    Example:
        >>> @register_fault(name="my_fault", description="My custom fault")
        ... class MyFault(BaseFaultPlugin):
        ...     def apply(self, time, process_state):
        ...         return [FaultEffect('feed_temp_d', 'additive', 5.0)]
        ...
        ...     def reset(self):
        ...         pass
    """
    def decorator(cls: Type[BaseFaultPlugin]) -> Type[BaseFaultPlugin]:
        FaultPluginRegistry.register(
            cls,
            name=name,
            description=description,
            category=category,
            default_params=default_params
        )
        return cls
    return decorator


class FaultManager:
    """
    Manages multiple fault plugins during simulation.

    Coordinates activation, deactivation, and application of faults
    to the process simulation.

    Example:
        >>> manager = FaultManager()
        >>> manager.add_fault('feed_ratio', magnitude=0.5, activate_at=1.0)
        >>> manager.add_fault('reactor_temp', activate_at=2.0)
        >>>
        >>> # During simulation loop:
        >>> effects = manager.apply_all(time, process_state)
    """

    def __init__(self):
        """Initialize fault manager."""
        self._faults: List[Tuple[BaseFaultPlugin, Optional[float]]] = []
        self._rng = np.random.default_rng()

    def add_fault(
        self,
        fault: BaseFaultPlugin | str,
        activate_at: Optional[float] = None,
        **kwargs
    ):
        """
        Add a fault to be managed.

        Args:
            fault: Fault instance or registered fault name
            activate_at: Time (hours) to activate fault (None = immediate)
            **kwargs: Parameters if fault is a name string
        """
        if isinstance(fault, str):
            fault = FaultPluginRegistry.create(fault, **kwargs)

        self._faults.append((fault, activate_at))

        # Activate immediately if no activation time
        if activate_at is None:
            fault.activate(0.0)

    def remove_fault(self, name: str):
        """
        Remove a fault by name.

        Args:
            name: Name of the fault to remove
        """
        self._faults = [
            (f, t) for f, t in self._faults if f.name != name
        ]

    def apply_all(
        self,
        time: float,
        process_state: Dict[str, Any]
    ) -> List[FaultEffect]:
        """
        Apply all active faults and collect their effects.

        This method also handles scheduled activations.

        Args:
            time: Current simulation time in hours
            process_state: Current process state dict

        Returns:
            Combined list of all fault effects
        """
        # Add random generator to process state
        process_state['random'] = self._rng

        effects = []
        for fault, activate_at in self._faults:
            # Handle scheduled activation
            if activate_at is not None and not fault.is_active:
                if time >= activate_at:
                    fault.activate(time)

            # Apply if active
            if fault.is_active:
                fault_effects = fault.apply(time, process_state)
                effects.extend(fault_effects)

        return effects

    def reset(self):
        """Reset all faults to initial state."""
        for fault, _ in self._faults:
            fault.reset()
            fault.deactivate()

    def get_active_faults(self) -> List[str]:
        """Get names of currently active faults."""
        return [f.name for f, _ in self._faults if f.is_active]

    def get_all_faults(self) -> List[Dict[str, Any]]:
        """Get info for all managed faults."""
        return [f.get_info() for f, _ in self._faults]

    def set_random_seed(self, seed: int):
        """Set random seed for stochastic faults."""
        self._rng = np.random.default_rng(seed)
