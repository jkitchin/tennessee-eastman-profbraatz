"""
Base classes and plugin registry for TEP controllers.

This module provides the foundation for a plugin-based controller system,
allowing users to easily create and register custom controllers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any, Callable
import numpy as np

from .constants import NUM_MANIPULATED_VARS, NUM_MEASUREMENTS


class BaseController(ABC):
    """
    Abstract base class for all TEP controllers.

    All controller plugins must inherit from this class and implement
    the required methods. Controllers receive measurements and current
    MV values, and return new MV values.

    Example:
        >>> class MyController(BaseController):
        ...     name = "my_controller"
        ...     description = "My custom controller"
        ...
        ...     def calculate(self, xmeas, xmv, step):
        ...         # Control logic here
        ...         return xmv.copy()
        ...
        ...     def reset(self):
        ...         pass
    """

    # Class attributes - override in subclasses
    name: str = "base"
    description: str = "Base controller class"
    version: str = "1.0.0"

    # Which MVs this controller manages (None = all)
    controlled_mvs: Optional[List[int]] = None

    @abstractmethod
    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        """
        Calculate new manipulated variable values.

        Args:
            xmeas: Current measurements (41 elements, 0-indexed)
                   See constants.MEASUREMENT_NAMES for descriptions
            xmv: Current manipulated variables (12 elements, 0-indexed)
                 See constants.MANIPULATED_VAR_NAMES for descriptions
            step: Current simulation step number (1 step = 1 second)

        Returns:
            Updated manipulated variables (12 elements)

        Note:
            If this controller only manages a subset of MVs (controlled_mvs),
            it should copy xmv and only modify the MVs it controls.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset controller state to initial conditions.

        Called when simulation is reinitialized. Should reset any
        internal state (error accumulators, setpoints, etc.).
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Get controller information.

        Returns:
            Dict with name, description, version, and controlled_mvs
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "controlled_mvs": self.controlled_mvs,
        }

    def set_parameter(self, name: str, value: Any):
        """
        Set a controller parameter.

        Override this method to support runtime parameter changes.

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
        Get controller parameters.

        Override this method to expose tunable parameters.

        Returns:
            Dict of parameter names and values
        """
        return {}


@dataclass
class ControllerConfig:
    """Configuration for a controller instance."""
    controller_class: Type[BaseController]
    name: str
    description: str
    default_params: Dict[str, Any] = field(default_factory=dict)


class ControllerRegistry:
    """
    Registry for controller plugins.

    Provides discovery, registration, and instantiation of controllers.

    Example:
        >>> # Register a controller
        >>> ControllerRegistry.register(MyController)
        >>>
        >>> # List available controllers
        >>> print(ControllerRegistry.list_available())
        ['decentralized', 'manual', 'my_controller']
        >>>
        >>> # Get a controller instance
        >>> controller = ControllerRegistry.create('my_controller', param1=value1)
    """

    _controllers: Dict[str, ControllerConfig] = {}

    @classmethod
    def register(
        cls,
        controller_class: Type[BaseController],
        name: str = None,
        description: str = None,
        default_params: Dict[str, Any] = None
    ):
        """
        Register a controller class.

        Args:
            controller_class: Controller class (must inherit from BaseController)
            name: Optional name override (defaults to class.name)
            description: Optional description override
            default_params: Default parameters for instantiation
        """
        if not issubclass(controller_class, BaseController):
            raise TypeError(
                f"Controller must inherit from BaseController, "
                f"got {controller_class.__bases__}"
            )

        reg_name = name or controller_class.name
        reg_desc = description or controller_class.description

        cls._controllers[reg_name] = ControllerConfig(
            controller_class=controller_class,
            name=reg_name,
            description=reg_desc,
            default_params=default_params or {}
        )

    @classmethod
    def unregister(cls, name: str):
        """
        Unregister a controller.

        Args:
            name: Controller name to remove
        """
        if name in cls._controllers:
            del cls._controllers[name]

    @classmethod
    def get(cls, name: str) -> Type[BaseController]:
        """
        Get a controller class by name.

        Args:
            name: Controller name

        Returns:
            Controller class

        Raises:
            KeyError: If controller not found
        """
        if name not in cls._controllers:
            available = ", ".join(cls._controllers.keys())
            raise KeyError(
                f"Controller '{name}' not found. "
                f"Available: {available}"
            )
        return cls._controllers[name].controller_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseController:
        """
        Create a controller instance.

        Args:
            name: Controller name
            **kwargs: Parameters passed to controller constructor

        Returns:
            Controller instance
        """
        config = cls._controllers.get(name)
        if config is None:
            available = ", ".join(cls._controllers.keys())
            raise KeyError(
                f"Controller '{name}' not found. "
                f"Available: {available}"
            )

        # Merge default params with provided kwargs
        params = {**config.default_params, **kwargs}
        return config.controller_class(**params)

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all registered controller names.

        Returns:
            List of controller names
        """
        return list(cls._controllers.keys())

    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about a registered controller.

        Args:
            name: Controller name

        Returns:
            Dict with controller info
        """
        if name not in cls._controllers:
            raise KeyError(f"Controller '{name}' not found")

        config = cls._controllers[name]
        return {
            "name": config.name,
            "description": config.description,
            "class": config.controller_class.__name__,
            "default_params": config.default_params,
        }

    @classmethod
    def list_all_info(cls) -> List[Dict[str, Any]]:
        """
        Get information about all registered controllers.

        Returns:
            List of controller info dicts
        """
        return [cls.get_info(name) for name in cls._controllers]

    @classmethod
    def clear(cls):
        """Clear all registered controllers (mainly for testing)."""
        cls._controllers.clear()


def register_controller(
    name: str = None,
    description: str = None,
    default_params: Dict[str, Any] = None
):
    """
    Decorator to register a controller class.

    Example:
        >>> @register_controller(name="my_ctrl", description="My controller")
        ... class MyController(BaseController):
        ...     pass
    """
    def decorator(cls: Type[BaseController]) -> Type[BaseController]:
        ControllerRegistry.register(
            cls,
            name=name,
            description=description,
            default_params=default_params
        )
        return cls
    return decorator


class CompositeController(BaseController):
    """
    Controller that combines multiple sub-controllers.

    Each sub-controller manages a subset of MVs. The composite
    controller dispatches to the appropriate sub-controller based
    on which MVs each controls.

    Example:
        >>> composite = CompositeController()
        >>> composite.add_controller(ReactorTempController(), mvs=[10])
        >>> composite.add_controller(LevelController(), mvs=[7, 8])
        >>> # MVs not covered by sub-controllers use the fallback
    """

    name = "composite"
    description = "Composite controller combining multiple sub-controllers"

    def __init__(self, fallback_controller: BaseController = None):
        """
        Initialize composite controller.

        Args:
            fallback_controller: Controller for MVs not covered by sub-controllers
                                If None, those MVs are held at their current values
        """
        self._sub_controllers: List[tuple] = []  # (controller, mv_indices)
        self._fallback = fallback_controller
        self._mv_assignment: Dict[int, BaseController] = {}

    def add_controller(
        self,
        controller: BaseController,
        mvs: List[int]
    ):
        """
        Add a sub-controller for specific MVs.

        Args:
            controller: Controller instance
            mvs: List of MV indices (1-12) this controller manages
        """
        # Convert to 0-indexed
        mv_indices = [mv - 1 if mv >= 1 else mv for mv in mvs]

        # Check for conflicts
        for idx in mv_indices:
            if idx in self._mv_assignment:
                existing = self._mv_assignment[idx]
                raise ValueError(
                    f"MV {idx + 1} already assigned to {existing.name}"
                )

        # Register
        self._sub_controllers.append((controller, mv_indices))
        for idx in mv_indices:
            self._mv_assignment[idx] = controller

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        """Calculate new MVs by dispatching to sub-controllers."""
        xmv_new = xmv.copy()

        # Apply each sub-controller
        for controller, mv_indices in self._sub_controllers:
            sub_xmv = controller.calculate(xmeas, xmv_new, step)
            for idx in mv_indices:
                xmv_new[idx] = sub_xmv[idx]

        # Apply fallback for unassigned MVs
        if self._fallback is not None:
            fallback_xmv = self._fallback.calculate(xmeas, xmv_new, step)
            for i in range(NUM_MANIPULATED_VARS):
                if i not in self._mv_assignment:
                    xmv_new[i] = fallback_xmv[i]

        return xmv_new

    def reset(self):
        """Reset all sub-controllers."""
        for controller, _ in self._sub_controllers:
            controller.reset()
        if self._fallback is not None:
            self._fallback.reset()

    def get_info(self) -> Dict[str, Any]:
        """Get info including sub-controllers."""
        info = super().get_info()
        info["sub_controllers"] = [
            {"name": ctrl.name, "mvs": [i + 1 for i in mvs]}
            for ctrl, mvs in self._sub_controllers
        ]
        if self._fallback:
            info["fallback"] = self._fallback.name
        return info
