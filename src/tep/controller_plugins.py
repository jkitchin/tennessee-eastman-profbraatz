"""
Example controller plugins for the Tennessee Eastman Process.

This module provides a range of example controllers demonstrating the
plugin system, from single-loop controllers to full process control.

Controllers are organized by complexity:
1. Single-loop: Control one MV based on one measurement
2. Subsystem: Control a related group of MVs (e.g., reactor)
3. Composition: Control product quality via cascade loops
4. Full process: Control all MVs with coordinated strategy

All controllers inherit from BaseController and are registered with
the ControllerRegistry for easy discovery and instantiation.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .controller_base import (
    BaseController,
    ControllerRegistry,
    register_controller,
    CompositeController,
)
from .controllers import PIController


# =============================================================================
# SINGLE-LOOP CONTROLLERS
# =============================================================================

@register_controller(
    name="reactor_temp",
    description="Single-loop reactor temperature control via cooling water"
)
class ReactorTemperatureController(BaseController):
    """
    Single-loop controller for reactor temperature.

    Controls reactor temperature (XMEAS 9) by adjusting reactor
    cooling water flow (XMV 10). This is a critical safety loop.

    This is the simplest example of a controller plugin - it manages
    just one MV based on one measurement.
    """

    name = "reactor_temp"
    description = "Single-loop reactor temperature control via cooling water"
    version = "1.0.0"
    controlled_mvs = [10]

    def __init__(
        self,
        setpoint: float = 120.40,
        gain: float = -1.56,
        taui: float = 0.403,  # 1452/3600 hours
    ):
        """
        Initialize reactor temperature controller.

        Args:
            setpoint: Target reactor temperature (deg C)
            gain: Controller gain (negative for reverse action)
            taui: Integral time constant (hours)
        """
        self.setpoint = setpoint
        self.gain = gain
        self.taui = taui

        self._controller = PIController(
            setpoint=setpoint,
            gain=gain,
            taui=taui,
            scale=100.0 / 150.0,
            output_min=0.0,
            output_max=100.0,
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        """Calculate cooling water valve position."""
        xmv_new = xmv.copy()

        # Only execute every 3 seconds (matches original timing)
        if step % 3 == 0:
            dt = 3.0 / 3600.0  # 3 seconds in hours
            reactor_temp = xmeas[8]  # XMEAS(9) is index 8
            xmv_new[9] = self._controller.calculate(
                reactor_temp, xmv_new[9], dt
            )

        return xmv_new

    def reset(self):
        """Reset controller state."""
        self._controller.reset()

    def get_parameters(self) -> Dict[str, Any]:
        """Get controller parameters."""
        return {
            "setpoint": self.setpoint,
            "gain": self.gain,
            "taui": self.taui,
        }


@register_controller(
    name="separator_level",
    description="Single-loop separator level control"
)
class SeparatorLevelController(BaseController):
    """
    Single-loop controller for separator level.

    Controls separator level (XMEAS 12) by adjusting separator
    underflow valve (XMV 7).
    """

    name = "separator_level"
    description = "Single-loop separator level control"
    version = "1.0.0"
    controlled_mvs = [7]

    def __init__(
        self,
        setpoint: float = 50.0,
        gain: float = -2.06,
    ):
        """
        Initialize separator level controller.

        Args:
            setpoint: Target level (%)
            gain: Controller gain (negative for reverse action)
        """
        self.setpoint = setpoint
        self.gain = gain

        self._controller = PIController(
            setpoint=setpoint,
            gain=gain,
            taui=0.0,  # P-only control
            scale=100.0 / 70.0,
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        """Calculate separator underflow valve position."""
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt = 3.0 / 3600.0
            sep_level = xmeas[11]  # XMEAS(12)
            xmv_new[6] = self._controller.calculate(
                sep_level, xmv_new[6], dt
            )

        return xmv_new

    def reset(self):
        """Reset controller state."""
        self._controller.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {"setpoint": self.setpoint, "gain": self.gain}


@register_controller(
    name="stripper_level",
    description="Single-loop stripper level control"
)
class StripperLevelController(BaseController):
    """
    Single-loop controller for stripper level.

    Controls stripper level (XMEAS 15) by adjusting stripper
    product flow valve (XMV 8).
    """

    name = "stripper_level"
    description = "Single-loop stripper level control"
    version = "1.0.0"
    controlled_mvs = [8]

    def __init__(
        self,
        setpoint: float = 50.0,
        gain: float = -1.62,
    ):
        self.setpoint = setpoint
        self.gain = gain

        self._controller = PIController(
            setpoint=setpoint,
            gain=gain,
            taui=0.0,
            scale=100.0 / 70.0,
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt = 3.0 / 3600.0
            strip_level = xmeas[14]  # XMEAS(15)
            xmv_new[7] = self._controller.calculate(
                strip_level, xmv_new[7], dt
            )

        return xmv_new

    def reset(self):
        self._controller.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {"setpoint": self.setpoint, "gain": self.gain}


# =============================================================================
# SUBSYSTEM CONTROLLERS
# =============================================================================

@register_controller(
    name="reactor_subsystem",
    description="Reactor subsystem control (temperature, level, cooling)"
)
class ReactorSubsystemController(BaseController):
    """
    Subsystem controller for the reactor section.

    Controls:
    - Reactor temperature via cooling water (XMV 10)
    - Reactor level via A+C feed (XMV 4)

    Uses cascade control where the reactor level controller adjusts
    the A+C feed flow setpoint.
    """

    name = "reactor_subsystem"
    description = "Reactor subsystem control (temperature, level, cooling)"
    version = "1.0.0"
    controlled_mvs = [4, 10]

    def __init__(
        self,
        temp_setpoint: float = 120.40,
        level_setpoint: float = 75.0,
    ):
        """
        Initialize reactor subsystem controller.

        Args:
            temp_setpoint: Target reactor temperature (deg C)
            level_setpoint: Target reactor level (%)
        """
        self.temp_setpoint = temp_setpoint
        self.level_setpoint = level_setpoint

        # Reactor cooling water temperature control (cascade secondary)
        self.cw_temp_setpoint = 94.599
        self._ctrl_cw_temp = PIController(
            setpoint=self.cw_temp_setpoint,
            gain=-1.56,
            taui=1452.0 / 3600.0,
            scale=100.0 / 150.0,
        )

        # Reactor temperature control (cascade primary -> CW temp setpoint)
        self._ctrl_reactor_temp = PIController(
            setpoint=temp_setpoint,
            gain=28.3,
            taui=982.0 / 3600.0,
            scale=100.0 / 150.0,
        )

        # A+C feed flow control (cascade secondary)
        self.ac_feed_setpoint = 9.3477
        self._ctrl_ac_feed = PIController(
            setpoint=self.ac_feed_setpoint,
            gain=1.0,
            taui=0.0,
            scale=100.0 / 15.25,
        )

        # Reactor level control (cascade primary -> A+C feed setpoint)
        self._ctrl_reactor_level = PIController(
            setpoint=level_setpoint,
            gain=1.11,
            taui=3168.0 / 3600.0,
            scale=100.0 / 50.0,
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt = 3.0 / 3600.0

            # Reactor temperature cascade
            # Primary: Reactor temp -> CW temp setpoint
            dxmv = self._ctrl_reactor_temp.calculate_change(xmeas[8], dt)
            self.cw_temp_setpoint += dxmv * 150.0 / 100.0
            self._ctrl_cw_temp.setpoint = self.cw_temp_setpoint

            # Secondary: CW temp -> XMV 10
            xmv_new[9] = self._ctrl_cw_temp.calculate(
                xmeas[20], xmv_new[9], dt
            )

            # Reactor level cascade
            # Primary: Level -> A+C feed setpoint
            dxmv = self._ctrl_reactor_level.calculate_change(xmeas[7], dt)
            self.ac_feed_setpoint += dxmv * 15.25 / 100.0
            self._ctrl_ac_feed.setpoint = self.ac_feed_setpoint

            # Secondary: A+C feed -> XMV 4
            xmv_new[3] = self._ctrl_ac_feed.calculate(
                xmeas[3], xmv_new[3], dt
            )

        return xmv_new

    def reset(self):
        self.cw_temp_setpoint = 94.599
        self.ac_feed_setpoint = 9.3477
        self._ctrl_cw_temp.reset()
        self._ctrl_reactor_temp.reset()
        self._ctrl_ac_feed.reset()
        self._ctrl_reactor_level.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "temp_setpoint": self.temp_setpoint,
            "level_setpoint": self.level_setpoint,
            "cw_temp_setpoint": self.cw_temp_setpoint,
            "ac_feed_setpoint": self.ac_feed_setpoint,
        }


@register_controller(
    name="separator_subsystem",
    description="Separator subsystem control (level, pressure)"
)
class SeparatorSubsystemController(BaseController):
    """
    Subsystem controller for the separator section.

    Controls:
    - Separator level via underflow (XMV 7)
    - Condenser cooling water (XMV 11)
    """

    name = "separator_subsystem"
    description = "Separator subsystem control (level, pressure)"
    version = "1.0.0"
    controlled_mvs = [7, 11]

    def __init__(
        self,
        level_setpoint: float = 50.0,
    ):
        self.level_setpoint = level_setpoint

        self._ctrl_level = PIController(
            setpoint=level_setpoint,
            gain=-2.06,
            taui=0.0,
            scale=100.0 / 70.0,
        )

        self._ctrl_condenser = PIController(
            setpoint=22.949,
            gain=1.09,
            taui=2600.0 / 3600.0,
            scale=100.0 / 46.0,
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt = 3.0 / 3600.0

            # Separator level
            xmv_new[6] = self._ctrl_level.calculate(
                xmeas[11], xmv_new[6], dt
            )

            # Condenser cooling water (based on stripper underflow)
            xmv_new[10] = self._ctrl_condenser.calculate(
                xmeas[16], xmv_new[10], dt
            )

        return xmv_new

    def reset(self):
        self._ctrl_level.reset()
        self._ctrl_condenser.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {"level_setpoint": self.level_setpoint}


@register_controller(
    name="feed_subsystem",
    description="Feed subsystem control (D, E, A, A+C feeds)"
)
class FeedSubsystemController(BaseController):
    """
    Subsystem controller for feed flows.

    Controls:
    - D feed flow (XMV 1)
    - E feed flow (XMV 2)
    - A feed flow (XMV 3)
    - A+C feed flow (XMV 4)

    These are typically flow control loops that track setpoints
    from higher-level composition controllers.
    """

    name = "feed_subsystem"
    description = "Feed subsystem control (D, E, A, A+C feeds)"
    version = "1.0.0"
    controlled_mvs = [1, 2, 3, 4]

    def __init__(
        self,
        d_feed_sp: float = 3664.0,
        e_feed_sp: float = 4509.3,
        a_feed_sp: float = 0.25052,
        ac_feed_sp: float = 9.3477,
    ):
        self.d_feed_sp = d_feed_sp
        self.e_feed_sp = e_feed_sp
        self.a_feed_sp = a_feed_sp
        self.ac_feed_sp = ac_feed_sp

        self._ctrl_d = PIController(
            setpoint=d_feed_sp, gain=1.0, taui=0.0,
            scale=100.0 / 5811.0
        )
        self._ctrl_e = PIController(
            setpoint=e_feed_sp, gain=1.0, taui=0.0,
            scale=100.0 / 8354.0
        )
        self._ctrl_a = PIController(
            setpoint=a_feed_sp, gain=1.0, taui=0.0,
            scale=100.0 / 1.017
        )
        self._ctrl_ac = PIController(
            setpoint=ac_feed_sp, gain=1.0, taui=0.0,
            scale=100.0 / 15.25
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt = 3.0 / 3600.0

            xmv_new[0] = self._ctrl_d.calculate(xmeas[1], xmv_new[0], dt)
            xmv_new[1] = self._ctrl_e.calculate(xmeas[2], xmv_new[1], dt)
            xmv_new[2] = self._ctrl_a.calculate(xmeas[0], xmv_new[2], dt)
            xmv_new[3] = self._ctrl_ac.calculate(xmeas[3], xmv_new[3], dt)

        return xmv_new

    def reset(self):
        self._ctrl_d.reset()
        self._ctrl_e.reset()
        self._ctrl_a.reset()
        self._ctrl_ac.reset()

    def set_setpoints(
        self,
        d_feed: float = None,
        e_feed: float = None,
        a_feed: float = None,
        ac_feed: float = None
    ):
        """Update feed flow setpoints."""
        if d_feed is not None:
            self.d_feed_sp = d_feed
            self._ctrl_d.setpoint = d_feed
        if e_feed is not None:
            self.e_feed_sp = e_feed
            self._ctrl_e.setpoint = e_feed
        if a_feed is not None:
            self.a_feed_sp = a_feed
            self._ctrl_a.setpoint = a_feed
        if ac_feed is not None:
            self.ac_feed_sp = ac_feed
            self._ctrl_ac.setpoint = ac_feed

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "d_feed_sp": self.d_feed_sp,
            "e_feed_sp": self.e_feed_sp,
            "a_feed_sp": self.a_feed_sp,
            "ac_feed_sp": self.ac_feed_sp,
        }


# =============================================================================
# COMPOSITION CONTROLLERS
# =============================================================================

@register_controller(
    name="product_quality",
    description="Product quality control via stripper temperature cascade"
)
class ProductQualityController(BaseController):
    """
    Product quality controller using cascade control.

    Controls product E composition (XMEAS 38) by adjusting stripper
    temperature, which in turn adjusts steam flow (XMV 9).

    This is a slow outer loop (15-minute sample time) that cascades
    to a faster temperature loop (3-second sample time).
    """

    name = "product_quality"
    description = "Product quality control via stripper temperature cascade"
    version = "1.0.0"
    controlled_mvs = [9]

    def __init__(
        self,
        product_e_setpoint: float = 0.8357,
    ):
        """
        Initialize product quality controller.

        Args:
            product_e_setpoint: Target product E composition (mol fraction)
        """
        self.product_e_setpoint = product_e_setpoint

        # Stripper temperature setpoint (adjusted by outer loop)
        self.stripper_temp_setpoint = 65.731

        # Steam flow setpoint (adjusted by temperature loop)
        self.steam_flow_setpoint = 230.31

        # Outer loop: Product E composition -> stripper temp setpoint
        self._ctrl_product_e = PIController(
            setpoint=product_e_setpoint,
            gain=-3.26,
            taui=12408.0 / 3600.0,
            scale=100.0 / 1.6,
        )

        # Middle loop: Stripper temp -> steam flow setpoint
        self._ctrl_stripper_temp = PIController(
            setpoint=self.stripper_temp_setpoint,
            gain=0.169,
            taui=236.0 / 3600.0,
            scale=100.0 / 130.0,
        )

        # Inner loop: Steam flow control
        self._ctrl_steam = PIController(
            setpoint=self.steam_flow_setpoint,
            gain=0.41,
            taui=0.0,
            scale=100.0 / 460.0,
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        # Fast loop (every 3 seconds)
        if step % 3 == 0:
            dt3 = 3.0 / 3600.0

            # Update steam flow setpoint from temperature controller
            self._ctrl_steam.setpoint = self.steam_flow_setpoint

            # Steam flow control
            xmv_new[8] = self._ctrl_steam.calculate(
                xmeas[18], xmv_new[8], dt3
            )

            # Stripper temperature -> steam setpoint
            dxmv = self._ctrl_stripper_temp.calculate_change(xmeas[17], dt3)
            self.steam_flow_setpoint += dxmv * 460.0 / 100.0

        # Slow loop (every 15 minutes = 900 seconds)
        if step % 900 == 0:
            dt900 = 900.0 / 3600.0

            # Product E composition -> stripper temp setpoint
            dxmv = self._ctrl_product_e.calculate_change(xmeas[37], dt900)
            self.stripper_temp_setpoint += dxmv * 130.0 / 100.0
            self._ctrl_stripper_temp.setpoint = self.stripper_temp_setpoint

        return xmv_new

    def reset(self):
        self.stripper_temp_setpoint = 65.731
        self.steam_flow_setpoint = 230.31
        self._ctrl_product_e.reset()
        self._ctrl_stripper_temp.reset()
        self._ctrl_steam.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "product_e_setpoint": self.product_e_setpoint,
            "stripper_temp_setpoint": self.stripper_temp_setpoint,
            "steam_flow_setpoint": self.steam_flow_setpoint,
        }


@register_controller(
    name="reactor_composition",
    description="Reactor feed composition control (A, D, E in feed)"
)
class ReactorCompositionController(BaseController):
    """
    Reactor feed composition controller.

    Controls reactor feed composition by adjusting feed flow setpoints.
    Uses slow (6-minute) cascade loops.

    Controls:
    - Feed A composition via A feed (affects XMV 3 setpoint)
    - Feed D composition via D feed (affects XMV 1 setpoint)
    - Feed E composition via E feed (affects XMV 2 setpoint)

    Note: This controller outputs setpoints, not direct MV values.
    It should be used with a FeedSubsystemController to execute
    the flow control.
    """

    name = "reactor_composition"
    description = "Reactor feed composition control (A, D, E in feed)"
    version = "1.0.0"
    controlled_mvs = [1, 2, 3]

    def __init__(
        self,
        comp_a_setpoint: float = 32.188,
        comp_d_setpoint: float = 6.882,
        comp_e_setpoint: float = 18.776,
    ):
        self.comp_a_setpoint = comp_a_setpoint
        self.comp_d_setpoint = comp_d_setpoint
        self.comp_e_setpoint = comp_e_setpoint

        # Feed flow setpoints (outputs of this controller)
        self.a_feed_setpoint = 0.25052
        self.d_feed_setpoint = 3664.0
        self.e_feed_setpoint = 4509.3

        # Composition controllers (outer loops)
        self._ctrl_comp_a = PIController(
            setpoint=comp_a_setpoint, gain=18.0,
            taui=3168.0 / 3600.0, scale=100.0 / 100.0
        )
        self._ctrl_comp_d = PIController(
            setpoint=comp_d_setpoint, gain=8.3,
            taui=3168.0 / 3600.0, scale=100.0 / 100.0
        )
        self._ctrl_comp_e = PIController(
            setpoint=comp_e_setpoint, gain=2.37,
            taui=5069.0 / 3600.0, scale=100.0 / 100.0
        )

        # Flow controllers (inner loops)
        self._ctrl_a_feed = PIController(
            setpoint=self.a_feed_setpoint, gain=1.0,
            taui=0.0, scale=100.0 / 1.017
        )
        self._ctrl_d_feed = PIController(
            setpoint=self.d_feed_setpoint, gain=1.0,
            taui=0.0, scale=100.0 / 5811.0
        )
        self._ctrl_e_feed = PIController(
            setpoint=self.e_feed_setpoint, gain=1.0,
            taui=0.0, scale=100.0 / 8354.0
        )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        # Fast loops (every 3 seconds) - flow control
        if step % 3 == 0:
            dt3 = 3.0 / 3600.0

            self._ctrl_a_feed.setpoint = self.a_feed_setpoint
            self._ctrl_d_feed.setpoint = self.d_feed_setpoint
            self._ctrl_e_feed.setpoint = self.e_feed_setpoint

            xmv_new[2] = self._ctrl_a_feed.calculate(xmeas[0], xmv_new[2], dt3)
            xmv_new[0] = self._ctrl_d_feed.calculate(xmeas[1], xmv_new[0], dt3)
            xmv_new[1] = self._ctrl_e_feed.calculate(xmeas[2], xmv_new[1], dt3)

        # Slow loops (every 6 minutes = 360 seconds) - composition control
        if step % 360 == 0:
            dt360 = 360.0 / 3600.0

            # Composition A -> A feed setpoint
            dxmv = self._ctrl_comp_a.calculate_change(xmeas[22], dt360)
            self.a_feed_setpoint += dxmv * 1.017 / 100.0

            # Composition D -> D feed setpoint
            dxmv = self._ctrl_comp_d.calculate_change(xmeas[25], dt360)
            self.d_feed_setpoint += dxmv * 5811.0 / 100.0

            # Composition E -> E feed setpoint
            dxmv = self._ctrl_comp_e.calculate_change(xmeas[26], dt360)
            self.e_feed_setpoint += dxmv * 8354.0 / 100.0

        return xmv_new

    def reset(self):
        self.a_feed_setpoint = 0.25052
        self.d_feed_setpoint = 3664.0
        self.e_feed_setpoint = 4509.3
        self._ctrl_comp_a.reset()
        self._ctrl_comp_d.reset()
        self._ctrl_comp_e.reset()
        self._ctrl_a_feed.reset()
        self._ctrl_d_feed.reset()
        self._ctrl_e_feed.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "comp_a_setpoint": self.comp_a_setpoint,
            "comp_d_setpoint": self.comp_d_setpoint,
            "comp_e_setpoint": self.comp_e_setpoint,
            "a_feed_setpoint": self.a_feed_setpoint,
            "d_feed_setpoint": self.d_feed_setpoint,
            "e_feed_setpoint": self.e_feed_setpoint,
        }


# =============================================================================
# FULL PROCESS CONTROLLERS
# =============================================================================

@register_controller(
    name="proportional_only",
    description="Simple proportional-only control for all loops"
)
class ProportionalOnlyController(BaseController):
    """
    Simple proportional-only controller for all MVs.

    This is a minimal full-process controller that uses P-only control
    for all loops. Useful as a baseline or for testing.
    """

    name = "proportional_only"
    description = "Simple proportional-only control for all loops"
    version = "1.0.0"
    controlled_mvs = list(range(1, 12))

    # Default setpoints and gains for each MV
    DEFAULT_CONFIG = {
        # (measurement_index, setpoint, gain, scale)
        0: (1, 3664.0, 1.0, 100.0 / 5811.0),     # XMV1: D feed
        1: (2, 4509.3, 1.0, 100.0 / 8354.0),     # XMV2: E feed
        2: (0, 0.25052, 1.0, 100.0 / 1.017),     # XMV3: A feed
        3: (3, 9.3477, 1.0, 100.0 / 15.25),      # XMV4: A+C feed
        4: (4, 26.902, -0.083, 100.0 / 53.0),    # XMV5: Recycle
        5: (9, 0.33712, 1.22, 100.0 / 1.0),      # XMV6: Purge
        6: (11, 50.0, -2.06, 100.0 / 70.0),      # XMV7: Sep level
        7: (14, 50.0, -1.62, 100.0 / 70.0),      # XMV8: Strip level
        8: (18, 230.31, 0.41, 100.0 / 460.0),    # XMV9: Steam
        9: (20, 94.599, -1.56, 100.0 / 150.0),   # XMV10: Reactor CW
        10: (16, 22.949, 1.09, 100.0 / 46.0),    # XMV11: Cond CW
    }

    def __init__(self):
        self._controllers = {}
        for mv_idx, (meas_idx, sp, gain, scale) in self.DEFAULT_CONFIG.items():
            self._controllers[mv_idx] = PIController(
                setpoint=sp,
                gain=gain,
                taui=0.0,  # P-only
                scale=scale,
            )

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        if step % 3 == 0:
            dt = 3.0 / 3600.0
            for mv_idx, (meas_idx, _, _, _) in self.DEFAULT_CONFIG.items():
                xmv_new[mv_idx] = self._controllers[mv_idx].calculate(
                    xmeas[meas_idx], xmv_new[mv_idx], dt
                )

        return xmv_new

    def reset(self):
        for ctrl in self._controllers.values():
            ctrl.reset()

    def get_parameters(self) -> Dict[str, Any]:
        return {
            f"xmv{i+1}_setpoint": self._controllers[i].setpoint
            for i in self._controllers
        }


@register_controller(
    name="economic_mpc",
    description="Economic MPC-style controller (simplified demonstration)"
)
class EconomicMPCController(BaseController):
    """
    Simplified economic MPC-style controller.

    This demonstrates how an MPC-style controller could be implemented.
    It doesn't use actual MPC optimization, but shows the structure
    for integrating more advanced control strategies.

    The controller prioritizes:
    1. Safety (reactor pressure/temperature limits)
    2. Product quality (product E composition)
    3. Production rate (maximize throughput)
    """

    name = "economic_mpc"
    description = "Economic MPC-style controller (simplified demonstration)"
    version = "1.0.0"
    controlled_mvs = list(range(1, 12))

    # Safety limits
    REACTOR_TEMP_MAX = 150.0
    REACTOR_TEMP_MIN = 100.0
    REACTOR_PRESSURE_MAX = 2900.0
    REACTOR_PRESSURE_MIN = 2700.0

    def __init__(
        self,
        production_rate_target: float = 1.0,  # Fraction of nominal
        product_e_target: float = 0.8357,
    ):
        """
        Initialize economic MPC controller.

        Args:
            production_rate_target: Target production rate (fraction of nominal)
            product_e_target: Target product E composition
        """
        self.production_rate_target = production_rate_target
        self.product_e_target = product_e_target

        # Base setpoints (nominal operation)
        self.base_setpoints = {
            "d_feed": 3664.0,
            "e_feed": 4509.3,
            "a_feed": 0.25052,
            "ac_feed": 9.3477,
            "recycle": 26.902,
            "purge": 0.33712,
            "sep_level": 50.0,
            "strip_level": 50.0,
            "steam": 230.31,
            "reactor_cw": 94.599,
            "cond_cw": 22.949,
        }

        # Initialize PI controllers for each loop
        self._init_controllers()

    def _init_controllers(self):
        """Initialize all PI controllers."""
        self._controllers = {
            0: PIController(self.base_setpoints["d_feed"], 1.0, 0.0, scale=100.0/5811.0),
            1: PIController(self.base_setpoints["e_feed"], 1.0, 0.0, scale=100.0/8354.0),
            2: PIController(self.base_setpoints["a_feed"], 1.0, 0.0, scale=100.0/1.017),
            3: PIController(self.base_setpoints["ac_feed"], 1.0, 0.0, scale=100.0/15.25),
            4: PIController(self.base_setpoints["recycle"], -0.083, 1.0/3600.0, scale=100.0/53.0),
            5: PIController(self.base_setpoints["purge"], 1.22, 0.0, scale=100.0/1.0),
            6: PIController(self.base_setpoints["sep_level"], -2.06, 0.0, scale=100.0/70.0),
            7: PIController(self.base_setpoints["strip_level"], -1.62, 0.0, scale=100.0/70.0),
            8: PIController(self.base_setpoints["steam"], 0.41, 0.0, scale=100.0/460.0),
            9: PIController(self.base_setpoints["reactor_cw"], -1.56, 1452.0/3600.0, scale=100.0/150.0),
            10: PIController(self.base_setpoints["cond_cw"], 1.09, 2600.0/3600.0, scale=100.0/46.0),
        }

        # Measurement indices for each controller
        self._meas_idx = {0: 1, 1: 2, 2: 0, 3: 3, 4: 4, 5: 9, 6: 11, 7: 14, 8: 18, 9: 20, 10: 16}

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        xmv_new = xmv.copy()

        # Safety layer - override if approaching limits
        reactor_temp = xmeas[8]
        reactor_pressure = xmeas[6]

        safety_mode = False
        if reactor_temp > self.REACTOR_TEMP_MAX - 5:
            # Increase cooling
            xmv_new[9] = min(xmv_new[9] + 2.0, 100.0)
            safety_mode = True
        elif reactor_temp < self.REACTOR_TEMP_MIN + 5:
            # Decrease cooling
            xmv_new[9] = max(xmv_new[9] - 2.0, 0.0)
            safety_mode = True

        if reactor_pressure > self.REACTOR_PRESSURE_MAX - 50:
            # Increase purge
            xmv_new[5] = min(xmv_new[5] + 5.0, 100.0)
            safety_mode = True

        # Normal control (if not in safety mode)
        if not safety_mode and step % 3 == 0:
            dt = 3.0 / 3600.0

            # Adjust setpoints based on production rate target
            rate_factor = self.production_rate_target
            self._controllers[0].setpoint = self.base_setpoints["d_feed"] * rate_factor
            self._controllers[1].setpoint = self.base_setpoints["e_feed"] * rate_factor

            # Execute all control loops
            for mv_idx, ctrl in self._controllers.items():
                meas_idx = self._meas_idx[mv_idx]
                xmv_new[mv_idx] = ctrl.calculate(xmeas[meas_idx], xmv_new[mv_idx], dt)

        return xmv_new

    def reset(self):
        self._init_controllers()

    def set_production_rate(self, rate: float):
        """Set production rate target (fraction of nominal)."""
        self.production_rate_target = np.clip(rate, 0.5, 1.2)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "production_rate_target": self.production_rate_target,
            "product_e_target": self.product_e_target,
            "base_setpoints": self.base_setpoints.copy(),
        }


# =============================================================================
# PASSTHROUGH CONTROLLER (for testing)
# =============================================================================

@register_controller(
    name="passthrough",
    description="Passthrough controller - holds MVs at current values"
)
class PassthroughController(BaseController):
    """
    Passthrough controller that holds MVs at their current values.

    Useful for testing or as a fallback in composite controllers.
    """

    name = "passthrough"
    description = "Passthrough controller - holds MVs at current values"
    version = "1.0.0"
    controlled_mvs = None  # Doesn't actively control any

    def calculate(
        self,
        xmeas: np.ndarray,
        xmv: np.ndarray,
        step: int
    ) -> np.ndarray:
        return xmv.copy()

    def reset(self):
        pass

    def get_parameters(self) -> Dict[str, Any]:
        return {}


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_composite_controller(
    subsystems: List[str],
    fallback: str = "passthrough"
) -> CompositeController:
    """
    Create a composite controller from named subsystem controllers.

    Args:
        subsystems: List of subsystem controller names to include
        fallback: Name of fallback controller for uncontrolled MVs

    Returns:
        CompositeController instance

    Example:
        >>> ctrl = create_composite_controller(
        ...     ["reactor_subsystem", "separator_subsystem"],
        ...     fallback="passthrough"
        ... )
    """
    # MV assignments for known subsystem controllers
    MV_ASSIGNMENTS = {
        "reactor_temp": [10],
        "separator_level": [7],
        "stripper_level": [8],
        "reactor_subsystem": [4, 10],
        "separator_subsystem": [7, 11],
        "feed_subsystem": [1, 2, 3, 4],
        "product_quality": [9],
        "reactor_composition": [1, 2, 3],
    }

    fallback_ctrl = ControllerRegistry.create(fallback)
    composite = CompositeController(fallback_controller=fallback_ctrl)

    for name in subsystems:
        ctrl = ControllerRegistry.create(name)
        mvs = MV_ASSIGNMENTS.get(name, ctrl.controlled_mvs or [])
        if mvs:
            composite.add_controller(ctrl, mvs)

    return composite
