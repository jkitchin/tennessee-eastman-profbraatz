"""
Streaming simulation server for Tennessee Eastman Process.

Provides a ZeroMQ PUB socket that streams simulation data to multiple
remote consumers. The simulation runs continuously with random fault
injection, automatically restarting on safety violations.

Usage:
    from tep.streaming import StreamingServer

    server = StreamingServer(port=5555)
    server.run()  # Runs forever

Consumers connect via ZMQ SUB socket:
    import zmq
    socket = zmq.Context().socket(zmq.SUB)
    socket.connect("tcp://hostname:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    while True:
        msg = socket.recv_json()
        print(msg)
"""

import json
import uuid
import time
import random
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

from .simulator import TEPSimulator, ControlMode
from .constants import (
    DISTURBANCE_NAMES,
    SAFETY_LIMITS,
    NUM_DISTURBANCES,
)


@dataclass
class FaultSchedule:
    """Schedule for a fault injection."""
    idv: int                    # Fault index (1-20)
    start_time: float           # When to activate (hours)
    end_time: Optional[float]   # When to deactivate (hours), None if until restart


@dataclass
class StreamingConfig:
    """Configuration for the streaming server."""
    port: int = 5555
    bind_address: str = "tcp://*"

    # Fault scheduling
    min_fault_start: float = 0.5      # Minimum time before fault (hours)
    max_fault_start: float = 2.0      # Maximum time before fault (hours)
    min_fault_duration: float = 1.0   # Minimum fault duration (hours)
    max_fault_duration: float = 4.0   # Maximum fault duration (hours)
    allowed_faults: Optional[List[int]] = None  # Which IDVs to use (None=all)

    # Simulation
    backend: str = "python"
    random_seed: Optional[int] = None
    data_interval: int = 1            # Send data every N steps (1=every second)

    # Run limits
    max_run_time: Optional[float] = None  # Max hours per run (None=unlimited)


class StreamingServer:
    """
    ZeroMQ streaming server for TEP simulation data.

    Runs a continuous simulation with random fault injection, publishing
    all data and events to connected subscribers.
    """

    def __init__(self, config: Optional[StreamingConfig] = None, **kwargs):
        """
        Initialize the streaming server.

        Args:
            config: StreamingConfig object, or pass individual params as kwargs
        """
        if not HAS_ZMQ:
            raise ImportError(
                "pyzmq is required for streaming. Install with: pip install pyzmq"
            )

        if config is None:
            config = StreamingConfig(**kwargs)
        self.config = config

        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)

        # Simulation state
        self.simulator: Optional[TEPSimulator] = None
        self.run_id: Optional[str] = None
        self.run_number: int = 0
        self.previous_run_id: Optional[str] = None
        self.fault_schedule: Optional[FaultSchedule] = None
        self.fault_active: bool = False

        # Statistics
        self.total_messages: int = 0
        self.total_restarts: int = 0

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        return uuid.uuid4().hex[:8]

    def _schedule_fault(self) -> FaultSchedule:
        """Create a random fault schedule for the current run."""
        # Pick a random fault
        if self.config.allowed_faults:
            idv = random.choice(self.config.allowed_faults)
        else:
            idv = random.randint(1, NUM_DISTURBANCES)

        # Random start time
        start_time = random.uniform(
            self.config.min_fault_start,
            self.config.max_fault_start
        )

        # Random duration
        duration = random.uniform(
            self.config.min_fault_duration,
            self.config.max_fault_duration
        )
        end_time = start_time + duration

        return FaultSchedule(idv=idv, start_time=start_time, end_time=end_time)

    def _check_safety(self) -> Optional[Tuple[str, float, float]]:
        """
        Check if any safety limits are violated.

        The simulator checks internal state variables (not XMEAS) against limits:
        - Reactor pressure > 3000 kPa
        - Reactor liquid volume outside 2-24 m³
        - Reactor temperature > 175°C
        - Separator liquid volume outside 1-12 m³
        - Stripper liquid volume outside 1-8 m³

        Returns:
            None if safe, or (reason, limit, actual) tuple if violated
        """
        # Use simulator's built-in shutdown detection
        # This checks internal state variables, not XMEAS values
        if self.simulator.is_shutdown():
            xmeas = self.simulator.get_measurements()
            # Try to determine which limit was exceeded from XMEAS values
            # Note: XMEAS values are approximations, actual limits are on internal vars
            if xmeas[6] > 2900:  # Close to pressure limit
                return ("reactor_pressure", 3000.0, xmeas[6])
            elif xmeas[8] > 170:  # Close to temperature limit
                return ("reactor_temp", 175.0, xmeas[8])
            else:
                # Generic shutdown - could be level limits on internal vars
                return ("safety_shutdown", 0.0, 0.0)
        return None

    def _publish(self, msg: Dict[str, Any]):
        """Publish a message to all subscribers."""
        self.socket.send_json(msg)
        self.total_messages += 1

    def _start_run(self):
        """Initialize a new simulation run."""
        self.previous_run_id = self.run_id
        self.run_id = self._generate_run_id()
        self.run_number += 1
        self.fault_active = False

        # Create new simulator
        seed = self.config.random_seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        self.simulator = TEPSimulator(
            random_seed=seed,
            control_mode=ControlMode.CLOSED_LOOP,
            backend=self.config.backend,
        )
        self.simulator.initialize()

        # Schedule a fault
        self.fault_schedule = self._schedule_fault()

        # Publish run_start
        self._publish({
            "type": "run_start",
            "run_id": self.run_id,
            "t": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "backend": self.config.backend,
            "scheduled_fault": {
                "idv": self.fault_schedule.idv,
                "start_time": self.fault_schedule.start_time,
                "end_time": self.fault_schedule.end_time,
            },
            "run_number": self.run_number,
            "previous_run_id": self.previous_run_id,
        })

    def _handle_restart(self, reason: str, limit: float, actual: float):
        """Handle a safety violation and restart."""
        self.total_restarts += 1
        next_run_id = self._generate_run_id()

        # Publish restart message
        self._publish({
            "type": "restart",
            "run_id": self.run_id,
            "t": self.simulator.time,
            "step": self.simulator.step_count,
            "reason": reason,
            "limit": limit,
            "actual": float(actual),
            "fault": self.fault_schedule.idv if self.fault_active else 0,
            "next_run_id": next_run_id,
        })

        # Start new run (will generate its own ID, overwriting next_run_id prediction)
        self._start_run()

    def _publish_data(self):
        """Publish current simulation data."""
        xmeas = self.simulator.get_measurements()
        xmv = self.simulator.get_manipulated_vars()

        self._publish({
            "type": "data",
            "run_id": self.run_id,
            "t": self.simulator.time,
            "step": self.simulator.step_count,
            "xmeas": xmeas.tolist(),
            "xmv": xmv.tolist(),
            "fault": self.fault_schedule.idv if self.fault_active else 0,
            "fault_active": self.fault_active,
        })

    def _publish_fault_on(self):
        """Publish fault activation event."""
        self._publish({
            "type": "fault_on",
            "run_id": self.run_id,
            "t": self.simulator.time,
            "step": self.simulator.step_count,
            "fault": self.fault_schedule.idv,
            "fault_name": DISTURBANCE_NAMES[self.fault_schedule.idv - 1],
        })

    def _publish_fault_off(self):
        """Publish fault deactivation event."""
        fault_duration = self.simulator.time - self.fault_schedule.start_time
        self._publish({
            "type": "fault_off",
            "run_id": self.run_id,
            "t": self.simulator.time,
            "step": self.simulator.step_count,
            "fault": self.fault_schedule.idv,
            "fault_duration": fault_duration,
        })

    def run(self, duration: Optional[float] = None):
        """
        Run the streaming server.

        Args:
            duration: Optional total runtime in hours (None=run forever)
        """
        # Bind socket
        bind_addr = f"{self.config.bind_address}:{self.config.port}"
        self.socket.bind(bind_addr)
        print(f"Streaming server listening on {bind_addr}")

        # Allow subscribers time to connect
        time.sleep(0.5)

        # Start first run
        self._start_run()

        start_wall_time = time.time()

        try:
            while True:
                # Check duration limit
                if duration is not None:
                    elapsed = (time.time() - start_wall_time) / 3600.0
                    if elapsed >= duration:
                        print(f"Duration limit reached ({duration} hours)")
                        break

                # Check max run time
                if self.config.max_run_time is not None:
                    if self.simulator.time >= self.config.max_run_time:
                        # Just restart with new fault schedule
                        self._start_run()
                        continue

                # Check for fault activation
                if not self.fault_active and self.simulator.time >= self.fault_schedule.start_time:
                    self.simulator.set_disturbance(self.fault_schedule.idv, 1)
                    self.fault_active = True
                    self._publish_fault_on()

                # Check for fault deactivation
                if self.fault_active and self.fault_schedule.end_time is not None:
                    if self.simulator.time >= self.fault_schedule.end_time:
                        self.simulator.set_disturbance(self.fault_schedule.idv, 0)
                        self.fault_active = False
                        self._publish_fault_off()
                        # Schedule next fault
                        self.fault_schedule = self._schedule_fault()
                        self.fault_schedule.start_time += self.simulator.time
                        if self.fault_schedule.end_time is not None:
                            self.fault_schedule.end_time += self.simulator.time

                # Step simulation
                running = self.simulator.step()

                # Check for numerical instability
                if not running:
                    self._handle_restart("numerical_instability", 0.0, 0.0)
                    continue

                # Check safety limits
                violation = self._check_safety()
                if violation is not None:
                    reason, limit, actual = violation
                    self._handle_restart(reason, limit, actual)
                    continue

                # Publish data at configured interval
                if self.simulator.step_count % self.config.data_interval == 0:
                    self._publish_data()

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.socket.close()
            self.context.term()
            print(f"Total messages: {self.total_messages}, restarts: {self.total_restarts}")

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "total_messages": self.total_messages,
            "total_restarts": self.total_restarts,
            "run_number": self.run_number,
            "current_run_id": self.run_id,
        }


def run_server(port: int = 5555, **kwargs):
    """
    Convenience function to run a streaming server.

    Args:
        port: Port to bind to
        **kwargs: Additional StreamingConfig parameters
    """
    config = StreamingConfig(port=port, **kwargs)
    server = StreamingServer(config)
    server.run()


def main():
    """CLI entry point for streaming server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TEP Streaming Server - streams simulation data via ZeroMQ"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=5555,
        help="Port to bind to (default: 5555)"
    )
    parser.add_argument(
        "--bind", "-b", default="tcp://*",
        help="Bind address (default: tcp://*)"
    )
    parser.add_argument(
        "--backend", choices=["python", "fortran"], default="python",
        help="Simulation backend (default: python)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--interval", type=int, default=1,
        help="Send data every N steps (default: 1)"
    )
    parser.add_argument(
        "--faults", type=int, nargs="+", default=None,
        help="Allowed fault indices, e.g., --faults 1 2 4 (default: all)"
    )
    parser.add_argument(
        "--min-fault-start", type=float, default=0.5,
        help="Minimum time before fault starts (hours, default: 0.5)"
    )
    parser.add_argument(
        "--max-fault-start", type=float, default=2.0,
        help="Maximum time before fault starts (hours, default: 2.0)"
    )
    parser.add_argument(
        "--min-fault-duration", type=float, default=1.0,
        help="Minimum fault duration (hours, default: 1.0)"
    )
    parser.add_argument(
        "--max-fault-duration", type=float, default=4.0,
        help="Maximum fault duration (hours, default: 4.0)"
    )
    parser.add_argument(
        "--max-run-time", type=float, default=None,
        help="Maximum run time before restart (hours, default: unlimited)"
    )

    args = parser.parse_args()

    config = StreamingConfig(
        port=args.port,
        bind_address=args.bind,
        backend=args.backend,
        random_seed=args.seed,
        data_interval=args.interval,
        allowed_faults=args.faults,
        min_fault_start=args.min_fault_start,
        max_fault_start=args.max_fault_start,
        min_fault_duration=args.min_fault_duration,
        max_fault_duration=args.max_fault_duration,
        max_run_time=args.max_run_time,
    )

    print("TEP Streaming Server")
    print(f"  Port: {config.port}")
    print(f"  Backend: {config.backend}")
    print(f"  Data interval: every {config.data_interval} step(s)")
    print(f"  Fault timing: {config.min_fault_start}-{config.max_fault_start}h start, "
          f"{config.min_fault_duration}-{config.max_fault_duration}h duration")
    if config.allowed_faults:
        print(f"  Allowed faults: IDV({', '.join(map(str, config.allowed_faults))})")
    print()

    server = StreamingServer(config)
    server.run()


if __name__ == "__main__":
    main()
