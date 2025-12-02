#!/usr/bin/env python
"""
Example subscriber for the TEP streaming server.

Connects to a streaming server and prints received messages.

Usage:
    # In one terminal, start the server:
    python -m tep.streaming

    # In another terminal, run this subscriber:
    python examples/streaming_subscriber.py

    # Or connect to a remote server:
    python examples/streaming_subscriber.py --host 192.168.1.100 --port 5555
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime

try:
    import zmq
except ImportError:
    print("pyzmq is required. Install with: pip install pyzmq")
    sys.exit(1)


def format_timestamp(iso_str: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except:
        return iso_str


def main():
    parser = argparse.ArgumentParser(description="TEP Streaming Subscriber")
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Only show events, not data messages")
    parser.add_argument("--stats-interval", type=int, default=100,
                        help="Show stats every N data messages (0=disable)")
    args = parser.parse_args()

    # Connect to server
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{args.host}:{args.port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages

    print(f"Connected to tcp://{args.host}:{args.port}")
    print("Waiting for messages... (Ctrl+C to quit)\n")

    # Statistics
    stats = defaultdict(int)
    current_run_id = None
    last_t = 0.0

    try:
        while True:
            msg = socket.recv_json()
            msg_type = msg.get("type", "unknown")
            stats[msg_type] += 1

            if msg_type == "run_start":
                current_run_id = msg["run_id"]
                fault = msg["scheduled_fault"]
                print(f"[RUN START] run={msg['run_id']} #{msg['run_number']}")
                print(f"  Scheduled: IDV({fault['idv']}) at t={fault['start_time']:.2f}h "
                      f"until t={fault['end_time']:.2f}h")
                if msg.get("previous_run_id"):
                    print(f"  Previous run: {msg['previous_run_id']}")
                print()

            elif msg_type == "data":
                last_t = msg["t"]
                if not args.quiet:
                    if args.stats_interval > 0 and stats["data"] % args.stats_interval == 0:
                        xmeas = msg["xmeas"]
                        print(f"[DATA] t={msg['t']:.4f}h step={msg['step']} "
                              f"fault={'IDV('+str(msg['fault'])+')' if msg['fault_active'] else 'none'}")
                        print(f"  Reactor: P={xmeas[6]:.1f}kPa L={xmeas[7]:.1f}% T={xmeas[8]:.1f}C")
                        print(f"  Separator: L={xmeas[11]:.1f}%  Stripper: L={xmeas[14]:.1f}%")
                        print()

            elif msg_type == "fault_on":
                print(f"[FAULT ON] IDV({msg['fault']}) at t={msg['t']:.4f}h")
                print(f"  {msg['fault_name']}")
                print()

            elif msg_type == "fault_off":
                print(f"[FAULT OFF] IDV({msg['fault']}) at t={msg['t']:.4f}h")
                print(f"  Duration: {msg['fault_duration']:.4f}h ({msg['fault_duration']*3600:.0f}s)")
                print()

            elif msg_type == "restart":
                print(f"[RESTART] Safety violation at t={msg['t']:.4f}h")
                print(f"  Reason: {msg['reason']}")
                print(f"  Limit: {msg['limit']}, Actual: {msg['actual']:.2f}")
                if msg.get("fault"):
                    print(f"  Active fault: IDV({msg['fault']})")
                print()

            else:
                print(f"[{msg_type.upper()}] {msg}")
                print()

    except KeyboardInterrupt:
        print("\n\nSubscriber statistics:")
        for msg_type, count in sorted(stats.items()):
            print(f"  {msg_type}: {count}")
        print(f"\nLast simulation time: {last_t:.4f}h ({last_t*3600:.0f}s)")

    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
