#!/usr/bin/env python3
"""
Arm Control Server (Raspberry Pi)

Runs on Raspberry Pi with CAN bus connected. Receives commands via
socket from remote machine (macOS) and controls the robot arm.

Usage:
    python arm_server.py [--port 5555] [--interface can0]
"""

import socket
import json
import time
import argparse
import signal
import sys
import threading
from typing import Optional

# Add parent directory to path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm_control import HardwareArmController
from arm_control.logging_config import setup_logging, Loggers


class ArmServer:
    """
    Socket server that receives joint commands and controls the arm.
    
    Protocol: Newline-delimited JSON messages
    
    Commands:
        {"command": "enable"}
        {"command": "disable"}
        {"command": "set_joints", "angles": [j1, j2, j3, j4, j5, j6], "speed": 50}
        {"command": "set_gripper", "position": 0.5}
        {"command": "get_status"}
        {"command": "emergency_stop"}
        {"command": "disconnect"}
    
    Responses:
        {"success": true, "data": {...}}
        {"success": false, "error": "..."}
    """

    def __init__(self, port: int = 5555, can_interface: str = "can0"):
        """
        Initialize arm server.

        Args:
            port: Socket port to listen on
            can_interface: CAN interface name
        """
        self.log = Loggers.hardware()
        self.log.info("=" * 60)
        self.log.info("ARM CONTROL SERVER INITIALIZING")
        self.log.info("=" * 60)

        self.port = port
        self.can_interface = can_interface

        self.controller: Optional[HardwareArmController] = None
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.running = False
        
        self.command_count = 0
        self.last_command_time = 0

        self.log.info(f"Port: {port}")
        self.log.info(f"CAN interface: {can_interface}")

    def start(self) -> bool:
        """Start the server."""
        self.log.info("Starting arm control server...")

        # Initialize hardware controller
        self.log.info(f"Initializing hardware controller on {self.can_interface}...")
        self.controller = HardwareArmController(self.can_interface)

        if not self.controller.connect():
            self.log.error("Failed to connect to arm hardware")
            return False

        self.log.info("Hardware controller connected")

        # Create server socket
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)  # Allow periodic checks

            self.log.info(f"Server listening on port {self.port}")
            self.log.info("=" * 60)
            self.log.info("READY TO ACCEPT CONNECTIONS")
            self.log.info("=" * 60)

            self.running = True
            return True

        except OSError as e:
            self.log.error(f"Failed to create server socket: {e}")
            return False

    def run(self) -> None:
        """Main server loop."""
        try:
            while self.running:
                try:
                    # Accept connection (with timeout)
                    self.client_socket, addr = self.server_socket.accept()
                    self.log.info(f"Client connected from {addr}")

                    # Handle client
                    self._handle_client()

                except socket.timeout:
                    continue  # Check running flag
                except Exception as e:
                    if self.running:
                        self.log.error(f"Accept error: {e}")

        except KeyboardInterrupt:
            self.log.info("Interrupted by user")

        finally:
            self.shutdown()

    def _handle_client(self) -> None:
        """Handle connected client."""
        self.client_socket.settimeout(30.0)  # 30s timeout for commands
        buffer = ""

        try:
            while self.running:
                # Receive data
                data = self.client_socket.recv(4096).decode()
                if not data:
                    self.log.info("Client disconnected")
                    break

                buffer += data

                # Process complete messages (newline-delimited JSON)
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        try:
                            command = json.loads(line)
                            response = self._process_command(command)
                            self._send_response(response)
                        except json.JSONDecodeError as e:
                            self.log.warning(f"Invalid JSON: {line[:100]}")
                            self._send_response({
                                "success": False,
                                "error": f"Invalid JSON: {e}"
                            })

        except socket.timeout:
            self.log.warning("Client timeout - no commands received")
        except Exception as e:
            self.log.error(f"Client handler error: {e}")
        finally:
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None

    def _process_command(self, command: dict) -> dict:
        """
        Process a command and return response.

        Args:
            command: Command dictionary

        Returns:
            Response dictionary
        """
        self.command_count += 1
        self.last_command_time = time.time()

        cmd_type = command.get("command", "unknown")
        cmd_id = command.get("id", self.command_count)

        self.log.debug(f"CMD#{cmd_id}: {cmd_type}")

        try:
            if cmd_type == "enable":
                success = self.controller.enable()
                return {
                    "success": success,
                    "error": None if success else "Enable failed"
                }

            elif cmd_type == "disable":
                self.controller.disable()
                return {"success": True}

            elif cmd_type == "set_joints":
                angles = command.get("angles", [])
                speed = command.get("speed", 50)

                if len(angles) != 6:
                    return {
                        "success": False,
                        "error": f"Expected 6 angles, got {len(angles)}"
                    }

                success = self.controller.set_joint_angles(angles, speed)
                return {"success": success}

            elif cmd_type == "set_gripper":
                position = command.get("position", 0.0)
                success = self.controller.set_gripper(position)
                return {"success": success}

            elif cmd_type == "get_status":
                status = self.controller.get_status()
                return {
                    "success": True,
                    "data": {
                        "state": status.state.value,
                        "joint_angles": status.joint_angles,
                        "timestamp": status.timestamp
                    }
                }

            elif cmd_type == "emergency_stop":
                self.controller.emergency_stop()
                return {"success": True}

            elif cmd_type == "disconnect":
                self.log.info("Client requested disconnect")
                return {"success": True}

            else:
                return {
                    "success": False,
                    "error": f"Unknown command: {cmd_type}"
                }

        except Exception as e:
            self.log.error(f"CMD#{cmd_id} error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _send_response(self, response: dict) -> None:
        """Send response to client."""
        if not self.client_socket:
            return

        try:
            message = json.dumps(response) + "\n"
            self.client_socket.sendall(message.encode())
        except Exception as e:
            self.log.error(f"Send response error: {e}")

    def shutdown(self) -> None:
        """Shutdown the server."""
        self.log.info("Shutting down arm server...")

        self.running = False

        # Disconnect client
        if self.client_socket:
            try:
                self.client_socket.close()
            except Exception:
                pass
            self.client_socket = None

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None

        # Disable and disconnect controller
        if self.controller:
            try:
                self.controller.disable()
                self.controller.disconnect()
            except Exception as e:
                self.log.warning(f"Controller shutdown error: {e}")
            self.controller = None

        self.log.info("=" * 60)
        self.log.info("ARM SERVER SHUTDOWN COMPLETE")
        self.log.info(f"Total commands processed: {self.command_count}")
        self.log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Arm Control Server (Raspberry Pi)")
    parser.add_argument('--port', '-p', type=int, default=5555,
                        help='Socket port (default: 5555)')
    parser.add_argument('--interface', '-i', default='can0',
                        help='CAN interface (default: can0)')
    args = parser.parse_args()

    # Set up logging
    setup_logging("arm_server", level=10)  # DEBUG level

    # Create and start server
    server = ArmServer(port=args.port, can_interface=args.interface)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nShutdown signal received")
        server.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start server
    if server.start():
        server.run()
    else:
        print("Failed to start server", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
