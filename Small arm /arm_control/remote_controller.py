#!/usr/bin/env python3
"""
Remote Arm Controller via SSH

Sends joint commands from local machine (macOS) to remote Raspberry Pi
over SSH, where the Pi controls the robot via CAN bus.

Architecture:
    [macOS] Video + Pose Detection → SSH → [Raspberry Pi] → CAN Bus → Robot
"""

import paramiko
import json
import time
import threading
from typing import List, Optional
from queue import Queue, Empty
import socket

from .controller import ArmControllerBase, ArmState, ArmStatus
from .logging_config import Loggers


class RemoteArmController(ArmControllerBase):
    """
    Remote arm controller that sends commands via SSH to a Raspberry Pi.
    
    The Pi runs a server (arm_server.py) that receives commands and
    forwards them to the real hardware via CAN bus.
    """

    def __init__(self,
                 host: str,
                 port: int = 22,
                 username: str = "pi",
                 password: Optional[str] = None,
                 key_filename: Optional[str] = None,
                 command_port: int = 5555):
        """
        Initialize remote controller.

        Args:
            host: Raspberry Pi hostname or IP address
            port: SSH port (default 22)
            username: SSH username (default "pi")
            password: SSH password (optional if using key)
            key_filename: Path to SSH private key (optional)
            command_port: Port for command socket on Pi (default 5555)
        """
        super().__init__()
        self.log = Loggers.hardware()

        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.key_filename = key_filename
        self.command_port = command_port

        self._ssh_client: Optional[paramiko.SSHClient] = None
        self._command_socket: Optional[socket.socket] = None
        self._connected = False
        self._command_count = 0
        self._response_queue: Queue = Queue()
        self._response_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.log.info(f"RemoteArmController initialized")
        self.log.info(f"  Target: {username}@{host}:{port}")
        self.log.info(f"  Command port: {command_port}")

    def connect(self) -> bool:
        """
        Connect to Raspberry Pi via SSH and establish command socket.

        Returns:
            True if connection successful
        """
        self.log.info(f"Connecting to Raspberry Pi at {self.host}...")

        try:
            # Establish SSH connection
            self._ssh_client = paramiko.SSHClient()
            self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            self.log.debug(f"SSH: Connecting to {self.username}@{self.host}:{self.port}")

            if self.key_filename:
                self._ssh_client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    key_filename=self.key_filename,
                    timeout=10
                )
            else:
                self._ssh_client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    timeout=10
                )

            self.log.info("SSH connection established")

            # Check if arm server is running on Pi
            stdin, stdout, stderr = self._ssh_client.exec_command(
                "pgrep -f 'arm_server.py' || echo 'NOT_RUNNING'"
            )
            output = stdout.read().decode().strip()

            if output == 'NOT_RUNNING':
                self.log.warning("Arm server not running on Pi - attempting to start...")
                if not self._start_arm_server():
                    self.log.error("Failed to start arm server on Pi")
                    return False

            # Connect command socket
            self._command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._command_socket.settimeout(5.0)
            self._command_socket.connect((self.host, self.command_port))
            self.log.info(f"Command socket connected on port {self.command_port}")

            # Start response listener
            self._stop_event.clear()
            self._response_thread = threading.Thread(target=self._response_listener, daemon=True)
            self._response_thread.start()

            self._connected = True
            self._state = ArmState.DISABLED

            self.log.info("Remote connection established successfully")
            return True

        except paramiko.AuthenticationException:
            self.log.error("SSH authentication failed")
            self.log.error("Check username/password or SSH key")
            return False
        
        except paramiko.SSHException as e:
            self.log.error(f"SSH error: {e}")
            return False

        except socket.timeout:
            self.log.error(f"Connection timeout to {self.host}")
            self.log.error("Check that Pi is powered on and reachable")
            return False

        except ConnectionRefusedError:
            self.log.error(f"Connection refused on port {self.command_port}")
            self.log.error("Arm server may not be running on Pi")
            return False

        except Exception as e:
            self.log.error(f"Connection failed: {e}")
            return False

    def _start_arm_server(self) -> bool:
        """Start arm server on Raspberry Pi."""
        try:
            self.log.info("Starting arm server on Pi...")
            
            # Start server in background
            command = (
                "cd ~/Robot\\ Rave\\ -\\ Bodhisattva/Small\\ arm && "
                "source venv/bin/activate && "
                "nohup python arm_control/arm_server.py > /tmp/arm_server.log 2>&1 &"
            )
            
            stdin, stdout, stderr = self._ssh_client.exec_command(command)
            time.sleep(2)  # Wait for server to start

            # Check if server is now running
            stdin, stdout, stderr = self._ssh_client.exec_command("pgrep -f 'arm_server.py'")
            output = stdout.read().decode().strip()

            if output:
                self.log.info(f"Arm server started (PID: {output})")
                return True
            else:
                self.log.error("Arm server failed to start")
                return False

        except Exception as e:
            self.log.error(f"Failed to start arm server: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Raspberry Pi."""
        self.log.info("Disconnecting from Raspberry Pi...")

        # Stop response thread
        self._stop_event.set()
        if self._response_thread:
            self._response_thread.join(timeout=2.0)

        # Close command socket
        if self._command_socket:
            try:
                self._send_command({"command": "disconnect"})
                self._command_socket.close()
            except Exception:
                pass
            self._command_socket = None

        # Close SSH connection
        if self._ssh_client:
            self._ssh_client.close()
            self._ssh_client = None

        self._connected = False
        self._state = ArmState.DISCONNECTED
        self.log.info("Disconnected from Raspberry Pi")

    def enable(self) -> bool:
        """
        Enable the arm motors on the Pi.

        Returns:
            True when enabled
        """
        self.log.info("Enabling arm motors on Pi...")

        if not self._connected:
            self.log.error("Not connected to Pi")
            return False

        response = self._send_command({"command": "enable"})
        
        if response and response.get("success"):
            self._state = ArmState.ENABLED
            self.log.info("Arm enabled on Pi")
            return True
        else:
            error = response.get("error", "Unknown error") if response else "No response"
            self.log.error(f"Enable failed: {error}")
            return False

    def disable(self) -> None:
        """Disable the arm motors."""
        self.log.info("Disabling arm motors on Pi...")

        if self._connected:
            self._send_command({"command": "disable"})

        self._state = ArmState.DISABLED
        self.log.info("Arm disabled")

    def set_joint_angles(self, angles: List[float], speed_percent: int = 50) -> bool:
        """
        Send joint angle command to Pi.

        Args:
            angles: List of 6 joint angles in radians
            speed_percent: Movement speed percentage

        Returns:
            True if command sent successfully
        """
        self._command_count += 1

        if not self._connected:
            self.log.warning(f"CMD#{self._command_count}: Not connected to Pi")
            return False

        if not self.is_ready:
            self.log.warning(f"CMD#{self._command_count}: Arm not enabled")
            return False

        # Validate and clamp
        valid, msg = self.validate_angles(angles)
        if not valid:
            self.log.warning(f"CMD#{self._command_count}: {msg}")

        clamped = self.clamp_angles(angles)
        self._last_command = clamped

        # Send command (non-blocking)
        try:
            command = {
                "command": "set_joints",
                "angles": clamped,
                "speed": speed_percent,
                "id": self._command_count
            }
            
            self._send_command_async(command)
            return True

        except Exception as e:
            self.log.error(f"CMD#{self._command_count}: Failed to send: {e}")
            return False

    def _send_command(self, command: dict, timeout: float = 2.0) -> Optional[dict]:
        """
        Send command and wait for response (blocking).

        Args:
            command: Command dictionary
            timeout: Response timeout in seconds

        Returns:
            Response dictionary or None
        """
        if not self._command_socket:
            return None

        try:
            # Send command
            message = json.dumps(command) + "\n"
            self._command_socket.sendall(message.encode())

            # Wait for response
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = self._response_queue.get(timeout=0.1)
                    return response
                except Empty:
                    continue

            self.log.warning(f"Command timeout: {command.get('command', 'unknown')}")
            return None

        except Exception as e:
            self.log.error(f"Send command error: {e}")
            return None

    def _send_command_async(self, command: dict) -> None:
        """
        Send command without waiting for response (non-blocking).
        Used for high-frequency joint commands.

        Args:
            command: Command dictionary
        """
        if not self._command_socket:
            return

        try:
            message = json.dumps(command) + "\n"
            self._command_socket.sendall(message.encode())
        except Exception as e:
            self.log.error(f"Send command async error: {e}")

    def _response_listener(self) -> None:
        """Listen for responses from Pi (runs in thread)."""
        self.log.debug("Response listener started")

        buffer = ""
        while not self._stop_event.is_set():
            try:
                data = self._command_socket.recv(4096).decode()
                if not data:
                    break

                buffer += data

                # Process complete messages (newline-delimited JSON)
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        try:
                            response = json.loads(line)
                            self._response_queue.put(response)
                        except json.JSONDecodeError:
                            self.log.warning(f"Invalid JSON response: {line}")

            except socket.timeout:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    self.log.error(f"Response listener error: {e}")
                break

        self.log.debug("Response listener stopped")

    def get_status(self) -> ArmStatus:
        """Get current arm status from Pi."""
        if not self._connected:
            return ArmStatus(
                state=self._state,
                joint_angles=[0.0] * 6,
                timestamp=time.time()
            )

        response = self._send_command({"command": "get_status"}, timeout=1.0)

        if response and response.get("success"):
            data = response.get("data", {})
            return ArmStatus(
                state=self._state,
                joint_angles=data.get("joint_angles", [0.0] * 6),
                timestamp=time.time()
            )
        else:
            return ArmStatus(
                state=self._state,
                joint_angles=self._last_command if self._last_command else [0.0] * 6,
                timestamp=time.time()
            )

    def get_joint_angles(self) -> List[float]:
        """Get current joint angles."""
        status = self.get_status()
        return status.joint_angles

    def set_gripper(self, position: float) -> bool:
        """Set gripper position."""
        if not self._connected:
            return False

        response = self._send_command({
            "command": "set_gripper",
            "position": position
        })

        return response and response.get("success", False)

    def emergency_stop(self) -> None:
        """Trigger emergency stop on Pi."""
        self.log.critical("!!! EMERGENCY STOP !!!")
        
        self._emergency_stop = True
        self._state = ArmState.EMERGENCY_STOP

        if self._connected:
            self._send_command({"command": "emergency_stop"})

    def get_command_count(self) -> int:
        """Get total number of commands sent."""
        return self._command_count
