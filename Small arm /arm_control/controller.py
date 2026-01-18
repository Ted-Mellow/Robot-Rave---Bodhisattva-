#!/usr/bin/env python3
"""
Abstract Arm Controller Interface

Defines the common interface for controlling the Piper arm,
whether through simulation or hardware.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np


class ArmState(Enum):
    """Arm operational states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    DISABLED = "disabled"
    ENABLED = "enabled"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class JointLimits:
    """Joint angle limits in radians"""
    # From Piper URDF and SDK documentation
    J1: Tuple[float, float] = (-2.6179, 2.6179)    # ±150°
    J2: Tuple[float, float] = (0.0, 3.14)          # 0→180°
    J3: Tuple[float, float] = (-2.967, 0.0)        # -170→0°
    J4: Tuple[float, float] = (-1.745, 1.745)      # ±100°
    J5: Tuple[float, float] = (-1.22, 1.22)        # ±70°
    J6: Tuple[float, float] = (-2.094, 2.094)      # ±120°

    def get_limits(self, joint_idx: int) -> Tuple[float, float]:
        """Get limits for joint index (0-5)"""
        limits = [self.J1, self.J2, self.J3, self.J4, self.J5, self.J6]
        return limits[joint_idx]

    def clamp(self, angles: List[float]) -> List[float]:
        """Clamp angles to joint limits"""
        clamped = []
        for i, angle in enumerate(angles):
            lo, hi = self.get_limits(i)
            clamped.append(max(lo, min(hi, angle)))
        return clamped


@dataclass
class ArmStatus:
    """Current arm status"""
    state: ArmState
    joint_angles: List[float]  # radians
    joint_velocities: Optional[List[float]] = None  # rad/s
    end_effector_pos: Optional[Tuple[float, float, float]] = None  # xyz meters
    end_effector_orn: Optional[Tuple[float, float, float, float]] = None  # quaternion
    gripper_position: float = 0.0  # 0=closed, 1=open
    timestamp: float = 0.0
    error_message: Optional[str] = None


class ArmControllerBase(ABC):
    """
    Abstract base class for arm controllers.

    Provides a unified interface for both simulation and hardware control.
    All angles are in radians, positions in meters.
    """

    JOINT_LIMITS = JointLimits()
    RAD_TO_DEG = 180.0 / np.pi
    DEG_TO_RAD = np.pi / 180.0

    def __init__(self):
        self._state = ArmState.DISCONNECTED
        self._last_command: Optional[List[float]] = None
        self._emergency_stop = False

    @property
    def state(self) -> ArmState:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state == ArmState.ENABLED

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the arm (CAN bus or simulation).
        Returns True if successful.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the arm safely."""
        pass

    @abstractmethod
    def enable(self) -> bool:
        """
        Enable the arm motors / gravity compensation.
        Must be called before sending commands.
        Returns True when fully enabled.
        """
        pass

    @abstractmethod
    def disable(self) -> None:
        """Disable the arm motors."""
        pass

    @abstractmethod
    def set_joint_angles(self, angles: List[float], speed_percent: int = 50) -> bool:
        """
        Set target joint angles.

        Args:
            angles: List of 6 joint angles in radians [J1, J2, J3, J4, J5, J6]
            speed_percent: Movement speed as percentage (0-100)

        Returns:
            True if command was accepted
        """
        pass

    @abstractmethod
    def get_status(self) -> ArmStatus:
        """Get current arm status including joint angles."""
        pass

    @abstractmethod
    def get_joint_angles(self) -> List[float]:
        """Get current joint angles in radians."""
        pass

    def set_gripper(self, position: float) -> bool:
        """
        Set gripper position.

        Args:
            position: 0.0 = fully closed, 1.0 = fully open

        Returns:
            True if command was accepted
        """
        # Default implementation - override in subclasses
        return False

    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        self._emergency_stop = True
        self._state = ArmState.EMERGENCY_STOP
        self.disable()

    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop. Returns True if successful."""
        if self._emergency_stop:
            self._emergency_stop = False
            self._state = ArmState.DISABLED
            return True
        return False

    def validate_angles(self, angles: List[float]) -> Tuple[bool, str]:
        """
        Validate joint angles are within limits.

        Returns:
            (is_valid, error_message)
        """
        if len(angles) != 6:
            return False, f"Expected 6 angles, got {len(angles)}"

        for i, angle in enumerate(angles):
            lo, hi = self.JOINT_LIMITS.get_limits(i)
            if angle < lo or angle > hi:
                return False, f"Joint {i+1} angle {angle:.3f} rad outside limits [{lo:.3f}, {hi:.3f}]"

        return True, ""

    def clamp_angles(self, angles: List[float]) -> List[float]:
        """Clamp angles to joint limits."""
        return self.JOINT_LIMITS.clamp(angles)


class ArmController:
    """
    Factory class for creating arm controllers.

    Usage:
        # For simulation
        controller = ArmController.create_simulation()

        # For hardware (local)
        controller = ArmController.create_hardware("can0")

        # For remote (via SSH to Raspberry Pi)
        controller = ArmController.create_remote("192.168.1.100", username="pi", password="raspberry")
    """

    @staticmethod
    def create_simulation(gui: bool = True) -> 'SimulationArmController':
        """Create a simulation controller."""
        from .simulation_controller import SimulationArmController
        return SimulationArmController(gui=gui)

    @staticmethod
    def create_hardware(can_interface: str = "can0") -> 'HardwareArmController':
        """Create a hardware controller for real arm."""
        from .hardware_controller import HardwareArmController
        return HardwareArmController(can_interface=can_interface)

    @staticmethod
    def create_remote(host: str, 
                     username: str = "pi",
                     password: str = None,
                     key_filename: str = None,
                     port: int = 22,
                     command_port: int = 5555) -> 'RemoteArmController':
        """
        Create a remote controller that connects via SSH.

        Args:
            host: Raspberry Pi hostname or IP address
            username: SSH username (default "pi")
            password: SSH password (optional if using key)
            key_filename: Path to SSH private key (optional)
            port: SSH port (default 22)
            command_port: Command socket port (default 5555)

        Returns:
            RemoteArmController instance
        """
        from .remote_controller import RemoteArmController
        return RemoteArmController(
            host=host,
            port=port,
            username=username,
            password=password,
            key_filename=key_filename,
            command_port=command_port
        )

    @staticmethod
    def create_signal_logger() -> 'SignalBridge':
        """Create a signal bridge for logging/export without arm."""
        from .signal_bridge import SignalBridge
        return SignalBridge()
