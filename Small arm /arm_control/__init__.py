#!/usr/bin/env python3
"""
Arm Control Package

Provides unified interface for controlling the Piper arm via:
- Simulation (PyBullet)
- Hardware (CAN bus via Piper SDK)
- Signal logging/export
- Synchronized video playback with arm dancing

Note: CAN bus requires Linux with SocketCAN. On macOS, the hardware
controller operates in "dry run" mode (commands logged but not sent).
"""

from .controller import ArmController, ArmControllerBase, ArmState, ArmStatus
from .hardware_controller import HardwareArmController
from .simulation_controller import SimulationArmController
from .remote_controller import RemoteArmController
from .signal_bridge import SignalBridge, JointSignal
from .can_setup import setup_can_interface, check_can_status
from .logging_config import setup_logging, Loggers

__all__ = [
    'ArmController',
    'ArmControllerBase',
    'ArmState',
    'ArmStatus',
    'HardwareArmController',
    'SimulationArmController',
    'RemoteArmController',
    'SignalBridge',
    'JointSignal',
    'setup_can_interface',
    'check_can_status',
    'setup_logging',
    'Loggers',
]
