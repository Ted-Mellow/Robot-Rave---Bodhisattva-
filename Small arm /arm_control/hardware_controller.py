#!/usr/bin/env python3
"""
Hardware Arm Controller

Controls the real Piper arm via CAN bus using the Piper SDK.
"""

import time
import sys
from typing import List, Optional, Tuple
import numpy as np

from .controller import ArmControllerBase, ArmState, ArmStatus
from .logging_config import Loggers


class HardwareArmController(ArmControllerBase):
    """
    Arm controller for real Piper hardware via CAN bus.

    Requires:
    - CAN interface configured (can0)
    - Piper SDK installed (piper-sdk)
    - Physical arm connected and powered

    Note: CAN bus requires Linux with SocketCAN. On macOS, this controller
    will operate in a "dry run" logging mode unless connected to a Linux
    system with proper CAN hardware.
    """

    # Conversion factor: radians to 0.001 degrees (SDK unit)
    RAD_TO_MILLI_DEG = 1000.0 * 180.0 / np.pi  # 57295.7795

    def __init__(self, can_interface: str = "can0", dry_run: bool = False):
        """
        Initialize hardware controller.

        Args:
            can_interface: CAN interface name (default "can0")
            dry_run: If True, log commands without sending to hardware
        """
        super().__init__()
        self.log = Loggers.hardware()
        self.can_log = Loggers.can_bus()

        self.can_interface = can_interface
        self._piper = None
        self._enable_timeout = 10.0  # seconds
        self._control_rate = 0.005  # 5ms control loop (200Hz)
        self._speed_percent = 50
        self._command_count = 0

        # Auto-detect if we should use dry run mode (macOS)
        self._dry_run = dry_run or sys.platform == 'darwin'

        if self._dry_run:
            self.log.warning("DRY RUN MODE: Commands will be logged but not sent to hardware")
            self.log.warning(f"Platform: {sys.platform} - CAN bus requires Linux with SocketCAN")

        self.log.info(f"HardwareArmController initialized")
        self.log.info(f"  CAN interface: {can_interface}")
        self.log.info(f"  Dry run mode: {self._dry_run}")

    def connect(self) -> bool:
        """
        Connect to the Piper arm via CAN bus.

        Returns:
            True if connection successful
        """
        self.log.info(f"Connecting to Piper arm on {self.can_interface}...")
        self.can_log.info(f"CAN CONNECT: interface={self.can_interface}")

        if self._dry_run:
            self._state = ArmState.DISABLED
            self.log.info("DRY RUN: Simulating successful connection")
            self.can_log.info("CAN CONNECT: [DRY RUN] Connection simulated")
            return True

        try:
            from piper_sdk import C_PiperInterface_V2

            self._state = ArmState.CONNECTING
            self.can_log.debug(f"CAN INIT: Creating C_PiperInterface_V2({self.can_interface})")

            # Initialize SDK - can_auto_init must be True for actual CAN connection
            self._piper = C_PiperInterface_V2(
                can_name=self.can_interface,
                can_auto_init=True,
                judge_flag=True
            )
            
            self.can_log.debug("CAN INIT: Calling ConnectPort()")
            self._piper.ConnectPort()

            self._state = ArmState.DISABLED
            self.log.info(f"Connected successfully to {self.can_interface}")
            self.can_log.info(f"CAN CONNECT: SUCCESS on {self.can_interface}")
            return True

        except ImportError as e:
            self.log.error("piper_sdk not installed")
            self.log.error("Install with: pip install piper-sdk")
            self.can_log.error(f"CAN CONNECT: FAILED - ImportError: {e}")
            self._state = ArmState.ERROR
            return False

        except ConnectionError as e:
            self.log.error(f"CAN connection failed: {e}")
            self.log.error(f"This usually means:")
            self.log.error(f"  - CAN interface '{self.can_interface}' does not exist")
            self.log.error(f"  - On macOS: SocketCAN not available (use Linux)")
            self.log.error(f"  - On Linux: Run 'sudo ip link set {self.can_interface} up'")
            self.can_log.error(f"CAN CONNECT: FAILED - ConnectionError: {e}")
            self._state = ArmState.ERROR
            return False

        except Exception as e:
            self.log.error(f"Connection failed: {e}")
            self.can_log.error(f"CAN CONNECT: FAILED - {type(e).__name__}: {e}")
            self._state = ArmState.ERROR
            return False

    def disconnect(self) -> None:
        """Disconnect from the arm."""
        self.log.info("Disconnecting from arm...")
        self.can_log.info("CAN DISCONNECT: Initiating disconnect")

        if self._piper is not None:
            try:
                self.disable()
            except Exception as e:
                self.log.warning(f"Error during disable on disconnect: {e}")
            self._piper = None

        self._state = ArmState.DISCONNECTED
        self.log.info("Disconnected")
        self.can_log.info("CAN DISCONNECT: Complete")

    def enable(self) -> bool:
        """
        Enable the arm motors and gravity compensation.

        This must complete before sending joint commands.
        Blocks until all motors report enabled or timeout.

        Returns:
            True when fully enabled
        """
        self.log.info("Enabling arm motors...")
        self.can_log.info("CAN ENABLE: Starting motor enable sequence")

        if self._dry_run:
            self._state = ArmState.ENABLED
            self.log.info("DRY RUN: Motors enabled (simulated)")
            self.can_log.info("CAN ENABLE: [DRY RUN] Enable simulated")
            return True

        if self._piper is None:
            self.log.error("Not connected - call connect() first")
            return False

        start_time = time.time()
        attempt = 0

        while (time.time() - start_time) < self._enable_timeout:
            attempt += 1
            try:
                self.can_log.debug(f"CAN ENABLE: Attempt {attempt} - calling EnablePiper()")
                if self._piper.EnablePiper():
                    self._state = ArmState.ENABLED
                    elapsed = time.time() - start_time
                    self.log.info(f"Arm enabled after {elapsed:.2f}s ({attempt} attempts)")
                    self.can_log.info(f"CAN ENABLE: SUCCESS after {attempt} attempts")
                    return True
            except Exception as e:
                self.log.error(f"Enable error on attempt {attempt}: {e}")
                self.can_log.error(f"CAN ENABLE: ERROR on attempt {attempt} - {e}")
                break

            time.sleep(0.01)

        self.log.error(f"Enable timeout after {self._enable_timeout}s")
        self.log.error("Check: arm power, CAN connection, emergency stop")
        self.can_log.error(f"CAN ENABLE: TIMEOUT after {attempt} attempts")
        return False

    def disable(self) -> None:
        """Disable the arm motors."""
        self.log.info("Disabling arm motors...")
        self.can_log.info("CAN DISABLE: Sending disable command")

        if self._dry_run:
            self._state = ArmState.DISABLED
            self.log.info("DRY RUN: Motors disabled (simulated)")
            return

        if self._piper is not None:
            try:
                self._piper.DisablePiper()
                self.can_log.info("CAN DISABLE: DisablePiper() sent")
            except Exception as e:
                self.log.error(f"Disable error: {e}")
                self.can_log.error(f"CAN DISABLE: ERROR - {e}")

        self._state = ArmState.DISABLED
        self.log.info("Arm disabled")

    def set_joint_angles(self, angles: List[float], speed_percent: int = 50) -> bool:
        """
        Send joint angle command to hardware.

        Args:
            angles: List of 6 joint angles in radians
            speed_percent: Movement speed as percentage (0-100)

        Returns:
            True if command sent successfully
        """
        self._command_count += 1
        cmd_id = self._command_count

        if not self.is_ready:
            self.log.warning(f"CMD#{cmd_id}: Not ready - state={self._state}")
            return False

        if self._emergency_stop:
            self.log.warning(f"CMD#{cmd_id}: Emergency stop active - command rejected")
            return False

        # Validate and clamp
        valid, msg = self.validate_angles(angles)
        if not valid:
            self.log.warning(f"CMD#{cmd_id}: Angle validation warning: {msg}")

        clamped = self.clamp_angles(angles)
        self._last_command = clamped
        self._speed_percent = max(0, min(100, speed_percent))

        # Convert radians to 0.001 degrees (SDK unit)
        j_milli = [int(round(a * self.RAD_TO_MILLI_DEG)) for a in clamped]

        # Log the command
        angles_deg = [a * 57.3 for a in clamped]
        self.log.debug(
            f"CMD#{cmd_id}: angles_rad={[f'{a:.3f}' for a in clamped]} "
            f"angles_deg={[f'{a:.1f}' for a in angles_deg]} speed={self._speed_percent}%"
        )
        self.can_log.debug(
            f"CAN TX CMD#{cmd_id}: JointCtrl({j_milli[0]}, {j_milli[1]}, {j_milli[2]}, "
            f"{j_milli[3]}, {j_milli[4]}, {j_milli[5]}) @ {self._speed_percent}%"
        )

        if self._dry_run:
            self.can_log.info(f"CAN TX CMD#{cmd_id}: [DRY RUN] Command logged but not sent")
            return True

        if self._piper is None:
            self.log.error(f"CMD#{cmd_id}: No piper interface")
            return False

        try:
            # Set motion mode: CAN control (0x01), MOVE J (0x01), speed %
            self._piper.MotionCtrl_2(0x01, 0x01, self._speed_percent, 0x00)
            self.can_log.debug(f"CAN TX: MotionCtrl_2(0x01, 0x01, {self._speed_percent}, 0x00)")

            # Send joint command
            self._piper.JointCtrl(j_milli[0], j_milli[1], j_milli[2],
                                   j_milli[3], j_milli[4], j_milli[5])
            self.can_log.debug(f"CAN TX: JointCtrl sent successfully")

            return True

        except Exception as e:
            self.log.error(f"CMD#{cmd_id}: Joint command error: {e}")
            self.can_log.error(f"CAN TX CMD#{cmd_id}: FAILED - {e}")
            self._state = ArmState.ERROR
            return False

    def get_status(self) -> ArmStatus:
        """Get current arm status from hardware."""
        if self._dry_run:
            # Return last commanded angles in dry run mode
            return ArmStatus(
                state=self._state,
                joint_angles=self._last_command if self._last_command else [0.0] * 6,
                timestamp=time.time()
            )

        if self._piper is None:
            return ArmStatus(
                state=self._state,
                joint_angles=[0.0] * 6,
                timestamp=time.time()
            )

        try:
            arm_status = self._piper.GetArmStatus()
            arm_joint = self._piper.GetArmJointMsgs()

            # Convert from SDK format to radians
            angles = [0.0] * 6
            if hasattr(arm_joint, 'joint_state'):
                js = arm_joint.joint_state
                if hasattr(js, 'joint_1'):
                    angles[0] = js.joint_1 / self.RAD_TO_MILLI_DEG
                if hasattr(js, 'joint_2'):
                    angles[1] = js.joint_2 / self.RAD_TO_MILLI_DEG
                if hasattr(js, 'joint_3'):
                    angles[2] = js.joint_3 / self.RAD_TO_MILLI_DEG
                if hasattr(js, 'joint_4'):
                    angles[3] = js.joint_4 / self.RAD_TO_MILLI_DEG
                if hasattr(js, 'joint_5'):
                    angles[4] = js.joint_5 / self.RAD_TO_MILLI_DEG
                if hasattr(js, 'joint_6'):
                    angles[5] = js.joint_6 / self.RAD_TO_MILLI_DEG

            self.can_log.debug(f"CAN RX: Joint angles = {[f'{a:.3f}' for a in angles]}")

            return ArmStatus(
                state=self._state,
                joint_angles=angles,
                timestamp=arm_status.time_stamp if hasattr(arm_status, 'time_stamp') else time.time()
            )

        except Exception as e:
            self.log.error(f"Get status error: {e}")
            self.can_log.error(f"CAN RX: GetArmStatus FAILED - {e}")
            return ArmStatus(
                state=ArmState.ERROR,
                joint_angles=[0.0] * 6,
                timestamp=time.time(),
                error_message=str(e)
            )

    def get_joint_angles(self) -> List[float]:
        """Get current joint angles in radians."""
        status = self.get_status()
        return status.joint_angles

    def set_gripper(self, position: float) -> bool:
        """
        Set gripper position.

        Args:
            position: 0.0 = fully closed, 1.0 = fully open

        Returns:
            True if command sent
        """
        gripper_pos = int(position * 70000)
        gripper_pos = max(0, min(70000, gripper_pos))

        self.log.debug(f"Gripper: position={position:.2f} -> {gripper_pos} (0.001mm units)")
        self.can_log.debug(f"CAN TX: GripperCtrl({gripper_pos}, 1000, 0x01, 0)")

        if self._dry_run:
            self.can_log.info("CAN TX: [DRY RUN] Gripper command logged")
            return True

        if self._piper is None:
            return False

        try:
            self._piper.GripperCtrl(gripper_pos, 1000, 0x01, 0)
            return True

        except Exception as e:
            self.log.error(f"Gripper error: {e}")
            self.can_log.error(f"CAN TX: GripperCtrl FAILED - {e}")
            return False

    def emergency_stop(self) -> None:
        """Trigger emergency stop - immediately disable motors."""
        self.log.critical("!!! EMERGENCY STOP TRIGGERED !!!")
        self.can_log.critical("CAN EMERGENCY: Immediate disable")

        self._emergency_stop = True
        self._state = ArmState.EMERGENCY_STOP

        if self._piper is not None and not self._dry_run:
            try:
                self._piper.DisablePiper()
                self.can_log.info("CAN EMERGENCY: DisablePiper() sent")
            except Exception as e:
                self.can_log.error(f"CAN EMERGENCY: DisablePiper failed - {e}")

    def send_continuous(self, angles: List[float]) -> bool:
        """
        Send joint command for continuous control loop.

        Use this for high-frequency control (200Hz).
        Logs at DEBUG level to avoid spam.

        Args:
            angles: Joint angles in radians

        Returns:
            True if sent
        """
        if not self.is_ready:
            return False

        if self._emergency_stop:
            return False

        clamped = self.clamp_angles(angles)
        j_milli = [int(round(a * self.RAD_TO_MILLI_DEG)) for a in clamped]

        self.can_log.debug(
            f"CAN TX CONT: [{j_milli[0]}, {j_milli[1]}, {j_milli[2]}, "
            f"{j_milli[3]}, {j_milli[4]}, {j_milli[5]}]"
        )

        if self._dry_run:
            self._last_command = clamped
            return True

        if self._piper is None:
            return False

        try:
            self._piper.MotionCtrl_2(0x01, 0x01, self._speed_percent, 0x00)
            self._piper.JointCtrl(j_milli[0], j_milli[1], j_milli[2],
                                   j_milli[3], j_milli[4], j_milli[5])
            return True

        except Exception:
            return False

    def get_command_count(self) -> int:
        """Get total number of commands sent."""
        return self._command_count
