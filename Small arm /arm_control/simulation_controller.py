#!/usr/bin/env python3
"""
Simulation Arm Controller

Wraps the PyBullet simulation to provide the standard ArmController interface.
"""

import time
from typing import List, Optional, Tuple
import numpy as np

from .controller import ArmControllerBase, ArmState, ArmStatus


class SimulationArmController(ArmControllerBase):
    """
    Arm controller using PyBullet simulation.

    Provides the same interface as hardware controller for testing
    trajectories before deploying to real arm.
    """

    def __init__(self, gui: bool = True, urdf_path: Optional[str] = None):
        """
        Initialize simulation controller.

        Args:
            gui: Show PyBullet GUI window
            urdf_path: Path to URDF file (auto-detected if None)
        """
        super().__init__()
        self.gui = gui
        self.urdf_path = urdf_path
        self._sim = None
        self._current_angles = [0.0] * 6
        self._target_angles = [0.0] * 6

    def connect(self) -> bool:
        """Initialize PyBullet simulation."""
        try:
            # Import here to avoid loading PyBullet unless needed
            import sys
            import os

            # Add simulation directory to path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            sim_dir = os.path.join(parent_dir, "simulation")
            if sim_dir not in sys.path:
                sys.path.insert(0, sim_dir)

            from piper_simultion_corrected import PiperSimulation

            self._state = ArmState.CONNECTING
            self._sim = PiperSimulation(urdf_path=self.urdf_path, gui=self.gui)
            self._state = ArmState.DISABLED
            print("[SIM] Connected to PyBullet simulation")
            return True

        except Exception as e:
            self._state = ArmState.ERROR
            print(f"[SIM] Failed to connect: {e}")
            return False

    def disconnect(self) -> None:
        """Close PyBullet simulation."""
        if self._sim is not None:
            try:
                self._sim.close()
            except Exception:
                pass
            self._sim = None
        self._state = ArmState.DISCONNECTED
        print("[SIM] Disconnected")

    def enable(self) -> bool:
        """
        Enable simulation (immediate in sim mode).
        """
        if self._sim is None:
            print("[SIM] Not connected - call connect() first")
            return False

        self._state = ArmState.ENABLED
        print("[SIM] Enabled")
        return True

    def disable(self) -> None:
        """Disable simulation."""
        self._state = ArmState.DISABLED
        print("[SIM] Disabled")

    def set_joint_angles(self, angles: List[float], speed_percent: int = 50) -> bool:
        """
        Set target joint angles in simulation.

        Args:
            angles: List of 6 joint angles in radians
            speed_percent: Ignored in simulation (instant positioning)

        Returns:
            True if command accepted
        """
        if not self.is_ready:
            print("[SIM] Not ready - call enable() first")
            return False

        if self._emergency_stop:
            print("[SIM] Emergency stop active")
            return False

        # Validate and clamp
        valid, msg = self.validate_angles(angles)
        if not valid:
            print(f"[SIM] Warning: {msg}")

        clamped = self.clamp_angles(angles)
        self._target_angles = clamped
        self._last_command = clamped

        # Send to simulation
        try:
            self._sim.set_joint_positions(clamped)
            return True
        except Exception as e:
            print(f"[SIM] Error setting joints: {e}")
            self._state = ArmState.ERROR
            return False

    def get_status(self) -> ArmStatus:
        """Get current simulation status."""
        if self._sim is None:
            return ArmStatus(
                state=self._state,
                joint_angles=[0.0] * 6,
                timestamp=time.time()
            )

        try:
            angles = self._sim.get_joint_positions()
            velocities = self._sim.get_joint_velocities()
            pos, orn = self._sim.get_end_effector_pose()

            return ArmStatus(
                state=self._state,
                joint_angles=angles,
                joint_velocities=velocities,
                end_effector_pos=pos,
                end_effector_orn=orn,
                timestamp=time.time()
            )
        except Exception as e:
            return ArmStatus(
                state=ArmState.ERROR,
                joint_angles=[0.0] * 6,
                timestamp=time.time(),
                error_message=str(e)
            )

    def get_joint_angles(self) -> List[float]:
        """Get current joint angles from simulation."""
        if self._sim is None:
            return [0.0] * 6
        try:
            return self._sim.get_joint_positions()
        except Exception:
            return [0.0] * 6

    def set_gripper(self, position: float) -> bool:
        """Set gripper position in simulation."""
        if self._sim is None:
            return False
        try:
            # Convert 0-1 to gripper opening (0.0-0.0425)
            opening = position * 0.0425
            self._sim.set_gripper(opening)
            return True
        except Exception:
            return False

    def step(self, steps: int = 1) -> None:
        """
        Step the simulation forward.

        Args:
            steps: Number of physics steps to run
        """
        if self._sim is not None:
            for _ in range(steps):
                try:
                    self._sim.step()
                except RuntimeError:
                    # GUI window closed
                    self._state = ArmState.DISCONNECTED
                    break

    def run_trajectory(self,
                       trajectory: List[Tuple[float, List[float]]],
                       realtime: bool = True) -> bool:
        """
        Run a trajectory through simulation.

        Args:
            trajectory: List of (timestamp, joint_angles) tuples
            realtime: If True, run at real-time speed

        Returns:
            True if completed successfully
        """
        if not self.is_ready:
            return False

        start_time = time.time()

        for target_time, angles in trajectory:
            if self._emergency_stop:
                print("[SIM] Trajectory interrupted by emergency stop")
                return False

            self.set_joint_angles(angles)

            if realtime:
                # Wait until target time
                elapsed = time.time() - start_time
                wait_time = target_time - elapsed
                if wait_time > 0:
                    # Step simulation while waiting
                    steps_needed = int(wait_time / (1.0/240.0))
                    for _ in range(steps_needed):
                        self.step()
            else:
                # Run 10 steps per waypoint
                self.step(10)

        return True
