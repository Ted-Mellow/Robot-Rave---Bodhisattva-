#!/usr/bin/env python3
"""
Trajectory Player (Raspberry Pi)

Plays back pre-recorded trajectories on the robot arm.
No video processing - just loads a trajectory JSON file and executes it.

Usage:
    python trajectory_player.py trajectory.json
    python trajectory_player.py trajectory.json --loop
    python trajectory_player.py trajectory.json --speed 0.8
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arm_control import HardwareArmController
from arm_control.logging_config import setup_logging, Loggers


class TrajectoryPlayer:
    """
    Simple trajectory playback for pre-recorded movements.
    
    Loads a trajectory JSON file and plays it back on the robot.
    """

    def __init__(self, trajectory_path: str, can_interface: str = "can0"):
        """
        Initialize trajectory player.

        Args:
            trajectory_path: Path to trajectory JSON file
            can_interface: CAN interface name
        """
        self.log = Loggers.playback()
        self.log.info("=" * 60)
        self.log.info("TRAJECTORY PLAYER INITIALIZING")
        self.log.info("=" * 60)

        self.trajectory_path = trajectory_path
        self.can_interface = can_interface
        
        self.controller: Optional[HardwareArmController] = None
        self.trajectory: List[Dict] = []
        self.frame_count = 0
        self.duration = 0.0

        self.log.info(f"Trajectory file: {trajectory_path}")
        self.log.info(f"CAN interface: {can_interface}")

    def load_trajectory(self) -> bool:
        """Load trajectory from JSON file."""
        try:
            with open(self.trajectory_path, 'r') as f:
                data = json.load(f)

            self.trajectory = data.get('frames', [])
            self.frame_count = len(self.trajectory)
            self.duration = data.get('duration', 0.0)

            self.log.info(f"Trajectory loaded: {self.frame_count} frames")
            self.log.info(f"Duration: {self.duration:.2f} seconds")
            self.log.info(f"Video source: {data.get('video_path', 'unknown')}")
            self.log.info(f"Arm side: {data.get('arm_side', 'unknown')}")
            
            return True

        except FileNotFoundError:
            self.log.error(f"Trajectory file not found: {self.trajectory_path}")
            return False
        except json.JSONDecodeError as e:
            self.log.error(f"Invalid JSON in trajectory file: {e}")
            return False
        except Exception as e:
            self.log.error(f"Failed to load trajectory: {e}")
            return False

    def initialize_arm(self) -> bool:
        """Initialize hardware controller."""
        self.log.info("Initializing arm controller...")

        try:
            self.controller = HardwareArmController(self.can_interface)

            if not self.controller.connect():
                self.log.error("Failed to connect to arm")
                return False

            if not self.controller.enable():
                self.log.error("Failed to enable arm")
                self.controller.disconnect()
                return False

            self.log.info("Arm controller ready")
            return True

        except Exception as e:
            self.log.error(f"Arm initialization failed: {e}")
            return False

    def play(self, speed_multiplier: float = 1.0, loop: bool = False) -> None:
        """
        Play the trajectory.

        Args:
            speed_multiplier: Speed adjustment (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
            loop: If True, loop the trajectory continuously
        """
        if not self.trajectory:
            self.log.error("No trajectory loaded")
            return

        if not self.controller:
            self.log.error("Arm controller not initialized")
            return

        self.log.info("=" * 60)
        self.log.info("STARTING PLAYBACK")
        self.log.info(f"Speed multiplier: {speed_multiplier}x")
        self.log.info(f"Loop: {loop}")
        self.log.info("=" * 60)

        iteration = 0
        
        try:
            while True:
                iteration += 1
                if loop:
                    self.log.info(f"Starting iteration {iteration}")

                start_time = time.time()
                last_timestamp = 0.0
                frames_sent = 0

                for frame in self.trajectory:
                    timestamp = frame['timestamp']
                    joint_angles = frame['joint_angles']
                    
                    # Calculate target time with speed adjustment
                    target_time = timestamp / speed_multiplier
                    
                    # Wait until target time
                    elapsed = time.time() - start_time
                    wait_time = target_time - elapsed
                    
                    if wait_time > 0:
                        time.sleep(wait_time)

                    # Send joint command
                    success = self.controller.set_joint_angles(joint_angles, speed_percent=50)
                    
                    if success:
                        frames_sent += 1
                        
                        # Log progress every 100 frames
                        if frames_sent % 100 == 0:
                            progress = (frames_sent / self.frame_count) * 100
                            self.log.info(f"Progress: {progress:.1f}% ({frames_sent}/{self.frame_count})")
                    
                    last_timestamp = timestamp

                elapsed_total = time.time() - start_time
                self.log.info(f"Playback complete: {frames_sent} frames in {elapsed_total:.2f}s")

                # If not looping, break
                if not loop:
                    break

                # Brief pause before next iteration
                time.sleep(1.0)

        except KeyboardInterrupt:
            self.log.info("Playback interrupted by user")
        except Exception as e:
            self.log.error(f"Playback error: {e}")
        finally:
            self.log.info("Playback stopped")

    def shutdown(self) -> None:
        """Shutdown the controller."""
        if self.controller:
            self.log.info("Shutting down...")
            try:
                self.controller.disable()
                self.controller.disconnect()
            except Exception as e:
                self.log.warning(f"Shutdown error: {e}")
            
            self.log.info("Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description="Trajectory Player - Play pre-recorded trajectories on robot")
    
    parser.add_argument('trajectory', help='Path to trajectory JSON file')
    parser.add_argument('--interface', '-i', default='can0',
                        help='CAN interface (default: can0)')
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                        help='Speed multiplier (default: 1.0)')
    parser.add_argument('--loop', '-l', action='store_true',
                        help='Loop playback continuously')
    
    args = parser.parse_args()

    # Set up logging
    setup_logging("trajectory_player", level=10)  # DEBUG level

    # Create player
    player = TrajectoryPlayer(
        trajectory_path=args.trajectory,
        can_interface=args.interface
    )

    # Load trajectory
    if not player.load_trajectory():
        print("Failed to load trajectory", file=sys.stderr)
        sys.exit(1)

    # Initialize arm
    if not player.initialize_arm():
        print("Failed to initialize arm", file=sys.stderr)
        sys.exit(1)

    # Play trajectory
    try:
        player.play(speed_multiplier=args.speed, loop=args.loop)
    finally:
        player.shutdown()


if __name__ == "__main__":
    main()
