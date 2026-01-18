#!/usr/bin/env python3
"""
Run mirrored dance - two Piper robots performing the same dance mirrored.
One robot on the left, one on the right, with mirrored movements.

Usage:
    python run_mirrored_dance.py <csv_file> [--speed <multiplier>] [--loop]
    
Example:
    python run_mirrored_dance.py ../csv_trajectories/mesmerizing_dance.csv
    python run_mirrored_dance.py ../csv_trajectories/mesmerizing_dance.csv --speed 0.8 --loop
"""

import sys
import os
import csv
import time
import argparse
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pybullet as p
import pybullet_data
from piper_simultion_corrected import PiperSimulation


class MirroredDanceRunner:
    """Run two robots with mirrored movements."""
    
    def __init__(self, csv_file, gui=True):
        """
        Initialize mirrored dance runner.
        
        Args:
            csv_file: Path to CSV trajectory file
            gui: Show PyBullet GUI
        """
        self.csv_file = Path(csv_file)
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        self.gui = gui
        self.waypoints = self.load_trajectory()
        print(f"✅ Loaded {len(self.waypoints)} waypoints from {self.csv_file.name}")
        
        # Initialize PyBullet
        if gui:
            self.client = p.connect(p.GUI)
            # Configure camera for better view of both robots (front view)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.8,
                cameraYaw=0,
                cameraPitch=-20,
                cameraTargetPosition=[0, 0, 0.3]
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1./240.)
        p.setGravity(0, 0, -9.81)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load two robots - left and right, rotated to face camera
        # Rotate 90 degrees around Z axis so robots face forward (toward camera)
        left_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])  # 90° rotation
        right_orientation = p.getQuaternionFromEuler([0, 0, -np.pi/2])  # -90° rotation (mirrored)
        
        self.robot_left = self._load_robot_at_position([-0.3, 0, 0], left_orientation, "left")
        self.robot_right = self._load_robot_at_position([0.3, 0, 0], right_orientation, "right")
        
        # Stabilize
        print("⏳ Stabilizing robots...")
        for _ in range(150):
            p.stepSimulation()
        
        print("✅ Both robots initialized and stabilized")
    
    def _load_robot_at_position(self, position, orientation, side):
        """Load a robot at a specific position and orientation."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        urdf_paths = [
            os.path.join(parent_dir, "robot_models", "piper_corrected.urdf"),
            os.path.join(parent_dir, "robot_models", "piper.urdf"),
            os.path.join(script_dir, "robot_models", "piper_corrected.urdf"),
            os.path.join(script_dir, "robot_models", "piper.urdf"),
        ]
        
        urdf_path = None
        for path in urdf_paths:
            if os.path.exists(path):
                urdf_path = path
                break
        
        if urdf_path is None:
            raise FileNotFoundError("Could not find URDF file")
        
        # Load robot with offset position and rotation
        robot_id = p.loadURDF(
            urdf_path,
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=1
        )
        
        # Get joint indices (first 6 controllable joints)
        num_joints = p.getNumJoints(robot_id)
        joint_indices = []
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:  # Revolute joint
                joint_indices.append(i)
                if len(joint_indices) >= 6:
                    break
        
        # Set initial joint positions (home)
        home_positions = [0.0, 3.2, 0.0, 0.0, 0.0, 0.0]
        for i, joint_idx in enumerate(joint_indices[:6]):
            p.setJointMotorControl2(
                robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=home_positions[i],
                force=500,
                positionGain=1.0,
                velocityGain=0.3
            )
        
        return {
            'id': robot_id,
            'joints': joint_indices[:6],
            'side': side
        }
    
    def load_trajectory(self):
        """Load trajectory from CSV file."""
        waypoints = []
        
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    waypoint = {
                        'time': float(row['time']),
                        'joints': [
                            float(row['joint1']),
                            float(row['joint2']),
                            float(row['joint3']),
                            float(row['joint4']),
                            float(row['joint5']),
                            float(row['joint6'])
                        ],
                        'description': row.get('description', '')
                    }
                    waypoints.append(waypoint)
                except (KeyError, ValueError) as e:
                    print(f"⚠️  Skipping invalid row: {e}")
                    continue
        
        if not waypoints:
            raise ValueError("No valid waypoints found in CSV file")
        
        waypoints.sort(key=lambda x: x['time'])
        return waypoints
    
    def set_robot_joints(self, robot, joints):
        """Set joint positions for a robot."""
        # Clamp joints to limits
        JOINT_LIMITS = {
            'lower': [-2.68781, 0, -3.05433, -1.85005, -1.30900, -1.74533],
            'upper': [2.68781, 3.40339, 0, 1.85005, 1.30900, 1.74533]
        }
        
        clamped = []
        for i, angle in enumerate(joints):
            clamped.append(max(JOINT_LIMITS['lower'][i], min(JOINT_LIMITS['upper'][i], angle)))
        
        # Set positions
        for i, joint_idx in enumerate(robot['joints']):
            if i in [1, 2]:  # J2 and J3 need more force
                pos_gain = 1.0
                vel_gain = 0.5
            else:
                pos_gain = 0.5
                vel_gain = 0.2
            
            p.setJointMotorControl2(
                robot['id'],
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=clamped[i],
                force=500,
                positionGain=pos_gain,
                velocityGain=vel_gain
            )
    
    def mirror_joints(self, joints):
        """Mirror joint angles for right robot (flip J1)."""
        mirrored = joints.copy()
        mirrored[0] = -mirrored[0]  # Flip base rotation (J1)
        return mirrored
    
    def interpolate(self, pos_start, pos_end, t):
        """Linear interpolation between two positions."""
        return [
            start + (end - start) * t
            for start, end in zip(pos_start, pos_end)
        ]
    
    def run(self, loop=False, speed_factor=1.0):
        """Execute mirrored trajectory."""
        print(f"\n{'='*60}")
        print(f"Running MIRRORED dance: {self.csv_file.name}")
        print(f"Speed factor: {speed_factor}x")
        print(f"Loop: {'Yes' if loop else 'No'}")
        print(f"{'='*60}\n")
        
        try:
            # Reset both robots to home
            home_joints = [0.0, 3.2, 0.0, 0.0, 0.0, 0.0]
            self.set_robot_joints(self.robot_left, home_joints)
            self.set_robot_joints(self.robot_right, home_joints)
            
            for _ in range(100):
                p.stepSimulation()
            
            time.sleep(0.5)
            
            run_count = 0
            while True:
                run_count += 1
                if loop:
                    print(f"\n🔁 Loop #{run_count}")
                
                # Execute trajectory
                for i in range(len(self.waypoints) - 1):
                    wp_start = self.waypoints[i]
                    wp_end = self.waypoints[i + 1]
                    
                    # Calculate time difference
                    dt = (wp_end['time'] - wp_start['time']) / speed_factor
                    steps = max(int(dt / (1./240.)), 10)
                    
                    # Print waypoint info
                    if wp_end['description'] and i % 10 == 0:
                        print(f"  → {wp_end['description']}")
                    
                    # Interpolate between waypoints
                    for step in range(steps):
                        t = (step + 1) / steps
                        interpolated_left = self.interpolate(
                            wp_start['joints'],
                            wp_end['joints'],
                            t
                        )
                        # Mirror for right robot
                        interpolated_right = self.mirror_joints(interpolated_left)
                        
                        # Set both robots simultaneously
                        self.set_robot_joints(self.robot_left, interpolated_left)
                        self.set_robot_joints(self.robot_right, interpolated_right)
                        
                        p.stepSimulation()
                
                if not loop:
                    break
                
                time.sleep(0.5)
            
            print(f"\n✅ Mirrored dance complete!")
            print("\nPress Ctrl+C to exit, or close the GUI window")
            
            # Keep simulation running
            while True:
                p.stepSimulation()
                
        except KeyboardInterrupt:
            print("\n⚠️  Dance stopped by user (Ctrl+C)")
        except RuntimeError as e:
            if "disconnected" in str(e).lower() or "closed" in str(e).lower():
                print("\n⚠️  GUI window closed")
            else:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
        finally:
            p.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Run mirrored dance with two Piper robots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run mirrored dance
  python run_mirrored_dance.py ../csv_trajectories/mesmerizing_dance.csv
  
  # Slower and loop
  python run_mirrored_dance.py ../csv_trajectories/mesmerizing_dance.csv --speed 0.8 --loop
        """
    )
    parser.add_argument("csv_file", help="Path to CSV trajectory file")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Speed multiplier (default: 1.0, 0.5 = half speed)")
    parser.add_argument("--loop", action="store_true",
                       help="Loop the dance continuously")
    parser.add_argument("--no-gui", action="store_true",
                       help="Run headless (no GUI)")
    
    args = parser.parse_args()
    
    if args.speed <= 0:
        print("ERROR: Speed multiplier must be positive")
        return 1
    
    try:
        runner = MirroredDanceRunner(
            args.csv_file,
            gui=not args.no_gui
        )
        runner.run(loop=args.loop, speed_factor=args.speed)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
