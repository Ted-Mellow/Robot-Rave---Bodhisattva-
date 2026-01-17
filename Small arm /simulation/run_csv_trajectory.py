#!/usr/bin/env python3
"""
CSV Trajectory Runner for Piper Robot Arm
Loads and executes trajectories from CSV files in PyBullet simulation

Usage:
    python run_csv_trajectory.py csv_trajectories/example_wave.csv
    python run_csv_trajectory.py csv_trajectories/example_dance.csv
"""

import sys
import os
import csv
import time
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from piper_pybullet_sim import PiperSimulation
import pybullet as p


class CSVTrajectoryRunner:
    """Run robot trajectories from CSV files"""
    
    def __init__(self, csv_file, gui=True, use_urdf=True):
        """
        Initialize trajectory runner
        
        Args:
            csv_file (str): Path to CSV trajectory file
            gui (bool): Show PyBullet GUI
            use_urdf (bool): Load URDF model (default True, uses robot_models/piper.urdf)
        """
        self.csv_file = Path(csv_file)
        self.gui = gui
        self.use_urdf = use_urdf
        
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        # Load trajectory
        self.waypoints = self.load_trajectory()
        print(f"✅ Loaded {len(self.waypoints)} waypoints from {self.csv_file.name}")
        
        # Initialize simulation
        self.sim = PiperSimulation(gui=gui)
        
        # Load URDF if requested
        if use_urdf:
            # Look for URDF in parent directory's robot_models folder
            urdf_path = Path(__file__).parent.parent / "robot_models" / "piper.urdf"
            if urdf_path.exists():
                print(f"✅ Loading URDF model: {urdf_path}")
                # Remove the primitive robot
                p.removeBody(self.sim.robot_id)
                
                # Load URDF
                self.sim.robot_id = p.loadURDF(
                    str(urdf_path),
                    basePosition=[0, 0, 0],
                    useFixedBase=True
                )
                
                # Update joint indices for URDF
                num_joints = p.getNumJoints(self.sim.robot_id)
                self.sim.joint_indices = []
                for i in range(num_joints):
                    joint_info = p.getJointInfo(self.sim.robot_id, i)
                    joint_name = joint_info[1].decode('utf-8')
                    if 'joint' in joint_name and joint_name != 'ee_joint':
                        self.sim.joint_indices.append(i)
                
                print(f"   URDF joints: {self.sim.joint_indices}")
            else:
                print(f"⚠️  URDF not found: {urdf_path}, using primitive shapes")
    
    def load_trajectory(self):
        """Load trajectory from CSV file"""
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
        
        # Sort by time
        waypoints.sort(key=lambda x: x['time'])
        
        return waypoints
    
    def interpolate(self, pos_start, pos_end, t):
        """Linear interpolation between two positions"""
        return [
            start + (end - start) * t
            for start, end in zip(pos_start, pos_end)
        ]
    
    def run(self, loop=False, speed_factor=1.0):
        """
        Execute the trajectory
        
        Args:
            loop (bool): Loop the trajectory continuously
            speed_factor (float): Speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
        """
        print(f"\n{'='*60}")
        print(f"Running trajectory: {self.csv_file.name}")
        print(f"Speed factor: {speed_factor}x")
        print(f"Loop: {'Yes' if loop else 'No'}")
        print(f"{'='*60}\n")
        
        try:
            # Reset to first position
            self.sim.reset()
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
                    steps = max(int(dt / self.sim.time_step), 10)
                    
                    # Print waypoint info
                    if wp_end['description']:
                        print(f"  → {wp_end['description']}")
                    else:
                        print(f"  → Waypoint {i+1}/{len(self.waypoints)-1}")
                    
                    # Interpolate between waypoints
                    for step in range(steps):
                        t = (step + 1) / steps
                        interpolated = self.interpolate(
                            wp_start['joints'],
                            wp_end['joints'],
                            t
                        )
                        
                        self.sim.set_joint_positions(interpolated)
                        self.sim.step()
                
                # If not looping, break
                if not loop:
                    break
                
                # Small pause between loops
                time.sleep(0.5)
            
            print(f"\n✅ Trajectory complete!")
            print("\nPress Ctrl+C to exit, or close the GUI window")
            
            # Keep simulation running
            while True:
                self.sim.step()
                
        except KeyboardInterrupt:
            print("\n⚠️  Trajectory stopped by user")
        except RuntimeError as e:
            if "disconnected" in str(e):
                print("\n⚠️  GUI window closed")
            else:
                raise
        finally:
            self.sim.close()


def main():
    parser = argparse.ArgumentParser(
        description='Run Piper robot trajectories from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_csv_trajectory.py csv_trajectories/example_wave.csv
  python run_csv_trajectory.py csv_trajectories/example_dance.csv --loop
  python run_csv_trajectory.py csv_trajectories/example_wave.csv --speed 0.5
  python run_csv_trajectory.py csv_trajectories/example_dance.csv --urdf --loop
        """
    )
    
    parser.add_argument('csv_file', help='Path to CSV trajectory file')
    parser.add_argument('--loop', action='store_true', help='Loop trajectory continuously')
    parser.add_argument('--speed', type=float, default=1.0, help='Speed factor (default: 1.0)')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    parser.add_argument('--no-urdf', dest='urdf', action='store_false', default=True, help='Use primitive shapes instead of URDF model')
    
    args = parser.parse_args()
    
    # Run trajectory
    runner = CSVTrajectoryRunner(
        args.csv_file,
        gui=not args.no_gui,
        use_urdf=args.urdf
    )
    runner.run(loop=args.loop, speed_factor=args.speed)


if __name__ == "__main__":
    main()
