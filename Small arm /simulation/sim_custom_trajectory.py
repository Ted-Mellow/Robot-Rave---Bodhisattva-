#!/usr/bin/env python3
"""
Custom Trajectory Simulator
Loads and runs trajectories from CSV files in csv_trajectories/ folder
"""

import sys
import os
import csv
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from piper_pybullet_sim import PiperSimulation
import time


def load_csv_trajectory(csv_file):
    """Load trajectory from CSV file with timing and descriptions"""
    waypoints = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            waypoints.append({
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
            })
    
    # Validate times are increasing
    for i in range(1, len(waypoints)):
        if waypoints[i]['time'] < waypoints[i-1]['time']:
            print(f"⚠️  Warning: Time decreased at waypoint {i+1}")
    
    return waypoints


def list_csv_files():
    """List available CSV trajectory files"""
    csv_dir = Path(__file__).parent.parent / "csv_trajectories"
    csv_files = list(csv_dir.glob("*.csv"))
    return sorted(csv_files)


def main():
    print("=" * 60)
    print("Custom Trajectory Simulator")
    print("=" * 60)
    
    # List available CSV files
    csv_files = list_csv_files()
    
    if not csv_files:
        print("❌ No CSV files found in csv_trajectories/")
        return
    
    print("\nAvailable trajectories:")
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {f.name}")
    
    # Get user selection
    try:
        choice = input(f"\nSelect trajectory (1-{len(csv_files)}) or Enter for first: ").strip()
        if choice == "":
            selected = csv_files[0]
        else:
            selected = csv_files[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid selection, using first file")
        selected = csv_files[0]
    
    print(f"\n✅ Loading: {selected.name}")
    
    # Load trajectory
    trajectory = load_csv_trajectory(selected)
    print(f"   {len(trajectory)} waypoints loaded")
    
    # Run simulation
    sim = PiperSimulation(gui=True)
    
    try:
        sim.reset()
        print("\n🔄 Executing trajectory...")
        time.sleep(0.5)
        
        # Execute trajectory with proper timing
        for i in range(len(trajectory)):
            wp = trajectory[i]
            
            # Display description if available
            if wp['description']:
                print(f"   {i+1}/{len(trajectory)}: {wp['description']}")
            else:
                print(f"   Waypoint {i+1}/{len(trajectory)}")
            
            # Calculate time to next waypoint
            if i < len(trajectory) - 1:
                time_delta = trajectory[i+1]['time'] - wp['time']
                steps = max(int(time_delta / sim.time_step), 10)  # At least 10 steps
            else:
                steps = 120  # Hold final position
            
            # Move to position
            sim.set_joint_positions(wp['joints'])
            
            # Interpolate over time
            for _ in range(steps):
                sim.step()
        
        print("\n✅ Trajectory complete!")
        print("Press Ctrl+C to exit\n")
        
        # Keep running
        while True:
            sim.step()
            
    except KeyboardInterrupt:
        print("\n⚠️  Stopped by user")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
