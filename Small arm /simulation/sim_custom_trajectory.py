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
    """Load trajectory from CSV file"""
    waypoints = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            waypoints.append([
                float(row['joint1']),
                float(row['joint2']),
                float(row['joint3']),
                float(row['joint4']),
                float(row['joint5']),
                float(row['joint6'])
            ])
    
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
        
        # Execute trajectory
        for i, waypoint in enumerate(trajectory, 1):
            print(f"   Waypoint {i}/{len(trajectory)}")
            sim.set_joint_positions(waypoint)
            
            # Hold position
            for _ in range(120):
                sim.step()
            
            time.sleep(0.3)
        
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
