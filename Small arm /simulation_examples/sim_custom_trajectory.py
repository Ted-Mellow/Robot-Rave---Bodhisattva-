#!/usr/bin/env python3
"""
Example: Custom Trajectory Template
Use this as a template to create your own custom trajectories
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from piper_pybullet_sim import PiperSimulation
import numpy as np
import time


def main():
    print("=" * 60)
    print("Piper Simulation: Custom Trajectory")
    print("=" * 60)
    
    sim = PiperSimulation(gui=True)
    
    try:
        # Reset to home
        sim.reset()
        print("✅ Robot initialized")
        time.sleep(1)
        
        print("\n🔄 Running custom trajectory...")
        print("   Modify this file to create your own motions!")
        
        # CUSTOMIZE YOUR TRAJECTORY HERE
        # ================================
        
        # Example: Define your own waypoints
        my_trajectory = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # Waypoint 1
            [0.5, 1.0, -1.0, 0.5, 0.5, 0.5],     # Waypoint 2
            [1.0, 1.5, -1.5, 1.0, 1.0, 1.0],     # Waypoint 3
            [-0.5, 0.8, -0.8, -0.3, -0.3, -0.3], # Waypoint 4
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # Return home
        ]
        
        # Execute trajectory
        for i, waypoint in enumerate(my_trajectory):
            print(f"   Moving to waypoint {i+1}/{len(my_trajectory)}")
            
            # Move to waypoint
            sim.set_joint_positions(waypoint)
            
            # Hold position for a moment
            for _ in range(120):
                sim.step()
            
            time.sleep(0.5)
        
        print("\n✅ Custom trajectory complete!")
        
        # OPTIONAL: Add continuous motion
        print("\n🔁 Starting continuous motion (Ctrl+C to stop)...")
        
        t = 0
        while True:
            # Example: Gentle sinusoidal motion
            target = [
                0.3 * np.sin(t * 0.5),
                1.0 + 0.2 * np.cos(t * 0.3),
                -1.0 + 0.2 * np.sin(t * 0.4),
                0.2 * np.cos(t * 0.6),
                0.1 * np.sin(t * 0.7),
                0.3 * np.cos(t * 0.5)
            ]
            
            sim.set_joint_positions(target)
            sim.step()
            
            t += 0.01
            
    except KeyboardInterrupt:
        print("\n⚠️  Simulation stopped by user")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
