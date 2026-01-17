#!/usr/bin/env python3
"""
Example: Pick and Place Simulation
Simulates a simple pick and place operation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from piper_pybullet_sim import PiperSimulation
import time


def move_to_position(sim, target, steps=120):
    """Smoothly move to target position"""
    current = sim.get_joint_positions()
    
    for i in range(steps):
        t = (i + 1) / steps  # 0 to 1
        interpolated = [
            c + (tgt - c) * t
            for c, tgt in zip(current, target)
        ]
        sim.set_joint_positions(interpolated)
        sim.step()


def main():
    print("=" * 60)
    print("Piper Simulation: Pick and Place Test")
    print("=" * 60)
    
    sim = PiperSimulation(gui=True)
    
    try:
        # Reset to home
        sim.reset()
        print("✅ Robot initialized")
        time.sleep(1)
        
        # Define waypoints for pick and place
        waypoints = [
            ("Home", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("Pre-pick", [0.8, 1.2, -1.5, 0.5, 0.3, 0.0]),
            ("Pick", [0.8, 1.5, -1.8, 0.5, 0.3, 0.0]),
            ("Lift", [0.8, 1.2, -1.5, 0.5, 0.3, 0.0]),
            ("Transfer", [-0.8, 1.0, -1.2, -0.5, 0.3, 1.5]),
            ("Pre-place", [-0.8, 1.3, -1.5, -0.5, 0.3, 1.5]),
            ("Place", [-0.8, 1.6, -1.8, -0.5, 0.3, 1.5]),
            ("Retract", [-0.8, 1.3, -1.5, -0.5, 0.3, 1.5]),
            ("Home", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        
        print("\n🔄 Executing pick and place sequence...")
        
        for i, (name, position) in enumerate(waypoints):
            print(f"\n   Step {i+1}/{len(waypoints)}: {name}")
            
            # Add gripper simulation (visual only)
            if "Pick" in name or "Place" in name:
                if "Pick" in name and i > 1:
                    print("   🤏 Closing gripper...")
                elif "Place" in name and i > len(waypoints) - 3:
                    print("   🖐️  Opening gripper...")
            
            move_to_position(sim, position)
            time.sleep(0.3)
        
        print("\n✅ Pick and place complete!")
        print("\nPress Ctrl+C to exit")
        
        while True:
            sim.step()
            
    except KeyboardInterrupt:
        print("\n⚠️  Simulation stopped by user")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
