#!/usr/bin/env python3
"""
Example: Joint Sweep Simulation
Sweeps each joint through its range of motion
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from piper_pybullet_sim import PiperSimulation
import numpy as np
import time


def main():
    print("=" * 60)
    print("Piper Simulation: Joint Sweep Test")
    print("=" * 60)
    
    sim = PiperSimulation(gui=True)
    
    try:
        # Reset to home
        sim.reset()
        print("✅ Robot initialized at home position")
        time.sleep(1)
        
        # Sweep each joint individually
        for joint_idx in range(6):
            print(f"\n🔄 Sweeping Joint {joint_idx + 1}...")
            
            lower = sim.JOINT_LIMITS['lower'][joint_idx]
            upper = sim.JOINT_LIMITS['upper'][joint_idx]
            mid = (lower + upper) / 2
            
            # Create sweep trajectory
            steps = 120
            angles = np.linspace(mid, upper, steps//2).tolist() + \
                     np.linspace(upper, lower, steps).tolist() + \
                     np.linspace(lower, mid, steps//2).tolist()
            
            for angle in angles:
                target = [0.0] * 6
                target[joint_idx] = angle
                sim.set_joint_positions(target)
                sim.step()
            
            print(f"   Range: [{np.rad2deg(lower):.1f}°, {np.rad2deg(upper):.1f}°]")
        
        # Return to home
        print("\n✅ Joint sweep complete! Returning home...")
        sim.reset()
        
        print("\nPress Ctrl+C to exit")
        while True:
            sim.step()
            
    except KeyboardInterrupt:
        print("\n⚠️  Simulation stopped by user")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
