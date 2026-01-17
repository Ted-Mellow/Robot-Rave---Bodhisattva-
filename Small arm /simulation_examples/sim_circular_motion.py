#!/usr/bin/env python3
"""
Example: Circular Motion Simulation
Creates smooth circular motion with the arm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from piper_pybullet_sim import PiperSimulation
import numpy as np
import time


def main():
    print("=" * 60)
    print("Piper Simulation: Circular Motion Test")
    print("=" * 60)
    
    sim = PiperSimulation(gui=True)
    
    try:
        # Reset to home
        sim.reset()
        print("✅ Robot initialized")
        time.sleep(1)
        
        print("🔄 Creating circular motion pattern...")
        
        # Create circular trajectory in joint space
        # This is a simplified example - real inverse kinematics would be better
        num_points = 200
        t = np.linspace(0, 2*np.pi, num_points)
        
        for i in range(3):  # 3 circles
            print(f"   Circle {i+1}/3")
            for angle in t:
                # Sinusoidal motion in multiple joints
                target = [
                    0.5 * np.sin(angle),           # J1
                    1.0 + 0.3 * np.cos(angle),     # J2
                    -1.0 + 0.3 * np.sin(angle*2),  # J3
                    0.3 * np.cos(angle*1.5),       # J4
                    0.2 * np.sin(angle*2),         # J5
                    0.5 * np.cos(angle)            # J6
                ]
                
                sim.set_joint_positions(target)
                sim.step()
        
        # Return to home
        print("\n✅ Circular motion complete! Returning home...")
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
