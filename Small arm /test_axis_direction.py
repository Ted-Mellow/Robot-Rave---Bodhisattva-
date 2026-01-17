#!/usr/bin/env python3
"""
Test to determine if J2 axis is inverted
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation'))

from piper_simultion_corrected import PiperSimulation
import pybullet as p
import time

def test_axis():
    print("="*60)
    print("TESTING J2 AXIS DIRECTION")
    print("="*60)
    
    sim = PiperSimulation(gui=True)
    
    try:
        # Start at home
        print("\n1. HOME (J2=0) - observe arm position")
        sim.set_joint_positions([0, 0, 0, 0, 0, 0])
        time.sleep(3)
        
        # Move J2 POSITIVE
        print("\n2. J2 = +0.5 rad (28.6°)")
        print("   EXPECTED: Arm should move UP/FORWARD")
        print("   IF INVERTED: Arm moves DOWN/BACK")
        sim.set_joint_positions([0, 0.5, 0, 0, 0, 0])
        time.sleep(3)
        
        # Check actual position
        actual = sim.get_joint_positions()
        print(f"   Actual J2: {actual[1]:.3f} rad")
        
        # Move J2 more positive
        print("\n3. J2 = +1.5 rad (85.9°)")
        print("   EXPECTED: Arm should move MORE UP")
        sim.set_joint_positions([0, 1.5, 0, 0, 0, 0])
        time.sleep(3)
        
        actual = sim.get_joint_positions()
        print(f"   Actual J2: {actual[1]:.3f} rad")
        
        # Move J2 NEGATIVE (back toward zero)
        print("\n4. J2 = -0.5 rad WOULD BE INVALID (below limit)")
        print("   Testing J2 = 0.1 (near zero)")
        sim.set_joint_positions([0, 0.1, 0, 0, 0, 0])
        time.sleep(3)
        
        print("\n" + "="*60)
        print("DIAGNOSIS:")
        print("-" * 60)
        print("Look at the arm motion in the simulation:")
        print()
        print("If J2=0 → J2=1.5 made arm go UP:")
        print("  ✅ Axis is CORRECT (positive = up)")
        print()
        print("If J2=0 → J2=1.5 made arm go DOWN:")
        print("  ❌ Axis is INVERTED (positive = down)")
        print("     Fix: Change URDF axis from '0 1 0' to '0 -1 0'")
        print()
        print("If arm DIDN'T MOVE AT ALL:")
        print("  ❌ Motor force still insufficient OR")
        print("     ❌ Joint is locked/broken in URDF")
        print("="*60)
        
        print("\nPress Ctrl+C to exit...")
        while True:
            sim.step()
            
    except KeyboardInterrupt:
        print("\n\nTest stopped")
    finally:
        sim.close()

if __name__ == "__main__":
    test_axis()
