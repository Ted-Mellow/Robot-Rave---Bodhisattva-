#!/usr/bin/env python3
"""
Diagnostic test for Joint 2 (Shoulder) motion
Tests if J2 can actually lift the arm against gravity
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation'))

from piper_simultion_corrected import PiperSimulation
import pybullet as p
import time

def test_j2_motion():
    print("="*60)
    print("DIAGNOSTIC: Testing Joint 2 (Shoulder) Motion")
    print("="*60)
    
    # Create simulation
    sim = PiperSimulation(gui=True)
    
    try:
        # Start at home (J2=0 - should be hanging down)
        print("\n1. Home position (J2=0) - arm should hang down")
        sim.set_joint_positions([0, 0, 0, 0, 0, 0])
        for _ in range(240):
            sim.step()
        time.sleep(2)
        
        # Get current J2 position
        current_pos = sim.get_joint_positions()
        print(f"   Current J2: {current_pos[1]:.3f} rad")
        
        # Try to move J2 to 1.57 (90 degrees - horizontal)
        print("\n2. Moving J2 to 1.57 rad (90°) - should be horizontal")
        print("   Applying command...")
        
        # Apply with VERY high force
        target = [0, 1.57, 0, 0, 0, 0]
        for _ in range(480):  # 2 seconds
            # Apply extra force to J2 specifically
            for i, joint_idx in enumerate(sim.joint_indices):
                if i == 1:  # J2
                    p.setJointMotorControl2(
                        sim.robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=target[i],
                        force=2000,  # Much higher force for J2
                        positionGain=0.5,
                        velocityGain=0.2
                    )
                else:
                    p.setJointMotorControl2(
                        sim.robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=target[i],
                        force=600,
                        positionGain=0.3,
                        velocityGain=0.1
                    )
            sim.step()
        
        current_pos = sim.get_joint_positions()
        print(f"   Current J2: {current_pos[1]:.3f} rad")
        print(f"   Target was: 1.57 rad")
        print(f"   Error: {abs(current_pos[1] - 1.57):.3f} rad")
        
        if abs(current_pos[1] - 1.57) > 0.1:
            print("   ❌ J2 did NOT reach target!")
            print("   PROBLEM: Arm cannot lift against gravity")
        else:
            print("   ✅ J2 reached target!")
        
        time.sleep(2)
        
        # Try to move J2 up further (J2=3.0 - nearly vertical)
        print("\n3. Moving J2 to 3.0 rad (172°) - should be nearly up")
        target = [0, 3.0, 0, 0, 0, 0]
        for _ in range(480):
            for i, joint_idx in enumerate(sim.joint_indices):
                if i == 1:  # J2
                    p.setJointMotorControl2(
                        sim.robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=target[i],
                        force=2000,
                        positionGain=0.5,
                        velocityGain=0.2
                    )
                else:
                    p.setJointMotorControl2(
                        sim.robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=target[i],
                        force=600,
                        positionGain=0.3,
                        velocityGain=0.1
                    )
            sim.step()
        
        current_pos = sim.get_joint_positions()
        print(f"   Current J2: {current_pos[1]:.3f} rad")
        print(f"   Target was: 3.0 rad")
        
        time.sleep(2)
        
        # Check all joint positions
        print("\n4. Final joint positions:")
        for i, pos in enumerate(current_pos):
            print(f"   J{i+1}: {pos:.3f} rad ({pos*57.3:.1f}°)")
        
        print("\n" + "="*60)
        print("Test complete. Observe the arm motion.")
        print("If arm didn't lift, we need to:")
        print("1. Increase motor force significantly")
        print("2. Check URDF joint axis direction")
        print("3. Verify mass/inertia values")
        print("="*60)
        
        # Keep running
        print("\nPress Ctrl+C to exit...")
        while True:
            sim.step()
            
    except KeyboardInterrupt:
        print("\n\nTest stopped by user")
    finally:
        sim.close()

if __name__ == "__main__":
    test_j2_motion()
