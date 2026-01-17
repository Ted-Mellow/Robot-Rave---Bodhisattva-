#!/usr/bin/env python3
"""
Circular Motion Demo for Piper Robot Arm
Demonstrates smooth continuous motion in a circular pattern
This helps verify the URDF model loads correctly and moves smoothly

Usage:
    python simulation/sim_circle_demo.py
"""

import sys
import os
import time
import math
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from piper_simultion_corrected import PiperSimulation


def generate_circle_trajectory(radius=0.5, height=0.3, num_points=60, duration=6.0):
    """
    Generate a circular trajectory in joint space
    
    Args:
        radius: Amplitude of circular motion (radians)
        height: Elevation angle for joint 2 (radians)
        num_points: Number of points in circle
        duration: Total duration for one complete circle (seconds)
    
    Returns:
        List of (time, joint_positions) tuples
    """
    trajectory = []
    
    for i in range(num_points):
        t = i / num_points
        angle = t * 2 * math.pi
        
        # Time stamp
        timestamp = t * duration
        
        # Create circular motion using joints 1 and 3
        # Joint 1 (base rotation) creates horizontal circle
        # Joint 2 (shoulder) stays elevated
        # Joint 3 (elbow) creates vertical component
        joint_positions = [
            radius * math.cos(angle),        # Joint 1: horizontal circle
            height + 0.2 * math.sin(2*angle), # Joint 2: elevated with wave
            0.3 * math.sin(angle),            # Joint 3: vertical component
            0.0,                              # Joint 4: neutral
            0.2 * math.cos(angle),            # Joint 5: subtle wrist motion
            0.0                               # Joint 6: neutral
        ]
        
        trajectory.append((timestamp, joint_positions))
    
    return trajectory


def main():
    print("="*60)
    print("🎯 Piper Robot - Circular Motion Demo")
    print("="*60)
    print("\nThis demo makes the robot perform smooth circular motions")
    print("to verify the URDF model loads and moves correctly.\n")
    print("Press Ctrl+C to stop, or close the GUI window\n")
    
    # Initialize simulation (URDF loaded by default)
    sim = PiperSimulation(gui=True)
    
    try:
        # Reset to home position
        print("📍 Moving to home position...")
        sim.reset_to_home()
        time.sleep(1)
        
        # Generate circular trajectory
        trajectory = generate_circle_trajectory(
            radius=0.5,      # 0.5 radians amplitude
            height=0.3,      # Elevated position
            num_points=100,  # Smooth motion
            duration=8.0     # 8 seconds per circle
        )
        
        print("✅ Starting circular motion (looping indefinitely)...\n")
        
        loop_count = 0
        while True:
            loop_count += 1
            print(f"🔁 Loop #{loop_count}")
            
            # Execute one circle
            for i in range(len(trajectory)):
                _, joint_positions = trajectory[i]
                
                # Move to position
                sim.set_joint_positions(joint_positions)
                
                # Step simulation multiple times for smooth motion
                for _ in range(10):
                    sim.step()
            
            # Brief pause between loops
            time.sleep(0.3)
    
    except KeyboardInterrupt:
        print("\n⚠️  Demo stopped by user")
    except RuntimeError as e:
        if "disconnected" in str(e):
            print("\n⚠️  GUI window closed")
        else:
            print(f"\n❌ Error: {e}")
            raise
    finally:
        print("🔴 Closing simulation...")
        sim.close()
        print("✅ Demo complete!\n")


if __name__ == "__main__":
    main()
