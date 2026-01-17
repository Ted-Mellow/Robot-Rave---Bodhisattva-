#!/usr/bin/env python3
"""
Motion Mapper - Converts Human Pose to Robot Joint Angles
Maps arm movements from pose detection to Piper robot arm joint angles
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import csv


class MotionMapper:
    """Maps human arm movements to robot joint angles"""

    # Piper robot joint limits (radians)
    JOINT_LIMITS = {
        'J1': (-2.68781, 2.68781),   # ±154° - Base rotation
        'J2': (0, 3.40339),            # 0→195° - Shoulder elevation
        'J3': (-3.05433, 0),           # -175→0° - Elbow
        'J4': (-1.85005, 1.85005),     # ±106° - Wrist roll
        'J5': (-1.30900, 1.30900),     # ±75° - Wrist pitch
        'J6': (-1.74533, 1.74533),     # ±100° - Wrist rotation
    }

    def __init__(self,
                 scaling_factor: float = 1.0,
                 z_offset: float = 0.0,
                 smooth_window: int = 5):
        """
        Initialize motion mapper

        Args:
            scaling_factor: Scale factor for motion amplitude (1.0 = no scaling)
            z_offset: Vertical offset for arm position
            smooth_window: Window size for temporal smoothing
        """
        self.scaling_factor = scaling_factor
        self.z_offset = z_offset
        self.smooth_window = smooth_window

        print(f"✅ MotionMapper initialized")
        print(f"   Scaling factor: {scaling_factor}")
        print(f"   Z offset: {z_offset}")
        print(f"   Smoothing window: {smooth_window}")

    def map_arm_pose_to_joints(self, arm_angles: Dict[str, float]) -> List[float]:
        """
        Map human arm angles to robot joint angles

        Args:
            arm_angles: Dictionary with shoulder_angle, elbow_angle, arm_elevation, lateral_position

        Returns:
            List of 6 joint angles [J1, J2, J3, J4, J5, J6] in radians
        """
        if not arm_angles:
            # Return neutral position if no data
            return [0.0, 1.2, -0.5, 0.0, 0.0, 0.0]

        # Extract arm measurements
        shoulder_angle = arm_angles.get('shoulder_angle', 90.0)
        elbow_angle = arm_angles.get('elbow_angle', 180.0)
        arm_elevation = arm_angles.get('arm_elevation', 0.0)
        lateral_pos = arm_angles.get('lateral_position', 0.0)

        # J1: Base rotation - maps to lateral arm position (left-right swing)
        # Lateral position is normalized [0, 1], where 0.5 is center
        # Map to robot's ±154° range
        j1 = self._map_range(lateral_pos, -0.3, 0.3, -2.6, 2.6)
        j1 = self._clamp(j1, *self.JOINT_LIMITS['J1'])

        # J2: Shoulder elevation - maps to arm elevation angle
        # When arm is down (elevation ~-90°), J2 should be low (~0.5 rad)
        # When arm is up (elevation ~90°), J2 should be high (~2.5 rad)
        # Arm elevation: -90° (down) to +90° (up)
        j2 = self._map_range(arm_elevation, -90, 90, 0.3, 2.8)
        j2 += self.z_offset
        j2 = self._clamp(j2, *self.JOINT_LIMITS['J2'])

        # J3: Elbow - maps to elbow angle
        # Human elbow: 180° = straight, 0° = fully bent
        # Robot J3: 0° = straight, -175° = fully bent
        # Invert the mapping
        j3 = self._map_range(elbow_angle, 180, 30, 0, -2.5)
        j3 = self._clamp(j3, *self.JOINT_LIMITS['J3'])

        # J4, J5, J6: Wrist joints - keep neutral for now
        # These could be mapped to hand orientation if needed
        j4 = 0.0
        j5 = 0.0
        j6 = 0.0

        # Apply scaling factor to movement (around neutral position)
        neutral = [0.0, 1.2, -0.5, 0.0, 0.0, 0.0]
        joints = [j1, j2, j3, j4, j5, j6]

        if self.scaling_factor != 1.0:
            for i in range(3):  # Only scale first 3 joints
                joints[i] = neutral[i] + (joints[i] - neutral[i]) * self.scaling_factor

        return joints

    def process_pose_sequence(self, pose_data_path: str, side: str = 'right') -> List[Dict]:
        """
        Process pose sequence and generate robot joint trajectories

        Args:
            pose_data_path: Path to JSON file from pose detector
            side: 'left' or 'right' arm to track

        Returns:
            List of motion keyframes with timestamp and joint angles
        """
        # Load pose data
        with open(pose_data_path, 'r') as f:
            data = json.load(f)

        poses = data['poses']
        fps = data['fps']

        print(f"\n🤖 Processing pose sequence...")
        print(f"   Frames: {len(poses)}")
        print(f"   FPS: {fps}")
        print(f"   Tracking: {side} arm")

        trajectory = []

        for pose in poses:
            timestamp = pose['timestamp']
            arm_key = f'{side}_arm'

            if arm_key not in pose:
                continue

            arm_angles = pose[arm_key]

            # Map to robot joints
            joint_angles = self.map_arm_pose_to_joints(arm_angles)

            trajectory.append({
                'timestamp': timestamp,
                'joints': joint_angles,
                'arm_angles': arm_angles  # Keep for debugging
            })

        # Apply temporal smoothing
        if self.smooth_window > 1:
            trajectory = self._smooth_trajectory(trajectory)

        print(f"✅ Generated {len(trajectory)} motion keyframes")

        return trajectory

    def _smooth_trajectory(self, trajectory: List[Dict]) -> List[Dict]:
        """Apply moving average smoothing to trajectory"""
        if len(trajectory) < self.smooth_window:
            return trajectory

        smoothed = []
        half_window = self.smooth_window // 2

        for i in range(len(trajectory)):
            # Get window range
            start = max(0, i - half_window)
            end = min(len(trajectory), i + half_window + 1)
            window = trajectory[start:end]

            # Average joint angles in window
            joint_sum = np.zeros(6)
            for frame in window:
                joint_sum += np.array(frame['joints'])

            avg_joints = (joint_sum / len(window)).tolist()

            smoothed.append({
                'timestamp': trajectory[i]['timestamp'],
                'joints': avg_joints,
                'arm_angles': trajectory[i]['arm_angles']
            })

        return smoothed

    def export_to_csv(self, trajectory: List[Dict], output_path: str, description: str = ""):
        """
        Export trajectory to CSV format compatible with Piper simulation

        Args:
            trajectory: Motion trajectory from process_pose_sequence()
            output_path: Output CSV file path
            description: Optional description prefix for each row
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(['time', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'description'])

            # Write trajectory
            for i, frame in enumerate(trajectory):
                time = frame['timestamp']
                joints = frame['joints']

                desc = f"{description} frame {i}" if description else f"Frame {i}"

                row = [f"{time:.3f}"] + [f"{j:.3f}" for j in joints] + [desc]
                writer.writerow(row)

        print(f"\n💾 CSV trajectory saved to: {output_path}")
        print(f"   Duration: {trajectory[-1]['timestamp']:.2f}s")
        print(f"   Keyframes: {len(trajectory)}")

    def _map_range(self, value: float, in_min: float, in_max: float,
                   out_min: float, out_max: float) -> float:
        """Map value from input range to output range"""
        # Clamp input
        value = max(in_min, min(in_max, value))
        # Linear mapping
        return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value to range"""
        return max(min_val, min(max_val, value))


def visualize_trajectory(trajectory: List[Dict]):
    """
    Visualize trajectory with matplotlib (optional)

    Args:
        trajectory: Motion trajectory from process_pose_sequence()
    """
    try:
        import matplotlib.pyplot as plt

        times = [frame['timestamp'] for frame in trajectory]
        joints = np.array([frame['joints'] for frame in trajectory])

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle('Robot Joint Trajectories', fontsize=16)

        joint_names = ['J1 (Base)', 'J2 (Shoulder)', 'J3 (Elbow)',
                      'J4 (Wrist Roll)', 'J5 (Wrist Pitch)', 'J6 (Wrist Rot)']

        for i in range(6):
            ax = axes[i // 2, i % 2]
            ax.plot(times, joints[:, i], linewidth=2)
            ax.set_title(joint_names[i])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Angle (rad)')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('trajectory_visualization.png', dpi=150)
        print(f"\n📊 Trajectory visualization saved to: trajectory_visualization.png")
        plt.show()

    except ImportError:
        print("\n⚠️  matplotlib not available - skipping visualization")


if __name__ == "__main__":
    print("=" * 70)
    print("MOTION MAPPER - POSE TO ROBOT JOINTS")
    print("=" * 70)

    # Configuration
    POSE_DATA_PATH = "pose_data.json"
    OUTPUT_CSV = "bodhisattva_dance.csv"

    # Create mapper
    mapper = MotionMapper(
        scaling_factor=0.8,  # Scale down movements slightly for safety
        z_offset=0.0,        # No vertical offset
        smooth_window=5      # Smooth over 5 frames
    )

    try:
        # Process pose sequence
        trajectory = mapper.process_pose_sequence(
            POSE_DATA_PATH,
            side='right'  # Track right arm
        )

        if trajectory:
            # Export to CSV
            mapper.export_to_csv(
                trajectory,
                OUTPUT_CSV,
                description="Bodhisattva"
            )

            # Visualize (optional)
            visualize_trajectory(trajectory)

            print("\n✅ Motion mapping complete!")
        else:
            print("\n⚠️  No trajectory generated - check pose data")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"   Make sure to run pose_detector.py first!")


