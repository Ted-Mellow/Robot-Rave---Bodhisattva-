#!/usr/bin/env python3
"""
Expressive Motion Mapper - More Dynamic Robot Movements

Improvements over original:
- Wider range mapping for more expressive movements
- Better elbow articulation
- More dramatic base rotation for left-right movements
- Optimized for dance performance (not safety-conservative)
"""

import numpy as np
from typing import List, Dict


class ExpressiveMotionMapper:
    """Maps human arm movements to robot with full expressiveness"""

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
                 scaling_factor: float = 1.2,  # Amplify movements
                 z_offset: float = 0.0,
                 smooth_window: int = 5):
        """
        Initialize expressive motion mapper
        
        Args:
            scaling_factor: Movement amplification (>1.0 = more dramatic)
            z_offset: Vertical offset for arm position
            smooth_window: Window size for temporal smoothing
        """
        self.scaling_factor = scaling_factor
        self.z_offset = z_offset
        self.smooth_window = smooth_window

        print(f"✅ ExpressiveMotionMapper initialized")
        print(f"   Scaling factor: {scaling_factor} (amplified for dance!)")
        print(f"   Z offset: {z_offset}")
        print(f"   Smoothing window: {smooth_window}")

    def map_arm_pose_to_joints(self, arm_angles: Dict[str, float]) -> List[float]:
        """
        Map human arm angles to robot joint angles with FULL expressiveness
        
        Args:
            arm_angles: Dictionary with shoulder_angle, elbow_angle, arm_elevation, lateral_position
        
        Returns:
            List of 6 joint angles [J1, J2, J3, J4, J5, J6] in radians
        """
        if not arm_angles:
            return [0.0, 1.5, -0.8, 0.0, 0.0, 0.0]  # More dynamic neutral

        # Extract measurements
        shoulder_angle = arm_angles.get('shoulder_angle', 90.0)
        elbow_angle = arm_angles.get('elbow_angle', 180.0)
        arm_elevation = arm_angles.get('arm_elevation', 0.0)
        lateral_pos = arm_angles.get('lateral_position', 0.0)
        
        # Get actual landmark positions for better mapping
        shoulder_x = arm_angles.get('shoulder_x', 0.5)
        elbow_y = arm_angles.get('elbow_y', 0.5)
        wrist_y = arm_angles.get('wrist_y', 0.5)

        # ==== J1: Base Rotation (Left-Right) ====
        # Use FULL range for dramatic left-right movements
        # Lateral position: negative = left, positive = right
        # Map aggressively to use ±2.4 rad (±137°) of the ±2.68 limit
        j1 = self._map_range(lateral_pos, -0.5, 0.5, -2.4, 2.4)
        j1 = self._clamp(j1, *self.JOINT_LIMITS['J1'])

        # ==== J2: Shoulder Elevation (Up-Down) ====
        # Map arm elevation to use FULL vertical range
        # -90° (down) → 0.3 rad, 0° (horizontal) → 1.6 rad, +90° (up) → 3.0 rad
        if arm_elevation < 0:
            # Arm pointing down
            j2 = self._map_range(arm_elevation, -90, 0, 0.3, 1.6)
        else:
            # Arm pointing up
            j2 = self._map_range(arm_elevation, 0, 90, 1.6, 3.0)
        
        j2 += self.z_offset
        j2 = self._clamp(j2, *self.JOINT_LIMITS['J2'])

        # ==== J3: Elbow Bend ====
        # Human elbow: 180° = straight, 90° = bent 90°, 0° = fully bent
        # Robot J3: 0 = straight, -1.5 = 90° bend, -3.0 = fully bent
        # Map with full range for visible bending
        j3 = self._map_range(elbow_angle, 180, 90, -0.1, -1.8)
        j3 = self._clamp(j3, *self.JOINT_LIMITS['J3'])

        # ==== J4: Wrist Roll ====
        # Slight rotation based on arm position for natural look
        j4 = self._map_range(lateral_pos, -0.3, 0.3, -0.3, 0.3)
        j4 = self._clamp(j4, *self.JOINT_LIMITS['J4'])

        # ==== J5: Wrist Pitch ====
        # Tilt wrist based on elevation for natural hand position
        j5 = self._map_range(arm_elevation, -45, 45, -0.4, 0.4)
        j5 = self._clamp(j5, *self.JOINT_LIMITS['J5'])

        # ==== J6: Wrist Rotation ====
        j6 = 0.0  # Keep neutral

        joints = [j1, j2, j3, j4, j5, j6]

        # Apply scaling for extra drama
        if self.scaling_factor != 1.0:
            neutral = [0.0, 1.5, -0.8, 0.0, 0.0, 0.0]
            for i in range(5):  # Scale all except J6
                joints[i] = neutral[i] + (joints[i] - neutral[i]) * self.scaling_factor
                joints[i] = self._clamp(joints[i], *self.JOINT_LIMITS[f'J{i+1}'])

        return joints

    def process_pose_sequence(self, pose_data_path: str, side: str = 'right') -> List[Dict]:
        """Process pose sequence and generate robot joint trajectories"""
        import json
        
        with open(pose_data_path, 'r') as f:
            data = json.load(f)

        poses = data['poses']
        fps = data['fps']

        print(f"\n🤖 Processing pose sequence (EXPRESSIVE mode)...")
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
            joint_angles = self.map_arm_pose_to_joints(arm_angles)

            trajectory.append({
                'timestamp': timestamp,
                'joints': joint_angles,
                'arm_angles': arm_angles
            })

        # Apply temporal smoothing
        if self.smooth_window > 1:
            trajectory = self._smooth_trajectory(trajectory)

        print(f"✅ Generated {len(trajectory)} motion keyframes")
        
        # Show range statistics
        if trajectory:
            j1_vals = [t['joints'][0] for t in trajectory]
            j2_vals = [t['joints'][1] for t in trajectory]
            j3_vals = [t['joints'][2] for t in trajectory]
            
            print(f"\n📊 Joint ranges:")
            print(f"   J1 (base): {min(j1_vals):.2f} to {max(j1_vals):.2f} rad")
            print(f"   J2 (shoulder): {min(j2_vals):.2f} to {max(j2_vals):.2f} rad")
            print(f"   J3 (elbow): {min(j3_vals):.2f} to {max(j3_vals):.2f} rad")

        return trajectory

    def _smooth_trajectory(self, trajectory: List[Dict]) -> List[Dict]:
        """Apply moving average smoothing to trajectory"""
        if len(trajectory) < self.smooth_window:
            return trajectory

        smoothed = []
        half_window = self.smooth_window // 2

        for i in range(len(trajectory)):
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
        """Export trajectory to CSV format"""
        from pathlib import Path
        import csv
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'description'])

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
        value = max(in_min, min(in_max, value))
        return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value to range"""
        return max(min_val, min(max_val, value))
