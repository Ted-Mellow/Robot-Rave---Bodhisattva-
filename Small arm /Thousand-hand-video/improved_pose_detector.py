#!/usr/bin/env python3
"""
Improved Pose Detection with Enhanced Single-Arm Tracking

Improvements:
- Higher confidence thresholds for stable tracking
- Visibility-based filtering
- Temporal smoothing with exponential moving average
- Outlier detection and rejection
- Focused single-arm tracking
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Optional, Deque
from collections import deque


class ImprovedPoseDetector:
    """Enhanced pose detector with smooth single-arm tracking"""

    def __init__(self,
                 model_complexity=2,
                 min_detection_confidence=0.7,  # Higher for stability
                 min_tracking_confidence=0.6,   # Higher for stability
                 min_visibility=0.5,  # Require visible landmarks
                 smoothing_alpha=0.3):  # EMA smoothing factor (lower = smoother)
        """
        Initialize improved pose detector
        
        Args:
            model_complexity: 0-2, higher = more accurate
            min_detection_confidence: Minimum confidence for detection (0.5-1.0)
            min_tracking_confidence: Minimum confidence for tracking (0.5-1.0)
            min_visibility: Minimum visibility score for landmarks (0-1)
            smoothing_alpha: Exponential moving average alpha (0.1-0.5)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.min_visibility = min_visibility
        self.smoothing_alpha = smoothing_alpha
        
        # Initialize MediaPipe Pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True,  # Enable MediaPipe internal smoothing
            enable_segmentation=False,
            smooth_segmentation=False
        )
        
        # Landmark indices
        self.LANDMARKS = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
        }
        
        # Smoothed landmark history (exponential moving average)
        self.smoothed_landmarks = {}
        
        # Previous valid pose for outlier detection
        self.prev_valid_pose = None
        
        # Detection stats
        self.frame_count = 0
        self.detection_count = 0
        self.rejection_count = 0
        
        print(f"✅ ImprovedPoseDetector initialized")
        print(f"   Model complexity: {model_complexity}")
        print(f"   Detection confidence: {min_detection_confidence}")
        print(f"   Tracking confidence: {min_tracking_confidence}")
        print(f"   Min visibility: {min_visibility}")
        print(f"   Smoothing alpha: {smoothing_alpha}")
    
    def process_frame(self, frame: np.ndarray, side: str = 'right') -> Optional[Dict]:
        """
        Process frame with enhanced tracking for one arm
        
        Args:
            frame: RGB image
            side: 'left' or 'right' - which arm to focus on
            
        Returns:
            Pose data dict or None if detection failed
        """
        self.frame_count += 1
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape
        
        # Check arm visibility first
        arm_landmarks = self._get_arm_landmarks(side)
        if not self._check_visibility(landmarks, arm_landmarks):
            self.rejection_count += 1
            return None  # Reject if arm not visible enough
        
        # Extract pose data
        pose_data = {
            'timestamp': None,
            'landmarks': {},
            'visibility': {},
            'raw_landmarks': results.pose_landmarks
        }
        
        # Extract and smooth landmarks for the target arm
        for name, idx in self.LANDMARKS.items():
            landmark = landmarks[idx]
            
            # Normalize coordinates
            coords = np.array([landmark.x, landmark.y, landmark.z])
            
            # Apply exponential moving average smoothing
            if name in self.smoothed_landmarks:
                smoothed = (self.smoothing_alpha * coords + 
                           (1 - self.smoothing_alpha) * self.smoothed_landmarks[name])
            else:
                smoothed = coords
            
            self.smoothed_landmarks[name] = smoothed
            
            # Store smoothed coordinates
            pose_data['landmarks'][name] = {
                'x': float(smoothed[0]),
                'y': float(smoothed[1]),
                'z': float(smoothed[2]),
                'px': int(smoothed[0] * w),
                'py': int(smoothed[1] * h)
            }
            pose_data['visibility'][name] = landmark.visibility
        
        # Outlier detection - check if pose is reasonable
        if not self._is_valid_pose(pose_data, side):
            self.rejection_count += 1
            return None
        
        self.prev_valid_pose = pose_data
        self.detection_count += 1
        
        return pose_data
    
    def _get_arm_landmarks(self, side: str):
        """Get landmark names for specified arm"""
        return [f'{side}_shoulder', f'{side}_elbow', f'{side}_wrist']
    
    def _check_visibility(self, landmarks, arm_landmark_names):
        """Check if arm landmarks are sufficiently visible"""
        for name in arm_landmark_names:
            idx = self.LANDMARKS[name]
            if landmarks[idx].visibility < self.min_visibility:
                return False
        return True
    
    def _is_valid_pose(self, pose_data: Dict, side: str) -> bool:
        """
        Check if pose is valid (outlier detection)
        
        Rejects poses with:
        - Impossible joint angles
        - Sudden large jumps from previous frame
        - Landmarks in wrong relative positions
        """
        if self.prev_valid_pose is None:
            return True  # First frame, accept it
        
        landmarks = pose_data['landmarks']
        prev_landmarks = self.prev_valid_pose['landmarks']
        
        # Check for sudden jumps (more than 20% of frame in one step)
        max_jump = 0.2
        arm_points = self._get_arm_landmarks(side)
        
        for name in arm_points:
            curr = np.array([landmarks[name]['x'], landmarks[name]['y']])
            prev = np.array([prev_landmarks[name]['x'], prev_landmarks[name]['y']])
            
            jump = np.linalg.norm(curr - prev)
            if jump > max_jump:
                return False  # Too large a jump, likely detection error
        
        # Check anatomical constraints
        shoulder = landmarks[f'{side}_shoulder']
        elbow = landmarks[f'{side}_elbow']
        wrist = landmarks[f'{side}_wrist']
        
        # Elbow should be between shoulder and wrist (roughly)
        shoulder_to_elbow = np.linalg.norm([
            elbow['x'] - shoulder['x'],
            elbow['y'] - shoulder['y']
        ])
        elbow_to_wrist = np.linalg.norm([
            wrist['x'] - elbow['x'],
            wrist['y'] - elbow['y']
        ])
        shoulder_to_wrist = np.linalg.norm([
            wrist['x'] - shoulder['x'],
            wrist['y'] - shoulder['y']
        ])
        
        # Reject if triangle inequality violated significantly
        # (indicates landmarks are in wrong order)
        if shoulder_to_wrist > (shoulder_to_elbow + elbow_to_wrist) * 1.3:
            return False
        
        return True
    
    def calculate_arm_angles(self, pose_data: Dict, side: str = 'right') -> Dict[str, float]:
        """Calculate arm angles from pose data"""
        if pose_data is None:
            return {}
        
        landmarks = pose_data['landmarks']
        
        shoulder = landmarks[f'{side}_shoulder']
        elbow = landmarks[f'{side}_elbow']
        wrist = landmarks[f'{side}_wrist']
        hip = landmarks[f'{side}_hip']
        
        # Calculate shoulder angle (relative to vertical)
        shoulder_hip_vec = np.array([hip['x'] - shoulder['x'], hip['y'] - shoulder['y']])
        shoulder_elbow_vec = np.array([elbow['x'] - shoulder['x'], elbow['y'] - shoulder['y']])
        shoulder_angle = self._angle_between_vectors(shoulder_hip_vec, shoulder_elbow_vec)
        
        # Calculate elbow angle
        elbow_shoulder_vec = np.array([shoulder['x'] - elbow['x'], shoulder['y'] - elbow['y']])
        elbow_wrist_vec = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])
        elbow_angle = self._angle_between_vectors(elbow_shoulder_vec, elbow_wrist_vec)
        
        # Calculate arm elevation (how high the arm is raised)
        arm_elevation = np.arctan2(-(elbow['y'] - shoulder['y']), elbow['x'] - shoulder['x'])
        arm_elevation_deg = np.degrees(arm_elevation)
        
        # Calculate lateral position
        lateral_angle = shoulder['x'] - hip['x']
        
        return {
            'shoulder_angle': shoulder_angle,
            'elbow_angle': elbow_angle,
            'arm_elevation': arm_elevation_deg,
            'lateral_position': lateral_angle,
            'shoulder_x': shoulder['x'],
            'shoulder_y': shoulder['y'],
            'elbow_x': elbow['x'],
            'elbow_y': elbow['y'],
            'wrist_x': wrist['x'],
            'wrist_y': wrist['y'],
        }
    
    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees"""
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        return np.degrees(angle_rad)
    
    def draw_pose(self, frame: np.ndarray, pose_data: Dict, side: str = 'right') -> np.ndarray:
        """Draw pose landmarks with focus on tracked arm"""
        if pose_data is None or 'raw_landmarks' not in pose_data:
            return frame
        
        annotated_frame = frame.copy()
        
        # Draw all landmarks
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            pose_data['raw_landmarks'],
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Highlight tracked arm
        landmarks = pose_data['landmarks']
        h, w, _ = frame.shape
        
        arm_points = [
            (landmarks[f'{side}_shoulder']['px'], landmarks[f'{side}_shoulder']['py']),
            (landmarks[f'{side}_elbow']['px'], landmarks[f'{side}_elbow']['py']),
            (landmarks[f'{side}_wrist']['px'], landmarks[f'{side}_wrist']['py'])
        ]
        
        # Draw thick line for tracked arm
        for i in range(len(arm_points) - 1):
            cv2.line(annotated_frame, arm_points[i], arm_points[i+1], (0, 255, 255), 5)
        
        # Draw circles on tracked arm joints
        for point in arm_points:
            cv2.circle(annotated_frame, point, 8, (0, 255, 255), -1)
        
        return annotated_frame
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
        rejection_rate = (self.rejection_count / self.frame_count * 100) if self.frame_count > 0 else 0
        
        return {
            'frames_processed': self.frame_count,
            'detections': self.detection_count,
            'rejections': self.rejection_count,
            'detection_rate': detection_rate,
            'rejection_rate': rejection_rate
        }
    
    def close(self):
        """Clean up resources"""
        self.pose.close()
