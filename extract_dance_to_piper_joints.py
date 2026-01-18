#!/usr/bin/env python3
"""
Extract dance movements from video using MediaPipe and convert to Piper robot joint angles.
Outputs a CSV file compatible with the trajectory runner.

Usage:
    python extract_dance_to_piper_joints.py <video_path> [--start <seconds>] [--end <seconds>] [--output <csv_file>]
    
Example:
    python extract_dance_to_piper_joints.py videoplayback.1768755854126.publer.com.mp4 --output dance_joints.csv
"""

import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
import mediapipe as mp
import os
import urllib.request
import math

# MediaPipe landmark indices
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15
LEFT_HIP = 23  # For reference frame

# Piper joint limits (radians)
J1_LIMIT = 2.688  # ±154°
J2_MIN, J2_MAX = 0.0, 3.403  # 0° to 195°
J3_MIN, J3_MAX = -3.054, 0.0  # -175° to 0°
J4_LIMIT = 1.850  # ±106°
J5_LIMIT = 1.309  # ±75°
J6_LIMIT = 1.745  # ±100°


def calculate_angle_3d(p1, p2, p3):
    """Calculate angle at p2 given three 3D points."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    # Normalize vectors
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
    
    # Calculate angle
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return angle


def calculate_angle_2d(p1, p2, p3):
    """Calculate angle at p2 given three 2D points (x, y)."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Normalize
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
    
    # Calculate angle
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return angle


def shoulder_angle_to_j2(shoulder_angle_deg):
    """
    Convert intuitive shoulder angle to Piper J2.
    
    MediaPipe gives us angle from vertical (0° = up, 90° = horizontal, 180° = down)
    Piper J2: 1.63 rad = up, 3.2 rad = horizontal, 0.5 rad = down
    
    Args:
        shoulder_angle_deg: Angle from vertical in degrees (0-180)
    
    Returns:
        J2 value in radians
    """
    # Map: 0° (up) -> 1.63 rad, 90° (horizontal) -> 3.2 rad, 180° (down) -> 0.5 rad
    if shoulder_angle_deg <= 90:
        # Up to horizontal: linear interpolation
        ratio = shoulder_angle_deg / 90.0
        return 1.63 + ratio * (3.2 - 1.63)
    else:
        # Horizontal to down: linear interpolation
        ratio = (shoulder_angle_deg - 90) / 90.0
        return 3.2 - ratio * (3.2 - 0.5)


def elbow_angle_to_j3(elbow_angle_deg):
    """
    Convert elbow angle to Piper J3.
    
    MediaPipe gives us angle at elbow (180° = straight, <180° = bent)
    Piper J3: 0 rad = straight, negative = bent
    
    Args:
        elbow_angle_deg: Elbow angle in degrees (typically 0-180)
    
    Returns:
        J3 value in radians (clamped to limits)
    """
    # 180° (straight) = 0 rad, bent = negative
    # Map: 180° -> 0 rad, 0° -> -3.054 rad (max bend)
    angle_rad = np.deg2rad(180 - elbow_angle_deg)
    # Clamp to limits
    return max(J3_MIN, min(J3_MAX, -angle_rad))


def calculate_base_rotation(shoulder, wrist, hip):
    """
    Calculate J1 (base rotation) from arm horizontal position.
    
    Args:
        shoulder: Shoulder position (x, y, z)
        wrist: Wrist position (x, y, z)
        hip: Hip position (x, y, z) for reference frame
    
    Returns:
        J1 value in radians
    """
    # Project arm to horizontal plane (ignore Y/vertical)
    # MediaPipe: x is left-right, z is depth
    arm_vec_2d = np.array([wrist[0] - shoulder[0], wrist[2] - shoulder[2]])
    
    # Calculate angle from forward direction (positive X in MediaPipe = right)
    forward = np.array([1.0, 0.0])
    
    if np.linalg.norm(arm_vec_2d) < 1e-6:
        return 0.0
    
    arm_norm = arm_vec_2d / np.linalg.norm(arm_vec_2d)
    cos_angle = np.clip(np.dot(arm_norm, forward), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    # Determine left/right from cross product (3D for proper sign)
    forward_3d = np.array([forward[0], 0.0, forward[1]])
    arm_norm_3d = np.array([arm_norm[0], 0.0, arm_norm[1]])
    cross_3d = np.cross(forward_3d, arm_norm_3d)
    if cross_3d[1] < 0:  # Y component of cross product
        angle = -angle
    
    # Clamp to limits
    return max(-J1_LIMIT, min(J1_LIMIT, angle))


class DanceToJointsExtractor:
    """Extract dance movements and convert to Piper joint angles."""
    
    def __init__(self):
        """Initialize MediaPipe Pose detector."""
        self.fps = 30.0
        
        # Try new API first (MediaPipe 0.10+)
        if hasattr(mp, 'tasks'):
            try:
                BaseOptions = mp.tasks.BaseOptions
                PoseLandmarker = mp.tasks.vision.PoseLandmarker
                PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode
                
                model_path = None
                local_model = os.path.join(os.getcwd(), 'pose_landmarker_full.task')
                if os.path.exists(local_model):
                    model_path = local_model
                else:
                    mp_path = os.path.dirname(mp.__file__)
                    model_dir = os.path.join(mp_path, 'tasks', 'data')
                    os.makedirs(model_dir, exist_ok=True)
                    
                    possible_paths = [
                        os.path.join(model_dir, 'pose_landmarker_full.task'),
                        os.path.join(model_dir, 'pose_landmarker_heavy.task'),
                        os.path.join(model_dir, 'pose_landmarker_lite.task'),
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            model_path = path
                            break
                    
                    if model_path is None:
                        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
                        model_path = os.path.join(model_dir, 'pose_landmarker_full.task')
                        print(f"Downloading MediaPipe model...")
                        urllib.request.urlretrieve(model_url, model_path)
                
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.3,
                    min_pose_presence_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
                
                self.pose_landmarker = PoseLandmarker.create_from_options(options)
                self.use_new_api = True
                self.timestamp_ms = 0
                print(f"✅ Using MediaPipe tasks API")
                return
            except Exception as e:
                print(f"Warning: Failed to initialize new API: {e}")
        
        # Fall back to old API
        if hasattr(mp, 'solutions'):
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_new_api = False
            print("✅ Using MediaPipe solutions API (legacy)")
        else:
            raise ImportError("MediaPipe not available")
    
    def extract_joints_from_frame(self, frame, frame_number, timestamp):
        """Extract joint angles from a single frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Run pose estimation
        if self.use_new_api:
            ImageFormat = mp.ImageFormat
            mp_image = mp.Image(image_format=ImageFormat.SRGB, data=frame_rgb)
            result = self.pose_landmarker.detect_for_video(mp_image, self.timestamp_ms)
            self.timestamp_ms += int(1000 / self.fps)
            
            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return None
            
            landmarks_list = result.pose_landmarks[0]
            class Landmark:
                def __init__(self, x, y, z, visibility):
                    self.x = x
                    self.y = y
                    self.z = z
                    self.visibility = visibility
            
            class Landmarks:
                def __init__(self, landmarks_list):
                    self.landmark = [None] * 33
                    for i, lm in enumerate(landmarks_list):
                        self.landmark[i] = Landmark(lm.x, lm.y, lm.z, lm.visibility)
            
            landmarks = Landmarks(landmarks_list)
        else:
            results = self.pose.process(frame_rgb)
            if not results.pose_landmarks:
                return None
            landmarks = results.pose_landmarks
        
        # Get key landmarks
        shoulder = landmarks.landmark[LEFT_SHOULDER]
        elbow = landmarks.landmark[LEFT_ELBOW]
        wrist = landmarks.landmark[LEFT_WRIST]
        hip = landmarks.landmark[LEFT_HIP]
        
        # Check visibility
        if shoulder.visibility < 0.3 or wrist.visibility < 0.3 or elbow.visibility < 0.3:
            return None
        
        # Convert to 3D coordinates (MediaPipe gives normalized 0-1)
        shoulder_3d = [shoulder.x, shoulder.y, shoulder.z]
        elbow_3d = [elbow.x, elbow.y, elbow.z]
        wrist_3d = [wrist.x, wrist.y, wrist.z]
        hip_3d = [hip.x, hip.y, hip.z]
        
        # Calculate angles properly
        # Shoulder angle: angle of arm relative to horizontal plane
        # We want the angle from horizontal (0°) to vertical (90°)
        arm_vec = np.array([wrist_3d[0] - shoulder_3d[0],
                           wrist_3d[1] - shoulder_3d[1],
                           wrist_3d[2] - shoulder_3d[2]])
        
        arm_length = np.linalg.norm(arm_vec)
        if arm_length < 1e-6:
            return None
        
        # Calculate elevation angle (angle from horizontal plane)
        # Y is vertical in MediaPipe (0 = top, 1 = bottom, so we invert)
        vertical_component = -arm_vec[1]  # Negative because Y increases downward
        horizontal_component = np.sqrt(arm_vec[0]**2 + arm_vec[2]**2)
        
        if horizontal_component < 1e-6:
            # Arm is straight up/down
            shoulder_angle_deg = 90.0 if vertical_component > 0 else -90.0
        else:
            # Calculate angle from horizontal (0° = horizontal, 90° = up)
            shoulder_angle_deg = np.degrees(np.arctan2(vertical_component, horizontal_component))
            # Normalize to 0-180 range
            if shoulder_angle_deg < 0:
                shoulder_angle_deg = 180 + shoulder_angle_deg
        
        # Elbow angle: angle at elbow joint
        upper_arm = np.array([elbow_3d[0] - shoulder_3d[0],
                              elbow_3d[1] - shoulder_3d[1],
                              elbow_3d[2] - shoulder_3d[2]])
        forearm = np.array([wrist_3d[0] - elbow_3d[0],
                           wrist_3d[1] - elbow_3d[1],
                           wrist_3d[2] - elbow_3d[2]])
        
        upper_arm_norm = upper_arm / (np.linalg.norm(upper_arm) + 1e-6)
        forearm_norm = forearm / (np.linalg.norm(forearm) + 1e-6)
        
        cos_elbow = np.clip(np.dot(upper_arm_norm, forearm_norm), -1.0, 1.0)
        elbow_angle_deg = np.degrees(np.arccos(cos_elbow))
        
        # Convert to Piper joints
        j1 = calculate_base_rotation(shoulder_3d, wrist_3d, hip_3d)
        j2 = shoulder_angle_to_j2(shoulder_angle_deg)
        j3 = elbow_angle_to_j3(elbow_angle_deg)
        j4 = 0.0  # Wrist roll - not easily determined from pose
        j5 = 0.0  # Wrist pitch - simplified
        j6 = 0.0  # Wrist rotation - simplified
        
        return {
            "frame_number": frame_number,
            "timestamp": round(timestamp, 3),
            "joint1": round(j1, 4),
            "joint2": round(j2, 4),
            "joint3": round(j3, 4),
            "joint4": round(j4, 4),
            "joint5": round(j5, 4),
            "joint6": round(j6, 4),
        }
    
    def smooth_joint_angles(self, joint_angles, window_size=5):
        """Apply moving average smoothing to reduce jerkiness."""
        if len(joint_angles) < window_size:
            return joint_angles
        
        smoothed = []
        for i in range(len(joint_angles)):
            # Get window of surrounding frames
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(joint_angles), i + window_size // 2 + 1)
            window = joint_angles[start_idx:end_idx]
            
            # Average joint values in window
            avg_joints = {
                'frame_number': joint_angles[i]['frame_number'],
                'timestamp': joint_angles[i]['timestamp'],
                'joint1': float(np.mean([w['joint1'] for w in window])),
                'joint2': float(np.mean([w['joint2'] for w in window])),
                'joint3': float(np.mean([w['joint3'] for w in window])),
                'joint4': float(np.mean([w['joint4'] for w in window])),
                'joint5': float(np.mean([w['joint5'] for w in window])),
                'joint6': float(np.mean([w['joint6'] for w in window])),
            }
            smoothed.append(avg_joints)
        
        return smoothed
    
    def process_video(self, video_path, start_time=0.0, end_time=None, output_path=None, downsample=2, smooth=True):
        """Process video and extract joint angles."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        self.fps = fps
        
        if end_time is None:
            end_time = duration
        
        print(f"Video info:")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Processing: {start_time:.2f}s to {end_time:.2f}s")
        print(f"  Downsampling: Every {downsample} frames (reduces jerkiness)")
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        if self.use_new_api:
            self.timestamp_ms = int(start_time * 1000)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        joint_angles = []
        frame_number = start_frame
        frames_processed = 0
        frames_with_pose = 0
        
        print(f"\nProcessing frames...")
        
        while frame_number < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Downsample: only process every Nth frame
            if (frame_number - start_frame) % downsample != 0:
                frame_number += 1
                continue
            
            timestamp = frame_number / fps
            joint_data = self.extract_joints_from_frame(frame, frame_number, timestamp)
            
            frames_processed += 1
            
            if joint_data:
                joint_angles.append(joint_data)
                frames_with_pose += 1
                
                if frames_with_pose % 30 == 0:
                    print(f"  Processed {frames_processed} frames, {frames_with_pose} with pose...")
            
            frame_number += 1
        
        cap.release()
        if self.use_new_api:
            self.pose_landmarker.close()
        else:
            self.pose.close()
        
        print(f"\nExtraction complete:")
        print(f"  Total frames processed: {frames_processed}")
        print(f"  Frames with pose: {frames_with_pose}")
        print(f"  Detection rate: {frames_with_pose/frames_processed*100:.1f}%")
        
        if frames_with_pose == 0:
            print("\n⚠️  No poses detected!")
            return None
        
        # Apply smoothing to reduce jerkiness
        if smooth and len(joint_angles) > 5:
            print(f"  Applying smoothing...")
            joint_angles = self.smooth_joint_angles(joint_angles, window_size=5)
            print(f"  Smoothed to {len(joint_angles)} waypoints")
        
        # Write CSV
        if output_path is None:
            output_path = video_path.stem + "_joints.csv"
        
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            f.write("time,joint1,joint2,joint3,joint4,joint5,joint6,description\n")
            for i, data in enumerate(joint_angles):
                desc = f"Frame {data['frame_number']}"
                f.write(f"{data['timestamp']:.3f},{data['joint1']:.4f},{data['joint2']:.4f},"
                       f"{data['joint3']:.4f},{data['joint4']:.4f},{data['joint5']:.4f},"
                       f"{data['joint6']:.4f},{desc}\n")
        
        print(f"\n✅ Joint angles saved to: {output_path}")
        print(f"   Total waypoints: {len(joint_angles)}")
        print(f"\nTo run the dance:")
        print(f"  cd 'Small arm /simulation'")
        # Calculate relative path from simulation directory to root
        csv_path = Path(output_path)
        if csv_path.is_absolute():
            # If absolute, calculate relative from simulation dir
            sim_dir = Path("Small arm /simulation").resolve()
            try:
                rel_path = csv_path.relative_to(sim_dir.parent.parent)
            except:
                rel_path = f"../../{csv_path.name}"
        else:
            # Relative path - from simulation dir, go up 2 levels
            rel_path = f"../../{csv_path.name}"
        print(f"  python run_csv_trajectory.py {rel_path}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract dance movements from video and convert to Piper joint angles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract entire video
  python extract_dance_to_piper_joints.py video.mp4
  
  # Extract specific time range with smoothing
  python extract_dance_to_piper_joints.py video.mp4 --start 10 --end 60 --downsample 3
  
  # Custom output with heavy smoothing
  python extract_dance_to_piper_joints.py video.mp4 --output dance.csv --downsample 2
        """
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--output", help="Output CSV file path")
    parser.add_argument("--downsample", type=int, default=2, 
                       help="Process every Nth frame (default: 2, reduces jerkiness)")
    parser.add_argument("--no-smooth", action="store_true",
                       help="Disable smoothing (faster but more jerky)")
    
    args = parser.parse_args()
    
    if args.downsample < 1:
        print("ERROR: Downsample must be >= 1")
        return 1
    
    try:
        extractor = DanceToJointsExtractor()
        extractor.process_video(
            args.video_path,
            start_time=args.start,
            end_time=args.end,
            output_path=args.output,
            downsample=args.downsample,
            smooth=not args.no_smooth
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
