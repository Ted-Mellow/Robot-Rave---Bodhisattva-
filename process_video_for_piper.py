#!/usr/bin/env python3
"""
Video Processing Script for PiPER Keypoint Extraction
Crops video to specified time range, extracts keypoints using MediaPipe,
and normalizes coordinates for robot control.

Usage:
    python process_video_for_piper.py <video_path> <start_time> <end_time> [output_json]

Example:
    python process_video_for_piper.py "Thousand-Hand Bodhisattva _ CCTV English.mp4" 10 70 output_sequence.json
    # This will extract frames from 10 seconds to 70 seconds (1 minute)
"""

import sys
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import mediapipe as mp
from collections import deque
import os
import urllib.request

# Try to load trained mapping model
try:
    from apply_trained_mapping import load_mapping_model, apply_mapping, default_mapping
    HAS_TRAINED_MAPPING = True
except ImportError:
    HAS_TRAINED_MAPPING = False
    def default_mapping(x_raw, y_raw):
        x = (x_raw + 1.0) / 2.0
        y = (y_raw + 1.0) / 2.0
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


class VideoKeypointExtractor:
    """Extract and normalize keypoints from video for PiPER robot control."""
    
    # MediaPipe landmark indices (same as pose_cartesian_app.py)
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    
    def __init__(self, mapping_model_path=None):
        """Initialize MediaPipe Pose detector."""
        self.fps = 30.0  # Will be updated when processing video
        
        # Load trained mapping model if available
        self.mapping_model = None
        if mapping_model_path and HAS_TRAINED_MAPPING:
            try:
                self.mapping_model = load_mapping_model(mapping_model_path)
                print(f"✅ Loaded trained mapping model: {mapping_model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load mapping model: {e}")
                print("   Using default mapping")
        elif HAS_TRAINED_MAPPING:
            # Try default location
            default_model = Path("keypoint_mapping_model.pkl")
            if default_model.exists():
                try:
                    self.mapping_model = load_mapping_model(str(default_model))
                    print(f"✅ Loaded trained mapping model: {default_model}")
                except Exception as e:
                    print(f"⚠️  Failed to load mapping model: {e}")
                    print("   Using default mapping")
        
        # Try new API first (MediaPipe 0.10+)
        if hasattr(mp, 'tasks'):
            try:
                BaseOptions = mp.tasks.BaseOptions
                PoseLandmarker = mp.tasks.vision.PoseLandmarker
                PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode
                
                # Try to find model - check local directory first, then MediaPipe install dir
                model_path = None
                
                # First, check if model exists in current directory (user's local file)
                local_model = os.path.join(os.getcwd(), 'pose_landmarker_full.task')
                if os.path.exists(local_model):
                    model_path = local_model
                    print(f"Using local model: {model_path}")
                else:
                    # Check MediaPipe install directory
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
                        # Download the full model (best accuracy)
                        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
                        model_path = os.path.join(model_dir, 'pose_landmarker_full.task')
                        print(f"Downloading MediaPipe pose model to {model_path}...")
                        print("This is a one-time download (~40MB)")
                        try:
                            urllib.request.urlretrieve(model_url, model_path)
                            print("Model downloaded successfully")
                        except Exception as e:
                            raise FileNotFoundError(
                                f"Failed to download MediaPipe model: {e}\n"
                                "Please download manually:\n"
                                f"  wget -O {model_path} {model_url}"
                            )
                
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.3,  # Lower threshold for better detection
                    min_pose_presence_confidence=0.3,   # Lower threshold for better tracking
                    min_tracking_confidence=0.3,        # Lower threshold for smoother tracking
                )
                
                self.pose_landmarker = PoseLandmarker.create_from_options(options)
                self.use_new_api = True
                self.timestamp_ms = 0
                print(f"Using MediaPipe tasks API with model: {model_path}")
                return
            except Exception as e:
                print(f"Warning: Failed to initialize new API: {e}")
                print("Falling back to old API...")
        
        # Fall back to old API (MediaPipe < 0.10)
        if hasattr(mp, 'solutions'):
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_new_api = False
            print("Using MediaPipe solutions API (legacy)")
        else:
            raise ImportError(
                "MediaPipe API not available. Please install/upgrade mediapipe:\n"
                "  pip install --upgrade mediapipe"
            )
        
    def calculate_angle(self, a, b, c):
        """Calculate angle at point b given three points a, b, c."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        return angle
    
    def calculate_angle_to_vertical(self, point_a, point_b):
        """Calculate angle of vector from point_a to point_b relative to vertical."""
        vec = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]])
        vertical = np.array([0, 1])
        
        cos_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle
    
    def extract_keypoints_from_frame(self, frame, frame_number, timestamp):
        """
        Extract normalized keypoint coordinates from a single frame.
        
        Returns:
            dict with keys: frame_number, timestamp, x, y, arm_length_pixels, 
                           shoulder_angle, elbow_angle, or None if no pose detected
        """
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Run pose estimation based on API version
        if self.use_new_api:
            # New API (MediaPipe 0.10+)
            ImageFormat = mp.ImageFormat
            mp_image = mp.Image(image_format=ImageFormat.SRGB, data=frame_rgb)
            result = self.pose_landmarker.detect_for_video(mp_image, self.timestamp_ms)
            self.timestamp_ms += int(1000 / self.fps)  # Use actual video FPS
            
            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return None
            
            # Get first pose landmarks
            landmarks_list = result.pose_landmarks[0]
            # Convert to similar format as old API (list of landmarks indexed by landmark index)
            class Landmark:
                def __init__(self, x, y, visibility):
                    self.x = x
                    self.y = y
                    self.visibility = visibility
            
            class Landmarks:
                def __init__(self, landmarks_list):
                    # Create a list with 33 elements (MediaPipe has 33 pose landmarks)
                    self.landmark = [None] * 33
                    for i, lm in enumerate(landmarks_list):
                        self.landmark[i] = Landmark(lm.x, lm.y, lm.visibility)
            
            landmarks = Landmarks(landmarks_list)
        else:
            # Old API (MediaPipe < 0.10)
            results = self.pose.process(frame_rgb)
            
            if not results.pose_landmarks:
                return None
            
            landmarks = results.pose_landmarks
        
        # Get key landmarks
        shoulder = landmarks.landmark[self.LEFT_SHOULDER]
        elbow = landmarks.landmark[self.LEFT_ELBOW]
        wrist = landmarks.landmark[self.LEFT_WRIST]
        
        # Check visibility - lower threshold for better detection in complex scenes
        if shoulder.visibility < 0.3 or wrist.visibility < 0.3:
            return None
        
        # Convert to pixel coordinates
        shoulder_pixel = (int(shoulder.x * w), int(shoulder.y * h))
        elbow_pixel = (int(elbow.x * w), int(elbow.y * h))
        wrist_pixel = (int(wrist.x * w), int(wrist.y * h))
        
        # Calculate arm length in pixels
        dx = wrist_pixel[0] - shoulder_pixel[0]
        dy = wrist_pixel[1] - shoulder_pixel[1]
        arm_length_pixels = np.sqrt(dx * dx + dy * dy)
        
        if arm_length_pixels < 1:
            return None
        
        # Calculate normalized position (shoulder-relative)
        # x: positive = arm extended away from body (right in image)
        # We need to handle both positive and negative x values
        x_raw = (wrist_pixel[0] - shoulder_pixel[0]) / arm_length_pixels
        
        # y: positive = upward (wrist above shoulder)
        # We need to handle both positive and negative y values
        y_raw = (shoulder_pixel[1] - wrist_pixel[1]) / arm_length_pixels
        
        # Apply trained mapping if available, otherwise use default
        if self.mapping_model and HAS_TRAINED_MAPPING:
            x, y = apply_mapping(x_raw, y_raw, self.mapping_model)
        else:
            # Default normalization: map [-1, 1] to [0, 1]
            x = (x_raw + 1.0) / 2.0
            y = (y_raw + 1.0) / 2.0
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
        
        # Calculate angles for reference
        shoulder_angle = None
        elbow_angle = None
        
        if elbow.visibility > 0.5:
            shoulder_angle = self.calculate_angle_to_vertical(
                [shoulder.x, shoulder.y], [elbow.x, elbow.y]
            )
            elbow_angle = self.calculate_angle(
                [shoulder.x, shoulder.y], [elbow.x, elbow.y], [wrist.x, wrist.y]
            )
        
        return {
            "frame_number": frame_number,
            "timestamp": round(timestamp, 3),
            "x": round(x, 3),
            "y": round(y, 3),
            "arm_length_pixels": round(arm_length_pixels, 1),
            "shoulder_angle": round(shoulder_angle, 1) if shoulder_angle else None,
            "elbow_angle": round(elbow_angle, 1) if elbow_angle else None,
        }
    
    def process_video(self, video_path, start_time, end_time, output_path=None):
        """
        Process video and extract keypoints.
        
        Args:
            video_path: Path to input video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to save JSON output (default: video_name_sequence.json)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video info:")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Processing: {start_time:.2f}s to {end_time:.2f}s ({end_time - start_time:.2f}s)")
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Update FPS for timestamp calculation in new API
        self.fps = fps
        if self.use_new_api:
            self.timestamp_ms = int(start_time * 1000)  # Reset timestamp to start
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract keypoints
        keypoints_sequence = []
        frame_number = start_frame
        frames_processed = 0
        frames_with_pose = 0
        
        print(f"\nProcessing frames {start_frame} to {end_frame}...")
        
        while frame_number < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_number / fps
            keypoint_data = self.extract_keypoints_from_frame(frame, frame_number, timestamp)
            
            frames_processed += 1
            
            if keypoint_data:
                keypoints_sequence.append(keypoint_data)
                frames_with_pose += 1
                
                # Progress indicator
                if frames_with_pose % 30 == 0:
                    print(f"  Processed {frames_processed} frames, {frames_with_pose} with pose detected...")
            
            frame_number += 1
        
        cap.release()
        if self.use_new_api:
            self.pose_landmarker.close()
        else:
            self.pose.close()
        
        print(f"\nExtraction complete:")
        print(f"  Total frames processed: {frames_processed}")
        print(f"  Frames with pose: {frames_with_pose}")
        print(f"  Pose detection rate: {frames_with_pose/frames_processed*100:.1f}%")
        
        if frames_with_pose == 0:
            print("\n⚠️  WARNING: No poses detected in video segment!")
            print("   Check that:")
            print("   - The time range contains visible people")
            print("   - People are clearly visible (not too far/occluded)")
            print("   - Video quality is sufficient")
            return None
        
        # Prepare output data
        output_data = {
            "source_video": str(video_path),
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "fps": fps,
            "total_frames": frames_processed,
            "frames_with_pose": frames_with_pose,
            "keypoints": keypoints_sequence
        }
        
        # Save to JSON
        if output_path is None:
            output_path = video_path.stem + "_sequence.json"
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Sequence saved to: {output_path}")
        print(f"   Total keypoints: {len(keypoints_sequence)}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract keypoints from video for PiPER robot control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 1 minute from 10s to 70s
  python process_video_for_piper.py "video.mp4" 10 70
  
  # Extract with custom output name
  python process_video_for_piper.py "video.mp4" 10 70 my_sequence.json
        """
    )
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("start_time", type=float, help="Start time in seconds")
    parser.add_argument("end_time", type=float, help="End time in seconds")
    parser.add_argument("output_json", nargs="?", help="Output JSON file path (optional)")
    parser.add_argument("--mapping-model", help="Path to trained mapping model (default: keypoint_mapping_model.pkl)")
    
    args = parser.parse_args()
    
    if args.end_time <= args.start_time:
        print("ERROR: end_time must be greater than start_time")
        sys.exit(1)
    
    if args.end_time - args.start_time > 120:
        print("WARNING: Duration exceeds 2 minutes. Consider shorter segments.")
    
    try:
        extractor = VideoKeypointExtractor(mapping_model_path=args.mapping_model)
        extractor.process_video(
            args.video_path,
            args.start_time,
            args.end_time,
            args.output_json
        )
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
