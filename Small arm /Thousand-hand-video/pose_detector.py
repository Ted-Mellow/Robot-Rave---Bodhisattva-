#!/usr/bin/env python3
"""
Thousand-Hand Bodhisattva Pose Detection
Tracks arm movements from video using MediaPipe Pose Detection
Supports both color and greyscale processing for improved accuracy
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


class PoseDetector:
    """Detects and tracks human pose from video using MediaPipe"""

    def __init__(self,
                 model_complexity=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 use_greyscale=False):
        """
        Initialize MediaPipe Pose detector

        Args:
            model_complexity: 0, 1, or 2 (higher = more accurate but slower)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            use_greyscale: If True, convert to greyscale for better accuracy
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.use_greyscale = use_greyscale

        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True
        )

        # Key landmarks for arm tracking (MediaPipe indices)
        self.LANDMARK_NAMES = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
        }

        print(f"✅ PoseDetector initialized")
        print(f"   Model complexity: {model_complexity}")
        print(f"   Greyscale mode: {use_greyscale}")

    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process a single frame and extract pose landmarks

        Args:
            frame: RGB image as numpy array

        Returns:
            Dictionary with pose data, or None if no pose detected
        """
        # Convert to greyscale if enabled (improves contrast and accuracy)
        if self.use_greyscale:
            # Convert to greyscale then back to RGB for MediaPipe
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalization to improve contrast
            grey = cv2.equalizeHist(grey)
            # Convert back to BGR for MediaPipe (which expects RGB)
            frame_to_process = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        else:
            frame_to_process = frame

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        # Extract landmark positions
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        pose_data = {
            'timestamp': None,  # Will be filled by video processor
            'landmarks': {},
            'visibility': {},
            'raw_landmarks': results.pose_landmarks
        }

        # Extract key landmarks with normalized and pixel coordinates
        for name, idx in self.LANDMARK_NAMES.items():
            landmark = landmarks[idx]
            pose_data['landmarks'][name] = {
                'x': landmark.x,  # Normalized [0, 1]
                'y': landmark.y,  # Normalized [0, 1]
                'z': landmark.z,  # Depth (relative to hips)
                'px': int(landmark.x * w),  # Pixel coordinates
                'py': int(landmark.y * h)
            }
            pose_data['visibility'][name] = landmark.visibility

        return pose_data

    def draw_pose(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """
        Draw pose landmarks on frame

        Args:
            frame: Image frame
            pose_data: Pose data from process_frame()

        Returns:
            Frame with pose drawn
        """
        if pose_data is None or 'raw_landmarks' not in pose_data:
            return frame

        # Draw pose landmarks
        annotated_frame = frame.copy()
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            pose_data['raw_landmarks'],
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        return annotated_frame

    def calculate_arm_angles(self, pose_data: Dict, side: str = 'right') -> Dict[str, float]:
        """
        Calculate arm joint angles from pose data

        Args:
            pose_data: Pose data from process_frame()
            side: 'left' or 'right'

        Returns:
            Dictionary with shoulder and elbow angles in degrees
        """
        if pose_data is None:
            return {}

        landmarks = pose_data['landmarks']

        # Get relevant landmarks
        shoulder_key = f'{side}_shoulder'
        elbow_key = f'{side}_elbow'
        wrist_key = f'{side}_wrist'
        hip_key = f'{side}_hip'

        shoulder = landmarks[shoulder_key]
        elbow = landmarks[elbow_key]
        wrist = landmarks[wrist_key]
        hip = landmarks[hip_key]

        # Calculate shoulder angle (relative to vertical)
        # Angle between shoulder-hip line and shoulder-elbow line
        shoulder_hip_vec = np.array([hip['x'] - shoulder['x'], hip['y'] - shoulder['y']])
        shoulder_elbow_vec = np.array([elbow['x'] - shoulder['x'], elbow['y'] - shoulder['y']])

        shoulder_angle = self._angle_between_vectors(shoulder_hip_vec, shoulder_elbow_vec)

        # Calculate elbow angle
        # Angle between shoulder-elbow and elbow-wrist
        elbow_shoulder_vec = np.array([shoulder['x'] - elbow['x'], shoulder['y'] - elbow['y']])
        elbow_wrist_vec = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])

        elbow_angle = self._angle_between_vectors(elbow_shoulder_vec, elbow_wrist_vec)

        # Calculate arm elevation (how high the arm is raised)
        # Angle from horizontal
        arm_elevation = np.arctan2(-(elbow['y'] - shoulder['y']), elbow['x'] - shoulder['x'])
        arm_elevation_deg = np.degrees(arm_elevation)

        # Calculate lateral position (left-right)
        # Positive = away from body, negative = across body
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
        # Normalize vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

        # Calculate angle
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def close(self):
        """Clean up resources"""
        self.pose.close()


class VideoProcessor:
    """Process video and extract pose data"""

    def __init__(self, video_path: str, detector: PoseDetector):
        """
        Initialize video processor

        Args:
            video_path: Path to video file
            detector: PoseDetector instance
        """
        self.video_path = Path(video_path)
        self.detector = detector

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        print(f"\n📹 Video loaded: {self.video_path.name}")
        print(f"   Resolution: {self.width}x{self.height}")
        print(f"   FPS: {self.fps:.2f}")
        print(f"   Frames: {self.frame_count}")
        print(f"   Duration: {self.duration:.2f}s")

    def process_video(self,
                     start_time: float = 0.0,
                     end_time: Optional[float] = None,
                     skip_frames: int = 0,
                     show_preview: bool = True,
                     target_person: str = 'center') -> List[Dict]:
        """
        Process video and extract pose data

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds (None = end of video)
            skip_frames: Process every Nth frame (0 = process all)
            show_preview: Show video preview with pose overlay
            target_person: Which person to track ('center', 'left', 'right', 'auto')

        Returns:
            List of pose data dictionaries
        """
        if end_time is None:
            end_time = self.duration

        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)

        # Seek to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        pose_sequence = []
        frame_idx = start_frame
        processed_count = 0

        print(f"\n🎬 Processing video...")
        print(f"   Range: {start_time:.2f}s - {end_time:.2f}s")
        print(f"   Skip frames: {skip_frames}")
        print(f"   Target person: {target_person}")

        while frame_idx < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Skip frames if needed
            if skip_frames > 0 and (frame_idx - start_frame) % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            # Process frame
            timestamp = frame_idx / self.fps
            pose_data = self.detector.process_frame(frame)

            if pose_data is not None:
                pose_data['timestamp'] = timestamp
                pose_data['frame'] = frame_idx

                # Calculate arm angles for both arms
                pose_data['left_arm'] = self.detector.calculate_arm_angles(pose_data, 'left')
                pose_data['right_arm'] = self.detector.calculate_arm_angles(pose_data, 'right')

                pose_sequence.append(pose_data)
                processed_count += 1

            # Show preview
            if show_preview and pose_data is not None:
                annotated = self.detector.draw_pose(frame, pose_data)

                # Add timestamp
                cv2.putText(annotated, f"Time: {timestamp:.2f}s", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Resize for display if too large
                if self.width > 1280:
                    scale = 1280 / self.width
                    display_frame = cv2.resize(annotated, None, fx=scale, fy=scale)
                else:
                    display_frame = annotated

                cv2.imshow('Pose Detection', display_frame)

                # ESC to quit, SPACE to pause
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\n⚠️  Processing interrupted by user")
                    break
                elif key == 32:  # SPACE
                    print("⏸️  Paused - press SPACE to continue")
                    cv2.waitKey(0)

            # Progress update
            if processed_count % 30 == 0:
                progress = (frame_idx - start_frame) / (end_frame - start_frame) * 100
                print(f"   Progress: {progress:.1f}% ({processed_count} frames processed)")

            frame_idx += 1

        if show_preview:
            cv2.destroyAllWindows()

        print(f"\n✅ Processing complete!")
        print(f"   Processed {processed_count} frames")
        print(f"   Detected pose in {len(pose_sequence)} frames")

        return pose_sequence

    def save_pose_data(self, pose_sequence: List[Dict], output_path: str):
        """
        Save pose data to JSON file

        Args:
            pose_sequence: List of pose data from process_video()
            output_path: Output JSON file path
        """
        # Remove raw landmarks (not JSON serializable)
        clean_sequence = []
        for pose in pose_sequence:
            clean_pose = {k: v for k, v in pose.items() if k != 'raw_landmarks'}
            clean_sequence.append(clean_pose)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                'video': str(self.video_path),
                'fps': self.fps,
                'frame_count': len(clean_sequence),
                'poses': clean_sequence
            }, f, indent=2)

        print(f"\n💾 Pose data saved to: {output_path}")

    def close(self):
        """Release video capture"""
        self.cap.release()


if __name__ == "__main__":
    print("=" * 70)
    print("THOUSAND-HAND BODHISATTVA - POSE DETECTION")
    print("=" * 70)

    # Configuration
    VIDEO_PATH = "Cropped_thousandhand.mp4"
    OUTPUT_PATH = "pose_data.json"

    # Test with first 10 seconds
    START_TIME = 0.0
    END_TIME = 10.0

    # Create detector (try greyscale mode for better accuracy)
    print("\n🔍 Testing with greyscale mode for improved accuracy...")
    detector = PoseDetector(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        use_greyscale=True  # Enable greyscale processing
    )

    # Process video
    processor = VideoProcessor(VIDEO_PATH, detector)

    try:
        pose_sequence = processor.process_video(
            start_time=START_TIME,
            end_time=END_TIME,
            skip_frames=0,  # Process every frame
            show_preview=True,
            target_person='center'
        )

        # Save results
        if pose_sequence:
            processor.save_pose_data(pose_sequence, OUTPUT_PATH)

            # Print some stats
            print("\n📊 Statistics:")
            print(f"   Total poses detected: {len(pose_sequence)}")
            if len(pose_sequence) > 0:
                first = pose_sequence[0]
                print(f"   Sample right arm angles:")
                print(f"      Shoulder: {first['right_arm'].get('shoulder_angle', 0):.1f}°")
                print(f"      Elbow: {first['right_arm'].get('elbow_angle', 0):.1f}°")
                print(f"      Elevation: {first['right_arm'].get('arm_elevation', 0):.1f}°")
        else:
            print("\n⚠️  No poses detected in video!")

    finally:
        processor.close()
        detector.close()

    print("\n✅ Done!")
