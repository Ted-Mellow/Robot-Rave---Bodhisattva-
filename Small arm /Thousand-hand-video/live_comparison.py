#!/usr/bin/env python3
"""
Live Comparison - Side-by-side video and simulation
Shows pose detection on video alongside robot simulation in real-time
"""

import sys
import os
from pathlib import Path
import cv2
import time
import numpy as np
from threading import Thread, Lock
import json

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from pose_detector import PoseDetector
from motion_mapper import MotionMapper

try:
    from simulation.piper_simultion_corrected import PiperSimulation
except ImportError:
    print("❌ Could not import PiperSimulation")
    sys.exit(1)


class LiveComparison:
    """Real-time comparison of pose detection and robot simulation"""

    def __init__(self, video_path: str, urdf_path: str = None):
        """
        Initialize live comparison

        Args:
            video_path: Path to video file
            urdf_path: Optional path to robot URDF
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print("\n🎬 Initializing Live Comparison...")

        # Initialize pose detector
        print("   Loading pose detector...")
        self.detector = PoseDetector(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            use_greyscale=True
        )

        # Initialize motion mapper
        print("   Loading motion mapper...")
        self.mapper = MotionMapper(
            scaling_factor=0.8,
            z_offset=0.0,
            smooth_window=3  # Less smoothing for more responsive live control
        )

        # Initialize simulation
        print("   Loading simulation...")
        self.sim = PiperSimulation(urdf_path=urdf_path, gui=True)

        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n✅ Live comparison ready!")
        print(f"   Video: {self.video_path.name}")
        print(f"   Resolution: {self.width}x{self.height}")
        print(f"   FPS: {self.fps:.2f}")
        print(f"   Frames: {self.frame_count}")

        # Smoothing buffer for joint angles
        self.joint_history = []
        self.max_history = 5

        # Statistics tracking
        self.stats = {
            'frames_processed': 0,
            'poses_detected': 0,
            'poses_failed': 0,
            'avg_confidence': 0.0
        }

    def _smooth_joints(self, joints):
        """Apply temporal smoothing to joint angles"""
        self.joint_history.append(joints)
        if len(self.joint_history) > self.max_history:
            self.joint_history.pop(0)

        # Average over history
        avg_joints = np.mean(self.joint_history, axis=0)
        return avg_joints.tolist()

    def _draw_robot_joint_overlays(self, frame, pose_data, current_joints, arm_side='right'):
        """Draw green dots showing robot joint axis positions on the human body"""
        if not pose_data or 'landmarks' not in pose_data:
            return frame

        landmarks = pose_data['landmarks']
        h, w = frame.shape[:2]

        # Define mapping of robot joints to body landmarks
        # J1 (Base rotation) -> Hip center
        # J2 (Shoulder elevation) -> Shoulder
        # J3 (Elbow) -> Elbow
        # J4 (Wrist roll) -> Wrist
        # J5 (Wrist pitch) -> Wrist (slightly offset)
        # J6 (Wrist rotate) -> Wrist (end)

        shoulder_key = f'{arm_side}_shoulder'
        elbow_key = f'{arm_side}_elbow'
        wrist_key = f'{arm_side}_wrist'
        hip_key = f'{arm_side}_hip'

        joint_positions = []
        joint_labels = []

        if hip_key in landmarks:
            # J1 - Base (at hip)
            hip = landmarks[hip_key]
            joint_positions.append((hip['px'], hip['py']))
            joint_labels.append(('J1', current_joints[0] if current_joints else 0))

        if shoulder_key in landmarks:
            # J2 - Shoulder
            shoulder = landmarks[shoulder_key]
            joint_positions.append((shoulder['px'], shoulder['py']))
            joint_labels.append(('J2', current_joints[1] if current_joints else 0))

        if elbow_key in landmarks:
            # J3 - Elbow
            elbow = landmarks[elbow_key]
            joint_positions.append((elbow['px'], elbow['py']))
            joint_labels.append(('J3', current_joints[2] if current_joints else 0))

        if wrist_key in landmarks:
            # J4, J5, J6 - Wrist area (stacked with offset)
            wrist = landmarks[wrist_key]
            joint_positions.append((wrist['px'], wrist['py']))
            joint_labels.append(('J4', current_joints[3] if current_joints else 0))

            # J5 - slightly above wrist
            joint_positions.append((wrist['px'], wrist['py'] - 15))
            joint_labels.append(('J5', current_joints[4] if current_joints else 0))

            # J6 - further above wrist
            joint_positions.append((wrist['px'], wrist['py'] - 30))
            joint_labels.append(('J6', current_joints[5] if current_joints else 0))

        # Draw the joint overlays
        for i, ((x, y), (label, angle)) in enumerate(zip(joint_positions, joint_labels)):
            # Draw green dot for joint position
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 10, (0, 200, 0), 2)

            # Draw label with joint name
            label_text = f"{label}"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Position label to the right of the dot
            label_x = x + 15
            label_y = y + 5

            # Draw background for text
            cv2.rectangle(frame,
                         (label_x - 2, label_y - text_size[1] - 2),
                         (label_x + text_size[0] + 2, label_y + 2),
                         (0, 0, 0), -1)

            # Draw text
            cv2.putText(frame, label_text, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw angle value below
            angle_text = f"{angle:.2f}"
            cv2.putText(frame, angle_text, (label_x, label_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw connection lines between joints to show arm chain
        if len(joint_positions) >= 3:
            # Connect J1 -> J2 -> J3 with green lines
            for i in range(min(3, len(joint_positions) - 1)):
                cv2.line(frame, joint_positions[i], joint_positions[i+1],
                        (0, 200, 0), 2, cv2.LINE_AA)

        # Add legend
        legend_x = 10
        legend_y = h - 120
        cv2.rectangle(frame, (legend_x, legend_y - 20), (legend_x + 180, h - 10), (0, 0, 0), -1)
        cv2.putText(frame, "Robot Joint Axes:", (legend_x + 5, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        joint_descriptions = [
            "J1: Base rotation",
            "J2: Shoulder elev.",
            "J3: Elbow bend",
            "J4-6: Wrist"
        ]

        for i, desc in enumerate(joint_descriptions):
            cv2.circle(frame, (legend_x + 10, legend_y + 20 + i * 20), 4, (0, 255, 0), -1)
            cv2.putText(frame, desc, (legend_x + 20, legend_y + 25 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def _create_info_overlay(self, frame, pose_data, current_joints, frame_idx):
        """Add information overlay to frame"""
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (w - 10, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (0, 255, 0)
        y_pos = 35

        # Frame info
        cv2.putText(frame, f"Frame: {frame_idx}/{self.frame_count}",
                   (20, y_pos), font, font_scale, color, thickness)
        y_pos += 25

        # Time
        timestamp = frame_idx / self.fps
        cv2.putText(frame, f"Time: {timestamp:.2f}s",
                   (20, y_pos), font, font_scale, color, thickness)
        y_pos += 25

        # Pose detection status
        if pose_data:
            status_text = "Pose: DETECTED"
            status_color = (0, 255, 0)

            # Show arm angles
            if 'right_arm' in pose_data:
                arm = pose_data['right_arm']
                y_pos += 5
                cv2.putText(frame, f"Arm Elevation: {arm.get('arm_elevation', 0):.1f}deg",
                           (20, y_pos), font, font_scale * 0.8, (100, 255, 255), 1)
                y_pos += 20
                cv2.putText(frame, f"Elbow Angle: {arm.get('elbow_angle', 0):.1f}deg",
                           (20, y_pos), font, font_scale * 0.8, (100, 255, 255), 1)
        else:
            status_text = "Pose: NOT DETECTED"
            status_color = (0, 0, 255)

        cv2.putText(frame, status_text, (20, 60), font, font_scale, status_color, thickness)

        # Robot joint angles (bottom right)
        if current_joints:
            joint_text_y = h - 150
            cv2.rectangle(frame, (w - 250, h - 160), (w - 10, h - 10), (0, 0, 0), -1)

            cv2.putText(frame, "Robot Joints (rad):",
                       (w - 240, joint_text_y), font, 0.5, (0, 255, 255), 1)
            joint_text_y += 20

            joint_names = ['J1 Base', 'J2 Shldr', 'J3 Elbow', 'J4 Roll', 'J5 Pitch', 'J6 Rot']
            for i, (name, angle) in enumerate(zip(joint_names, current_joints)):
                cv2.putText(frame, f"{name}: {angle:.2f}",
                           (w - 240, joint_text_y + i * 20),
                           font, 0.45, (255, 255, 255), 1)

        # Statistics (top right)
        stats_x = w - 280
        cv2.rectangle(frame, (stats_x - 10, 10), (w - 10, 100), (0, 0, 0), -1)
        cv2.putText(frame, "Statistics:", (stats_x, 35), font, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Processed: {self.stats['frames_processed']}",
                   (stats_x, 55), font, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"Detected: {self.stats['poses_detected']}",
                   (stats_x, 75), font, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, f"Failed: {self.stats['poses_failed']}",
                   (stats_x, 95), font, 0.45, (0, 100, 255), 1)

        return frame

    def run(self,
            start_time: float = 0.0,
            end_time: float = None,
            speed: float = 1.0,
            arm_side: str = 'right',
            save_comparison: bool = False):
        """
        Run live comparison

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds (None = full video)
            speed: Playback speed multiplier
            arm_side: Which arm to track ('left' or 'right')
            save_comparison: Save comparison video to file
        """
        if end_time is None:
            end_time = self.frame_count / self.fps

        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)

        # Seek to start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Video writer for saving
        video_writer = None
        if save_comparison:
            output_path = "comparison_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, fourcc, self.fps / speed,
                (self.width, self.height)
            )
            print(f"\n💾 Saving comparison to: {output_path}")

        print("\n▶️  Starting live comparison...")
        print(f"   Time range: {start_time:.2f}s - {end_time:.2f}s")
        print(f"   Speed: {speed}x")
        print(f"   Tracking: {arm_side} arm")
        print(f"\n   Controls:")
        print(f"   - SPACE: Pause/Resume")
        print(f"   - ESC: Exit")
        print(f"   - S: Save current frame")
        print(f"   - R: Reset to start")
        print(f"   - +/-: Adjust speed")

        current_joints = [0.0, 1.2, -0.5, 0.0, 0.0, 0.0]
        frame_idx = start_frame
        paused = False
        playback_speed = speed

        try:
            while frame_idx < end_frame:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    # IMPORTANT: Process pose on COLOR frame (greyscale breaks MediaPipe detection!)
                    # MediaPipe's neural network needs color information for accurate detection
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.detector.pose.process(rgb_frame)

                    # Convert to greyscale for display (user preference)
                    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    grey = cv2.equalizeHist(grey)
                    display_frame = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

                    # Build pose_data from MediaPipe results
                    self.stats['frames_processed'] += 1
                    pose_data = None

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        h, w, _ = frame.shape
                        pose_data = {
                            'timestamp': frame_idx / self.fps,
                            'frame': frame_idx,
                            'landmarks': {},
                            'visibility': {},
                            'raw_landmarks': results.pose_landmarks
                        }

                        # Extract key landmarks
                        for name, idx in self.detector.LANDMARK_NAMES.items():
                            landmark = landmarks[idx]
                            pose_data['landmarks'][name] = {
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z,
                                'px': int(landmark.x * w),
                                'py': int(landmark.y * h)
                            }
                            pose_data['visibility'][name] = landmark.visibility

                    if pose_data:
                        self.stats['poses_detected'] += 1
                        pose_data['left_arm'] = self.detector.calculate_arm_angles(pose_data, 'left')
                        pose_data['right_arm'] = self.detector.calculate_arm_angles(pose_data, 'right')

                        # Map to robot joints
                        arm_key = f'{arm_side}_arm'
                        if arm_key in pose_data:
                            joints = self.mapper.map_arm_pose_to_joints(pose_data[arm_key])
                            current_joints = self._smooth_joints(joints)

                        # Draw pose on display frame (greyscale background)
                        display_frame = self.detector.draw_pose(display_frame, pose_data)

                        # Draw robot joint axis overlays (green dots on body)
                        display_frame = self._draw_robot_joint_overlays(display_frame, pose_data, current_joints, arm_side)
                    else:
                        self.stats['poses_failed'] += 1

                    # Add info overlay
                    frame = self._create_info_overlay(display_frame, pose_data, current_joints, frame_idx)

                    # Update robot
                    self.sim.set_joint_positions(current_joints)
                    frame_idx += 1

                # Step simulation
                self.sim.step()

                # Add paused indicator
                if paused:
                    h, w = frame.shape[:2]
                    cv2.rectangle(frame, (w//2 - 100, h//2 - 50),
                                (w//2 + 100, h//2 + 50), (0, 0, 0), -1)
                    cv2.putText(frame, "PAUSED", (w//2 - 80, h//2 + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

                # Resize if too large
                if self.width > 1280:
                    scale = 1280 / self.width
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)

                cv2.imshow('Pose Detection + Robot Simulation', frame)

                # Save frame if recording
                if save_comparison and video_writer and not paused:
                    video_writer.write(frame)

                # Handle keyboard input
                wait_time = max(1, int(1000 / (self.fps * playback_speed)))
                key = cv2.waitKey(wait_time) & 0xFF

                if key == 27:  # ESC
                    print("\n⚠️  Stopped by user")
                    break
                elif key == 32:  # SPACE
                    paused = not paused
                    print(f"   {'⏸️  Paused' if paused else '▶️  Resumed'}")
                elif key == ord('s') or key == ord('S'):
                    screenshot_path = f"comparison_frame_{frame_idx}.png"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"   📸 Saved screenshot: {screenshot_path}")
                elif key == ord('r') or key == ord('R'):
                    print("   🔄 Resetting to start...")
                    frame_idx = start_frame
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    self.joint_history = []
                    paused = False
                elif key == ord('+') or key == ord('='):
                    playback_speed = min(2.0, playback_speed + 0.1)
                    print(f"   Speed: {playback_speed:.1f}x")
                elif key == ord('-') or key == ord('_'):
                    playback_speed = max(0.1, playback_speed - 0.1)
                    print(f"   Speed: {playback_speed:.1f}x")

                # Progress update
                if frame_idx % 100 == 0:
                    progress = (frame_idx - start_frame) / (end_frame - start_frame) * 100
                    detection_rate = (self.stats['poses_detected'] / self.stats['frames_processed'] * 100) if self.stats['frames_processed'] > 0 else 0
                    print(f"   Progress: {progress:.1f}% | Detection rate: {detection_rate:.1f}%")

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if video_writer:
                video_writer.release()
                print(f"✅ Comparison video saved")

            # Print final statistics
            print(f"\n📊 Final Statistics:")
            print(f"   Frames processed: {self.stats['frames_processed']}")
            print(f"   Poses detected: {self.stats['poses_detected']}")
            print(f"   Detection rate: {self.stats['poses_detected'] / self.stats['frames_processed'] * 100:.1f}%")

    def close(self):
        """Clean up resources"""
        self.detector.close()
        self.sim.close()


def main():
    """Main entry point"""
    print("=" * 70)
    print("LIVE COMPARISON - POSE DETECTION + ROBOT SIMULATION")
    print("=" * 70)

    import argparse

    parser = argparse.ArgumentParser(
        description='Live side-by-side comparison of pose detection and robot simulation'
    )

    parser.add_argument('video', nargs='?', default='Cropped_thousandhand.mp4',
                       help='Input video file')
    parser.add_argument('--start', type=float, default=0.0,
                       help='Start time in seconds (default: 0.0)')
    parser.add_argument('--end', type=float, default=None,
                       help='End time in seconds (default: full video)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed (default: 1.0)')
    parser.add_argument('--arm', choices=['left', 'right'], default='right',
                       help='Which arm to track (default: right)')
    parser.add_argument('--save', action='store_true',
                       help='Save comparison video to file')
    parser.add_argument('--urdf', type=str,
                       help='Path to robot URDF file')

    args = parser.parse_args()

    # Check video exists
    if not Path(args.video).exists():
        print(f"❌ Video file not found: {args.video}")
        return 1

    # Create comparison
    try:
        comparison = LiveComparison(args.video, urdf_path=args.urdf)

        comparison.run(
            start_time=args.start,
            end_time=args.end,
            speed=args.speed,
            arm_side=args.arm,
            save_comparison=args.save
        )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if 'comparison' in locals():
            comparison.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
