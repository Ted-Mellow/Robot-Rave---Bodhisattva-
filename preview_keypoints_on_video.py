#!/usr/bin/env python3
"""
Preview Keypoints on Video
Loads a video and JSON sequence, then displays the video with keypoints overlaid
to verify the extraction accuracy.

Usage:
    python preview_keypoints_on_video.py <video_path> <sequence_json> [--start-time <seconds>] [--end-time <seconds>]
"""

import sys
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import mediapipe as mp

# Try to load trained mapping model
try:
    from apply_trained_mapping import load_mapping_model, apply_mapping
    HAS_TRAINED_MAPPING = True
except ImportError:
    HAS_TRAINED_MAPPING = False


def draw_mediapipe_landmarks_on_frame(frame, keypoint_data):
    """
    Draw actual MediaPipe landmarks on frame by re-detecting pose.
    This shows where MediaPipe actually detected the arm landmarks.
    """
    # Initialize MediaPipe if not already done
    if not hasattr(draw_mediapipe_landmarks_on_frame, 'initialized'):
        draw_mediapipe_landmarks_on_frame.initialized = True
        
        if hasattr(mp, 'tasks'):
            # New API - use tasks
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            # Try to find model - check local directory first, then MediaPipe install dir
            import os
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
                
                # Try full model first, then fallback to others
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
                    # Download full model if not found
                    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
                    model_path = os.path.join(model_dir, 'pose_landmarker_full.task')
                    if not os.path.exists(model_path):
                        print(f"Downloading MediaPipe full pose model to {model_path}...")
                        print("This is a one-time download (~40MB)")
                        try:
                            import urllib.request
                            urllib.request.urlretrieve(model_url, model_path)
                            print("Model downloaded successfully")
                        except Exception as e:
                            print(f"Failed to download model: {e}")
                            model_path = None
            
            if model_path and os.path.exists(model_path):
                try:
                    options = PoseLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=model_path),
                        running_mode=VisionRunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=0.3,  # Lower for better detection
                        min_pose_presence_confidence=0.3,   # Lower for better tracking
                    )
                    draw_mediapipe_landmarks_on_frame.pose_landmarker = PoseLandmarker.create_from_options(options)
                    draw_mediapipe_landmarks_on_frame.use_new_api = True
                    print(f"✅ MediaPipe Tasks API initialized with {os.path.basename(model_path)}")
                except Exception as e:
                    print(f"⚠️ Failed to initialize MediaPipe Tasks API: {e}")
                    draw_mediapipe_landmarks_on_frame.pose_landmarker = None
                    draw_mediapipe_landmarks_on_frame.use_new_api = True
            else:
                print(f"⚠️ MediaPipe model not found. Please download pose_landmarker_full.task")
                draw_mediapipe_landmarks_on_frame.pose_landmarker = None
                draw_mediapipe_landmarks_on_frame.use_new_api = True
        else:
            # Old API
            try:
                draw_mediapipe_landmarks_on_frame.mp_pose = mp.solutions.pose
                draw_mediapipe_landmarks_on_frame.pose = draw_mediapipe_landmarks_on_frame.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                draw_mediapipe_landmarks_on_frame.use_new_api = False
                print("✅ MediaPipe Solutions API initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize MediaPipe Solutions API: {e}")
                draw_mediapipe_landmarks_on_frame.pose = None
                draw_mediapipe_landmarks_on_frame.use_new_api = False
    
    h, w, _ = frame.shape
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run pose detection
    if draw_mediapipe_landmarks_on_frame.use_new_api:
        if not hasattr(draw_mediapipe_landmarks_on_frame, 'pose_landmarker') or draw_mediapipe_landmarks_on_frame.pose_landmarker is None:
            # Draw a message if MediaPipe not initialized
            cv2.putText(frame, "MediaPipe not initialized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        try:
            ImageFormat = mp.ImageFormat
            mp_image = mp.Image(image_format=ImageFormat.SRGB, data=frame_rgb)
            result = draw_mediapipe_landmarks_on_frame.pose_landmarker.detect(mp_image)
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks_list = result.pose_landmarks[0]
                
                # Draw all landmarks (yellow dots) - MediaPipe has 33 landmarks
                for i, landmark in enumerate(landmarks_list):
                    if hasattr(landmark, 'visibility') and landmark.visibility > 0.3:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # Yellow
                        cv2.circle(frame, (cx, cy), 7, (0, 0, 0), 2)  # Black outline
                
                # Draw arm connections (left arm: indices 11, 13, 15)
                LEFT_SHOULDER = 11
                LEFT_ELBOW = 13
                LEFT_WRIST = 15
                
                if len(landmarks_list) > 15:
                    connections = [(LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST)]
                    for start_idx, end_idx in connections:
                        if start_idx < len(landmarks_list) and end_idx < len(landmarks_list):
                            start = landmarks_list[start_idx]
                            end = landmarks_list[end_idx]
                            if hasattr(start, 'visibility') and hasattr(end, 'visibility') and start.visibility > 0.3 and end.visibility > 0.3:
                                start_pt = (int(start.x * w), int(start.y * h))
                                end_pt = (int(end.x * w), int(end.y * h))
                                cv2.line(frame, start_pt, end_pt, (0, 255, 0), 4)  # Green line, thicker
                                
                                # Highlight key points with larger circles
                                if start_idx == LEFT_SHOULDER:
                                    cv2.circle(frame, start_pt, 15, (0, 255, 0), 4)  # Green for shoulder
                                    cv2.putText(frame, "S", (start_pt[0] + 20, start_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                if start_idx == LEFT_ELBOW:
                                    cv2.circle(frame, start_pt, 12, (255, 255, 0), 3)  # Yellow for elbow
                                if end_idx == LEFT_WRIST:
                                    cv2.circle(frame, end_pt, 15, (0, 0, 255), 4)  # Red for wrist
                                    cv2.putText(frame, "W", (end_pt[0] + 20, end_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # No pose detected
                cv2.putText(frame, "NO POSE DETECTED", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        except Exception as e:
            cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        if draw_mediapipe_landmarks_on_frame.pose is None:
            return frame
            
        results = draw_mediapipe_landmarks_on_frame.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Draw all landmarks
            for landmark in results.pose_landmarks.landmark:
                if landmark.visibility > 0.3:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)
                    cv2.circle(frame, (cx, cy), 7, (0, 0, 0), 1)
            
            # Draw arm connections
            LEFT_SHOULDER = 11
            LEFT_ELBOW = 13
            LEFT_WRIST = 15
            
            connections = [(LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST)]
            for start_idx, end_idx in connections:
                start = results.pose_landmarks.landmark[start_idx]
                end = results.pose_landmarks.landmark[end_idx]
                if start.visibility > 0.3 and end.visibility > 0.3:
                    start_pt = (int(start.x * w), int(start.y * h))
                    end_pt = (int(end.x * w), int(end.y * h))
                    cv2.line(frame, start_pt, end_pt, (0, 255, 0), 3)
                    
                    # Highlight key points
                    if start_idx == LEFT_SHOULDER:
                        cv2.circle(frame, start_pt, 10, (0, 255, 0), 3)  # Green for shoulder
                    if end_idx == LEFT_WRIST:
                        cv2.circle(frame, end_pt, 10, (0, 0, 255), 3)  # Red for wrist
    
    return frame


def draw_keypoints_on_frame(frame, keypoint_data, arm_length_pixels=None, mapping_model=None):
    """
    Draw normalized coordinate info as text overlay.
    The actual MediaPipe landmarks are drawn separately.
    """
    h, w, _ = frame.shape
    
    if keypoint_data is None or keypoint_data.get('x') is None or keypoint_data.get('y') is None:
        return frame
    
    x_norm = keypoint_data['x']
    y_norm = keypoint_data['y']
    
    # Draw coordinate text overlay
    coord_text = f"Normalized: x={x_norm:.3f} y={y_norm:.3f}"
    if mapping_model:
        coord_text += " [TRAINED]"
    cv2.putText(frame, coord_text, (w - 450, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, coord_text, (w - 450, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Draw angle info if available
    if keypoint_data.get('shoulder_angle') is not None:
        angle_text = f"Shoulder: {keypoint_data['shoulder_angle']:.0f}°  Elbow: {keypoint_data.get('elbow_angle', 0):.0f}°"
        cv2.putText(frame, angle_text, (w - 450, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, angle_text, (w - 450, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return frame


def preview_video_with_keypoints(video_path, sequence_json_path, start_time=None, end_time=None):
    """
    Preview video with keypoints overlaid.
    
    Args:
        video_path: Path to video file
        sequence_json_path: Path to JSON sequence file
        start_time: Optional start time in seconds (from sequence)
        end_time: Optional end time in seconds (from sequence)
    """
    # Load trained mapping model if available
    mapping_model = None
    if HAS_TRAINED_MAPPING:
        default_model = Path("keypoint_mapping_model.pkl")
        if default_model.exists():
            try:
                mapping_model = load_mapping_model(str(default_model))
                print(f"✅ Using trained mapping model for preview")
            except Exception as e:
                print(f"⚠️  Could not load mapping model: {e}")
    
    # Load sequence
    print(f"Loading sequence from: {sequence_json_path}")
    with open(sequence_json_path, 'r') as f:
        sequence_data = json.load(f)
    
    keypoints = sequence_data.get('keypoints', [])
    if not keypoints:
        print("ERROR: No keypoints found in sequence")
        return
    
    print(f"Loaded {len(keypoints)} keypoints")
    
    # Create lookup dict: frame_number -> keypoint_data
    keypoint_map = {kp['frame_number']: kp for kp in keypoints}
    
    # Open video
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine frame range
    seq_start_time = sequence_data.get('start_time', 0)
    seq_end_time = sequence_data.get('end_time', total_frames / fps)
    
    if start_time is None:
        start_time = seq_start_time
    if end_time is None:
        end_time = seq_end_time
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    print(f"Video: {fps:.2f} FPS, {total_frames} frames")
    print(f"Sequence: {seq_start_time:.2f}s to {seq_end_time:.2f}s")
    print(f"Preview: {start_time:.2f}s to {end_time:.2f}s (frames {start_frame} to {end_frame})")
    print("\nControls:")
    print("  SPACE: Pause/Resume")
    print("  'q': Quit")
    print("  's': Step frame (when paused)")
    print("  LEFT/RIGHT: Seek backward/forward")
    print("  '+'/'-': Speed up/slow down")
    print()
    
    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Playback state
    paused = False
    playback_speed = 1.0
    current_frame = start_frame
    
    # Statistics
    frames_with_keypoints = 0
    frames_without_keypoints = 0
    
    try:
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # Get keypoint for this frame
            keypoint_data = keypoint_map.get(current_frame)
            
            # Always draw MediaPipe landmarks first (shows actual detection)
            # This re-detects pose on each frame to show what MediaPipe sees
            frame = draw_mediapipe_landmarks_on_frame(frame, keypoint_data)
            
            # Then overlay normalized coordinate info
            if keypoint_data:
                frames_with_keypoints += 1
                frame = draw_keypoints_on_frame(frame, keypoint_data, mapping_model=mapping_model)
                status_text = "KEYPOINT DETECTED"
                status_color = (0, 255, 0)
            else:
                frames_without_keypoints += 1
                status_text = "NO KEYPOINT"
                status_color = (0, 0, 255)
            
            # Draw status
            timestamp = current_frame / fps
            status_bg = np.ones((40, 200, 3), dtype=np.uint8) * 128
            frame[10:50, 10:210] = status_bg
            cv2.putText(frame, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame, f"Frame: {current_frame}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Time: {timestamp:.2f}s", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw progress bar
            progress = (current_frame - start_frame) / (end_frame - start_frame)
            bar_width = int(w * 0.8)
            bar_height = 10
            bar_x = int(w * 0.1)
            bar_y = h - 30
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
            
            # Display
            cv2.imshow('Keypoint Preview - Press Q to quit, SPACE to pause', frame)
            
            # Handle keyboard input
            if paused:
                key = cv2.waitKey(0) & 0xFF
            else:
                delay = int(1000 / (fps * playback_speed))
                key = cv2.waitKey(delay) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('s') and paused:
                current_frame += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                continue
            elif key == 81 or key == 2:  # Left arrow
                current_frame = max(start_frame, current_frame - int(fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key == 83 or key == 3:  # Right arrow
                current_frame = min(end_frame - 1, current_frame + int(fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif key == ord('+') or key == ord('='):
                playback_speed = min(5.0, playback_speed * 1.2)
                print(f"Playback speed: {playback_speed:.1f}x")
            elif key == ord('-'):
                playback_speed = max(0.1, playback_speed / 1.2)
                print(f"Playback speed: {playback_speed:.1f}x")
            else:
                if not paused:
                    current_frame += 1
        
        print(f"\nPreview complete:")
        print(f"  Frames with keypoints: {frames_with_keypoints}")
        print(f"  Frames without keypoints: {frames_without_keypoints}")
        print(f"  Coverage: {frames_with_keypoints/(frames_with_keypoints+frames_without_keypoints)*100:.1f}%")
    
    except KeyboardInterrupt:
        print("\nPreview interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Preview keypoints overlaid on video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview entire sequence
  python preview_keypoints_on_video.py "video.mp4" sequence.json
  
  # Preview specific time range
  python preview_keypoints_on_video.py "video.mp4" sequence.json --start-time 20 --end-time 30
        """
    )
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("sequence_json", help="Path to JSON sequence file")
    parser.add_argument("--start-time", type=float, help="Start time in seconds (from sequence start)")
    parser.add_argument("--end-time", type=float, help="End time in seconds (from sequence start)")
    
    args = parser.parse_args()
    
    preview_video_with_keypoints(
        args.video_path,
        args.sequence_json,
        args.start_time,
        args.end_time
    )


if __name__ == "__main__":
    main()
