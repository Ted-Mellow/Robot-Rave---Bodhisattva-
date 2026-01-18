#!/usr/bin/env python3
"""
Dance Player - Synchronized Video and Arm Playback

A GUI application that:
1. Pre-processes video to extract arm movement trajectory
2. Plays the video on screen (with configurable delay)
3. Sends movement commands to the arm in sync

The video is delayed so the robot appears synchronized when dancing.

Usage:
    python dance_player.py                      # Launch GUI
    python dance_player.py video.mp4            # Load video on launch
    python dance_player.py --preprocess video.mp4  # Pre-process and save trajectory
"""

import argparse
import sys
import os
import time
import json
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
from queue import Queue, Empty

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from arm_control import ArmController, SignalBridge, JointSignal
from arm_control.logging_config import Loggers, setup_logging


@dataclass
class TrajectoryFrame:
    """A single frame in the trajectory"""
    timestamp: float
    joint_angles: List[float]
    frame_number: int


@dataclass
class PlaybackState:
    """Current playback state"""
    is_playing: bool = False
    is_paused: bool = False
    current_time: float = 0.0
    total_duration: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    arm_connected: bool = False
    video_delay: float = 0.5  # seconds


class DancePlayer:
    """
    Synchronized video and arm dance player.

    Pre-processes video to extract trajectory, then plays back
    with configurable video delay so robot appears in sync.
    """

    def __init__(self,
                 video_path: Optional[str] = None,
                 video_delay: float = 0.5,
                 arm_side: str = 'right',
                 use_simulation: bool = False,
                 can_interface: str = 'can0',
                 remote_host: Optional[str] = None,
                 remote_user: str = 'pi',
                 remote_password: Optional[str] = None,
                 remote_key: Optional[str] = None,
                 smooth_window: int = 15):
        """
        Initialize dance player.

        Args:
            video_path: Optional video file to load
            video_delay: Delay video display by this many seconds (arm leads)
            arm_side: Which arm to track ('left' or 'right')
            use_simulation: Use PyBullet simulation instead of hardware
            can_interface: CAN interface for hardware mode (if not remote)
            remote_host: Raspberry Pi hostname/IP for remote control
            remote_user: SSH username for remote connection
            remote_password: SSH password (optional if using key)
            remote_key: Path to SSH private key
            smooth_window: Moving average window for trajectory smoothing (larger = smoother, default: 15)
        """
        self.log = Loggers.playback()
        self.log.info("=" * 60)
        self.log.info("DANCE PLAYER INITIALIZING")
        self.log.info("=" * 60)

        # Configuration
        self.video_path = video_path
        self.video_delay = video_delay
        self.arm_side = arm_side
        self.use_simulation = use_simulation
        self.can_interface = can_interface
        self.remote_host = remote_host
        self.remote_user = remote_user
        self.remote_password = remote_password
        self.remote_key = remote_key
        self.smooth_window = smooth_window

        self.log.info(f"Video delay: {video_delay}s (arm leads video)")
        self.log.info(f"Tracking: {arm_side} arm")
        
        if use_simulation:
            self.log.info(f"Mode: Simulation")
        elif remote_host:
            self.log.info(f"Mode: Remote ({remote_user}@{remote_host})")
        else:
            self.log.info(f"Mode: Hardware ({can_interface})")

        # State
        self.state = PlaybackState(video_delay=video_delay)
        self.trajectory: List[TrajectoryFrame] = []
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.video_fps: float = 30.0
        self.video_size: tuple = (640, 480)

        # Components
        self.controller = None
        self.signal_bridge = None
        self.pose_detector = None
        self.motion_mapper = None

        # Threading
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_queue: Queue = Queue(maxsize=30)

        # Callbacks
        self._on_frame_callback: Optional[Callable] = None
        self._on_state_change_callback: Optional[Callable] = None

    def initialize_arm(self, pybullet_gui: bool = True) -> bool:
        """
        Initialize arm controller.
        
        Args:
            pybullet_gui: Show PyBullet GUI window (set False for headless/video-only mode)
        """
        self.log.info("Initializing arm controller...")

        try:
            if self.use_simulation:
                mode = "GUI" if pybullet_gui else "headless"
                self.log.info(f"Creating simulation controller ({mode})...")
                self.controller = ArmController.create_simulation(gui=pybullet_gui)
            elif self.remote_host:
                self.log.info(f"Creating remote controller for {self.remote_host}...")
                self.controller = ArmController.create_remote(
                    host=self.remote_host,
                    username=self.remote_user,
                    password=self.remote_password,
                    key_filename=self.remote_key
                )
            else:
                self.log.info(f"Creating hardware controller on {self.can_interface}...")
                self.controller = ArmController.create_hardware(self.can_interface)

            if not self.controller.connect():
                self.log.error("Failed to connect to arm")
                return False

            if not self.controller.enable():
                self.log.error("Failed to enable arm")
                self.controller.disconnect()
                return False

            self.signal_bridge = SignalBridge(
                controller=self.controller,
                smooth_window=5,
                max_velocity=2.0
            )

            self.state.arm_connected = True
            self.log.info("Arm controller ready")
            return True

        except Exception as e:
            self.log.error(f"Arm initialization failed: {e}")
            import traceback
            self.log.error(traceback.format_exc())
            return False

    def shutdown_arm(self) -> None:
        """Shutdown arm controller."""
        if self.controller:
            self.log.info("Shutting down arm controller...")
            try:
                if not self.use_simulation:
                    self.controller.disable()
                self.controller.disconnect()
            except Exception as e:
                self.log.warning(f"Error during shutdown: {e}")

            self.controller = None
            self.state.arm_connected = False
            self.log.info("Arm controller shutdown complete")

    def load_video(self, video_path: str) -> bool:
        """
        Load a video file.

        Args:
            video_path: Path to video file

        Returns:
            True if loaded successfully
        """
        self.log.info(f"Loading video: {video_path}")

        if not os.path.exists(video_path):
            self.log.error(f"Video file not found: {video_path}")
            return False

        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            self.log.error(f"Could not open video: {video_path}")
            return False

        self.video_path = video_path
        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.state.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.state.total_duration = self.state.total_frames / self.video_fps
        self.video_size = (
            int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

        self.log.info(f"Video loaded: {self.state.total_frames} frames @ {self.video_fps:.1f} fps")
        self.log.info(f"Duration: {self.state.total_duration:.1f}s, Size: {self.video_size}")

        return True

    def preprocess_video(self,
                         progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """
        Pre-process video to extract trajectory.

        Args:
            progress_callback: Called with progress 0.0-1.0

        Returns:
            True if successful
        """
        if self.video_cap is None:
            self.log.error("No video loaded")
            return False

        self.log.info("Pre-processing video to extract trajectory...")

        # Initialize vision components
        if not self._init_vision():
            return False

        # Reset video to start
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.trajectory = []

        frame_idx = 0
        poses_detected = 0

        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break

            timestamp = frame_idx / self.video_fps

            # Detect pose and map to joints
            joint_angles = self._process_frame_for_trajectory(frame)

            if joint_angles is not None:
                self.trajectory.append(TrajectoryFrame(
                    timestamp=timestamp,
                    joint_angles=joint_angles,
                    frame_number=frame_idx
                ))
                poses_detected += 1

            frame_idx += 1

            # Progress callback
            if progress_callback and frame_idx % 10 == 0:
                progress = frame_idx / self.state.total_frames
                progress_callback(progress)

            # Log periodic progress
            if frame_idx % 100 == 0:
                self.log.info(f"Preprocessing: {frame_idx}/{self.state.total_frames} frames, "
                             f"{poses_detected} poses detected")

        # Reset video position
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        detection_rate = (poses_detected / self.state.total_frames * 100) if self.state.total_frames > 0 else 0
        self.log.info(f"Preprocessing complete: {len(self.trajectory)} trajectory frames")
        self.log.info(f"Detection rate: {detection_rate:.1f}%")

        return len(self.trajectory) > 0

    def _init_vision(self) -> bool:
        """Initialize pose detection components."""
        try:
            # Import from Thousand-hand-video
            thousand_hand_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "Thousand-hand-video"
            )
            sys.path.insert(0, thousand_hand_dir)

            from pose_detector import PoseDetector
            from motion_mapper import MotionMapper

            self.pose_detector = PoseDetector(
                model_complexity=1,
                min_detection_confidence=0.3,  # Lower for better detection
                min_tracking_confidence=0.3
            )
            self.motion_mapper = MotionMapper(
                scaling_factor=0.8,
                smooth_window=self.smooth_window
            )

            self.log.info("Vision components initialized")
            return True

        except ImportError as e:
            self.log.error(f"Failed to import vision components: {e}")
            self.log.error("Make sure Thousand-hand-video/ directory exists")
            return False

    def _process_frame_for_trajectory(self, frame) -> Optional[List[float]]:
        """Process a single frame and return joint angles."""
        try:
            result = self.pose_detector.process_frame(frame)
            if result is None:
                return None

            # result is a dict with pose data
            arm_angles = self.pose_detector.calculate_arm_angles(
                result,
                side=self.arm_side
            )

            if not arm_angles:
                self.log.debug("extract_arm_angles returned empty")
                return None

            joints = self.motion_mapper.map_arm_pose_to_joints(arm_angles)
            return joints

        except Exception as e:
            self.log.error(f"Frame processing error: {e}")
            import traceback
            self.log.error(traceback.format_exc())
            return None

    def save_trajectory(self, path: str) -> bool:
        """Save trajectory to JSON file."""
        if not self.trajectory:
            self.log.warning("No trajectory to save")
            return False

        data = {
            'version': '1.0',
            'video_path': self.video_path,
            'video_fps': self.video_fps,
            'arm_side': self.arm_side,
            'frame_count': len(self.trajectory),
            'duration': self.trajectory[-1].timestamp if self.trajectory else 0,
            'frames': [
                {
                    'timestamp': f.timestamp,
                    'joint_angles': f.joint_angles,
                    'frame_number': f.frame_number
                }
                for f in self.trajectory
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        self.log.info(f"Trajectory saved to: {path}")
        return True

    def load_trajectory(self, path: str) -> bool:
        """Load trajectory from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self.trajectory = [
                TrajectoryFrame(
                    timestamp=frame['timestamp'],
                    joint_angles=frame['joint_angles'],
                    frame_number=frame['frame_number']
                )
                for frame in data['frames']
            ]

            self.log.info(f"Trajectory loaded: {len(self.trajectory)} frames from {path}")
            return True

        except Exception as e:
            self.log.error(f"Failed to load trajectory: {e}")
            return False

    def set_video_delay(self, delay: float) -> None:
        """Set video delay in seconds."""
        self.video_delay = delay
        self.state.video_delay = delay
        self.log.info(f"Video delay set to {delay}s")

    def play(self) -> bool:
        """Start playback."""
        if not self.trajectory:
            self.log.error("No trajectory loaded - preprocess video first")
            return False

        if self.video_cap is None:
            self.log.error("No video loaded")
            return False

        if self.state.is_playing:
            self.log.warning("Already playing")
            return False

        self.log.info("=" * 40)
        self.log.info("STARTING PLAYBACK")
        self.log.info(f"Video delay: {self.video_delay}s")
        self.log.info(f"Arm connected: {self.state.arm_connected}")
        self.log.info("=" * 40)

        # Reset
        self._stop_event.clear()
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.state.current_frame = 0
        self.state.current_time = 0.0
        self.state.is_playing = True
        self.state.is_paused = False

        # Start playback thread
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

        return True

    def pause(self) -> None:
        """Pause playback."""
        if self.state.is_playing:
            self.state.is_paused = not self.state.is_paused
            self.log.info(f"Playback {'paused' if self.state.is_paused else 'resumed'}")

    def stop(self) -> None:
        """Stop playback."""
        self.log.info("Stopping playback...")
        self._stop_event.set()
        self.state.is_playing = False
        self.state.is_paused = False

        if self._playback_thread:
            self._playback_thread.join(timeout=2.0)
            self._playback_thread = None

        self.log.info("Playback stopped")

    def _playback_loop(self) -> None:
        """Main playback loop (runs in thread)."""
        self.log.info("Playback loop started")

        start_time = time.time()
        trajectory_idx = 0
        video_started = False
        frame_time = 1.0 / self.video_fps
        last_displayed_frame = -1

        try:
            while not self._stop_event.is_set():
                if self.state.is_paused:
                    time.sleep(0.01)
                    continue

                # Calculate current playback time
                elapsed = time.time() - start_time
                self.state.current_time = elapsed

                # --- ARM CONTROL (leads video) ---
                # Find trajectory frame for current time
                while trajectory_idx < len(self.trajectory):
                    traj_frame = self.trajectory[trajectory_idx]
                    if traj_frame.timestamp <= elapsed:
                        # Send to arm
                        if self.state.arm_connected and self.signal_bridge:
                            signal = JointSignal(
                                timestamp=traj_frame.timestamp,
                                joint_angles=traj_frame.joint_angles,
                                source='playback',
                                frame_id=traj_frame.frame_number
                            )
                            self.signal_bridge.send_signal(signal)

                            # Step simulation if needed
                            if self.use_simulation and hasattr(self.controller, 'step'):
                                self.controller.step(2)

                        trajectory_idx += 1
                    else:
                        break

                # --- VIDEO DISPLAY (delayed) ---
                # Video time = elapsed - delay
                video_time = elapsed - self.video_delay

                if video_time >= 0:
                    video_started = True
                    target_frame = int(video_time * self.video_fps)

                    # Seek to frame if needed (only for large jumps to avoid jitter)
                    current_video_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if abs(target_frame - current_video_frame) > 5:
                        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

                    # Read and display frame (only if it's a new frame to avoid redundancy)
                    if target_frame != last_displayed_frame:
                        ret, frame = self.video_cap.read()
                        if ret:
                            self.state.current_frame = target_frame
                            last_displayed_frame = target_frame

                            # Add overlay
                            frame = self._add_playback_overlay(frame, elapsed, trajectory_idx)

                            # Call frame callback or put in queue for main thread
                            if self._on_frame_callback:
                                self._on_frame_callback(frame)
                            else:
                                # Put frame in queue for main thread to display
                                # Clear old frames if queue is full to avoid lag
                                if self._frame_queue.full():
                                    try:
                                        self._frame_queue.get_nowait()  # Remove oldest frame
                                    except:
                                        pass
                                try:
                                    self._frame_queue.put(frame, block=False)
                                except:
                                    pass  # Should not happen after clearing above
                        else:
                            # End of video
                            self.log.info("Video playback complete")
                            break

                # Check if trajectory is complete
                if trajectory_idx >= len(self.trajectory) and not video_started:
                    # All trajectory sent, waiting for video to start
                    pass
                elif trajectory_idx >= len(self.trajectory) and \
                     self.state.current_frame >= self.state.total_frames - 1:
                    # Both complete
                    self.log.info("Playback complete")
                    break

                # Frame timing - sleep to maintain smooth playback without CPU spin
                # Target: 60 updates/sec for smooth robot control + video display
                time.sleep(0.016)  # ~60 fps update rate

        except Exception as e:
            self.log.error(f"Playback error: {e}")

        finally:
            self.state.is_playing = False
            cv2.destroyAllWindows()
            self.log.info("Playback loop ended")

    def _add_playback_overlay(self, frame, elapsed: float, traj_idx: int):
        """Add status overlay to frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Status text
        y = 30
        cv2.putText(frame, f"Time: {elapsed:.1f}s / {self.state.total_duration:.1f}s",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20

        cv2.putText(frame, f"Video delay: {self.video_delay:.1f}s",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += 20

        cv2.putText(frame, f"Frame: {self.state.current_frame}/{self.state.total_frames}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20

        arm_status = "Connected" if self.state.arm_connected else "Not Connected"
        color = (0, 255, 0) if self.state.arm_connected else (0, 0, 255)
        cv2.putText(frame, f"Arm: {arm_status}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 20

        cv2.putText(frame, f"Trajectory: {traj_idx}/{len(self.trajectory)}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Paused indicator
        if self.state.is_paused:
            cv2.putText(frame, "PAUSED", (w//2 - 50, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return frame

    def run_gui(self) -> None:
        """Run simple OpenCV-based GUI."""
        self.log.info("Starting GUI...")

        window_name = 'Dance Player - Press Q to quit, SPACE to play/pause'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Status display when no video
        status_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        while True:
            # Update status frame
            status_frame.fill(30)

            y = 50
            cv2.putText(status_frame, "DANCE PLAYER", (50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            y += 50

            cv2.putText(status_frame, f"Video: {self.video_path or 'None loaded'}",
                        (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y += 30

            cv2.putText(status_frame, f"Trajectory: {len(self.trajectory)} frames",
                        (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y += 30

            arm_status = "Connected" if self.state.arm_connected else "Not Connected"
            color = (0, 255, 0) if self.state.arm_connected else (0, 100, 255)
            cv2.putText(status_frame, f"Arm: {arm_status}",
                        (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y += 30

            cv2.putText(status_frame, f"Video delay: {self.video_delay}s",
                        (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y += 50

            cv2.putText(status_frame, "Controls:", (50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            y += 25
            cv2.putText(status_frame, "  SPACE - Play/Pause", (50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20
            cv2.putText(status_frame, "  P - Preprocess video", (50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20
            cv2.putText(status_frame, "  A - Connect arm", (50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20
            cv2.putText(status_frame, "  +/- - Adjust delay", (50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20
            cv2.putText(status_frame, "  Q - Quit", (50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if not self.state.is_playing:
                cv2.imshow(window_name, status_frame)

            key = cv2.waitKey(100) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                if self.state.is_playing:
                    self.pause()
                elif self.trajectory:
                    self.play()
            elif key == ord('p'):
                if self.video_path and not self.state.is_playing:
                    self.preprocess_video()
            elif key == ord('a'):
                if not self.state.arm_connected:
                    self.initialize_arm()
            elif key == ord('+') or key == ord('='):
                self.set_video_delay(self.video_delay + 0.1)
            elif key == ord('-'):
                self.set_video_delay(max(0, self.video_delay - 0.1))

        cv2.destroyAllWindows()
        self.stop()
        self.shutdown_arm()


def main():
    parser = argparse.ArgumentParser(description="Dance Player - Synchronized video and arm playback")

    parser.add_argument('video', nargs='?', help='Video file to load')
    parser.add_argument('--delay', '-d', type=float, default=0.5,
                        help='Video delay in seconds (default: 0.5)')
    parser.add_argument('--arm', default='right', choices=['left', 'right'],
                        help='Which arm to track (default: right)')
    parser.add_argument('--sim', action='store_true',
                        help='Use simulation instead of hardware')
    parser.add_argument('--interface', '-i', default='can0',
                        help='CAN interface (default: can0)')
    
    # Remote connection options
    remote_group = parser.add_argument_group('Remote control (SSH to Raspberry Pi)')
    remote_group.add_argument('--remote', '-r', metavar='HOST',
                              help='Raspberry Pi hostname or IP address')
    remote_group.add_argument('--user', '-u', default='pi',
                              help='SSH username (default: pi)')
    remote_group.add_argument('--password', '-p',
                              help='SSH password')
    remote_group.add_argument('--key', '-k',
                              help='Path to SSH private key')
    
    # Processing options
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess video and save trajectory')
    parser.add_argument('--trajectory', '-t',
                        help='Load pre-saved trajectory file')
    parser.add_argument('--auto-play', action='store_true',
                        help='Start playing immediately after loading')
    parser.add_argument('--smooth', '-s', type=int, default=15,
                        help='Smoothing window size (frames). Higher = smoother but more delay. (default: 15)')
    parser.add_argument('--show-both', action='store_true',
                        help='Show both video AND PyBullet windows (may be unstable on macOS)')

    args = parser.parse_args()

    # Initialize logging
    setup_logging("dance_player", level=10)  # DEBUG level

    # Create player
    player = DancePlayer(
        video_path=args.video,
        video_delay=args.delay,
        arm_side=args.arm,
        use_simulation=args.sim,
        can_interface=args.interface,
        remote_host=args.remote,
        remote_user=args.user,
        remote_password=args.password,
        remote_key=args.key,
        smooth_window=args.smooth
    )

    # Load video if provided
    if args.video:
        if not player.load_video(args.video):
            print("Failed to load video")
            return

    # Load trajectory if provided
    if args.trajectory:
        player.load_trajectory(args.trajectory)
    elif args.video and args.preprocess:
        # Preprocess video
        print("Preprocessing video...")
        if player.preprocess_video(lambda p: print(f"Progress: {p*100:.1f}%")):
            # Save trajectory
            traj_path = os.path.splitext(args.video)[0] + '_trajectory.json'
            player.save_trajectory(traj_path)
            print(f"Trajectory saved to: {traj_path}")
        else:
            print("Preprocessing failed")
            return

    # Initialize arm
    if args.sim or args.auto_play:
        # Show PyBullet GUI if --show-both flag is set, otherwise headless for stability
        pybullet_gui = args.show_both if args.auto_play else True
        player.initialize_arm(pybullet_gui=pybullet_gui)

    # Auto-play or run GUI
    if args.auto_play and player.trajectory:
        # Create OpenCV window BEFORE starting playback (required for thread safety)
        cv2.namedWindow('Dance Player', cv2.WINDOW_NORMAL)
        # Resize to half width if showing both windows side-by-side
        window_width = 720 if args.show_both else 1440
        window_height = 540 if args.show_both else 1080
        cv2.resizeWindow('Dance Player', window_width, window_height)
        if args.show_both:
            # Position on left side of screen
            cv2.moveWindow('Dance Player', 0, 100)
        
        player.play()
        # Main thread handles OpenCV display (required on macOS)
        try:
            frame_displayed = False
            while player.state.is_playing:
                try:
                    # Get frame from playback thread
                    frame = player._frame_queue.get(timeout=0.05)
                    cv2.imshow('Dance Player', frame)
                    frame_displayed = True
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        player.stop()
                        break
                    elif key == ord(' '):
                        player.pause()
                except Empty:
                    # No frame available yet, keep window responsive
                    if frame_displayed:
                        cv2.waitKey(1)
                    else:
                        # Show loading message
                        loading = np.zeros((200, 400, 3), dtype=np.uint8)
                        cv2.putText(loading, "Starting playback...", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.imshow('Dance Player', loading)
                        cv2.waitKey(10)
        except KeyboardInterrupt:
            player.stop()
        finally:
            cv2.destroyAllWindows()
    else:
        player.run_gui()


if __name__ == "__main__":
    main()
