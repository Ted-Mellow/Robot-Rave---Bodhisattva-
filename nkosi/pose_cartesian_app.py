import tkinter as tk
import cv2
import numpy as np
import time
from collections import deque
from datetime import datetime
import threading
import json
import os
from enum import Enum

import websocket
from dotenv import load_dotenv

# Import MediaPipe with better error handling
try:
    import mediapipe as mp
    # Verify MediaPipe has solutions attribute
    if not hasattr(mp, 'solutions'):
        raise ImportError(
            "MediaPipe installation appears incomplete. "
            "Please reinstall: pip install --upgrade mediapipe"
        )
except ImportError as e:
    raise ImportError(
        "MediaPipe is not installed. "
        "Please install it: pip install mediapipe"
    ) from e


class CalibrationState(Enum):
    WAITING = "waiting"
    COUNTDOWN = "countdown"
    CALIBRATED = "calibrated"


class PoseCartesianApp:
    # Arm-only connections (landmark indices)
    ARM_CONNECTIONS = [
        # Left arm (in MediaPipe terms - actually your right arm due to mirror)
        (11, 13),  # left shoulder -> left elbow
        (13, 15),  # left elbow -> left wrist
        (15, 17),  # left wrist -> left pinky
        (15, 19),  # left wrist -> left index
        (15, 21),  # left wrist -> left thumb
    ]

    # Arm landmark indices for drawing points
    ARM_LANDMARKS = {11, 13, 15, 17, 19, 21}

    # Calibration thresholds (degrees)
    ELBOW_STRAIGHT_THRESHOLD = 15  # How close to 180 degrees
    SHOULDER_HORIZONTAL_THRESHOLD = 15  # How close to 90 degrees
    COUNTDOWN_DURATION = 3.0  # seconds

    def __init__(self):
        # Load environment configuration
        load_dotenv()
        self.ws_url = os.getenv('WEBSOCKET_URL', 'ws://localhost:8000/ws')

        # Initialize Tkinter window
        self.window = tk.Tk()
        self.window.title("Pose Cartesian - Arm Tracking")
        self.window.geometry("800x650")
        self.window.protocol("WM_DELETE_WINDOW", self.cleanup)

        # Calibration state
        self.calibration_state = CalibrationState.WAITING
        self.countdown_start_time = None
        self.arm_length_pixels = None

        # Current angle values (for calibration check)
        self.current_left_shoulder = None
        self.current_left_elbow = None

        # Current normalized position
        self.current_x = None
        self.current_y = None

        # Raw pixel positions (for calculations)
        self.shoulder_pixel = None
        self.wrist_pixel = None

        # Ring buffers for logging (maxlen=100)
        self.position_buffer = deque(maxlen=100)
        self.received_buffer = deque(maxlen=100)
        self.buffer_lock = threading.Lock()

        # WebSocket state
        self.ws = None
        self.ws_thread = None
        self.ws_connected = False
        self.is_streaming = False

        # Throttling for logging (controlled by slider)
        self.last_log_time = 0

        # Setup UI
        self._setup_ui()

        # Force window to appear
        self.window.update()

        print("Initializing webcam...")
        # Initialize webcam - try multiple camera indices
        self.cap = None
        for camera_idx in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print(f"Webcam initialized on camera {camera_idx}")
                    break
                else:
                    cap.release()
            else:
                cap.release()
        
        if self.cap is None:
            raise RuntimeError("Failed to initialize webcam. No camera found on indices 0, 1, or 2.")

        print("Initializing MediaPipe...")
        # Initialize MediaPipe Pose with better error handling
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe initialized")
        except AttributeError as e:
            raise ImportError(
                "MediaPipe 'solutions' module not found. "
                "Please install/upgrade mediapipe: pip install --upgrade mediapipe"
            ) from e

        # FPS tracking
        self.prev_time = time.time()
        self.fps = 0

        # Store photo reference
        self.photo = None

        # Start the update loop
        self.window.after(100, self.update_frame)

    def _setup_ui(self):
        """Build the UI layout with video, controls, and log textboxes."""
        # Main container
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Top: Video display
        self.video_frame = tk.Frame(self.main_frame)
        self.video_frame.pack(side=tk.TOP, fill=tk.X)

        self.label = tk.Label(self.video_frame, text="Initializing camera...",
                              width=640, height=480)
        self.label.pack()

        # Bottom: Controls and logs
        self.bottom_frame = tk.Frame(self.main_frame)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=5)

        # Left side: Controls
        self.control_frame = tk.Frame(self.bottom_frame, width=120)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.control_frame.pack_propagate(False)

        # Calibration status
        self.calib_label = tk.Label(self.control_frame, text="Not Calibrated",
                                    fg="orange", font=("TkDefaultFont", 10, "bold"))
        self.calib_label.pack(pady=5)

        self.recalib_btn = tk.Button(self.control_frame, text="Recalibrate",
                                     command=self._on_recalibrate_click, width=10)
        self.recalib_btn.pack(pady=5)

        # Separator
        tk.Frame(self.control_frame, height=2, bg="gray").pack(fill=tk.X, pady=10)

        self.connect_btn = tk.Button(self.control_frame, text="Connect",
                                     command=self._on_connect_click, width=10)
        self.connect_btn.pack(pady=5)

        self.stream_btn = tk.Button(self.control_frame, text="Stream",
                                    command=self._on_stream_click, width=10,
                                    state=tk.DISABLED)
        self.stream_btn.pack(pady=5)

        self.status_label = tk.Label(self.control_frame, text="Disconnected",
                                     fg="red", wraplength=100)
        self.status_label.pack(pady=10)

        # Send rate slider (0ms = realtime, 5000ms = 5 seconds)
        self.rate_label = tk.Label(self.control_frame, text="Send Rate", font=("TkDefaultFont", 9))
        self.rate_label.pack(pady=(10, 0))

        self.rate_var = tk.IntVar(value=200)  # Default 200ms
        self.rate_slider = tk.Scale(
            self.control_frame,
            from_=0,
            to=5000,
            orient=tk.HORIZONTAL,
            variable=self.rate_var,
            length=100,
            command=self._on_rate_change
        )
        self.rate_slider.pack()

        self.rate_display = tk.Label(self.control_frame, text="200ms", font=("TkDefaultFont", 8))
        self.rate_display.pack()

        # Right side: Two log textboxes
        self.logs_frame = tk.Frame(self.bottom_frame)
        self.logs_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Position log (left textbox)
        self.sent_frame = tk.LabelFrame(self.logs_frame, text="Position Log")
        self.sent_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        self.sent_text = tk.Text(self.sent_frame, height=8, width=25, state=tk.DISABLED)
        self.sent_scrollbar = tk.Scrollbar(self.sent_frame, command=self.sent_text.yview)
        self.sent_text.config(yscrollcommand=self.sent_scrollbar.set)
        self.sent_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sent_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Received log (right textbox)
        self.recv_frame = tk.LabelFrame(self.logs_frame, text="Received Messages")
        self.recv_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2)

        self.recv_text = tk.Text(self.recv_frame, height=8, width=25, state=tk.DISABLED)
        self.recv_scrollbar = tk.Scrollbar(self.recv_frame, command=self.recv_text.yview)
        self.recv_text.config(yscrollcommand=self.recv_scrollbar.set)
        self.recv_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.recv_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def calculate_angle(self, a, b, c):
        """Calculate angle at point b given three points a, b, c.

        Returns angle in degrees (0-180).
        """
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
        """Calculate angle of vector from point_a to point_b relative to vertical (downward).

        Returns angle in degrees where 0 = straight down, 90 = horizontal, 180 = straight up.
        """
        vec = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]])
        vertical = np.array([0, 1])

        cos_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def check_calibration_pose(self):
        """Check if the arm is in the calibration pose (straight arm, horizontal)."""
        if self.current_left_elbow is None or self.current_left_shoulder is None:
            return False

        elbow_ok = abs(180 - self.current_left_elbow) <= self.ELBOW_STRAIGHT_THRESHOLD
        shoulder_ok = abs(90 - self.current_left_shoulder) <= self.SHOULDER_HORIZONTAL_THRESHOLD

        return elbow_ok and shoulder_ok

    def perform_calibration(self):
        """Perform calibration by calculating arm length in pixels."""
        if self.shoulder_pixel is None or self.wrist_pixel is None:
            return False

        dx = self.wrist_pixel[0] - self.shoulder_pixel[0]
        dy = self.wrist_pixel[1] - self.shoulder_pixel[1]
        self.arm_length_pixels = np.sqrt(dx * dx + dy * dy)

        print(f"Calibrated! Arm length: {self.arm_length_pixels:.1f} pixels")
        return True

    def calculate_normalized_position(self):
        """Calculate normalized (x, y) position of wrist relative to shoulder."""
        if self.arm_length_pixels is None or self.arm_length_pixels < 1:
            return None, None
        if self.shoulder_pixel is None or self.wrist_pixel is None:
            return None, None

        # x: positive = arm extended away from body
        # wrist.x > shoulder.x when arm is extended (in mirrored view, your right arm extends to the right)
        x = (self.wrist_pixel[0] - self.shoulder_pixel[0]) / self.arm_length_pixels

        # y: positive = upward (shoulder.y > wrist.y when arm is raised, since y increases downward)
        y = (self.shoulder_pixel[1] - self.wrist_pixel[1]) / self.arm_length_pixels

        # Clamp both values between 0 and 1
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        return round(x, 3), round(y, 3)

    def draw_arm_landmarks(self, frame, landmarks):
        """Draw arm landmarks and calibration UI on the frame."""
        h, w, _ = frame.shape

        # Get landmark positions
        shoulder = landmarks.landmark[11]  # Left shoulder
        elbow = landmarks.landmark[13]     # Left elbow
        wrist = landmarks.landmark[15]     # Left wrist

        # Store pixel positions
        if shoulder.visibility > 0.5:
            self.shoulder_pixel = (int(shoulder.x * w), int(shoulder.y * h))
        if wrist.visibility > 0.5:
            self.wrist_pixel = (int(wrist.x * w), int(wrist.y * h))

        # Calculate angles
        self.current_left_shoulder = None
        self.current_left_elbow = None

        if shoulder.visibility > 0.5 and elbow.visibility > 0.5:
            self.current_left_shoulder = self.calculate_angle_to_vertical(
                [shoulder.x, shoulder.y], [elbow.x, elbow.y]
            )

        if shoulder.visibility > 0.5 and elbow.visibility > 0.5 and wrist.visibility > 0.5:
            self.current_left_elbow = self.calculate_angle(
                [shoulder.x, shoulder.y], [elbow.x, elbow.y], [wrist.x, wrist.y]
            )

        # Draw connections (lines)
        for start_idx, end_idx in self.ARM_CONNECTIONS:
            start = landmarks.landmark[start_idx]
            end = landmarks.landmark[end_idx]

            if start.visibility > 0.5 and end.visibility > 0.5:
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 3)

        # Draw landmark points
        for idx in self.ARM_LANDMARKS:
            landmark = landmarks.landmark[idx]
            if landmark.visibility > 0.5:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2)

        # Draw angles for debugging
        if self.current_left_shoulder is not None and self.shoulder_pixel:
            cv2.putText(frame, f"S:{int(self.current_left_shoulder)}",
                        (self.shoulder_pixel[0] + 10, self.shoulder_pixel[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if self.current_left_elbow is not None and elbow.visibility > 0.5:
            ex, ey = int(elbow.x * w), int(elbow.y * h)
            cv2.putText(frame, f"E:{int(self.current_left_elbow)}",
                        (ex + 10, ey - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Draw calibration UI
        self._draw_calibration_ui(frame, w, h)

        # Draw position if calibrated
        if self.calibration_state == CalibrationState.CALIBRATED:
            self.current_x, self.current_y = self.calculate_normalized_position()
            if self.current_x is not None:
                cv2.putText(frame, f"x: {self.current_x:.2f}  y: {self.current_y:.2f}",
                            (w // 2 - 100, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def _draw_calibration_ui(self, frame, w, h):
        """Draw calibration status and instructions on the frame."""
        if self.calibration_state == CalibrationState.WAITING:
            # Show instructions
            cv2.putText(frame, "Hold arm straight & horizontal",
                        (w // 2 - 180, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show progress indicators
            if self.current_left_elbow is not None:
                elbow_diff = abs(180 - self.current_left_elbow)
                elbow_color = (0, 255, 0) if elbow_diff <= self.ELBOW_STRAIGHT_THRESHOLD else (0, 0, 255)
                cv2.putText(frame, f"Elbow: {int(self.current_left_elbow)}/180",
                            (10, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, elbow_color, 2)

            if self.current_left_shoulder is not None:
                shoulder_diff = abs(90 - self.current_left_shoulder)
                shoulder_color = (0, 255, 0) if shoulder_diff <= self.SHOULDER_HORIZONTAL_THRESHOLD else (0, 0, 255)
                cv2.putText(frame, f"Shoulder: {int(self.current_left_shoulder)}/90",
                            (10, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, shoulder_color, 2)

        elif self.calibration_state == CalibrationState.COUNTDOWN:
            # Show countdown
            elapsed = time.time() - self.countdown_start_time
            remaining = self.COUNTDOWN_DURATION - elapsed
            if remaining > 0:
                cv2.putText(frame, f"Hold... {remaining:.1f}s",
                            (w // 2 - 80, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # Draw countdown circle
            center = (w // 2, h // 2)
            radius = 80
            angle = int(360 * (elapsed / self.COUNTDOWN_DURATION))
            cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, (0, 255, 255), 5)

        elif self.calibration_state == CalibrationState.CALIBRATED:
            cv2.putText(frame, "CALIBRATED",
                        (w // 2 - 70, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _update_calibration_state(self):
        """Update the calibration state machine."""
        pose_ok = self.check_calibration_pose()

        if self.calibration_state == CalibrationState.WAITING:
            if pose_ok:
                self.calibration_state = CalibrationState.COUNTDOWN
                self.countdown_start_time = time.time()
                print("Calibration pose detected, starting countdown...")

        elif self.calibration_state == CalibrationState.COUNTDOWN:
            if not pose_ok:
                # Pose broken, reset
                self.calibration_state = CalibrationState.WAITING
                self.countdown_start_time = None
                print("Pose broken, resetting countdown")
            else:
                elapsed = time.time() - self.countdown_start_time
                if elapsed >= self.COUNTDOWN_DURATION:
                    # Countdown complete, calibrate!
                    if self.perform_calibration():
                        self.calibration_state = CalibrationState.CALIBRATED
                        self.calib_label.config(text="Calibrated", fg="green")
                    else:
                        self.calibration_state = CalibrationState.WAITING

        # Update UI label
        if self.calibration_state == CalibrationState.WAITING:
            self.calib_label.config(text="Not Calibrated", fg="orange")
        elif self.calibration_state == CalibrationState.COUNTDOWN:
            self.calib_label.config(text="Hold pose...", fg="yellow")

    def _on_recalibrate_click(self):
        """Reset calibration to allow re-calibration."""
        self.calibration_state = CalibrationState.WAITING
        self.countdown_start_time = None
        self.arm_length_pixels = None
        self.calib_label.config(text="Not Calibrated", fg="orange")
        print("Recalibration requested")

    def frame_to_photoimage(self, frame):
        """Convert OpenCV frame to Tkinter PhotoImage using PPM format."""
        h, w, _ = frame.shape
        ppm_header = f"P6\n{w} {h}\n255\n".encode()
        ppm_data = ppm_header + frame.tobytes()
        return tk.PhotoImage(data=ppm_data)

    def _create_log_entry(self):
        """Create a formatted log entry string from current position."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.current_x is not None and self.current_y is not None:
            return f"[{timestamp}] x:{self.current_x:+.2f} y:{self.current_y:+.2f}"
        return f"[{timestamp}] --"

    def _update_sent_log(self):
        """Refresh the position log textbox from the ring buffer."""
        self.sent_text.config(state=tk.NORMAL)
        self.sent_text.delete(1.0, tk.END)
        with self.buffer_lock:
            for entry in self.position_buffer:
                self.sent_text.insert(tk.END, entry + "\n")
        self.sent_text.see(tk.END)
        self.sent_text.config(state=tk.DISABLED)

    def _update_received_log(self):
        """Refresh the received messages textbox from the ring buffer."""
        self.recv_text.config(state=tk.NORMAL)
        self.recv_text.delete(1.0, tk.END)
        with self.buffer_lock:
            for entry in self.received_buffer:
                self.recv_text.insert(tk.END, entry + "\n")
        self.recv_text.see(tk.END)
        self.recv_text.config(state=tk.DISABLED)

    def _on_rate_change(self, value):
        """Handle send rate slider change."""
        ms = int(value)
        if ms == 0:
            self.rate_display.config(text="Realtime")
        elif ms >= 1000:
            self.rate_display.config(text=f"{ms/1000:.1f}s")
        else:
            self.rate_display.config(text=f"{ms}ms")

    def _maybe_log_and_send(self):
        """Log position at throttled rate and optionally send over WebSocket."""
        # Only log/send when calibrated
        if self.calibration_state != CalibrationState.CALIBRATED:
            return

        current_time = time.time()
        send_interval = self.rate_var.get() / 1000.0
        if current_time - self.last_log_time < send_interval:
            return

        if self.current_x is None or self.current_y is None:
            return

        self.last_log_time = current_time

        # Create log entry and add to buffer
        entry = self._create_log_entry()
        with self.buffer_lock:
            self.position_buffer.append(entry)
        self._update_sent_log()

        # Send over WebSocket
        if self.is_streaming and self.ws_connected and self.ws:
            message = {"target": [self.current_x, self.current_y]}
            try:
                self.ws.send(json.dumps(message))
            except Exception as e:
                print(f"WebSocket send error: {e}")

    def _on_connect_click(self):
        """Handle Connect button click - toggle WebSocket connection."""
        if self.ws_connected:
            self.is_streaming = False
            if self.ws:
                self.ws.close()
            self.connect_btn.config(text="Connect")
            self.stream_btn.config(state=tk.DISABLED, text="Stream")
            self.status_label.config(text="Disconnected", fg="red")
        else:
            self.connect_btn.config(state=tk.DISABLED, text="Connecting...")
            self.ws_thread = threading.Thread(target=self._ws_thread_run, daemon=True)
            self.ws_thread.start()

    def _on_stream_click(self):
        """Handle Stream button click - toggle streaming state."""
        if self.is_streaming:
            self.is_streaming = False
            self.stream_btn.config(text="Stream")
        else:
            self.is_streaming = True
            self.stream_btn.config(text="Stop")

    def _ws_thread_run(self):
        """WebSocket thread main loop."""
        def on_open(ws):
            self.ws_connected = True
            self.window.after(0, self._on_ws_connected)

        def on_message(ws, message):
            timestamp = datetime.now().strftime("%H:%M:%S")
            with self.buffer_lock:
                self.received_buffer.append(f"[{timestamp}] {message}")
            self.window.after(0, self._update_received_log)

        def on_error(ws, error):
            print(f"WebSocket error: {error}")
            self.window.after(0, lambda: self._on_ws_error(str(error)))

        def on_close(ws, close_status_code, close_msg):
            self.ws_connected = False
            self.window.after(0, self._on_ws_disconnected)

        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        self.ws.run_forever()

    def _on_ws_connected(self):
        """Called on main thread when WebSocket connects."""
        self.connect_btn.config(state=tk.NORMAL, text="Disconnect")
        self.stream_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Connected", fg="green")

    def _on_ws_error(self, error_msg):
        """Called on main thread when WebSocket has an error."""
        self.status_label.config(text="Error", fg="red")

    def _on_ws_disconnected(self):
        """Called on main thread when WebSocket disconnects."""
        self.ws_connected = False
        self.is_streaming = False
        self.connect_btn.config(state=tk.NORMAL, text="Connect")
        self.stream_btn.config(state=tk.DISABLED, text="Stream")
        self.status_label.config(text="Disconnected", fg="red")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run pose estimation
            results = self.pose.process(frame_rgb)

            # Draw arm landmarks if pose detected
            if results.pose_landmarks:
                self.draw_arm_landmarks(frame_rgb, results.pose_landmarks)

            # Update calibration state machine
            self._update_calibration_state()

            # Log position and optionally send over WebSocket (throttled)
            self._maybe_log_and_send()

            # Calculate FPS
            current_time = time.time()
            self.fps = 1 / (current_time - self.prev_time + 1e-6)
            self.prev_time = current_time

            # Draw FPS on frame
            cv2.putText(
                frame_rgb,
                f"FPS: {self.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Convert to PhotoImage and display
            self.photo = self.frame_to_photoimage(frame_rgb)
            self.label.configure(image=self.photo)

        # Schedule next update
        self.window.after(1, self.update_frame)

    def cleanup(self):
        """Release resources and close window."""
        self.is_streaming = False
        if self.ws:
            self.ws.close()

        self.pose.close()
        self.cap.release()
        self.window.destroy()

    def run(self):
        """Start the application."""
        self.window.mainloop()


if __name__ == "__main__":
    app = PoseCartesianApp()
    app.run()
