import tkinter as tk
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from datetime import datetime
import threading
import json
import os

import websocket
from dotenv import load_dotenv


class PoseApp:
    # Arm-only connections (landmark indices)
    ARM_CONNECTIONS = [
        # Left arm
        (11, 13),  # left shoulder -> left elbow
        (13, 15),  # left elbow -> left wrist
        (15, 17),  # left wrist -> left pinky
        (15, 19),  # left wrist -> left index
        (15, 21),  # left wrist -> left thumb
        # Right arm
        (12, 14),  # right shoulder -> right elbow
        (14, 16),  # right elbow -> right wrist
        (16, 18),  # right wrist -> right pinky
        (16, 20),  # right wrist -> right index
        (16, 22),  # right wrist -> right thumb
    ]

    # Arm landmark indices for drawing points
    ARM_LANDMARKS = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}

    def __init__(self):
        # Load environment configuration
        load_dotenv()
        self.ws_url = os.getenv('WEBSOCKET_URL', 'ws://localhost:8000/ws')

        # Initialize Tkinter window
        self.window = tk.Tk()
        self.window.title("Pose Estimation - Arms Only")
        self.window.geometry("800x650")
        self.window.protocol("WM_DELETE_WINDOW", self.cleanup)

        # Ring buffers for logging (maxlen=100)
        self.angle_buffer = deque(maxlen=100)
        self.received_buffer = deque(maxlen=100)
        self.buffer_lock = threading.Lock()

        # Current angle values (updated each frame)
        # Shoulder angles (relative to vertical)
        self.current_left_shoulder = None
        self.current_right_shoulder = None
        # Elbow angles
        self.current_left_elbow = None
        self.current_right_elbow = None
        # Wrist angles
        self.current_left_wrist = None
        self.current_right_wrist = None

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
        # Initialize webcam
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Webcam initialized")

        print("Initializing MediaPipe...")
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe initialized")

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

        # Sent log (left textbox)
        self.sent_frame = tk.LabelFrame(self.logs_frame, text="Angle Log")
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

        # Vectors from elbow to shoulder and elbow to wrist
        ba = a - b
        bc = c - b

        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Handle floating point errors
        angle = np.degrees(np.arccos(cosine_angle))

        return angle

    def draw_arm_landmarks(self, frame, landmarks):
        """Draw only arm landmarks and connections on the frame."""
        h, w, _ = frame.shape

        # Draw connections (lines)
        for start_idx, end_idx in self.ARM_CONNECTIONS:
            start = landmarks.landmark[start_idx]
            end = landmarks.landmark[end_idx]

            # Only draw if both points are visible
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

        # Draw arm angles (shoulder, elbow, wrist)
        self.draw_arm_angles(frame, landmarks, w, h)

    def calculate_angle_to_vertical(self, point_a, point_b):
        """Calculate angle of vector from point_a to point_b relative to vertical (downward).

        Returns angle in degrees where 0° = straight down, 90° = horizontal, 180° = straight up.
        """
        # Vector from shoulder to elbow (or reference point to target)
        vec = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]])
        # Vertical reference pointing down (positive y in image coordinates)
        vertical = np.array([0, 1])

        # Calculate angle using dot product
        cos_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def draw_arm_angles(self, frame, landmarks, w, h):
        """Draw shoulder, elbow, and wrist angles and store current values."""
        # Reset all current angles
        self.current_left_shoulder = None
        self.current_right_shoulder = None
        self.current_left_elbow = None
        self.current_right_elbow = None
        self.current_left_wrist = None
        self.current_right_wrist = None

        # Arm configurations: (shoulder_idx, elbow_idx, wrist_idx, index_finger_idx, side)
        arms = [
            (11, 13, 15, 19, "left"),   # Left arm
            (12, 14, 16, 20, "right"),  # Right arm
        ]

        for shoulder_idx, elbow_idx, wrist_idx, index_idx, side in arms:
            shoulder = landmarks.landmark[shoulder_idx]
            elbow = landmarks.landmark[elbow_idx]
            wrist = landmarks.landmark[wrist_idx]
            index_finger = landmarks.landmark[index_idx]

            # --- SHOULDER ANGLE (relative to vertical) ---
            if shoulder.visibility > 0.5 and elbow.visibility > 0.5:
                shoulder_pos = [shoulder.x, shoulder.y]
                elbow_pos = [elbow.x, elbow.y]

                shoulder_angle = self.calculate_angle_to_vertical(shoulder_pos, elbow_pos)

                if side == "left":
                    self.current_left_shoulder = round(shoulder_angle, 1)
                else:
                    self.current_right_shoulder = round(shoulder_angle, 1)

                # Draw shoulder angle near shoulder
                sx, sy = int(shoulder.x * w), int(shoulder.y * h)
                cv2.putText(frame, f"S:{int(shoulder_angle)}", (sx + 10, sy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Cyan

            # --- ELBOW ANGLE ---
            if (shoulder.visibility > 0.5 and elbow.visibility > 0.5 and wrist.visibility > 0.5):
                shoulder_pos = [shoulder.x, shoulder.y]
                elbow_pos = [elbow.x, elbow.y]
                wrist_pos = [wrist.x, wrist.y]

                elbow_angle = self.calculate_angle(shoulder_pos, elbow_pos, wrist_pos)

                if side == "left":
                    self.current_left_elbow = round(elbow_angle, 1)
                else:
                    self.current_right_elbow = round(elbow_angle, 1)

                # Draw elbow angle near elbow
                ex, ey = int(elbow.x * w), int(elbow.y * h)
                cv2.putText(frame, f"E:{int(elbow_angle)}", (ex + 10, ey - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)  # Yellow

            # --- WRIST ANGLE ---
            if (elbow.visibility > 0.5 and wrist.visibility > 0.5 and index_finger.visibility > 0.5):
                elbow_pos = [elbow.x, elbow.y]
                wrist_pos = [wrist.x, wrist.y]
                index_pos = [index_finger.x, index_finger.y]

                wrist_angle = self.calculate_angle(elbow_pos, wrist_pos, index_pos)

                if side == "left":
                    self.current_left_wrist = round(wrist_angle, 1)
                else:
                    self.current_right_wrist = round(wrist_angle, 1)

                # Draw wrist angle near wrist
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, f"W:{int(wrist_angle)}", (wx + 10, wy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)  # Magenta

    def frame_to_photoimage(self, frame):
        """Convert OpenCV frame to Tkinter PhotoImage using PPM format."""
        # Frame should already be RGB
        h, w, _ = frame.shape
        # Create PPM header and data
        ppm_header = f"P6\n{w} {h}\n255\n".encode()
        ppm_data = ppm_header + frame.tobytes()
        return tk.PhotoImage(data=ppm_data)

    def _create_log_entry(self):
        """Create a formatted log entry string from current angles."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Format: S=shoulder, E=elbow, W=wrist
        ls = f"{int(self.current_left_shoulder)}" if self.current_left_shoulder else "-"
        le = f"{int(self.current_left_elbow)}" if self.current_left_elbow else "-"
        lw = f"{int(self.current_left_wrist)}" if self.current_left_wrist else "-"
        rs = f"{int(self.current_right_shoulder)}" if self.current_right_shoulder else "-"
        re = f"{int(self.current_right_elbow)}" if self.current_right_elbow else "-"
        rw = f"{int(self.current_right_wrist)}" if self.current_right_wrist else "-"
        return f"[{timestamp}] L:{ls}/{le}/{lw} R:{rs}/{re}/{rw}"

    def _update_sent_log(self):
        """Refresh the sent/angle log textbox from the ring buffer."""
        self.sent_text.config(state=tk.NORMAL)
        self.sent_text.delete(1.0, tk.END)
        with self.buffer_lock:
            for entry in self.angle_buffer:
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
        """Log angles at throttled rate and optionally send over WebSocket."""
        current_time = time.time()
        # Use slider value (ms) converted to seconds
        send_interval = self.rate_var.get() / 1000.0
        if current_time - self.last_log_time < send_interval:
            return  # Throttled

        # Check if we have any angle data for logging
        has_left = any([self.current_left_shoulder, self.current_left_elbow, self.current_left_wrist])
        has_right = any([self.current_right_shoulder, self.current_right_elbow, self.current_right_wrist])
        if not has_left and not has_right:
            return

        self.last_log_time = current_time

        # Create log entry and add to buffer
        entry = self._create_log_entry()
        with self.buffer_lock:
            self.angle_buffer.append(entry)
        self._update_sent_log()

        # Send over WebSocket only if ALL 6 angles are available
        if self.is_streaming and self.ws_connected and self.ws:
            all_angles_present = all([
                self.current_left_shoulder is not None,
                self.current_left_elbow is not None,
                self.current_left_wrist is not None,
                self.current_right_shoulder is not None,
                self.current_right_elbow is not None,
                self.current_right_wrist is not None
            ])
            if all_angles_present:
                message = {
                    "left": [0, self.current_left_shoulder, self.current_left_elbow, self.current_left_wrist],
                    "right": [0, self.current_right_shoulder, self.current_right_elbow, self.current_right_wrist]
                }
                try:
                    self.ws.send(json.dumps(message))
                except Exception as e:
                    print(f"WebSocket send error: {e}")

    def _on_connect_click(self):
        """Handle Connect button click - toggle WebSocket connection."""
        if self.ws_connected:
            # Disconnect
            self.is_streaming = False
            if self.ws:
                self.ws.close()
            self.connect_btn.config(text="Connect")
            self.stream_btn.config(state=tk.DISABLED, text="Stream")
            self.status_label.config(text="Disconnected", fg="red")
        else:
            # Connect
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
        self.status_label.config(text=f"Error", fg="red")

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

            # Log angles and optionally send over WebSocket (throttled)
            self._maybe_log_and_send()

            # Calculate FPS
            current_time = time.time()
            self.fps = 1 / (current_time - self.prev_time)
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

            # Convert to PhotoImage and display (using PPM, no PIL needed)
            self.photo = self.frame_to_photoimage(frame_rgb)
            self.label.configure(image=self.photo)

        # Schedule next update
        self.window.after(1, self.update_frame)

    def cleanup(self):
        """Release resources and close window."""
        # Stop streaming and close WebSocket
        self.is_streaming = False
        if self.ws:
            self.ws.close()

        # Close pose estimation
        self.pose.close()

        # Release camera
        self.cap.release()

        # Destroy window
        self.window.destroy()

    def run(self):
        """Start the application."""
        self.window.mainloop()


if __name__ == "__main__":
    app = PoseApp()
    app.run()
