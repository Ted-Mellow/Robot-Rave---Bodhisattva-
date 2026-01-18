#!/usr/bin/env python3
"""
Signal Bridge - Connects Vision Pipeline to Arm Control

Handles the flow of joint signals from video pose detection
to arm controllers (simulation or hardware).
"""

import time
import csv
import json
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path
from enum import Enum
import threading
from queue import Queue, Empty
import numpy as np


@dataclass
class JointSignal:
    """A single joint command signal"""
    timestamp: float
    joint_angles: List[float]  # 6 angles in radians
    source: str = "unknown"  # e.g., "video", "trajectory", "manual"
    frame_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'joint_angles': self.joint_angles,
            'source': self.source,
            'frame_id': self.frame_id,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'JointSignal':
        return cls(
            timestamp=data['timestamp'],
            joint_angles=data['joint_angles'],
            source=data.get('source', 'unknown'),
            frame_id=data.get('frame_id', 0),
            metadata=data.get('metadata', {})
        )


class SignalMode(Enum):
    """Signal bridge operating modes"""
    PASSTHROUGH = "passthrough"  # Signals go directly to controller
    BUFFERED = "buffered"        # Signals are buffered for smoothing
    RECORDING = "recording"      # Signals are recorded but not sent
    PLAYBACK = "playback"        # Playing back recorded signals


class SignalBridge:
    """
    Bridge between vision/trajectory input and arm controller output.

    Features:
    - Accepts signals from multiple sources (video, CSV, manual)
    - Optional smoothing/filtering
    - Recording and playback
    - Export to various formats
    - Thread-safe for real-time operation
    """

    def __init__(self,
                 controller=None,
                 smooth_window: int = 3,
                 max_velocity: float = 2.0):
        """
        Initialize signal bridge.

        Args:
            controller: Optional ArmControllerBase instance
            smooth_window: Window size for moving average smoothing
            max_velocity: Maximum joint velocity (rad/s) for safety limiting
        """
        self.controller = controller
        self.smooth_window = smooth_window
        self.max_velocity = max_velocity

        self._mode = SignalMode.PASSTHROUGH
        self._signal_queue: Queue = Queue(maxsize=100)
        self._signal_history: List[JointSignal] = []
        self._recorded_signals: List[JointSignal] = []

        self._last_signal: Optional[JointSignal] = None
        self._last_send_time: float = 0.0
        self._min_send_interval = 0.005  # 200Hz max

        self._callbacks: List[Callable[[JointSignal], None]] = []
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Smoothing buffer
        self._smooth_buffer: List[List[float]] = []

    @property
    def mode(self) -> SignalMode:
        return self._mode

    def set_mode(self, mode: SignalMode) -> None:
        """Change operating mode."""
        self._mode = mode
        if mode == SignalMode.RECORDING:
            self._recorded_signals = []
        print(f"[Bridge] Mode set to {mode.value}")

    def set_controller(self, controller) -> None:
        """Set or change the arm controller."""
        self.controller = controller

    def add_callback(self, callback: Callable[[JointSignal], None]) -> None:
        """Add a callback to be called for each signal."""
        self._callbacks.append(callback)

    def send_signal(self, signal: JointSignal) -> bool:
        """
        Send a joint signal through the bridge.

        Args:
            signal: JointSignal to send

        Returns:
            True if signal was processed
        """
        # Apply smoothing
        smoothed_angles = self._apply_smoothing(signal.joint_angles)
        smoothed_signal = JointSignal(
            timestamp=signal.timestamp,
            joint_angles=smoothed_angles,
            source=signal.source,
            frame_id=signal.frame_id,
            metadata=signal.metadata
        )

        # Apply velocity limiting
        if self._last_signal is not None:
            dt = signal.timestamp - self._last_signal.timestamp
            if dt > 0:
                smoothed_signal.joint_angles = self._limit_velocity(
                    self._last_signal.joint_angles,
                    smoothed_signal.joint_angles,
                    dt
                )

        # Store in history
        self._signal_history.append(smoothed_signal)
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-500:]

        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(smoothed_signal)
            except Exception as e:
                print(f"[Bridge] Callback error: {e}")

        # Process based on mode
        result = False
        if self._mode == SignalMode.PASSTHROUGH:
            result = self._send_to_controller(smoothed_signal)
        elif self._mode == SignalMode.BUFFERED:
            try:
                self._signal_queue.put_nowait(smoothed_signal)
                result = True
            except Exception:
                pass
        elif self._mode == SignalMode.RECORDING:
            self._recorded_signals.append(smoothed_signal)
            result = True

        self._last_signal = smoothed_signal
        return result

    def send_angles(self,
                    angles: List[float],
                    timestamp: Optional[float] = None,
                    source: str = "manual") -> bool:
        """
        Convenience method to send raw angles.

        Args:
            angles: 6 joint angles in radians
            timestamp: Optional timestamp (uses current time if None)
            source: Signal source identifier

        Returns:
            True if processed
        """
        if timestamp is None:
            timestamp = time.time()

        signal = JointSignal(
            timestamp=timestamp,
            joint_angles=angles,
            source=source
        )
        return self.send_signal(signal)

    def _send_to_controller(self, signal: JointSignal) -> bool:
        """Send signal to the arm controller."""
        if self.controller is None:
            return False

        # Rate limiting
        now = time.time()
        if (now - self._last_send_time) < self._min_send_interval:
            return False

        try:
            result = self.controller.set_joint_angles(signal.joint_angles)
            self._last_send_time = now
            return result
        except Exception as e:
            print(f"[Bridge] Controller error: {e}")
            return False

    def _apply_smoothing(self, angles: List[float]) -> List[float]:
        """Apply moving average smoothing."""
        if self.smooth_window <= 1:
            return angles

        self._smooth_buffer.append(angles)
        if len(self._smooth_buffer) > self.smooth_window:
            self._smooth_buffer.pop(0)

        # Average over buffer
        arr = np.array(self._smooth_buffer)
        return np.mean(arr, axis=0).tolist()

    def _limit_velocity(self,
                        prev_angles: List[float],
                        target_angles: List[float],
                        dt: float) -> List[float]:
        """Limit joint velocities for safety."""
        if dt <= 0:
            return target_angles

        max_delta = self.max_velocity * dt
        limited = []

        for prev, target in zip(prev_angles, target_angles):
            delta = target - prev
            if abs(delta) > max_delta:
                delta = max_delta if delta > 0 else -max_delta
            limited.append(prev + delta)

        return limited

    # Recording and Playback

    def start_recording(self) -> None:
        """Start recording signals."""
        self.set_mode(SignalMode.RECORDING)
        self._recorded_signals = []
        print("[Bridge] Recording started")

    def stop_recording(self) -> List[JointSignal]:
        """Stop recording and return recorded signals."""
        signals = self._recorded_signals.copy()
        self.set_mode(SignalMode.PASSTHROUGH)
        print(f"[Bridge] Recording stopped - {len(signals)} signals captured")
        return signals

    def play_recording(self, signals: Optional[List[JointSignal]] = None,
                       speed: float = 1.0) -> None:
        """
        Play back recorded signals.

        Args:
            signals: Signals to play (uses recorded if None)
            speed: Playback speed multiplier
        """
        if signals is None:
            signals = self._recorded_signals

        if not signals:
            print("[Bridge] No signals to play")
            return

        self.set_mode(SignalMode.PLAYBACK)
        print(f"[Bridge] Playing {len(signals)} signals at {speed}x speed")

        start_time = time.time()
        first_ts = signals[0].timestamp

        for signal in signals:
            if self._mode != SignalMode.PLAYBACK:
                break

            # Calculate target time
            target_time = start_time + (signal.timestamp - first_ts) / speed
            now = time.time()

            if target_time > now:
                time.sleep(target_time - now)

            self._send_to_controller(signal)

        self.set_mode(SignalMode.PASSTHROUGH)
        print("[Bridge] Playback complete")

    # Export Methods

    def export_to_csv(self,
                      path: str,
                      signals: Optional[List[JointSignal]] = None) -> None:
        """
        Export signals to CSV format compatible with run_csv_trajectory.py

        Args:
            path: Output file path
            signals: Signals to export (uses recorded if None)
        """
        if signals is None:
            signals = self._recorded_signals

        if not signals:
            print("[Bridge] No signals to export")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'joint1', 'joint2', 'joint3',
                            'joint4', 'joint5', 'joint6', 'description'])

            for i, signal in enumerate(signals):
                row = [f"{signal.timestamp:.4f}"]
                row.extend([f"{a:.4f}" for a in signal.joint_angles])
                row.append(f"{signal.source} frame {signal.frame_id}")
                writer.writerow(row)

        print(f"[Bridge] Exported {len(signals)} signals to {path}")

    def export_to_json(self,
                       path: str,
                       signals: Optional[List[JointSignal]] = None) -> None:
        """Export signals to JSON format."""
        if signals is None:
            signals = self._recorded_signals

        if not signals:
            print("[Bridge] No signals to export")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'version': '1.0',
            'signal_count': len(signals),
            'duration': signals[-1].timestamp - signals[0].timestamp if len(signals) > 1 else 0,
            'signals': [s.to_dict() for s in signals]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[Bridge] Exported {len(signals)} signals to {path}")

    def load_from_csv(self, path: str) -> List[JointSignal]:
        """
        Load signals from CSV trajectory file.

        Args:
            path: CSV file path

        Returns:
            List of JointSignal objects
        """
        signals = []

        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                angles = [
                    float(row['joint1']),
                    float(row['joint2']),
                    float(row['joint3']),
                    float(row['joint4']),
                    float(row['joint5']),
                    float(row['joint6']),
                ]
                signal = JointSignal(
                    timestamp=float(row['time']),
                    joint_angles=angles,
                    source='csv',
                    frame_id=i,
                    metadata={'description': row.get('description', '')}
                )
                signals.append(signal)

        print(f"[Bridge] Loaded {len(signals)} signals from {path}")
        return signals

    def load_from_json(self, path: str) -> List[JointSignal]:
        """Load signals from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        signals = [JointSignal.from_dict(s) for s in data['signals']]
        print(f"[Bridge] Loaded {len(signals)} signals from {path}")
        return signals

    # Statistics

    def get_statistics(self) -> Dict[str, Any]:
        """Get signal statistics."""
        if not self._signal_history:
            return {'count': 0}

        angles = np.array([s.joint_angles for s in self._signal_history])

        return {
            'count': len(self._signal_history),
            'duration': self._signal_history[-1].timestamp - self._signal_history[0].timestamp,
            'mean_angles': np.mean(angles, axis=0).tolist(),
            'std_angles': np.std(angles, axis=0).tolist(),
            'min_angles': np.min(angles, axis=0).tolist(),
            'max_angles': np.max(angles, axis=0).tolist(),
        }

    def clear_history(self) -> None:
        """Clear signal history."""
        self._signal_history = []
        self._smooth_buffer = []
        print("[Bridge] History cleared")
