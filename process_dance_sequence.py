#!/usr/bin/env python3
"""
Process dance sequence to make it slower, smoother, and more natural.
Applies smoothing, speed adjustment, and natural motion curves to match
the graceful movements of the Thousand-Hand Bodhisattva dance.

Usage:
    python process_dance_sequence.py bodhisattva_sequence.json [--speed 0.6] [--smooth 0.3] [--output processed_sequence.json]
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

# Try to import numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("ERROR: numpy is required. Install with: pip install numpy")
    exit(1)

# Try to import scipy, fallback to simpler methods if not available
HAS_SCIPY = False
try:
    from scipy import signal
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except (ImportError, KeyboardInterrupt):
    # If scipy import is interrupted or fails, continue without it
    HAS_SCIPY = False


class DanceSequenceProcessor:
    """Process dance sequences to make them slower, smoother, and more natural."""
    
    def __init__(self, speed_multiplier: float = 0.6, smoothing_factor: float = 0.3):
        """
        Initialize processor.
        
        Args:
            speed_multiplier: How much to slow down (0.6 = 60% speed, more graceful)
            smoothing_factor: EMA smoothing factor (0.0-1.0, higher = more smoothing)
        """
        self.speed_multiplier = speed_multiplier
        self.smoothing_factor = smoothing_factor
    
    def apply_exponential_smoothing(self, values: List[float], alpha: float) -> List[float]:
        """Apply exponential moving average smoothing."""
        if len(values) == 0:
            return values
        
        smoothed = [values[0]]  # Start with first value
        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[i-1])
        return smoothed
    
    def apply_savitzky_golay_filter(self, values: List[float], window_length: int = 11, polyorder: int = 3) -> List[float]:
        """
        Apply Savitzky-Golay filter for smooth, natural motion.
        Preserves important features while reducing noise.
        Falls back to exponential smoothing if scipy is not available.
        """
        # Lazy import of scipy to avoid slow startup
        global HAS_SCIPY
        if not HAS_SCIPY:
            try:
                from scipy import signal
                HAS_SCIPY = True
            except (ImportError, KeyboardInterrupt):
                # Use stronger exponential smoothing as fallback
                return self.apply_exponential_smoothing(values, self.smoothing_factor * 0.7)
        
        if len(values) < window_length:
            # If too few points, use simple smoothing
            return self.apply_exponential_smoothing(values, self.smoothing_factor)
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Clamp window_length to data size
        window_length = min(window_length, len(values))
        if window_length % 2 == 0:
            window_length -= 1
        
        try:
            from scipy import signal
            smoothed = signal.savgol_filter(values, window_length, polyorder)
            return smoothed.tolist()
        except Exception:
            # Fallback to exponential smoothing
            return self.apply_exponential_smoothing(values, self.smoothing_factor)
    
    def resample_sequence(self, keypoints: List[Dict], original_fps: float, target_fps: float) -> List[Dict]:
        """
        Resample sequence to different frame rate (for speed adjustment).
        
        Args:
            keypoints: List of keypoint dictionaries
            original_fps: Original frame rate
            target_fps: Target frame rate (lower = slower)
        """
        if len(keypoints) == 0:
            return keypoints
        
        # Extract timestamps and values
        timestamps = [kp.get("timestamp", i / original_fps) for i, kp in enumerate(keypoints)]
        x_values = [kp.get("x", 0.5) for kp in keypoints]
        y_values = [kp.get("y", 0.5) for kp in keypoints]
        
        # Generate new timestamps at target FPS
        duration = timestamps[-1] - timestamps[0]
        new_timestamps = np.linspace(timestamps[0], timestamps[-1], int(duration * target_fps) + 1)
        
        # Lazy import of scipy interpolation
        global HAS_SCIPY
        if not HAS_SCIPY:
            try:
                from scipy.interpolate import interp1d
                HAS_SCIPY = True
            except (ImportError, KeyboardInterrupt):
                HAS_SCIPY = False
        
        if HAS_SCIPY:
            try:
                # Use scipy interpolation for smooth cubic interpolation
                x_interp = interp1d(timestamps, x_values, kind='cubic', bounds_error=False, fill_value='extrapolate')
                y_interp = interp1d(timestamps, y_values, kind='cubic', bounds_error=False, fill_value='extrapolate')
                new_x = x_interp(new_timestamps)
                new_y = y_interp(new_timestamps)
            except Exception:
                # Fallback to numpy if scipy fails
                HAS_SCIPY = False
        
        if not HAS_SCIPY:
            # Fallback to numpy linear interpolation
            new_x = np.interp(new_timestamps, timestamps, x_values)
            new_y = np.interp(new_timestamps, timestamps, y_values)
        
        # Create new keypoints
        new_keypoints = []
        for i, t in enumerate(new_timestamps):
            # Find original frame number (approximate)
            original_frame_idx = min(int((t - timestamps[0]) * original_fps), len(keypoints) - 1)
            original_kp = keypoints[original_frame_idx]
            
            new_kp = {
                "frame_number": i,
                "timestamp": float(t),
                "x": float(new_x[i]),
                "y": float(new_y[i]),
                "arm_length_pixels": original_kp.get("arm_length_pixels", 70.0),
                "shoulder_angle": original_kp.get("shoulder_angle"),
                "elbow_angle": original_kp.get("elbow_angle"),
            }
            new_keypoints.append(new_kp)
        
        return new_keypoints
    
    def process_sequence(self, sequence_data: Dict) -> Dict:
        """
        Process entire sequence: smooth, slow down, and make natural.
        
        Args:
            sequence_data: Original sequence dictionary
            
        Returns:
            Processed sequence dictionary
        """
        keypoints = sequence_data.get("keypoints", [])
        if not keypoints:
            raise ValueError("No keypoints found in sequence")
        
        original_fps = sequence_data.get("fps", 25.0)
        original_duration = sequence_data.get("duration", len(keypoints) / original_fps)
        
        print(f"Processing sequence:")
        print(f"  Original: {len(keypoints)} keypoints, {original_duration:.2f}s @ {original_fps:.1f} FPS")
        
        # Step 1: Extract x and y values
        x_values = [kp.get("x", 0.5) for kp in keypoints]
        y_values = [kp.get("y", 0.5) for kp in keypoints]
        
        print(f"  Step 1: Applying Savitzky-Golay smoothing...")
        # Step 2: Apply Savitzky-Golay filter for natural motion
        # Window length: ~0.4 seconds of frames (for graceful curves)
        window_frames = max(5, int(original_fps * 0.4))
        if window_frames % 2 == 0:
            window_frames += 1
        
        x_smoothed = self.apply_savitzky_golay_filter(x_values, window_length=window_frames, polyorder=3)
        y_smoothed = self.apply_savitzky_golay_filter(y_values, window_length=window_frames, polyorder=3)
        
        # Step 3: Additional exponential smoothing for extra smoothness
        print(f"  Step 2: Applying exponential smoothing (alpha={self.smoothing_factor})...")
        x_final = self.apply_exponential_smoothing(x_smoothed, self.smoothing_factor)
        y_final = self.apply_exponential_smoothing(y_smoothed, self.smoothing_factor)
        
        # Step 4: Update keypoints with smoothed values
        smoothed_keypoints = []
        for i, kp in enumerate(keypoints):
            new_kp = kp.copy()
            new_kp["x"] = round(x_final[i], 4)
            new_kp["y"] = round(y_final[i], 4)
            smoothed_keypoints.append(new_kp)
        
        # Step 5: Resample to slower speed
        target_fps = original_fps * self.speed_multiplier
        print(f"  Step 3: Resampling to {target_fps:.1f} FPS ({self.speed_multiplier*100:.0f}% speed)...")
        resampled_keypoints = self.resample_sequence(smoothed_keypoints, original_fps, target_fps)
        
        # Calculate new duration
        new_duration = resampled_keypoints[-1]["timestamp"] - resampled_keypoints[0]["timestamp"]
        
        print(f"  Result: {len(resampled_keypoints)} keypoints, {new_duration:.2f}s @ {target_fps:.1f} FPS")
        print(f"  Duration change: {original_duration:.2f}s → {new_duration:.2f}s ({new_duration/original_duration:.1f}x)")
        
        # Create processed sequence
        processed_sequence = {
            "source_video": sequence_data.get("source_video", ""),
            "start_time": sequence_data.get("start_time", 0.0),
            "end_time": sequence_data.get("end_time", 0.0),
            "duration": new_duration,
            "fps": target_fps,
            "total_frames": len(resampled_keypoints),
            "frames_with_pose": len(resampled_keypoints),
            "processing": {
                "original_fps": original_fps,
                "speed_multiplier": self.speed_multiplier,
                "smoothing_factor": self.smoothing_factor,
                "smoothing_method": "savitzky_golay + exponential"
            },
            "keypoints": resampled_keypoints
        }
        
        return processed_sequence


def main():
    parser = argparse.ArgumentParser(
        description="Process dance sequence to make it slower, smoother, and more natural",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default settings (60% speed, moderate smoothing)
  python process_dance_sequence.py bodhisattva_sequence.json
  
  # Very slow and smooth (50% speed, heavy smoothing)
  python process_dance_sequence.py bodhisattva_sequence.json --speed 0.5 --smooth 0.5
  
  # Moderate slow, light smoothing (70% speed, light smoothing)
  python process_dance_sequence.py bodhisattva_sequence.json --speed 0.7 --smooth 0.2
  
  # Custom output file
  python process_dance_sequence.py bodhisattva_sequence.json --output graceful_dance.json
        """
    )
    parser.add_argument("input_json", help="Path to input JSON sequence file")
    parser.add_argument("--speed", type=float, default=0.6,
                       help="Speed multiplier (default: 0.6 = 60%% speed, slower and more graceful)")
    parser.add_argument("--smooth", type=float, default=0.3,
                       help="Smoothing factor 0.0-1.0 (default: 0.3, higher = more smoothing)")
    parser.add_argument("--output", help="Output JSON file path (default: input_name_processed.json)")
    
    args = parser.parse_args()
    
    if args.speed <= 0 or args.speed > 2.0:
        print("ERROR: Speed multiplier should be between 0.0 and 2.0")
        return 1
    
    if args.smooth < 0 or args.smooth > 1.0:
        print("ERROR: Smoothing factor should be between 0.0 and 1.0")
        return 1
    
    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_processed.json"
    
    print("=" * 70)
    print("🎭 DANCE SEQUENCE PROCESSOR")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Load sequence
    try:
        with open(input_path, 'r') as f:
            sequence_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load sequence: {e}")
        return 1
    
    # Process sequence
    try:
        processor = DanceSequenceProcessor(
            speed_multiplier=args.speed,
            smoothing_factor=args.smooth
        )
        processed_sequence = processor.process_sequence(sequence_data)
    except Exception as e:
        print(f"ERROR: Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save processed sequence
    try:
        with open(output_path, 'w') as f:
            json.dump(processed_sequence, f, indent=2)
        print()
        print("=" * 70)
        print(f"✅ Processed sequence saved to: {output_path}")
        print()
        print("To replay the processed sequence:")
        print(f"  python replay_sequence_to_simulation.py {output_path}")
        print()
        print("Or with custom speed:")
        print(f"  python replay_sequence_to_simulation.py {output_path} --speed 1.0")
        print("=" * 70)
    except Exception as e:
        print(f"ERROR: Failed to save processed sequence: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
