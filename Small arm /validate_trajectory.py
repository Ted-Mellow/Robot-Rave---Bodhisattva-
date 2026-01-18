#!/usr/bin/env python3
"""
Trajectory Validator

Validates trajectory JSON files before sending to robot.
Checks for:
- Joint limit violations
- Excessive velocities
- Large jumps between frames
- File format validity

Usage:
    python validate_trajectory.py trajectory.json
    python validate_trajectory.py trajectory.json --strict
"""

import json
import argparse
import sys
from typing import List, Tuple
import math


# Joint limits from robot specifications (radians)
JOINT_LIMITS = {
    0: (-2.6879, 2.6879),    # J1: ±154°
    1: (0.0, 3.4034),         # J2: 0→195°
    2: (-3.0543, 0.0),        # J3: -175→0°
    3: (-1.8501, 1.8501),     # J4: ±106°
    4: (-1.3090, 1.3090),     # J5: ±75°
    5: (-1.7453, 1.7453),     # J6: ±100°
}

# Maximum joint velocities (rad/s) from robot specs
MAX_VELOCITIES = {
    0: 3.1416,   # J1: 180°/s
    1: 3.4034,   # J2: 195°/s
    2: 3.1416,   # J3: 180°/s
    3: 3.9270,   # J4: 225°/s
    4: 3.9270,   # J5: 225°/s
    5: 3.9270,   # J6: 225°/s
}

# Warning thresholds
VELOCITY_WARNING_FACTOR = 0.8  # Warn if 80% of max velocity
JUMP_WARNING_THRESHOLD = 0.5   # Warn if >0.5 rad jump between frames


class TrajectoryValidator:
    """Validates trajectory files."""

    def __init__(self, trajectory_path: str, strict: bool = False):
        self.trajectory_path = trajectory_path
        self.strict = strict
        self.trajectory = None
        self.errors = []
        self.warnings = []

    def validate(self) -> bool:
        """
        Run all validation checks.
        
        Returns:
            True if trajectory is valid (no errors)
        """
        print(f"Validating: {self.trajectory_path}")
        print("=" * 60)
        
        # Load file
        if not self._load_file():
            return False
        
        # Run checks
        self._check_format()
        self._check_joint_limits()
        self._check_velocities()
        self._check_jumps()
        
        # Report
        self._print_report()
        
        return len(self.errors) == 0

    def _load_file(self) -> bool:
        """Load and parse JSON file."""
        try:
            with open(self.trajectory_path, 'r') as f:
                self.trajectory = json.load(f)
            return True
        except FileNotFoundError:
            self.errors.append(f"File not found: {self.trajectory_path}")
            return False
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Failed to load file: {e}")
            return False

    def _check_format(self):
        """Check file format and required fields."""
        required_fields = ['version', 'frames']
        for field in required_fields:
            if field not in self.trajectory:
                self.errors.append(f"Missing required field: {field}")
        
        if 'frames' not in self.trajectory:
            return
        
        frames = self.trajectory['frames']
        if not isinstance(frames, list):
            self.errors.append("'frames' must be a list")
            return
        
        if len(frames) == 0:
            self.errors.append("Trajectory has no frames")
            return
        
        # Check first frame format
        first_frame = frames[0]
        required_frame_fields = ['timestamp', 'joint_angles']
        for field in required_frame_fields:
            if field not in first_frame:
                self.errors.append(f"Frame missing required field: {field}")
        
        if 'joint_angles' in first_frame:
            if len(first_frame['joint_angles']) != 6:
                self.errors.append(f"Expected 6 joint angles, got {len(first_frame['joint_angles'])}")

    def _check_joint_limits(self):
        """Check if joint angles respect limits."""
        if 'frames' not in self.trajectory:
            return
        
        violations = []
        for i, frame in enumerate(self.trajectory['frames']):
            if 'joint_angles' not in frame:
                continue
            
            for j, angle in enumerate(frame['joint_angles']):
                lo, hi = JOINT_LIMITS[j]
                if angle < lo or angle > hi:
                    violations.append({
                        'frame': i,
                        'joint': j,
                        'angle': angle,
                        'limit': (lo, hi)
                    })
        
        if violations:
            self.errors.append(f"Joint limit violations: {len(violations)} frames")
            # Show first few
            for v in violations[:5]:
                deg = v['angle'] * 180.0 / math.pi
                lo_deg = v['limit'][0] * 180.0 / math.pi
                hi_deg = v['limit'][1] * 180.0 / math.pi
                msg = (f"  Frame {v['frame']}: J{v['joint']+1} = {deg:.1f}° "
                       f"(limit: [{lo_deg:.1f}°, {hi_deg:.1f}°])")
                self.errors.append(msg)
            
            if len(violations) > 5:
                self.errors.append(f"  ... and {len(violations) - 5} more violations")

    def _check_velocities(self):
        """Check if velocities are within limits."""
        if 'frames' not in self.trajectory:
            return
        
        frames = self.trajectory['frames']
        if len(frames) < 2:
            return
        
        high_velocities = []
        
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            
            dt = curr_frame['timestamp'] - prev_frame['timestamp']
            if dt <= 0:
                continue
            
            for j in range(6):
                da = curr_frame['joint_angles'][j] - prev_frame['joint_angles'][j]
                velocity = abs(da / dt)
                max_vel = MAX_VELOCITIES[j]
                
                if velocity > max_vel:
                    high_velocities.append({
                        'frame': i,
                        'joint': j,
                        'velocity': velocity,
                        'max': max_vel
                    })
                elif velocity > max_vel * VELOCITY_WARNING_FACTOR:
                    # Warning: approaching limit
                    deg_per_s = velocity * 180.0 / math.pi
                    max_deg_per_s = max_vel * 180.0 / math.pi
                    self.warnings.append(
                        f"Frame {i}: J{j+1} velocity {deg_per_s:.1f}°/s "
                        f"(max: {max_deg_per_s:.1f}°/s)"
                    )
        
        if high_velocities:
            self.errors.append(f"Velocity violations: {len(high_velocities)} instances")
            for v in high_velocities[:5]:
                deg_per_s = v['velocity'] * 180.0 / math.pi
                max_deg_per_s = v['max'] * 180.0 / math.pi
                msg = (f"  Frame {v['frame']}: J{v['joint']+1} = {deg_per_s:.1f}°/s "
                       f"(max: {max_deg_per_s:.1f}°/s)")
                self.errors.append(msg)
            
            if len(high_velocities) > 5:
                self.errors.append(f"  ... and {len(high_velocities) - 5} more violations")

    def _check_jumps(self):
        """Check for large discontinuous jumps."""
        if 'frames' not in self.trajectory:
            return
        
        frames = self.trajectory['frames']
        if len(frames) < 2:
            return
        
        large_jumps = []
        
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            
            for j in range(6):
                da = abs(curr_frame['joint_angles'][j] - prev_frame['joint_angles'][j])
                if da > JUMP_WARNING_THRESHOLD:
                    deg = da * 180.0 / math.pi
                    large_jumps.append({
                        'frame': i,
                        'joint': j,
                        'jump': deg
                    })
        
        if large_jumps:
            self.warnings.append(f"Large jumps detected: {len(large_jumps)} instances")
            for j in large_jumps[:3]:
                self.warnings.append(
                    f"  Frame {j['frame']}: J{j['joint']+1} jumped {j['jump']:.1f}°"
                )

    def _print_report(self):
        """Print validation report."""
        print()
        print("Validation Results:")
        print("-" * 60)
        
        if self.trajectory:
            frame_count = len(self.trajectory.get('frames', []))
            duration = self.trajectory.get('duration', 0)
            print(f"Frames: {frame_count}")
            print(f"Duration: {duration:.2f}s")
            print()
        
        if not self.errors and not self.warnings:
            print("✓ PASSED - Trajectory is valid")
            print()
            return
        
        if self.errors:
            print(f"✗ ERRORS: {len(self.errors)}")
            for error in self.errors:
                print(f"  {error}")
            print()
        
        if self.warnings:
            print(f"⚠ WARNINGS: {len(self.warnings)}")
            for warning in self.warnings[:10]:
                print(f"  {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
            print()
        
        if self.errors:
            print("✗ FAILED - Do not send to robot")
        elif self.strict and self.warnings:
            print("⚠ WARNINGS PRESENT - Review before sending")
        else:
            print("✓ PASSED WITH WARNINGS - Safe to send but review recommended")
        print()


def main():
    parser = argparse.ArgumentParser(description="Validate trajectory files before sending to robot")
    parser.add_argument('trajectory', help='Path to trajectory JSON file')
    parser.add_argument('--strict', action='store_true',
                        help='Fail on warnings as well as errors')
    
    args = parser.parse_args()
    
    validator = TrajectoryValidator(args.trajectory, strict=args.strict)
    is_valid = validator.validate()
    
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
