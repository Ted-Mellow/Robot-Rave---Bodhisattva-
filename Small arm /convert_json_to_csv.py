#!/usr/bin/env python3
"""
Convert trajectory JSON to CSV for robot-only simulation viewing.

Usage:
    python convert_json_to_csv.py input.json output.csv
    python convert_json_to_csv.py Thousand-hand-video/Cropped_thousandhand_trajectory.json
"""

import json
import csv
import sys
from pathlib import Path

def convert_trajectory(json_path: str, csv_path: str = None):
    """Convert trajectory JSON to CSV format."""
    json_path = Path(json_path)
    
    if csv_path is None:
        csv_path = Path("csv_trajectories") / f"{json_path.stem}.csv"
    else:
        csv_path = Path(csv_path)
    
    # Create output directory
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load JSON
    print(f"📖 Loading: {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    
    frames = data.get('frames', [])
    if not frames:
        print("❌ No frames found in trajectory")
        return False
    
    # Write CSV
    print(f"💾 Writing: {csv_path}")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'description'])
        
        for i, frame in enumerate(frames):
            timestamp = frame['timestamp']
            joints = frame.get('joint_angles') or frame.get('joints')
            
            if not joints:
                print(f"⚠️  Frame {i}: No joint data found")
                continue
            
            if len(joints) != 6:
                print(f"⚠️  Frame {i}: Expected 6 joints, got {len(joints)}")
                continue
            
            row = [f"{timestamp:.3f}"] + [f"{j:.3f}" for j in joints] + [f"Frame {i}"]
            writer.writerow(row)
    
    print(f"✅ Converted {len(frames)} frames")
    print(f"   Duration: {frames[-1]['timestamp']:.2f}s")
    print(f"\n🤖 View in simulation:")
    print(f"   python simulation/run_csv_trajectory.py {csv_path}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert_json_to_csv.py <input.json> [output.csv]")
        print("\nExample:")
        print("  python convert_json_to_csv.py Thousand-hand-video/Cropped_thousandhand_trajectory.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        convert_trajectory(json_path, csv_path)
    except FileNotFoundError:
        print(f"❌ File not found: {json_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
