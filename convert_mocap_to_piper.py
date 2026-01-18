#!/usr/bin/env python3
"""
Convert motion capture CSV format to Piper robot joint format.

The motion capture format has columns like:
- time_s, l_shoulder_pitch, l_elbow_flex, etc.

Piper format needs:
- time, joint1, joint2, joint3, joint4, joint5, joint6, description

Usage:
    python convert_mocap_to_piper.py <input_csv> [--output <output_csv>]
"""

import sys
import csv
import argparse
from pathlib import Path
import math


def mocap_to_piper_joints(row):
    """
    Convert motion capture row to Piper joint angles.
    
    Motion capture values appear to be small offsets/angles in radians.
    We need to map these to Piper's joint space with proper neutral positions.
    
    Mapping:
    - l_shoulder_pitch -> J2 (shoulder) - offset from horizontal (3.2 rad)
    - l_elbow_flex -> J3 (elbow) - offset from straight (0 rad)
    - Pelvis/spine rotation -> J1 (base)
    - Wrist angles -> J4, J5, J6
    """
    try:
        time_s = float(row.get('time_s', row.get('time', 0.0)))
        
        # Get left arm values (appear to be small angle offsets in radians)
        l_shoulder_pitch = float(row.get('l_shoulder_pitch', 0.0))
        l_elbow_flex = float(row.get('l_elbow_flex', 0.0))
        l_wrist_flex = float(row.get('l_wrist_flex', 0.0))
        
        # Get base rotation from pelvis/spine
        pelvis_yaw = float(row.get('pelvis_yaw', 0.0))
        ribcage_yaw = float(row.get('ribcage_yaw', 0.0))
        
        # J1 (base): Combine pelvis and ribcage yaw, scale appropriately
        # Values seem to be in radians already, but might need scaling
        # Scale by 10x to get meaningful rotation (adjust based on actual range)
        j1 = (pelvis_yaw + ribcage_yaw * 0.5) * 10.0
        
        # J2 (shoulder): Start from horizontal (3.2 rad) and add offset
        # l_shoulder_pitch: positive = up, negative = down
        # But Piper J2: smaller = up (1.63), larger = down (toward 0)
        # So we need to invert: positive offset -> subtract from 3.2
        # Scale the offset appropriately
        shoulder_offset = l_shoulder_pitch * 5.0  # Scale factor
        j2 = 3.2 - shoulder_offset  # Invert: positive offset moves toward up (smaller J2)
        
        # J3 (elbow): Start from straight (0 rad), negative = bent
        # l_elbow_flex: positive = more flexed (bent)
        # Scale appropriately
        j3 = -l_elbow_flex * 3.0  # Negative because bent = negative J3
        
        # J4 (wrist roll): Map from wrist flex
        j4 = l_wrist_flex * 2.0
        
        # J5 (wrist pitch): Small offset
        j5 = l_wrist_flex * 0.5
        
        # J6 (wrist rotation): Keep at 0
        j6 = 0.0
        
        # Clamp to limits
        j1 = max(-2.688, min(2.688, j1))
        j2 = max(0.0, min(3.403, j2))
        j3 = max(-3.054, min(0.0, j3))
        j4 = max(-1.850, min(1.850, j4))
        j5 = max(-1.309, min(1.309, j5))
        j6 = max(-1.745, min(1.745, j6))
        
        description = row.get('description', row.get('section', ''))
        
        return {
            'time': time_s,
            'joint1': round(j1, 4),
            'joint2': round(j2, 4),
            'joint3': round(j3, 4),
            'joint4': round(j4, 4),
            'joint5': round(j5, 4),
            'joint6': round(j6, 4),
            'description': description
        }
    except (KeyError, ValueError) as e:
        return None


def convert_csv(input_path, output_path=None):
    """Convert motion capture CSV to Piper format."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_piper.csv"
    else:
        output_path = Path(output_path)
    
    waypoints = []
    
    print(f"Reading: {input_path}")
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            waypoint = mocap_to_piper_joints(row)
            if waypoint:
                waypoints.append(waypoint)
    
    if not waypoints:
        raise ValueError("No valid waypoints found in input file")
    
    print(f"Converted {len(waypoints)} waypoints")
    
    # Write Piper format CSV
    print(f"Writing: {output_path}")
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'description'])
        
        for wp in waypoints:
            writer.writerow([
                wp['time'],
                wp['joint1'],
                wp['joint2'],
                wp['joint3'],
                wp['joint4'],
                wp['joint5'],
                wp['joint6'],
                wp['description']
            ])
    
    print(f"✅ Converted to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert motion capture CSV to Piper robot joint format"
    )
    parser.add_argument("input_csv", help="Input motion capture CSV file")
    parser.add_argument("--output", help="Output CSV file path")
    
    args = parser.parse_args()
    
    try:
        convert_csv(args.input_csv, args.output)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
