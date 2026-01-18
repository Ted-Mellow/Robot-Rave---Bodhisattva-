#!/usr/bin/env python3
"""
Replay Pre-recorded Keypoint Sequence to PiPER Simulation
Reads a JSON sequence file and sends keypoints to the simulation via WebSocket.

Usage:
    python replay_sequence_to_simulation.py <sequence_json> [--speed <multiplier>] [--loop]

Example:
    python replay_sequence_to_simulation.py output_sequence.json
    python replay_sequence_to_simulation.py output_sequence.json --speed 0.5  # Half speed
    python replay_sequence_to_simulation.py output_sequence.json --loop  # Loop continuously
"""

import sys
import json
import time
import argparse
import websocket
from pathlib import Path
from typing import List, Dict


class SequenceReplayer:
    """Replay keypoint sequences to simulation via WebSocket."""
    
    def __init__(self, ws_url="ws://localhost:8000/ws"):
        """Initialize replayer with WebSocket URL."""
        self.ws_url = ws_url
        self.ws = None
        self.connected = False
    
    def connect(self):
        """Connect to WebSocket server."""
        print(f"Connecting to {self.ws_url}...")
        try:
            self.ws = websocket.create_connection(self.ws_url, timeout=5)
            self.connected = True
            print("✅ Connected to simulation")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            print("   Make sure the simulation is running:")
            print("   python Small\\ arm\\ /simulation/realtime_ik_trajectory.py")
            return False
    
    def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.ws:
            self.ws.close()
            self.connected = False
            print("Disconnected from simulation")
    
    def send_keypoint(self, x, y):
        """Send a single keypoint to simulation."""
        if not self.connected:
            return False
        
        message = json.dumps({"target": [x, y]})
        try:
            self.ws.send(message)
            return True
        except Exception as e:
            print(f"Error sending keypoint: {e}")
            self.connected = False
            return False
    
    def replay_sequence(self, sequence_data: Dict, speed_multiplier: float = 1.0, loop: bool = False):
        """
        Replay a keypoint sequence.
        
        Args:
            sequence_data: Dictionary containing keypoints sequence
            speed_multiplier: Speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
            loop: Whether to loop the sequence continuously
        """
        if not self.connected:
            print("Not connected to simulation")
            return
        
        keypoints = sequence_data.get("keypoints", [])
        if not keypoints:
            print("No keypoints found in sequence")
            return
        
        fps = sequence_data.get("fps", 30.0)
        duration = sequence_data.get("duration", len(keypoints) / fps)
        
        print(f"\nReplaying sequence:")
        print(f"  Keypoints: {len(keypoints)}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Speed: {speed_multiplier}x")
        print(f"  Loop: {loop}")
        print("\nStarting replay... (Press Ctrl+C to stop)\n")
        
        frame_interval = 1.0 / (fps * speed_multiplier)
        
        try:
            while True:
                last_timestamp = None
                
                for i, kp in enumerate(keypoints):
                    if not self.connected:
                        break
                    
                    # Calculate delay based on timestamp difference
                    current_timestamp = kp.get("timestamp", i / fps)
                    
                    if last_timestamp is not None:
                        # Wait based on actual time difference
                        time_diff = (current_timestamp - last_timestamp) / speed_multiplier
                        if time_diff > 0:
                            time.sleep(time_diff)
                    else:
                        # First frame, use frame interval
                        time.sleep(frame_interval)
                    
                    # Send keypoint
                    x = kp.get("x")
                    y = kp.get("y")
                    
                    if x is not None and y is not None:
                        self.send_keypoint(x, y)
                        
                        # Progress indicator
                        if i % 30 == 0:
                            progress = (i / len(keypoints)) * 100
                            print(f"  Progress: {progress:.1f}% ({i}/{len(keypoints)} keypoints)")
                    
                    last_timestamp = current_timestamp
                
                if not loop:
                    break
                
                print("\nLooping sequence...\n")
                time.sleep(0.5)  # Brief pause between loops
                
        except KeyboardInterrupt:
            print("\n\nReplay stopped by user")
        except Exception as e:
            print(f"\nError during replay: {e}")
    
    def replay_from_file(self, json_path: Path, speed_multiplier: float = 1.0, loop: bool = False):
        """Load and replay sequence from JSON file."""
        if not json_path.exists():
            print(f"ERROR: Sequence file not found: {json_path}")
            return False
        
        print(f"Loading sequence from: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                sequence_data = json.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load sequence file: {e}")
            return False
        
        if not self.connect():
            return False
        
        try:
            self.replay_sequence(sequence_data, speed_multiplier, loop)
        finally:
            self.disconnect()
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Replay pre-recorded keypoint sequence to PiPER simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay at normal speed
  python replay_sequence_to_simulation.py output_sequence.json
  
  # Replay at half speed
  python replay_sequence_to_simulation.py output_sequence.json --speed 0.5
  
  # Replay in loop
  python replay_sequence_to_simulation.py output_sequence.json --loop
  
  # Custom WebSocket URL
  python replay_sequence_to_simulation.py output_sequence.json --ws-url ws://localhost:8000/ws
        """
    )
    parser.add_argument("sequence_json", help="Path to JSON sequence file")
    parser.add_argument("--speed", type=float, default=1.0, 
                       help="Speed multiplier (default: 1.0, 0.5 = half speed, 2.0 = double speed)")
    parser.add_argument("--loop", action="store_true",
                       help="Loop the sequence continuously")
    parser.add_argument("--ws-url", default="ws://localhost:8000/ws",
                       help="WebSocket URL (default: ws://localhost:8000/ws)")
    
    args = parser.parse_args()
    
    if args.speed <= 0:
        print("ERROR: Speed multiplier must be positive")
        sys.exit(1)
    
    replayer = SequenceReplayer(ws_url=args.ws_url)
    success = replayer.replay_from_file(
        Path(args.sequence_json),
        speed_multiplier=args.speed,
        loop=args.loop
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
