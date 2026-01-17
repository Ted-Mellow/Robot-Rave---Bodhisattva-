#!/usr/bin/env python3
"""
Simulate Dance - Play CSV choreography in PyBullet simulation
Uses the Piper simulation from the parent directory
"""

import sys
import os
from pathlib import Path
import csv
import time

# Add parent directory to path to import simulation
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from simulation.piper_simultion_corrected import PiperSimulation
except ImportError:
    print("❌ Could not import PiperSimulation")
    print("   Make sure piper_simultion_corrected.py exists in ../simulation/")
    sys.exit(1)


class DanceSimulator:
    """Simulate choreography from CSV file"""

    def __init__(self, csv_path: str, urdf_path: str = None):
        """
        Initialize dance simulator

        Args:
            csv_path: Path to CSV choreography file
            urdf_path: Optional path to robot URDF
        """
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load choreography
        self.choreography = self._load_csv()

        print(f"\n💃 Dance choreography loaded: {self.csv_path.name}")
        print(f"   Keyframes: {len(self.choreography)}")
        if self.choreography:
            duration = self.choreography[-1]['time']
            print(f"   Duration: {duration:.2f}s")

        # Initialize simulation
        print("\n🤖 Initializing PyBullet simulation...")
        self.sim = PiperSimulation(urdf_path=urdf_path, gui=True)

    def _load_csv(self) -> list:
        """Load choreography from CSV file"""
        choreography = []

        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Skip empty rows
                if not row.get('time'):
                    continue

                try:
                    keyframe = {
                        'time': float(row['time']),
                        'joints': [
                            float(row['joint1']),
                            float(row['joint2']),
                            float(row['joint3']),
                            float(row['joint4']),
                            float(row['joint5']),
                            float(row['joint6']),
                        ],
                        'description': row.get('description', '')
                    }
                    choreography.append(keyframe)
                except (ValueError, KeyError) as e:
                    print(f"⚠️  Skipping invalid row: {e}")
                    continue

        return choreography

    def play(self, speed: float = 1.0, loop: bool = False):
        """
        Play the choreography in simulation

        Args:
            speed: Playback speed multiplier (1.0 = normal)
            loop: If True, loop the choreography
        """
        if not self.choreography:
            print("❌ No choreography to play!")
            return

        print(f"\n▶️  Playing choreography...")
        print(f"   Speed: {speed}x")
        print(f"   Loop: {loop}")
        print(f"   Press Ctrl+C to stop")

        try:
            while True:
                start_real_time = time.time()

                for i, keyframe in enumerate(self.choreography):
                    target_time = keyframe['time'] / speed
                    current_time = time.time() - start_real_time

                    # Wait until target time
                    wait_time = target_time - current_time
                    if wait_time > 0:
                        # Step simulation while waiting
                        steps = int(wait_time / self.sim.time_step)
                        for _ in range(steps):
                            self.sim.step()

                    # Set joint positions
                    self.sim.set_joint_positions(keyframe['joints'])

                    # Show progress
                    if i % 30 == 0:  # Every 30 frames
                        progress = i / len(self.choreography) * 100
                        print(f"   Progress: {progress:.1f}% - {keyframe['description']}")

                print(f"✅ Playback complete!")

                if not loop:
                    break

                print(f"🔁 Looping...")

        except KeyboardInterrupt:
            print("\n\n⚠️  Playback stopped by user")

    def step_through(self):
        """
        Step through choreography frame by frame
        Press ENTER to advance to next keyframe
        """
        print(f"\n⏯️  Step-through mode")
        print(f"   Press ENTER to advance, Ctrl+C to exit")

        try:
            for i, keyframe in enumerate(self.choreography):
                print(f"\n📍 Keyframe {i+1}/{len(self.choreography)}")
                print(f"   Time: {keyframe['time']:.3f}s")
                print(f"   Description: {keyframe['description']}")
                print(f"   Joints: {[f'{j:.2f}' for j in keyframe['joints']]}")

                # Set position
                self.sim.set_joint_positions(keyframe['joints'])

                # Step simulation to reach position
                for _ in range(60):  # 0.25 seconds
                    self.sim.step()

                # Wait for user
                input("   Press ENTER for next keyframe...")

        except KeyboardInterrupt:
            print("\n\n⚠️  Step-through stopped")

    def compare_with_video(self, video_path: str):
        """
        Play choreography alongside original video for comparison

        Args:
            video_path: Path to original video file
        """
        try:
            import cv2
        except ImportError:
            print("❌ OpenCV not available - cannot play video")
            return

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"\n🎬 Playing with video comparison...")
        print(f"   Video FPS: {fps}")
        print(f"   Press ESC to stop")

        try:
            start_time = time.time()
            keyframe_idx = 0

            while cap.isOpened() and keyframe_idx < len(self.choreography):
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time() - start_time

                # Update robot position if needed
                while (keyframe_idx < len(self.choreography) and
                       self.choreography[keyframe_idx]['time'] <= current_time):

                    keyframe = self.choreography[keyframe_idx]
                    self.sim.set_joint_positions(keyframe['joints'])
                    keyframe_idx += 1

                # Step simulation
                self.sim.step()

                # Show video
                cv2.imshow('Original Video', frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

            cap.release()
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            print("\n\n⚠️  Comparison stopped")
            cap.release()
            cv2.destroyAllWindows()

    def close(self):
        """Clean up resources"""
        self.sim.close()


def main():
    """Main entry point"""
    print("=" * 70)
    print("PIPER DANCE SIMULATOR")
    print("=" * 70)

    import argparse

    parser = argparse.ArgumentParser(description='Simulate robot dance from CSV')
    parser.add_argument('csv_file', nargs='?', default='bodhisattva_dance.csv',
                       help='CSV choreography file')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed (default: 1.0)')
    parser.add_argument('--loop', action='store_true',
                       help='Loop the choreography')
    parser.add_argument('--step', action='store_true',
                       help='Step through frame by frame')
    parser.add_argument('--video', type=str,
                       help='Compare with original video')
    parser.add_argument('--urdf', type=str,
                       help='Path to robot URDF file')

    args = parser.parse_args()

    # Create simulator
    try:
        simulator = DanceSimulator(args.csv_file, urdf_path=args.urdf)

        if args.step:
            simulator.step_through()
        elif args.video:
            simulator.compare_with_video(args.video)
        else:
            simulator.play(speed=args.speed, loop=args.loop)

        # Keep window open after playback
        print("\n✅ Simulation complete!")
        print("   Close the PyBullet window to exit")

        while True:
            simulator.sim.step()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if 'simulator' in locals():
            simulator.close()


if __name__ == "__main__":
    main()
