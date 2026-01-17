#!/usr/bin/env python3
"""
Complete Pipeline - Video to Robot Dance
Processes video through pose detection, motion mapping, and simulation
"""

import argparse
from pathlib import Path
import sys

from pose_detector import PoseDetector, VideoProcessor
from motion_mapper import MotionMapper, visualize_trajectory
from simulate_dance import DanceSimulator
from live_comparison import LiveComparison


def run_complete_pipeline(
    video_path: str,
    output_csv: str,
    start_time: float = 0.0,
    end_time: float = None,
    use_greyscale: bool = True,
    skip_frames: int = 0,
    arm_side: str = 'right',
    scaling_factor: float = 0.8,
    smooth_window: int = 5,
    show_preview: bool = True,
    simulate: bool = True,
    visualize: bool = False
):
    """
    Run complete pipeline from video to simulation

    Args:
        video_path: Path to input video
        output_csv: Output CSV file path
        start_time: Start time in seconds
        end_time: End time in seconds (None = full video)
        use_greyscale: Use greyscale processing for better accuracy
        skip_frames: Skip every N frames for faster processing
        arm_side: Which arm to track ('left' or 'right')
        scaling_factor: Motion scaling factor
        smooth_window: Temporal smoothing window size
        show_preview: Show video preview during processing
        simulate: Run simulation after processing
        visualize: Create trajectory visualization
    """
    print("=" * 70)
    print("THOUSAND-HAND BODHISATTVA - COMPLETE PIPELINE")
    print("=" * 70)

    # Intermediate files
    pose_data_json = Path(output_csv).with_suffix('.json')

    # Step 1: Pose Detection
    print("\n" + "=" * 70)
    print("STEP 1: POSE DETECTION")
    print("=" * 70)

    detector = PoseDetector(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        use_greyscale=use_greyscale
    )

    processor = VideoProcessor(video_path, detector)

    try:
        # Process video
        pose_sequence = processor.process_video(
            start_time=start_time,
            end_time=end_time,
            skip_frames=skip_frames,
            show_preview=show_preview,
            target_person='center'
        )

        if not pose_sequence:
            print("\n❌ No poses detected in video!")
            return False

        # Save pose data
        processor.save_pose_data(pose_sequence, str(pose_data_json))

    finally:
        processor.close()
        detector.close()

    # Step 2: Motion Mapping
    print("\n" + "=" * 70)
    print("STEP 2: MOTION MAPPING")
    print("=" * 70)

    mapper = MotionMapper(
        scaling_factor=scaling_factor,
        z_offset=0.0,
        smooth_window=smooth_window
    )

    # Process pose sequence
    trajectory = mapper.process_pose_sequence(
        str(pose_data_json),
        side=arm_side
    )

    if not trajectory:
        print("\n❌ Failed to generate trajectory!")
        return False

    # Export to CSV
    mapper.export_to_csv(trajectory, output_csv, description="Bodhisattva")

    # Visualize trajectory
    if visualize:
        print("\n" + "=" * 70)
        print("STEP 3: VISUALIZATION")
        print("=" * 70)
        visualize_trajectory(trajectory)

    # Step 3: Simulation
    if simulate:
        print("\n" + "=" * 70)
        print("STEP 4: SIMULATION")
        print("=" * 70)

        try:
            simulator = DanceSimulator(output_csv)
            print("\n▶️  Playing choreography in simulation...")
            print("   Press Ctrl+C to stop")

            simulator.play(speed=1.0, loop=False)

            # Keep window open
            print("\n✅ Pipeline complete!")
            print("   Close the PyBullet window to exit")

            while True:
                simulator.sim.step()

        except KeyboardInterrupt:
            print("\n\n⚠️  Simulation stopped by user")
        finally:
            simulator.close()

    else:
        print("\n✅ Pipeline complete (simulation skipped)")

    return True


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Convert video of Thousand-Hand Bodhisattva to robot dance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process first 10 seconds with greyscale
  python run_pipeline.py --end 10 --greyscale

  # Process full video, right arm, skip preview
  python run_pipeline.py --no-preview

  # Process and visualize, no simulation
  python run_pipeline.py --visualize --no-simulate

  # Fast processing (skip frames)
  python run_pipeline.py --skip-frames 2 --end 5
        """
    )

    parser.add_argument('video', nargs='?',
                       default='Cropped_thousandhand.mp4',
                       help='Input video file')
    parser.add_argument('-o', '--output', default='bodhisattva_dance.csv',
                       help='Output CSV file (default: bodhisattva_dance.csv)')

    # Time range
    parser.add_argument('--start', type=float, default=0.0,
                       help='Start time in seconds (default: 0.0)')
    parser.add_argument('--end', type=float, default=None,
                       help='End time in seconds (default: full video)')

    # Processing options
    parser.add_argument('--greyscale', action='store_true',
                       help='Use greyscale processing for better accuracy')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Process every Nth frame (0 = all frames)')
    parser.add_argument('--arm', choices=['left', 'right'], default='right',
                       help='Which arm to track (default: right)')

    # Mapping options
    parser.add_argument('--scale', type=float, default=0.8,
                       help='Motion scaling factor (default: 0.8)')
    parser.add_argument('--smooth', type=int, default=5,
                       help='Smoothing window size (default: 5)')

    # Display options
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable video preview during processing')
    parser.add_argument('--no-simulate', action='store_true',
                       help='Skip simulation after processing')
    parser.add_argument('--visualize', action='store_true',
                       help='Create trajectory visualization plots')
    parser.add_argument('--live', action='store_true',
                       help='Live comparison mode: show pose detection + simulation side-by-side in real-time')

    args = parser.parse_args()

    # Check video exists
    if not Path(args.video).exists():
        print(f"❌ Video file not found: {args.video}")
        return 1

    # Live comparison mode - skip CSV generation, run directly
    if args.live:
        print("\n🎬 Starting LIVE COMPARISON mode...")
        print("   This will show pose detection and simulation in real-time")
        print("   (Processing + simulation happens simultaneously)\n")

        try:
            comparison = LiveComparison(args.video)
            comparison.run(
                start_time=args.start,
                end_time=args.end,
                speed=1.0,
                arm_side=args.arm,
                save_comparison=False
            )
            comparison.close()
        except Exception as e:
            print(f"\n❌ Error in live comparison: {e}")
            return 1

        return 0

    # Run pipeline
    success = run_complete_pipeline(
        video_path=args.video,
        output_csv=args.output,
        start_time=args.start,
        end_time=args.end,
        use_greyscale=args.greyscale,
        skip_frames=args.skip_frames,
        arm_side=args.arm,
        scaling_factor=args.scale,
        smooth_window=args.smooth,
        show_preview=not args.no_preview,
        simulate=not args.no_simulate,
        visualize=args.visualize
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
