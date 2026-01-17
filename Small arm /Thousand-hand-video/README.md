# Thousand-Hand Bodhisattva - Computer Vision to Robot Dance

Convert video of the Thousand-Hand Bodhisattva dance performance into choreography for the Piper robot arm using computer vision and pose detection.

## Overview

This system uses MediaPipe pose detection to track arm movements from video, maps human motion to robot joint angles, and plays the choreography in PyBullet simulation.

### Pipeline

1. **Pose Detection** - Extract arm movements from video using MediaPipe
2. **Motion Mapping** - Convert human pose to robot joint angles
3. **CSV Export** - Generate choreography in robot-compatible format
4. **Simulation** - Visualize and test in PyBullet

### Features

- **Greyscale Processing** - Improved accuracy through contrast enhancement
- **Temporal Smoothing** - Smooth trajectories for natural motion
- **Flexible Configuration** - Adjust scaling, time range, frame skipping
- **Live Preview** - See pose detection in real-time during processing
- **Side-by-side Comparison** - Play simulation alongside original video

## Quick Start

### 1. Install Dependencies

```bash
pip install opencv-python mediapipe numpy pybullet matplotlib
```

Or use the requirements file from the parent directory:

```bash
cd ..
pip install -r requirements.txt
```

### 2. Process Video (Quick Test - 10 seconds)

```bash
python3 run_pipeline.py --end 10 --greyscale
```

This will:
- Process the first 10 seconds of `Cropped_thousandhand.mp4`
- Use greyscale mode for better accuracy
- Show live preview of pose detection
- Generate `bodhisattva_dance.csv`
- Play the choreography in PyBullet simulation

**Note**: The system uses `Cropped_thousandhand.mp4` (23MB) by default, which is a focused crop of the performance for faster processing and better pose detection.

### 3. Process Full Video

```bash
python run_pipeline.py --greyscale
```

### 4. 🆕 Live Comparison Mode (See Pose Detection + Simulation Together!)

To see the pose detection and robot simulation side-by-side in real-time:

```bash
python run_pipeline.py --live --end 10
```

This shows:
- Video with pose detection overlay on the left/top
- PyBullet robot simulation on the right/bottom
- Real-time joint angles and statistics
- Detection accuracy metrics

**Perfect for:** Identifying inaccuracies, tuning parameters, and understanding how pose detection translates to robot motion!

## Usage

### Complete Pipeline

The easiest way to run the entire system:

```bash
# Process first 10 seconds with greyscale (recommended for testing)
python run_pipeline.py --end 10 --greyscale

# Process full video, right arm
python run_pipeline.py --arm right --greyscale

# Fast processing (skip every other frame)
python run_pipeline.py --skip-frames 2 --end 30

# Generate CSV only (no simulation)
python run_pipeline.py --no-simulate --end 10

# With visualization plots
python run_pipeline.py --visualize --end 10
```

### Live Comparison Mode

See pose detection and simulation together in real-time:

```bash
# Quick test with live comparison
python run_pipeline.py --live --end 10

# Or run standalone
python live_comparison.py

# Standalone with options
python live_comparison.py --end 20 --speed 0.5 --arm right

# Save comparison video to file
python live_comparison.py --end 10 --save
```

**Interactive controls during playback:**
- `SPACE` - Pause/Resume
- `ESC` - Exit
- `S` - Save current frame as screenshot
- `R` - Reset to beginning
- `+/-` - Increase/decrease playback speed

**What you see:**
- Video frame with pose skeleton overlay
- **🆕 Green dots showing 6 robot joint axes on the human body**
- Detection statistics (frame count, detection rate)
- Human arm angles (elevation, elbow)
- Robot joint angles in real-time (at each green dot)
- PyBullet simulation window showing synchronized robot motion
- Legend explaining joint mapping

**Use this mode to:**
- Verify pose detection accuracy
- See how human motion maps to robot joints
- Identify problematic frames or sections
- Tune scaling and smoothing parameters

### Individual Components

#### 1. Pose Detection Only

Extract pose data from video:

```bash
python pose_detector.py
```

Edit the script to configure:
- `VIDEO_PATH` - Input video file
- `START_TIME`, `END_TIME` - Time range to process
- `use_greyscale=True` - Enable greyscale mode

Outputs: `pose_data.json`

#### 2. Motion Mapping Only

Convert pose data to robot joints:

```bash
python motion_mapper.py
```

Requires: `pose_data.json` from pose detector

Outputs: `bodhisattva_dance.csv`

#### 3. Simulation Only

Play existing choreography:

```bash
# Normal playback
python simulate_dance.py bodhisattva_dance.csv

# Slow motion (0.5x speed)
python simulate_dance.py bodhisattva_dance.csv --speed 0.5

# Loop continuously
python simulate_dance.py bodhisattva_dance.csv --loop

# Step through frame by frame
python simulate_dance.py bodhisattva_dance.csv --step

# Compare with original video
python simulate_dance.py bodhisattva_dance.csv --video YTDown.com_*.mp4
```

## Configuration Options

### run_pipeline.py Options

```
Positional:
  video                 Input video file (default: Cropped_thousandhand.mp4)

Optional:
  -o, --output         Output CSV file (default: bodhisattva_dance.csv)
  --start SECONDS      Start time (default: 0.0)
  --end SECONDS        End time (default: full video)
  --greyscale          Use greyscale for better accuracy
  --skip-frames N      Process every Nth frame (0 = all)
  --arm left|right     Which arm to track (default: right)
  --scale FACTOR       Motion scaling (default: 0.8)
  --smooth SIZE        Smoothing window (default: 5)
  --no-preview         Disable video preview
  --no-simulate        Skip simulation
  --visualize          Create trajectory plots
```

### Greyscale Mode

The `--greyscale` flag enables preprocessing that:
1. Converts video to greyscale
2. Applies histogram equalization for contrast
3. Improves pose detection accuracy in complex lighting

**Recommended** for videos with:
- Multiple overlapping performers
- Complex backgrounds
- Varying lighting conditions

### Motion Scaling

The `--scale` parameter adjusts movement amplitude:
- `1.0` = Full range motion
- `0.8` = 80% scaled (safer, default)
- `0.5` = Half range motion

Lower values keep the robot within safer joint limits.

### Temporal Smoothing

The `--smooth` parameter sets the moving average window:
- `1` = No smoothing
- `5` = Moderate smoothing (default)
- `10` = Heavy smoothing (may lag)

Larger values create smoother but less responsive motion.

## File Outputs

### pose_data.json

Raw pose detection data with landmarks and joint angles:

```json
{
  "video": "path/to/video.mp4",
  "fps": 30.0,
  "frame_count": 300,
  "poses": [
    {
      "timestamp": 0.0,
      "frame": 0,
      "landmarks": { ... },
      "left_arm": { "shoulder_angle": ..., "elbow_angle": ... },
      "right_arm": { ... }
    }
  ]
}
```

### bodhisattva_dance.csv

Robot choreography in Piper-compatible format:

```csv
time,joint1,joint2,joint3,joint4,joint5,joint6,description
0.000,0.000,1.200,-0.500,0.000,0.000,0.000,Bodhisattva frame 0
0.033,0.450,1.600,-0.300,0.000,0.000,0.000,Bodhisattva frame 1
...
```

Columns:
- `time` - Timestamp in seconds
- `joint1` - Base rotation (rad)
- `joint2` - Shoulder elevation (rad)
- `joint3` - Elbow (rad)
- `joint4-6` - Wrist joints (rad)
- `description` - Human-readable label

## Technical Details

### Joint Mapping

Human arm movements map to robot joints as follows:

| Human Motion | Robot Joint | Range | Mapping |
|-------------|-------------|-------|---------|
| Lateral swing (left-right) | J1 (Base) | ±154° | Lateral position |
| Arm elevation (up-down) | J2 (Shoulder) | 0→195° | Arm elevation angle |
| Elbow bend | J3 (Elbow) | -175→0° | Elbow angle (inverted) |
| Hand orientation | J4-J6 (Wrist) | Various | Not currently used |

### Coordinate Systems

- **Video**: Origin at top-left, Y-axis down
- **MediaPipe**: Normalized [0, 1], origin at top-left
- **Robot**: Right-hand rule, Z-axis up

### Performance Tips

1. **Fast Testing**: Use `--end 10 --skip-frames 2` to process quickly
2. **Quality**: Use `--greyscale` for better pose detection
3. **Smoothness**: Increase `--smooth 10` for smoother motion
4. **Safety**: Reduce `--scale 0.6` to limit joint range

## Troubleshooting

### "No poses detected in video"

**Cause**: Pose detector cannot find person in frame

**Solutions**:
- Try `--greyscale` for better detection
- Adjust `start_time` to frame with clear view
- Check video plays correctly with a video player

### "Video file not found"

**Cause**: Video path is incorrect

**Solutions**:
- Ensure video is in `Thousand-hand-video/` folder
- Check filename matches exactly (case-sensitive)
- Use absolute path: `python run_pipeline.py /full/path/to/video.mp4`

### "Could not import PiperSimulation"

**Cause**: Simulation module not found

**Solutions**:
- Ensure `piper_simultion_corrected.py` exists in `../simulation/`
- Check file structure matches expected layout
- Run from `Thousand-hand-video/` directory

### Jerky motion in simulation

**Cause**: Not enough smoothing or too fast playback

**Solutions**:
- Increase smoothing: `--smooth 10`
- Process more frames: `--skip-frames 0`
- Slow playback: `python simulate_dance.py --speed 0.5`

### Arm not moving far enough

**Cause**: Motion scaling too conservative

**Solutions**:
- Increase scaling: `--scale 1.0`
- Check pose detection is tracking correctly (enable preview)
- Verify joint limits in motion mapper

## File Structure

```
Thousand-hand-video/
├── README.md                          # This file
├── QUICKREF.md                        # Quick reference card
├── OVERVIEW.md                        # System architecture
├── Cropped_thousandhand.mp4          # Input video (23MB, cropped)
├── YTDown.com_*.mp4                   # Original full video (126MB)
├── run_pipeline.py                    # Complete pipeline
├── pose_detector.py                   # Pose detection module
├── motion_mapper.py                   # Motion mapping module
├── simulate_dance.py                  # Simulation player
├── live_comparison.py                 # 🆕 Live pose + simulation viewer
├── quickstart.sh                      # Interactive setup script
├── requirements.txt                   # Python dependencies
├── pose_data.json                     # Generated pose data
├── bodhisattva_dance.csv             # Generated choreography
└── trajectory_visualization.png       # Generated plot (if --visualize)
```

## Examples

### Example 1: Quick Test

Process 5 seconds and check if it works:

```bash
python run_pipeline.py --end 5 --greyscale
```

### Example 2: High Quality Full Video

Process entire video with best quality settings:

```bash
python run_pipeline.py --greyscale --smooth 7 --scale 0.9
```

### Example 3: Fast Prototype

Quickly test with low frame rate:

```bash
python run_pipeline.py --end 10 --skip-frames 3 --no-preview
```

### Example 4: Generate CSV Only

Extract choreography without simulation:

```bash
python run_pipeline.py --end 20 --greyscale --no-simulate
```

Then test later:

```bash
python simulate_dance.py bodhisattva_dance.csv --loop
```

### Example 5: Side-by-Side Comparison

See robot motion next to original video:

```bash
python simulate_dance.py bodhisattva_dance.csv --video Cropped_thousandhand.mp4
```

## Next Steps

1. **Tune Parameters**: Adjust `--scale` and `--smooth` for desired motion
2. **Select Best Segments**: Use `--start` and `--end` to focus on good parts
3. **Track Multiple Arms**: Process left and right arms separately
4. **Export to Robot**: Load CSV on real Piper hardware
5. **Create Variations**: Adjust scaling/smoothing for different styles

## Dependencies

- `opencv-python` >= 4.5.0 - Video processing
- `mediapipe` >= 0.10.0 - Pose detection
- `numpy` >= 1.19.0 - Numerical operations
- `pybullet` >= 3.2.0 - Robot simulation
- `matplotlib` >= 3.3.0 - Visualization (optional)

## Credits

- Performance: Thousand-Hand Bodhisattva by China Disabled People's Performing Art Troupe
- Pose Detection: Google MediaPipe
- Robot Simulation: PyBullet
- Robot: Piper Agile X

## License

Educational and research use only. The original Thousand-Hand Bodhisattva performance is copyrighted by its respective owners.
