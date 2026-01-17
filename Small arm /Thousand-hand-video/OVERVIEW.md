# Thousand-Hand Bodhisattva - System Overview

## What This Does

Converts video of the Thousand-Hand Bodhisattva dance into robot arm choreography using computer vision and pose detection.

```
VIDEO → POSE DETECTION → MOTION MAPPING → CSV → SIMULATION
  📹         🎭              🤖            📄       🦾
```

## Quick Start (Copy & Paste)

```bash
# 1. Install dependencies
pip install opencv-python mediapipe numpy pybullet matplotlib

# 2. Run the complete pipeline (5 second test)
cd "Small arm /Thousand-hand-video"
python run_pipeline.py --end 5 --greyscale

# 3. That's it! The simulation will open automatically.
```

## What You Get

### Input
- `Cropped_thousandhand.mp4` - Cropped video of Thousand-Hand Bodhisattva performance (23MB)
- `YTDown.com_*.mp4` - Original full video (126MB, alternative)

### Outputs
- `pose_data.json` - Detected pose landmarks and arm angles
- `bodhisattva_dance.csv` - Robot choreography (compatible with Piper simulator)
- PyBullet simulation showing the robot performing the dance

## Files Created

### Core Modules

1. **pose_detector.py** (15KB)
   - Uses MediaPipe to track human pose from video
   - Extracts arm joint angles (shoulder, elbow, wrist)
   - Supports greyscale mode for better accuracy
   - Outputs: `pose_data.json`

2. **motion_mapper.py** (10KB)
   - Converts human arm angles to robot joint angles
   - Maps 3D human motion to 6-axis robot arm
   - Applies temporal smoothing for natural motion
   - Outputs: `bodhisattva_dance.csv`

3. **simulate_dance.py** (8.8KB)
   - Loads CSV choreography
   - Plays in PyBullet simulation
   - Supports playback speed control, looping, step-through
   - Can play side-by-side with original video

4. **run_pipeline.py** (7KB)
   - Runs complete pipeline in one command
   - Configurable parameters (time range, scaling, smoothing)
   - Command-line interface with help

### Supporting Files

5. **README.md** (9.9KB)
   - Complete documentation
   - Usage examples
   - Troubleshooting guide
   - Configuration options

6. **requirements.txt** (615B)
   - Python dependencies list
   - Quick installation

7. **quickstart.sh** (3.1KB, executable)
   - Interactive setup script
   - Checks dependencies
   - Multiple processing options

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         VIDEO INPUT                              │
│  Cropped_thousandhand.mp4 (23MB, cropped for better detection)  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POSE DETECTION                                │
│  pose_detector.py                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • MediaPipe Pose (model complexity 2)                    │  │
│  │ • Greyscale preprocessing (optional)                      │  │
│  │ • Histogram equalization for contrast                     │  │
│  │ • Tracks 33 body landmarks                                │  │
│  │ • Calculates arm joint angles                             │  │
│  │ • Real-time preview                                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼ pose_data.json
┌─────────────────────────────────────────────────────────────────┐
│                    MOTION MAPPING                                │
│  motion_mapper.py                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Maps human arm angles to robot joints:                 │  │
│  │   - Lateral swing → J1 (Base rotation)                   │  │
│  │   - Arm elevation → J2 (Shoulder)                        │  │
│  │   - Elbow bend → J3 (Elbow)                              │  │
│  │ • Motion scaling (safety limits)                          │  │
│  │ • Temporal smoothing (moving average)                     │  │
│  │ • Joint limit clamping                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼ bodhisattva_dance.csv
┌─────────────────────────────────────────────────────────────────┐
│                      SIMULATION                                  │
│  simulate_dance.py                                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Loads CSV choreography                                  │  │
│  │ • Uses PiperSimulation (from parent dir)                  │  │
│  │ • PyBullet physics engine                                 │  │
│  │ • Real-time playback                                      │  │
│  │ • Speed control, looping, step-through                    │  │
│  │ • Side-by-side video comparison                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Greyscale Processing
Improves pose detection accuracy by:
- Converting to greyscale
- Applying histogram equalization
- Enhancing contrast in complex scenes
- Better for multiple overlapping performers

**Enable with**: `--greyscale` flag

### 2. Motion Scaling
Scales movement amplitude for safety:
- `1.0` = Full human motion range
- `0.8` = 80% range (default, recommended)
- `0.5` = Half range (very safe)

**Configure with**: `--scale FACTOR`

### 3. Temporal Smoothing
Smooths trajectories using moving average:
- Removes jitter from pose detection
- Creates natural, fluid motion
- Window size: 1 (none) to 10 (heavy)

**Configure with**: `--smooth SIZE`

### 4. Frame Skipping
Process every Nth frame for speed:
- `0` = Process all frames (slow, accurate)
- `1` = Process every other frame (2x faster)
- `2` = Process every 3rd frame (3x faster)

**Configure with**: `--skip-frames N`

## Joint Mapping Details

### Human to Robot Joint Mapping

| Human Motion | Measurement | Robot Joint | Range | Formula |
|-------------|-------------|-------------|-------|---------|
| Left-Right Swing | Lateral position | J1 (Base) | ±154° (-2.68 to 2.68 rad) | Linear mapping from lateral position |
| Arm Up-Down | Elevation angle | J2 (Shoulder) | 0→195° (0 to 3.40 rad) | Map -90° (down) to 90° (up) → 0.3 to 2.8 rad |
| Elbow Bend | Elbow angle | J3 (Elbow) | -175→0° (-3.05 to 0 rad) | Invert: 180° (straight) → 0 rad, 30° (bent) → -2.5 rad |
| Wrist Roll | - | J4 | ±106° | Not used (0°) |
| Wrist Pitch | - | J5 | ±75° | Not used (0°) |
| Wrist Rotate | - | J6 | ±100° | Not used (0°) |

### Coordinate Systems

**Video Frame**:
- Origin: Top-left corner
- X-axis: Right
- Y-axis: Down

**MediaPipe Normalized**:
- Origin: Top-left corner
- Coordinates: [0, 1] range
- X: 0 (left) to 1 (right)
- Y: 0 (top) to 1 (bottom)
- Z: Depth relative to hips

**Robot (Piper)**:
- Origin: Base center
- Right-hand coordinate system
- Z-axis: Up (against gravity)

## Usage Examples

### Example 1: Absolute Beginner
```bash
# Install everything
pip install opencv-python mediapipe numpy pybullet

# Run with defaults (5 seconds)
python run_pipeline.py --end 5 --greyscale
```

### Example 2: Test Different Time Ranges
```bash
# First 10 seconds
python run_pipeline.py --end 10 --greyscale

# Seconds 30-60
python run_pipeline.py --start 30 --end 60 --greyscale

# Full video (will take a while!)
python run_pipeline.py --greyscale
```

### Example 3: Optimize for Speed
```bash
# Skip every other frame
python run_pipeline.py --end 10 --skip-frames 2 --no-preview
```

### Example 4: Optimize for Quality
```bash
# Process all frames, high smoothing
python run_pipeline.py --end 30 --greyscale --smooth 10 --scale 0.9
```

### Example 5: Generate CSV, Test Later
```bash
# Generate CSV without simulation
python run_pipeline.py --end 20 --greyscale --no-simulate

# Test simulation separately
python simulate_dance.py bodhisattva_dance.csv --loop
```

## Command Reference

### run_pipeline.py
Complete pipeline from video to simulation.

```bash
python run_pipeline.py [VIDEO] [OPTIONS]

# Main options:
--end SECONDS          # Process until this time
--start SECONDS        # Start from this time
--greyscale            # Use greyscale mode (recommended)
--skip-frames N        # Skip every N frames
--arm left|right       # Which arm to track
--scale FACTOR         # Motion scaling (0.0-1.0)
--smooth SIZE          # Smoothing window (1-10)
--no-preview           # Disable video preview
--no-simulate          # Skip simulation
--visualize            # Create plots
-o FILE.csv            # Output CSV filename
```

### simulate_dance.py
Play choreography in PyBullet.

```bash
python simulate_dance.py FILE.csv [OPTIONS]

# Main options:
--speed FACTOR         # Playback speed (0.5 = half speed)
--loop                 # Loop continuously
--step                 # Step through frame-by-frame
--video FILE.mp4       # Compare with original video
--urdf FILE.urdf       # Use custom robot model
```

## Performance

### Processing Time (approximate)
- **5 seconds** of video: ~30 seconds processing
- **30 seconds** of video: ~3 minutes processing
- **Full video** (2+ minutes): ~10-20 minutes processing

Variables affecting speed:
- Video resolution (1080p is slower)
- Greyscale mode (slightly faster)
- Frame skipping (linear speedup)
- CPU/GPU performance

### Memory Usage
- ~500MB for pose detection
- ~100MB for simulation
- Video loading is streaming (low memory)

## Troubleshooting

### Common Issues

**"No poses detected"**
→ Try `--greyscale` flag
→ Check video plays correctly
→ Adjust time range to clear frames

**"Import error: mediapipe"**
→ `pip install mediapipe`

**"Could not import PiperSimulation"**
→ Ensure `../simulation/piper_simultion_corrected.py` exists

**Jerky motion**
→ Increase `--smooth` parameter
→ Don't skip frames (`--skip-frames 0`)

**Arm barely moving**
→ Increase `--scale` to 1.0
→ Check pose detection preview

## Next Steps

1. **Test the system**: Run quickstart with 5 seconds
2. **Tune parameters**: Adjust scale/smooth to your liking
3. **Process longer segments**: Gradually increase time range
4. **Export to real robot**: Load CSV onto Piper hardware
5. **Track multiple performers**: Process different time segments

## Technical Stack

- **Python 3.7+**
- **OpenCV 4.5+** - Video I/O and preprocessing
- **MediaPipe 0.10+** - Pose detection (Google)
- **NumPy 1.19+** - Numerical operations
- **PyBullet 3.2+** - Physics simulation
- **Matplotlib 3.3+** - Visualization (optional)

## Credits

- **Performance**: Thousand-Hand Bodhisattva by China Disabled People's Performing Art Troupe
- **Pose Detection**: Google MediaPipe
- **Physics Engine**: PyBullet (Bullet Physics)
- **Robot Model**: Piper Agile X
- **Implementation**: Computer vision pipeline for robot choreography

---

**For full documentation, see [README.md](README.md)**

**For quick start, run: `./quickstart.sh`**
