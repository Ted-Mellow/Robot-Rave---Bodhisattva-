# Thousand-Hand Bodhisattva → Robot Dance

Convert video of dance performance into robot arm choreography using computer vision.

## Quick Fix for MediaPipe Error

If you see `AttributeError: module 'mediapipe' has no attribute 'solutions'`:

```bash
./FIX_MEDIAPIPE.sh
```

Or manually:
```bash
pip install --force-reinstall mediapipe==0.10.9 pybullet
```

## Quick Start

### 1. Install Dependencies
```bash
pip install mediapipe==0.10.9 opencv-python pybullet numpy matplotlib
```

### 2. Run Live Comparison (See Pose + Simulation + Green Dots)
```bash
python run_pipeline.py --live --end 10
```

This shows:
- ✅ Video with pose detection
- ✅ **Green dots on body showing 6 robot joint axes**
- ✅ Real-time joint angles at each green dot
- ✅ Robot simulation in separate window
- ✅ Legend showing joint mapping

### 3. Generate CSV for Robot
```bash
python run_pipeline.py --end 10 --greyscale
```

Outputs: `bodhisattva_dance.csv` (ready for robot)

## What You Get

### Live Comparison Mode (`--live`)
- Video window with:
  - MediaPipe skeleton (white/red lines)
  - **Green dots showing robot joints:**
    - J1 at hip (base rotation)
    - J2 at shoulder (elevation)
    - J3 at elbow (bend)
    - J4-J6 at wrist (roll/pitch/rotate)
  - Joint angles displayed at each dot
  - Green lines connecting joints
  - Legend in corner
- Separate PyBullet window with 3D robot

### Controls
- `SPACE` - Pause/Resume
- `ESC` - Exit
- `S` - Save screenshot
- `+/-` - Speed up/slow down

## Files

**Scripts:**
- `run_pipeline.py` - Main entry point
- `pose_detector.py` - MediaPipe pose detection
- `motion_mapper.py` - Maps human→robot joints
- `simulate_dance.py` - PyBullet player
- `live_comparison.py` - Live view with green overlays

**Input:**
- `Cropped_thousandhand.mp4` - Video (23MB)

**Output:**
- `pose_data.json` - Detected poses
- `bodhisattva_dance.csv` - Robot choreography

## Commands

```bash
# Live comparison with green joint dots
python run_pipeline.py --live --end 10

# Process and save CSV
python run_pipeline.py --end 10 --greyscale

# Replay simulation
python simulate_dance.py bodhisattva_dance.csv

# Just pose detection
python pose_detector.py
```

## Troubleshooting

### MediaPipe Error
```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```
**Fix:** `pip install --force-reinstall mediapipe==0.10.9`

### No Green Dots Visible
- Green dots only appear in **live mode**: `--live`
- Regular mode (`--greyscale`) doesn't show them
- Make sure pose is detected (white skeleton visible)

### Import Error
**Fix:** `pip install opencv-python mediapipe==0.10.9 pybullet`

## Full Documentation

- `README.md` - Complete guide (detailed)
- `QUICKREF.md` - Quick reference
- `JOINT_MAPPING.md` - Joint visualization explained
- `LIVE_COMPARISON.md` - Live mode details
- `INSTALL.md` - Installation guide

## Green Dot Mapping

```
● J6 (0.00) - Wrist top
● J5 (0.00) - Wrist
● J4 (0.00) - Wrist
|
● J3 (-0.82) - Elbow
|
● J2 (1.65) - Shoulder
|
● J1 (0.85) - Hip
```

Each dot shows:
- Joint position on body
- Joint label (J1-J6)
- Current angle in radians

## Requirements

- Python 3.7+
- **MediaPipe 0.10.9** (important!)
- OpenCV
- PyBullet
- NumPy

---

**Key Points:**
- Use `--live` to see green joint overlays
- Use `--greyscale` to generate CSV files
- MediaPipe must be version 0.10.9
- Run `./FIX_MEDIAPIPE.sh` if you get errors
