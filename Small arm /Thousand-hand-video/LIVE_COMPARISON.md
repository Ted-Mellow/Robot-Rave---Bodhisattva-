# Live Comparison Mode - Visual Guide

## What is Live Comparison?

Live Comparison Mode shows you **exactly** how the computer vision system interprets the video and controls the robot, in real-time. This is perfect for:

✅ **Spotting inaccuracies** - See immediately when pose detection fails or misinterprets movements
✅ **Understanding the mapping** - Watch how human arm angles translate to robot joint angles
✅ **Tuning parameters** - Adjust scaling and smoothing by seeing real-time results
✅ **Quality control** - Verify the choreography looks correct before exporting

## Quick Start

```bash
python run_pipeline.py --live --end 10
```

or standalone:

```bash
python live_comparison.py --end 10
```

## What You See

### Display Window Layout

```
╔══════════════════════════════════════════════════════════════════╗
║  Pose Detection + Robot Simulation - Live Comparison            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌─ Statistics ────┐                    ┌─ Frame Info ─┐       ║
║  │ Processed: 234  │                    │ Frame: 234   │       ║
║  │ Detected: 230   │                    │ Time: 7.80s  │       ║
║  │ Failed: 4       │                    │ Pose: DETECT │       ║
║  └─────────────────┘                    └──────────────┘       ║
║                                                                  ║
║            ┌──────────────────────────────┐                     ║
║            │                              │                     ║
║            │    VIDEO WITH POSE OVERLAY   │                     ║
║            │                              │                     ║
║            │  [Skeleton drawn on person]  │                     ║
║            │                              │                     ║
║            │  • Green lines = detected    │                     ║
║            │  • Shows arm elevation       │                     ║
║            │  • Shows elbow angles        │                     ║
║            │                              │                     ║
║            └──────────────────────────────┘                     ║
║                                                                  ║
║  ┌─ Arm Angles ────────────────┐                                ║
║  │ Arm Elevation: 45.2°        │                                ║
║  │ Elbow Angle: 135.7°         │                                ║
║  └─────────────────────────────┘                                ║
║                                                                  ║
║                    ┌─ Robot Joint Angles (rad) ┐                ║
║                    │ J1 Base:   0.85           │                ║
║                    │ J2 Shldr:  1.65           │                ║
║                    │ J3 Elbow: -0.82           │                ║
║                    │ J4 Roll:   0.00           │                ║
║                    │ J5 Pitch:  0.00           │                ║
║                    │ J6 Rot:    0.00           │                ║
║                    └───────────────────────────┘                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

PLUS: Separate PyBullet window showing robot moving in real-time!
```

### Display Elements

1. **Video Frame with Pose Overlay**
   - Shows original video with MediaPipe skeleton drawn on top
   - Green lines connect detected body landmarks
   - Arms, shoulders, and torso clearly marked

2. **Frame Information (Top Left)**
   - Current frame number and timestamp
   - Pose detection status (DETECTED or NOT DETECTED)
   - Updates every frame

3. **Statistics (Top Right)**
   - Total frames processed
   - Successful pose detections (green)
   - Failed detections (red)
   - Detection rate percentage

4. **Arm Angles (Middle)**
   - Arm elevation angle (how high arm is raised)
   - Elbow bend angle
   - Measured from detected pose landmarks

5. **Robot Joint Angles (Bottom Right)**
   - Real-time joint values sent to robot
   - All 6 joints displayed (J1-J6)
   - Values in radians

6. **PyBullet Simulation Window**
   - Separate window showing 3D robot
   - Moves synchronously with video
   - Same joint angles as displayed

## Interactive Controls

While the comparison is running, you can control playback:

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume playback |
| `ESC` | Exit the comparison |
| `S` | Save current frame as screenshot |
| `R` | Reset to beginning |
| `+` or `=` | Increase playback speed |
| `-` or `_` | Decrease playback speed |

## Usage Examples

### Basic Usage

```bash
# Quick 10 second test
python live_comparison.py --end 10

# Process full video
python live_comparison.py

# Specific time range (seconds 5-15)
python live_comparison.py --start 5 --end 15
```

### With Options

```bash
# Slow motion (half speed) for detailed analysis
python live_comparison.py --end 10 --speed 0.5

# Track left arm instead of right
python live_comparison.py --end 10 --arm left

# Save the comparison as a video file
python live_comparison.py --end 30 --save
```

### Through run_pipeline.py

```bash
# Use --live flag with pipeline
python run_pipeline.py --live --end 10

# With other options
python run_pipeline.py --live --end 20 --arm left
```

## 🆕 Robot Joint Axis Visualization

The live comparison now displays **green dots** on the human body showing exactly where each of the 6 robot joints maps to:

### Joint Mapping Overlay

- **J1 (Base)** - Green dot at hip → Robot base rotation (left-right swing)
- **J2 (Shoulder)** - Green dot at shoulder → Shoulder elevation (up-down)
- **J3 (Elbow)** - Green dot at elbow → Elbow bend
- **J4 (Wrist Roll)** - Green dot at wrist → Wrist rotation
- **J5 (Wrist Pitch)** - Green dot above wrist → Wrist pitch
- **J6 (Wrist Rotate)** - Green dot top of wrist → End effector rotation

### Visual Features

Each green dot shows:
- ✅ Joint position on the body
- ✅ Joint label (J1, J2, J3, etc.)
- ✅ Real-time angle value in radians
- ✅ Green connecting lines showing kinematic chain (J1→J2→J3)

### Legend

A legend in the bottom-left corner explains:
- Which body part controls which robot joint
- The function of each joint

This makes it immediately clear how the pose detection translates to robot motion!

## What to Look For

### Good Detection

✅ **Skeleton tracks person smoothly**
- Green lines follow body movements
- No jittering or jumping
- Joints stay in correct positions

✅ **Arm angles make sense**
- Elevation angle matches visual height of arm
- Elbow angle matches how bent the arm looks

✅ **Robot motion looks natural**
- Robot arm follows human arm smoothly
- No sudden jerky movements
- Motion is fluid and synchronized

### Problem Detection

❌ **Skeleton not appearing**
- "Pose: NOT DETECTED" shows in red
- No green lines on video
- Person may be obscured or off-screen

❌ **Skeleton jumps around**
- Detection is unstable
- May need greyscale mode
- Try adjusting detection confidence

❌ **Robot motion too small/large**
- Adjust `--scale` parameter in mapping
- Default is 0.8 (80% of human motion)
- Try 1.0 for full range, 0.6 for safer motion

❌ **Robot motion is jerky**
- Increase smoothing window
- Default is 3 for live mode (responsive)
- Try higher values for smoother motion

## Troubleshooting

### No pose detected
**Symptom:** Red "Pose: NOT DETECTED" message

**Solutions:**
- Check if person is clearly visible in frame
- Try greyscale mode (not available in live mode yet, use regular pipeline first)
- Adjust start time to a clearer section

### Robot barely moves
**Symptom:** Joint angles very small, robot stays still

**Solutions:**
- Check that arm angles are being detected (shown on display)
- May need to adjust motion mapping scaling
- Verify robot is not at joint limits

### Video and robot out of sync
**Symptom:** Robot lags behind video or vice versa

**Cause:** This is normal - slight delay for processing
**Note:** The system prioritizes smooth motion over perfect sync

### Comparison window too large
**Symptom:** Window doesn't fit on screen

**Automatic:** System auto-scales display if video is > 1280px wide

## Advanced Usage

### Save Comparison Video

```bash
python live_comparison.py --end 30 --save
```

Saves to `comparison_output.mp4` with:
- Video frame with pose overlay
- All statistics and angles overlaid
- Same as what you see on screen

### Compare Multiple Segments

```bash
# Test beginning
python live_comparison.py --start 0 --end 10

# Test middle
python live_comparison.py --start 30 --end 40

# Test end
python live_comparison.py --start 60 --end 70
```

### Performance Mode

```bash
# Faster processing (may be less smooth)
python live_comparison.py --end 10 --speed 1.5
```

## Technical Details

### Frame Processing Flow

1. **Read video frame** (30 FPS typical)
2. **Pose detection** (~50-100ms per frame)
3. **Calculate arm angles** (~1ms)
4. **Map to robot joints** (~1ms)
5. **Apply smoothing** (moving average over 3-5 frames)
6. **Update robot** (send joint commands)
7. **Render display** (add overlays, show frame)
8. **Step simulation** (PyBullet physics)

**Total:** ~60-120ms per frame (depends on CPU/GPU)

### Smoothing

Live mode uses less smoothing than batch mode for responsiveness:
- **Live mode:** 3 frame window (responsive)
- **Batch mode:** 5-10 frame window (smoother but lags)

### Joint Update Rate

- Video: 30 FPS (or video's native FPS)
- Pose detection: Every frame
- Robot update: Every frame
- Simulation: 240 Hz (PyBullet internal rate)

## Comparison with Other Modes

| Mode | Use Case | Speed | Accuracy View |
|------|----------|-------|---------------|
| **Live Comparison** | Real-time validation | Slow (processes as you watch) | ✅ Best - see everything |
| **Regular Pipeline** | Batch processing | Fast (no display) | ❌ None - blind processing |
| **Simulation Replay** | Review after processing | Fast (pre-computed) | ⚠️ Only robot, no pose data |

## Best Practices

1. **Start with short segments** (10 seconds) to test
2. **Use live comparison** to find good sections of video
3. **Batch process** those sections with regular pipeline
4. **Review** with simulation replay for final check

## Summary

Live Comparison Mode gives you **x-ray vision** into your computer vision pipeline:

- 🎥 See what the pose detector sees
- 📐 See the angles it measures
- 🤖 See how those map to robot joints
- ✅ Catch problems immediately
- 🎯 Perfect for debugging and validation

**Start with:**
```bash
python run_pipeline.py --live --end 10
```

And watch your video come to life on the robot! 🦾✨
