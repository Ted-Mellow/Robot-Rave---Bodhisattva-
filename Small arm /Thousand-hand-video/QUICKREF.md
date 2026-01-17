# Quick Reference - Thousand-Hand Bodhisattva

## One-Command Quick Start

```bash
cd "Small arm /Thousand-hand-video"
python run_pipeline.py --end 10 --greyscale
```

This processes the first 10 seconds of `Cropped_thousandhand.mp4` and opens the simulation.

## 🆕 Live Comparison Mode (See Pose + Simulation Side-by-Side!)

```bash
python run_pipeline.py --live --end 10
```

This shows pose detection on video **alongside** robot simulation in real-time, so you can see exactly how the pose detection translates to robot motion and spot any inaccuracies!

---

## Common Commands

### Process Video
```bash
# Quick test (10 seconds)
python run_pipeline.py --end 10 --greyscale

# Full video
python run_pipeline.py --greyscale

# Specific time range (seconds 5-15)
python run_pipeline.py --start 5 --end 15 --greyscale

# Fast processing (skip frames)
python run_pipeline.py --end 20 --skip-frames 2 --greyscale
```

### Replay Simulation
```bash
# Normal playback
python simulate_dance.py bodhisattva_dance.csv

# Slow motion
python simulate_dance.py bodhisattva_dance.csv --speed 0.5

# Loop
python simulate_dance.py bodhisattva_dance.csv --loop

# Step through
python simulate_dance.py bodhisattva_dance.csv --step

# With video comparison
python simulate_dance.py bodhisattva_dance.csv --video Cropped_thousandhand.mp4
```

### Use Different Video
```bash
# Process the full original video instead of cropped
python run_pipeline.py YTDown.com_YouTube_*.mp4 --end 10 --greyscale
```

### Live Comparison (NEW!)
```bash
# See pose detection + simulation in real-time
python run_pipeline.py --live --end 10

# Or run standalone
python live_comparison.py --end 10

# With playback speed control
python live_comparison.py --end 20 --speed 0.5

# Save the comparison video
python live_comparison.py --end 10 --save
```

**Live comparison shows:**
- Video with pose detection overlay
- Real-time robot simulation
- Joint angles and statistics
- Detection accuracy metrics

**Controls during playback:**
- `SPACE` - Pause/Resume
- `ESC` - Exit
- `S` - Save screenshot
- `R` - Reset to start
- `+/-` - Adjust playback speed

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--end SECONDS` | Full video | Stop processing at this time |
| `--start SECONDS` | 0.0 | Start processing at this time |
| `--greyscale` | Off | **Recommended**: Better accuracy |
| `--skip-frames N` | 0 | Skip every N frames (faster) |
| `--scale FACTOR` | 0.8 | Motion scaling (0.6-1.0) |
| `--smooth SIZE` | 5 | Smoothing window (1-10) |
| `--arm left\|right` | right | Which arm to track |
| `--no-preview` | Off | Disable video preview |
| `--no-simulate` | Off | Skip simulation |

---

## Files

**Input:**
- `Cropped_thousandhand.mp4` - **Default**, 23MB, focused crop
- `YTDown.com_*.mp4` - Original full video, 126MB

**Output:**
- `pose_data.json` - Raw pose detection data
- `bodhisattva_dance.csv` - Robot choreography
- `trajectory_visualization.png` - Plots (if `--visualize`)

---

## Troubleshooting

**No poses detected:**
```bash
# Use greyscale mode
python run_pipeline.py --end 10 --greyscale
```

**Motion too small:**
```bash
# Increase scaling
python run_pipeline.py --end 10 --greyscale --scale 1.0
```

**Motion too jerky:**
```bash
# Increase smoothing
python run_pipeline.py --end 10 --greyscale --smooth 10
```

**Processing too slow:**
```bash
# Skip frames
python run_pipeline.py --end 10 --skip-frames 2 --no-preview
```

---

## Video Comparison

The cropped video (`Cropped_thousandhand.mp4`, 23MB) is:
- ✅ Smaller file size (faster to process)
- ✅ Focused on single performer (better pose detection)
- ✅ Better for testing and development
- ✅ **Recommended for most use cases**

The original video (`YTDown.com_*.mp4`, 126MB) is:
- Full performance view
- Multiple performers visible
- May need specific time ranges for best results

---

## Help

```bash
# Full help
python run_pipeline.py --help
python simulate_dance.py --help

# Read docs
cat README.md
cat OVERVIEW.md
```

---

**Quick tip:** Always use `--greyscale` for best results with the Thousand-Hand Bodhisattva video!
