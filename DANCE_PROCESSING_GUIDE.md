# Dance Sequence Processing Guide

## Overview

The `process_dance_sequence.py` script processes dance sequences to make them:
- **Slower** - More graceful and deliberate movements
- **Smoother** - Natural motion curves without jitter
- **More natural** - Matches the elegant flow of the Thousand-Hand Bodhisattva dance

## Quick Start

### 1. Process the Bodhisattva Sequence

```bash
# Default settings (60% speed, moderate smoothing)
python process_dance_sequence.py bodhisattva_sequence.json

# Very slow and smooth (50% speed, heavy smoothing)
python process_dance_sequence.py bodhisattva_sequence.json --speed 0.5 --smooth 0.5

# Moderate slow, light smoothing (70% speed, light smoothing)
python process_dance_sequence.py bodhisattva_sequence.json --speed 0.7 --smooth 0.2

# Custom output file
python process_dance_sequence.py bodhisattva_sequence.json --output graceful_dance.json
```

### 2. Replay the Processed Sequence

```bash
# Start the simulation server first (in one terminal)
cd "Small arm /simulation"
python realtime_ik_trajectory.py

# Then replay the processed sequence (in another terminal)
python replay_sequence_to_simulation.py bodhisattva_sequence_processed.json

# Or with custom speed
python replay_sequence_to_simulation.py bodhisattva_sequence_processed.json --speed 1.0

# Loop continuously
python replay_sequence_to_simulation.py bodhisattva_sequence_processed.json --loop
```

## Processing Parameters

### Speed Multiplier (`--speed`)
- **0.5** = 50% speed (very slow, meditative)
- **0.6** = 60% speed (default, graceful)
- **0.7** = 70% speed (moderately slow)
- **1.0** = 100% speed (original)

### Smoothing Factor (`--smooth`)
- **0.1** = Light smoothing (preserves more detail)
- **0.3** = Moderate smoothing (default, balanced)
- **0.5** = Heavy smoothing (very smooth, may lose some detail)
- **0.7+** = Very heavy smoothing (extremely smooth)

## What the Script Does

1. **Savitzky-Golay Filtering**: Applies polynomial smoothing that preserves important movement features while reducing noise
2. **Exponential Smoothing**: Additional layer of smoothing for extra fluidity
3. **Temporal Resampling**: Slows down the sequence by resampling to a lower frame rate with smooth interpolation
4. **Natural Motion Curves**: Uses cubic interpolation to create smooth transitions between keypoints

## Recommended Settings for Thousand-Hand Bodhisattva

For the traditional, graceful movements of the Thousand-Hand Bodhisattva dance:

```bash
# Recommended: Slow, smooth, meditative
python process_dance_sequence.py bodhisattva_sequence.json --speed 0.55 --smooth 0.35 --output bodhisattva_graceful.json
```

This creates movements that are:
- **55% of original speed** - Slow enough to be meditative and deliberate
- **35% smoothing** - Smooth enough to be graceful without losing character
- **Natural curves** - Movements flow like water, matching the dance's aesthetic

## Output Format

The processed sequence maintains the same JSON structure as the original, with additional metadata:

```json
{
  "source_video": "...",
  "duration": 67.27,
  "fps": 15.0,
  "processing": {
    "original_fps": 25.0,
    "speed_multiplier": 0.6,
    "smoothing_factor": 0.3,
    "smoothing_method": "savitzky_golay + exponential"
  },
  "keypoints": [...]
}
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"
Install dependencies:
```bash
pip install numpy scipy
```

### "ModuleNotFoundError: No module named 'scipy'"
The script will work without scipy, but with less sophisticated smoothing:
```bash
pip install scipy  # Recommended for best results
```

### Movements still too fast
Try lower speed multiplier:
```bash
python process_dance_sequence.py bodhisattva_sequence.json --speed 0.4
```

### Movements too jerky
Increase smoothing:
```bash
python process_dance_sequence.py bodhisattva_sequence.json --smooth 0.5
```

## Workflow

1. **Extract** keypoints from video (if needed):
   ```bash
   python process_video_for_piper.py "video.mp4" 10 70
   ```

2. **Process** the sequence for natural movement:
   ```bash
   python process_dance_sequence.py output_sequence.json --speed 0.6 --smooth 0.3
   ```

3. **Preview** in simulation:
   ```bash
   # Terminal 1: Start simulation
   cd "Small arm /simulation" && python realtime_ik_trajectory.py
   
   # Terminal 2: Replay sequence
   python replay_sequence_to_simulation.py output_sequence_processed.json --loop
   ```

4. **Adjust** parameters as needed and reprocess until satisfied

## Tips

- Start with default settings, then adjust based on what you see
- For very graceful movements, use `--speed 0.5 --smooth 0.4`
- For more dynamic movements, use `--speed 0.7 --smooth 0.2`
- The `--loop` flag in replay is great for testing and refinement
- Process multiple versions with different settings to compare
