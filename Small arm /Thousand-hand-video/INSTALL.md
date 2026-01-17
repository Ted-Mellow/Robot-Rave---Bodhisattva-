# Installation Guide

## Quick Install

```bash
pip3 install opencv-python mediapipe==0.10.9 numpy pybullet matplotlib
```

## Step-by-Step Installation

### 1. Check Python Version

```bash
python3 --version
```

You need Python 3.7 or later. The system was tested with Python 3.9.

### 2. Install Dependencies

```bash
cd "Small arm /Thousand-hand-video"
pip3 install -r requirements.txt
```

Or install manually:

```bash
pip3 install opencv-python>=4.5.0
pip3 install mediapipe==0.10.9
pip3 install numpy
pip3 install pybullet>=3.2.0
pip3 install matplotlib>=3.3.0
```

### 3. Verify Installation

```bash
python3 -c "import cv2, mediapipe, numpy, pybullet, matplotlib; print('All dependencies installed!')"
```

### 4. Test the System

```bash
python3 run_pipeline.py --end 5 --greyscale
```

## Troubleshooting

### MediaPipe AttributeError

**Error:** `AttributeError: module 'mediapipe' has no attribute 'solutions'`

**Cause:** Wrong MediaPipe version (0.10.31+ has different API)

**Solution:**
```bash
pip3 install --force-reinstall "mediapipe==0.10.9"
```

### PyBullet Installation Error

**Error:** `ERROR: Cannot uninstall pybullet, RECORD file not found`

**Solution:**
```bash
rm -rf ~/Library/Python/3.9/lib/python/site-packages/pybullet*
pip3 install pybullet
```

### Import Error

**Error:** `ModuleNotFoundError: No module named 'mediapipe'`

**Solution:** Dependencies not installed
```bash
pip3 install -r requirements.txt
```

### NumPy Version Conflict

**Error:** `requires numpy<1.28.0, but you have numpy 2.0.2`

**Solution:** This warning can be ignored. The system works with numpy 2.0.2.

## Platform-Specific Notes

### macOS

- Tested on macOS with Apple Silicon (M1)
- May need Xcode Command Line Tools: `xcode-select --install`

### Linux

```bash
# May need these system packages first
sudo apt-get install python3-dev python3-pip
sudo apt-get install libgl1-mesa-glx  # For OpenCV
```

### Windows

```bash
# Use PowerShell or Command Prompt
pip install -r requirements.txt
```

## Dependency Versions

The system has been tested with:

- Python: 3.9
- opencv-python: 4.8.1+
- mediapipe: **0.10.9** (important!)
- numpy: 2.0.2
- pybullet: 3.2.7
- matplotlib: 3.9.4

## Minimal Installation

If you only want pose detection (no simulation):

```bash
pip3 install opencv-python mediapipe==0.10.9 numpy
```

## Full Installation

For all features including simulation and visualization:

```bash
pip3 install opencv-python mediapipe==0.10.9 numpy pybullet matplotlib
```

## Verification Commands

```bash
# Test pose detection
python3 pose_detector.py

# Test motion mapping
python3 motion_mapper.py

# Test simulation (requires PyBullet)
python3 simulate_dance.py bodhisattva_dance.csv

# Test complete pipeline
python3 run_pipeline.py --end 5 --greyscale
```

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Verify Python version: `python3 --version`
3. Check installed packages: `pip3 list | grep -E "(mediapipe|pybullet|opencv)"`
4. Read the error message carefully
5. Check [README.md](README.md) for usage examples

## Success!

Once installed, you should be able to run:

```bash
python3 run_pipeline.py --end 10 --greyscale
```

And see the pose detection processing followed by the robot simulation! 🎉
