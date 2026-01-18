# Piper Robot - Video to Dance Movement

**Workflow:** Video preprocessing → Simulate & test → Send to Pi → Play on robot

## 🎯 Complete Workflow

```bash
cd "Small arm " && source venv/bin/activate

# STEP 1: Extract trajectory from video (~60 sec, 90% detection rate)
python arm_control/dance_player.py Thousand-hand-video/Cropped_thousandhand.mp4 --preprocess
# → Saves: Thousand-hand-video/Cropped_thousandhand_trajectory.json

# STEP 2a: Test - VIDEO with CV detection (stable)
python arm_control/dance_player.py Thousand-hand-video/Cropped_thousandhand.mp4 \
    -t Thousand-hand-video/Cropped_thousandhand_trajectory.json --sim --auto-play --smooth 20
# Shows: Video with pose detection overlays (robot simulates in background)
# Press Q to quit, SPACE to pause

# STEP 2b: Test - View ROBOT ONLY (to see exact movements)
python convert_json_to_csv.py Thousand-hand-video/Cropped_thousandhand_trajectory.json
python simulation/run_csv_trajectory.py csv_trajectories/Cropped_thousandhand_trajectory.csv
# Shows: PyBullet robot performing the extracted dance movements
# Run this ALONGSIDE Step 2a (different terminal) to see both

# STEP 2c: Adjust smoothing if movements jerky (try 10-30, default: 15)
python arm_control/dance_player.py Thousand-hand-video/Cropped_thousandhand.mp4 \
    -t Thousand-hand-video/Cropped_thousandhand_trajectory.json --sim --auto-play --smooth 25

# STEP 3: Send to Pi when happy (validates limits, then deploys)
./send_to_pi.sh Thousand-hand-video/Cropped_thousandhand_trajectory.json --play --speed 0.5
# → Sends to: robot@172.20.10.12:~/piper_control/trajectories/
```

## ⚠️ macOS Limitation: Video + Robot Windows

**Problem:** Running `--show-both` crashes due to OpenCV/PyBullet OpenGL conflict on macOS.

**Solution:** Run video and robot in **separate terminals**:
```bash
# Terminal 1: Video with CV detection
python arm_control/dance_player.py Thousand-hand-video/Cropped_thousandhand.mp4 \
    -t Thousand-hand-video/Cropped_thousandhand_trajectory.json --sim --auto-play

# Terminal 2: Robot simulation (convert JSON → CSV first)
python convert_json_to_csv.py Thousand-hand-video/Cropped_thousandhand_trajectory.json
python simulation/run_csv_trajectory.py csv_trajectories/Cropped_thousandhand_trajectory.csv
```
Arrange windows side-by-side to observe the **isomorphism** between human pose and robot movements.

## Test CSV Movements (No Video)

```bash
# 160bpm choreography
python simulation/run_csv_trajectory.py csv_trajectories/160bpm.csv --loop

# Full range test
python simulation/run_csv_trajectory.py csv_trajectories/movement_testing.csv --loop
```

## Pi Setup (One-Time)

```bash
# On Pi (robot@<IP>, password: RAVE)
sudo apt install -y python3-venv can-utils
mkdir -p ~/piper_control && cd ~/piper_control
python3 -m venv venv && source venv/bin/activate
pip install piper-sdk python-can numpy

# Enable CAN bus
sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up

# Test trajectory player is copied (done by send_to_pi.sh)
ls -lh arm_control/trajectory_player.py
```

## Pi Package (What Gets Deployed)

**On macOS (development):**
- Trajectory files: `Thousand-hand-video/*.json` (generated from video)
- Pi player script: `arm_control/trajectory_player.py`
- Deployment script: `send_to_pi.sh`

**On Pi (after deployment):**
```
~/piper_control/
├── arm_control/trajectory_player.py   ← Player script
├── trajectories/*.json                 ← Trajectory files (robot-specific)
└── venv/                               ← Python environment
```

**To deploy:** `./send_to_pi.sh <file>.json` validates safety then copies to Pi

## Files

- `arm_control/dance_player.py` - Main script (preprocess, simulate, test)
- `simulation/run_csv_trajectory.py` - Test CSV choreography patterns
- `send_to_pi.sh` - Validate & deploy trajectories to Pi
- `validate_trajectory.py` - Safety checks (limits, velocities, accelerations)

## Joint Limits

| Joint | Range (rad) | Max Speed (rad/s) |
|-------|-------------|-------------------|
| J1 base | ±2.69 | 3.14 |
| J2 shoulder | 0→3.40 | 3.40 |
| J3 elbow | -3.05→0 | 3.14 |
| J4-J6 wrist | ±1.3-1.9 | 3.93 |
