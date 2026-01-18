# Piper Robot - Video to Dance Movement

**Workflow:** Video preprocessing → Simulate & test → Send to Pi → Play on robot

## 🎯 Complete Workflow

```bash
cd "Small arm " && source venv/bin/activate

# STEP 1: Extract trajectory from video (~60 sec)
python arm_control/dance_player.py Thousand-hand-video/Cropped_thousandhand.mp4 --preprocess

# STEP 2: Test in simulation (opens GUI, video plays alongside)
python arm_control/dance_player.py Thousand-hand-video/Cropped_thousandhand.mp4 \
    -t Thousand-hand-video/Cropped_thousandhand_trajectory.json --sim --auto-play

# STEP 3: Send to Pi when happy (validates first)
./send_to_pi.sh Thousand-hand-video/Cropped_thousandhand_trajectory.json --play --speed 0.5
```

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

## Files

- `arm_control/dance_player.py` - Main script (preprocess video, simulate, control)
- `simulation/run_csv_trajectory.py` - Test CSV patterns
- `send_to_pi.sh` - Validate & deploy to Pi
- `validate_trajectory.py` - Safety checks (limits, velocities)

## Joint Limits

| Joint | Range (rad) | Max Speed (rad/s) |
|-------|-------------|-------------------|
| J1 base | ±2.69 | 3.14 |
| J2 shoulder | 0→3.40 | 3.40 |
| J3 elbow | -3.05→0 | 3.14 |
| J4-J6 wrist | ±1.3-1.9 | 3.93 |
