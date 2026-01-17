# Piper Robot Arm - Quick Start

## 🚀 Setup
```bash
cd "Small arm "
source venv/bin/activate
```

## 🎮 Run Simulations

**CSV Trajectories (Recommended):**
```bash
python simulation/run_csv_trajectory.py csv_trajectories/example_wave.csv
python simulation/run_csv_trajectory.py csv_trajectories/example_dance.csv --urdf
python simulation/run_csv_trajectory.py csv_trajectories/example_wave.csv --loop --speed 0.5
```

**Python Simulations:**
```bash
python simulation/sim_custom_trajectory.py      # Interactive CSV selector
python simulation/piper_simultion_corrected.py  # Full demo with gripper
```

**Stop:** `Ctrl+C` or close GUI window

## ✏️ Create Trajectories

**CSV Format** (in `csv_trajectories/`):
```csv
time,joint1,joint2,joint3,joint4,joint5,joint6,description
0.0,0.0,0.0,0.0,0.0,0.0,0.0,Home
1.0,0.5,1.0,-1.0,0.5,0.5,0.5,Move
```

**Joint Limits (radians):** J1: ±2.62, J2: 0→3.14, J3: -2.97→0, J4: ±1.75, J5: ±1.22, J6: ±2.09

## 📁 Structure
```
Small arm/
├── piper_pybullet_sim.py       # Compatibility wrapper
├── simulation/
│   ├── run_csv_trajectory.py   # CSV runner
│   ├── sim_custom_trajectory.py
│   └── piper_simultion_corrected.py
├── csv_trajectories/           # Your CSV files here
├── robot_models/
│   └── piper.urdf              # Robot model (with gripper)
└── piper_sdk/                  # Real robot SDK
```

## 🤖 Real Robot
```bash
python piper_sdk/piper_sdk/demo/V2/piper_ctrl_joint.py
```
