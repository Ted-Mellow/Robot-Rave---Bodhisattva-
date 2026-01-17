# Piper Robot Arm - Quick Start

## 🚀 Setup
```bash
cd "Small arm "
source venv/bin/activate
```

## 🎮 Run Simulations

**CSV Trajectories (Recommended):**
```bash
python run_csv_trajectory.py csv_trajectories/example_wave.csv
python run_csv_trajectory.py csv_trajectories/example_dance.csv --urdf
python run_csv_trajectory.py csv_trajectories/example_wave.csv --loop --speed 0.5
```

**Python Simulations:**
```bash
python simulation/sim_custom_trajectory.py  # Loads from CSV
python piper_pybullet_sim.py                # Basic demo
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
├── run_csv_trajectory.py      # CSV runner
├── csv_trajectories/           # Your CSV files here
├── robot_models/piper.urdf     # Robot model
├── simulation/                 # Python examples
└── piper_sdk/                  # Real robot SDK
```

## 🤖 Real Robot
```bash
python piper_sdk/piper_sdk/demo/V2/piper_ctrl_joint.py
```
