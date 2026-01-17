# Piper Robot Arm - Quick Start Guide

## 🚀 Setup

Activate the virtual environment:
```bash
cd "Small arm "
source venv/bin/activate
```

## 🎮 Running Simulations

### CSV Trajectory Runner (Recommended)
Run pre-defined trajectories from CSV files:
```bash
# Basic usage
python run_csv_trajectory.py csv_trajectories/example_wave.csv

# With URDF model
python run_csv_trajectory.py csv_trajectories/example_dance.csv --urdf

# Loop continuously
python run_csv_trajectory.py csv_trajectories/example_wave.csv --loop

# Adjust speed (0.5 = half speed, 2.0 = double speed)
python run_csv_trajectory.py csv_trajectories/example_dance.csv --speed 0.5
```

### Basic Simulation Test
```bash
python piper_pybullet_sim.py
```

### Example Simulations

**Joint Sweep** - Test each joint's range of motion:
```bash
python simulation_examples/sim_joint_sweep.py
```

**Custom Trajectory** - Template for your own motions:
```bash
python simulation_examples/sim_custom_trajectory.py
```

## 🛑 Stopping Simulations

**Kill the program:**
- Press `Ctrl+C` in the terminal
- Close the PyBullet GUI window

The simulation will automatically cleanup and disconnect.

## ✏️ Creating Custom Trajectories

### Method 1: CSV Files (Easiest)
Create a CSV file in `csv_trajectories/` folder:

```csv
time,joint1,joint2,joint3,joint4,joint5,joint6,description
0.0,0.0,0.0,0.0,0.0,0.0,0.0,Start position
1.0,0.5,1.0,-1.0,0.5,0.5,0.5,Target position
2.0,0.0,0.0,0.0,0.0,0.0,0.0,Return home
```

Then run:
```bash
python run_csv_trajectory.py csv_trajectories/your_file.csv
```

### Method 2: Modify Python Template
Edit `simulation_examples/sim_custom_trajectory.py` and customize the waypoints:

```python
my_trajectory = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Joint angles in radians
    [0.5, 1.0, -1.0, 0.5, 0.5, 0.5],
    # Add your waypoints here...
]
```

### Method 3: Create New Python File
```python
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from piper_pybullet_sim import PiperSimulation

sim = PiperSimulation(gui=True)
try:
    sim.reset()
    
    # Your code here
    target = [0.5, 1.0, -1.0, 0.5, 0.5, 1.0]  # radians
    sim.set_joint_positions(target)
    
    for _ in range(240):
        sim.step()
    
    while True:
        sim.step()
except KeyboardInterrupt:
    pass
finally:
    sim.close()
```

Save as `simulation_examples/my_simulation.py` and run:
```bash
python simulation_examples/my_simulation.py
```

## 📊 Joint Limits (Radians)

| Joint | Min | Max |
|-------|-----|-----|
| J1 | -2.62 | 2.62 |
| J2 | 0.00 | 3.14 |
| J3 | -2.97 | 0.00 |
| J4 | -1.75 | 1.75 |
| J5 | -1.22 | 1.22 |
| J6 | -2.09 | 2.09 |

## 🔧 Key API Methods

```python
# Movement
sim.set_joint_positions([j1, j2, j3, j4, j5, j6])  # Set target position
sim.get_joint_positions()                           # Get current positions
sim.get_end_effector_pose()                         # Get end effector pose

# Control
sim.step()                                          # Step simulation forward
sim.reset()                                         # Return to home position
sim.close()                                         # Cleanup and exit
```

## 📁 File Structure

```
Small arm /
├── README.md                    # This file
├── run_csv_trajectory.py        # CSV trajectory runner
├── piper_pybullet_sim.py       # Main simulation class
├── csv_trajectories/            # Place your CSV trajectory files here
│   ├── example_wave.csv
│   ├── example_dance.csv
│   └── README.md
├── robot_models/                # URDF robot models
│   └── piper.urdf
├── simulation_examples/         # Example Python simulations
│   ├── sim_joint_sweep.py
│   └── sim_custom_trajectory.py
├── piper_sdk/                   # Piper SDK (for real robot)
├── src/                         # Source code modules
└── venv/                        # Python virtual environment
```

## 🤖 Real Robot Control

To control the actual robot (not simulation), use the Piper SDK:
```bash
python piper_sdk/piper_sdk/demo/V2/piper_ctrl_joint.py
```

See `AgileX_Piper_Development_Guide.md` for complete documentation.

---

**Quick Reference:**
- ✅ Run: `python simulation_examples/<filename>.py`
- 🛑 Stop: `Ctrl+C`
- ✏️ Edit: Modify waypoints in trajectory files
- 📖 Docs: `AgileX_Piper_Development_Guide.md`
