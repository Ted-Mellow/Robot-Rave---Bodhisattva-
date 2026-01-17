# Piper Robot Arm - Quick Start

## 🚀 Setup
```bash
cd "Small arm "
source venv/bin/activate
```

## 🎮 Run Simulations

**Single command to run any CSV choreography:**
```bash
python simulation/run_csv_trajectory.py csv_trajectories/YOUR_FILE.csv
python simulation/run_csv_trajectory.py csv_trajectories/example_wave.csv --loop --speed 0.5

```

**Examples:**
```bash
# Run a single trajectory
python simulation/run_csv_trajectory.py csv_trajectories/example_wave.csv

# Loop continuously
python simulation/run_csv_trajectory.py csv_trajectories/armwave.csv --loop

# Slow motion (0.5x speed)
python simulation/run_csv_trajectory.py csv_trajectories/movement_testing.csv --speed 0.5

# Combined: Loop at half speed
python simulation/run_csv_trajectory.py csv_trajectories/example_dance.csv --loop --speed 0.5
```

**Features:**
- ✅ Full Piper URDF model with gripper (loaded automatically)
- ✅ No initial flopping - gravity compensation on startup  
- ✅ Graceful exit on GUI window close or Ctrl+C
- ✅ Real-time physics with PyBullet
- ✅ Joint angle validation and clamping

**Stop Simulation:**
- Press `Ctrl+C` in terminal, OR
- Close the PyBullet GUI window
- Both methods exit cleanly without errors

## ✏️ Create Your Own Choreography

**CSV Format** (place in `csv_trajectories/` folder):
```csv
time,joint1,joint2,joint3,joint4,joint5,joint6,description
0.0,0.0,0.0,0.0,0.0,0.0,0.0,Home position
1.0,0.5,1.0,-1.0,0.5,0.5,0.5,Dance move
2.0,0.0,0.0,0.0,0.0,0.0,0.0,Return home
```

**⚠️ IMPORTANT:** Joint angles must be in **radians** (not degrees!)

**Joint Limits (from physical hardware specs):**
| Joint | Position Range (rad) | Position Range (°) | Max Speed (rad/s) | Max Speed (°/s) |
|-------|---------------------|-------------------|-------------------|-----------------|
| J1 (base) | ±2.688 | ±154° | 3.142 | 180°/s |
| J2 (shoulder) | 0 → 3.403 | 0° → 195° | 3.403 | 195°/s |
| J3 (elbow) | -3.054 → 0 | -175° → 0° | 3.142 | 180°/s |
| J4 (wrist roll) | ±1.850 | ±106° | 3.927 | 225°/s |
| J5 (wrist pitch) | ±1.309 | ±75° | 3.927 | 225°/s |
| J6 (wrist rotate) | ±1.745 | ±100° | 3.927 | 225°/s |

## 📁 Structure
```
Small arm/
├── simulation/
│   ├── piper_simultion_corrected.py  # Main simulation class
│   └── run_csv_trajectory.py         # CSV runner (use this!)
├── csv_trajectories/                 # Put your CSV choreography files here
│   ├── example_wave.csv
│   ├── example_dance.csv
│   ├── armwave.csv
│   └── movement_testing.csv
├── robot_models/
│   └── piper.urdf                    # 6-axis robot model (with gripper)
├── piper_sdk/                        # Real robot SDK (for hardware)
└── ROBOT_INITIALIZATION.md           # Gravity compensation documentation
```

## 🤖 Real Robot

**IMPORTANT:** Real robot requires motor enable before commands:
```python
from piper_sdk import *

piper = C_PiperInterface_V2("can0")
piper.ConnectPort()

# Wait for motors to engage (prevents flopping)
while not piper.EnablePiper():
    time.sleep(0.01)

# Now safe to send commands
piper.JointCtrl(0, 0, 0, 0, 0, 0)
```

See `ROBOT_INITIALIZATION.md` for full documentation on gravity compensation.

## 🔧 Troubleshooting

**Simulation flops on startup?**
- Should be fixed in latest version (gravity compensation active)
- If still happening, check you're using `simulation/piper_simultion_corrected.py`

**"Not connected to physics server" errors?**
- Fixed in latest version
- Close and restart simulation if persists

**Robot moves incorrectly?**
- Check CSV angles are in **radians** not degrees
- Verify angles are within joint limits (see above)
- Use `--speed 0.5` to slow down and observe

**Need help?**
- See `ROBOT_INITIALIZATION.md` for gravity compensation details
- Check `csv_trajectories/README.md` for CSV format details
