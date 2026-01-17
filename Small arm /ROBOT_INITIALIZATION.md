# Robot Initialization & Gravity Compensation

## Simulation (PyBullet)

The simulation automatically handles gravity compensation on startup:

1. **Immediate Joint Control**: When the robot URDF loads, joint motors are immediately set to hold home position (all zeros)
2. **Stabilization Period**: Physics runs for 250+ steps before control to ensure the robot is fully stable
3. **No Visible Flopping**: The robot appears already held in position when the GUI opens

**Implementation** (in `piper_simultion_corrected.py`):
- Joints are set to position control immediately after loading
- High force (500N) and damping (gain 0.3, velocity 0.1) prevent gravity drop
- Multiple stabilization steps before rendering

## Real Robot Hardware

The real Piper robot has built-in gravity compensation once enabled:

1. **Enable Robot**: Call `piper.EnablePiper()` and wait for confirmation
2. **Motors Engage**: Motors automatically hold current position
3. **Safe to Move**: Robot won't flop once enabled

**Best Practice** (from SDK examples):
```python
from piper_sdk import *

# Initialize
piper = C_PiperInterface_V2("can0")
piper.ConnectPort()

# IMPORTANT: Enable and wait for motors to engage
while not piper.EnablePiper():
    time.sleep(0.01)

# Now safe to send commands - robot won't flop
piper.JointCtrl(0, 0, 0, 0, 0, 0)  # Home position
```

## Key Differences

| Aspect | Simulation | Real Robot |
|--------|-----------|------------|
| Gravity Compensation | Manual (PID control) | Automatic (motor holding) |
| Startup State | All zeros (stabilized) | Current position (held) |
| Enable Required | No | Yes (EnablePiper) |
| Flopping Risk | Fixed in code | Fixed by hardware |

## Troubleshooting

**Simulation flopping?**
- Ensure using `piper_simultion_corrected.py` (not old versions)
- Check that `max_force` is ≥ 500N in `set_joint_positions()`
- Verify stabilization steps run after URDF loading

**Real robot flopping?**
- Always call `EnablePiper()` before sending commands
- Wait for enable confirmation (while loop)
- Check CAN connection is stable
- Verify motor power is on

## Safety Notes

⚠️ **Real Robot**: Never send commands before `EnablePiper()` returns True
⚠️ **Simulation**: Always use the corrected simulation class with stabilization
⚠️ **Both**: Start movements from known safe positions (like home)
