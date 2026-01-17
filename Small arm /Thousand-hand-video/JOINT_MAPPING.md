# Robot Joint Axis Mapping - Visual Guide

## Overview

The live comparison mode shows **green dots** overlaid on the human body, representing where each of the 6 robot joint axes map to human anatomy. This makes it crystal clear how human motion translates to robot motion.

## Visual Representation

```
        J6 ●  (Wrist top - End effector rotation)
           |
        J5 ●  (Wrist - Wrist pitch)
           |
        J4 ●  (Wrist - Wrist roll)
           |
           |  Green connecting line
           |
        J3 ●  (Elbow - Elbow bend)
           |
           |  Green connecting line
           |
        J2 ●  (Shoulder - Shoulder elevation)
           |
           |  Green connecting line
           |
        J1 ●  (Hip - Base rotation)
```

## Joint Mapping Details

### J1 - Base Rotation (Hip)
- **Location:** Hip center
- **Human Motion:** Left-right body rotation / arm swing
- **Robot Motion:** Base rotates left-right
- **Range:** ±154° (-2.69 to 2.69 rad)
- **Visual:** Green dot at hip with "J1" label

### J2 - Shoulder Elevation (Shoulder)
- **Location:** Shoulder joint
- **Human Motion:** Arm raises up or down
- **Robot Motion:** Shoulder lifts arm vertically
- **Range:** 0° to 195° (0 to 3.40 rad)
- **Visual:** Green dot at shoulder with "J2" label

### J3 - Elbow Bend (Elbow)
- **Location:** Elbow joint
- **Human Motion:** Elbow bends/extends
- **Robot Motion:** Elbow joint bends
- **Range:** -175° to 0° (-3.05 to 0 rad)
- **Visual:** Green dot at elbow with "J3" label

### J4 - Wrist Roll (Wrist)
- **Location:** Wrist
- **Human Motion:** Hand rotation (palm up/down)
- **Robot Motion:** Wrist rotates
- **Range:** ±106° (-1.85 to 1.85 rad)
- **Visual:** Green dot at wrist with "J4" label
- **Note:** Currently set to 0° (neutral)

### J5 - Wrist Pitch (Wrist)
- **Location:** Slightly above wrist (15px offset)
- **Human Motion:** Hand tilts up/down
- **Robot Motion:** Wrist pitches
- **Range:** ±75° (-1.31 to 1.31 rad)
- **Visual:** Green dot above wrist with "J5" label
- **Note:** Currently set to 0° (neutral)

### J6 - Wrist Rotate (Wrist)
- **Location:** Further above wrist (30px offset)
- **Human Motion:** Hand twists
- **Robot Motion:** End effector rotates
- **Range:** ±100° (-1.75 to 1.75 rad)
- **Visual:** Green dot at top with "J6" label
- **Note:** Currently set to 0° (neutral)

## Display Elements

### Green Dots
- **Size:** 8px radius filled circle
- **Color:** Bright green (0, 255, 0)
- **Border:** Darker green ring (0, 200, 0)
- **Purpose:** Show exact position of robot joint axis on human body

### Labels
- **Format:** "J1", "J2", "J3", etc.
- **Position:** To the right of each green dot
- **Background:** Black rectangle for readability
- **Font:** Green text, bold

### Angle Values
- **Format:** Numerical value in radians (e.g., "1.23")
- **Position:** Below each joint label
- **Purpose:** Show real-time joint angle
- **Updates:** Every frame (30+ FPS)

### Connection Lines
- **Color:** Green (0, 200, 0)
- **Connects:** J1 → J2 → J3
- **Purpose:** Show kinematic chain of main arm joints
- **Style:** Solid line, 2px width, anti-aliased

### Legend (Bottom-Left)
Shows quick reference:
```
Robot Joint Axes:
● J1: Base rotation
● J2: Shoulder elev.
● J3: Elbow bend
● J4-6: Wrist
```

## What the Colors Mean

- **Green Dots & Lines** - Robot joint positions and connections
- **White/Red/Yellow Skeleton** - MediaPipe pose detection landmarks
- **Cyan Text** - Information displays (statistics, labels)
- **Black Backgrounds** - Text boxes for readability

## Usage Tips

### Verify Joint Mapping
1. Raise your arm slowly
2. Watch J2 (shoulder) green dot - angle should increase
3. Watch robot in PyBullet - shoulder should lift

### Check Elbow Tracking
1. Bend your elbow
2. Watch J3 (elbow) green dot - angle should become more negative
3. Watch robot - elbow should bend

### Understand Base Rotation
1. Swing arm left-right across body
2. Watch J1 (hip) green dot - angle changes
3. Watch robot - base rotates

### Wrist Joints (J4-J6)
- Currently stacked at wrist location
- All show 0.00 (not actively controlled yet)
- Future: Could map to hand orientation

## Keyboard Shortcuts During Viewing

- **SPACE** - Pause to examine joint positions
- **S** - Save screenshot with joint overlays
- **+/-** - Adjust speed to see joints move slower/faster
- **R** - Reset to beginning

## Technical Details

### Coordinate System
- **Input:** MediaPipe normalized coordinates [0, 1]
- **Display:** Pixel coordinates (px, py)
- **Robot:** Radians for all joint angles

### Update Rate
- **Pose Detection:** 24-30 FPS (video framerate)
- **Joint Display:** Every frame
- **Smoothing:** 3-frame moving average
- **Latency:** ~50-100ms (1-3 frames)

### Accuracy
- **Joint Position:** Within 5-10 pixels of actual landmark
- **Angle Display:** Rounded to 2 decimal places
- **Kinematic Chain:** Direct mapping from pose landmarks

## Common Observations

### J1 (Hip/Base)
- Moves when arm swings left or right
- Stays ~0 when arm moves straight up
- Can range widely: -2.6 to +2.6 rad

### J2 (Shoulder)
- **Most active joint** during arm raises
- Ranges from 0.3 (arm down) to 2.8 (arm up)
- Very responsive to vertical arm motion

### J3 (Elbow)
- Negative values = bent elbow
- 0 = straight arm
- Typically ranges -0.5 to -2.5 rad

### J4-J6 (Wrist)
- Currently show 0.00 (neutral)
- Visually stacked at wrist area
- Could be expanded to track hand orientation

## Example Scenarios

### Scenario 1: Arm Raise
```
Start (Arm down):
J1: 0.00  J2: 0.50  J3: -0.30

Middle (Arm halfway):
J1: 0.00  J2: 1.50  J3: -0.50

End (Arm up):
J1: 0.00  J2: 2.50  J3: -0.20
```

### Scenario 2: Arm Swing
```
Center:
J1: 0.00  J2: 1.20  J3: -0.50

Left:
J1: -2.00  J2: 1.30  J3: -0.40

Right:
J1: +2.00  J2: 1.30  J3: -0.40
```

### Scenario 3: Elbow Bend
```
Straight arm:
J1: 0.00  J2: 1.50  J3: -0.10

Bent elbow:
J1: 0.00  J2: 1.50  J3: -2.50

Fully bent:
J1: 0.00  J2: 1.50  J3: -3.00
```

## Troubleshooting

### Green dots not appearing
- Check that pose is detected (white skeleton visible)
- Verify arm is in frame and clearly visible
- Try greyscale mode for better detection

### Dots in wrong position
- This is expected - they map to human body landmarks
- J1 at hip, J2 at shoulder, J3 at elbow, J4-6 at wrist
- Not an error, just showing the mapping!

### Angle values not changing
- Verify robot simulation is running
- Check that motion mapper is active
- Try moving arm more dramatically

### Lines not connecting properly
- Normal if only part of body is visible
- Requires J1, J2, and J3 all detected
- Move so full arm is in frame

## Summary

The green dot overlay system provides:

✅ **Visual Feedback** - See exactly where robot joints map to body
✅ **Real-time Angles** - Watch joint values update live
✅ **Kinematic Chain** - Understand how joints connect
✅ **Debugging Aid** - Spot mapping issues immediately
✅ **Learning Tool** - Understand robot kinematics

**To see it in action:**
```bash
python3 run_pipeline.py --live --end 10
```

The green dots will appear on the detected pose, showing you exactly how the 6-axis robot arm maps to human motion! 🤖✨
