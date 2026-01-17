# Piper Robot Joint Coordinate Reference

## Understanding Joint 2 (Shoulder) - IMPORTANT!

The shoulder joint (J2) **does NOT use intuitive angles**. Here's the actual behavior:

### J2 Reference Positions
```
J2 = 0.0 rad (0°)      → Arm near lower limit (angled down/back)
J2 = 1.63 rad (93°)    → Arm pointing UP (vertical) ⬆️
J2 = 3.2 rad (183°)    → Arm HORIZONTAL (forward) ➡️
J2 = 3.40 rad (195°)   → Upper limit (slightly past horizontal)
```

### Why Is This Confusing?

The robot's mechanical zero (J2=0) is NOT "arm hanging down." Due to DH parameter theta offsets:
- J2 joint angle ≠ actual arm angle in space
- J2 = 0 corresponds to approximately -172° in DH frame
- This shifts the entire reference frame

## Choreography Quick Reference

### For Dance Moves:

**UP Movement** (arm pointing to sky):
```csv
time,joint1,joint2,joint3,joint4,joint5,joint6,description
0.0,0.0,1.63,-0.1,0.0,0.0,0.0,Arm UP
```

**HORIZONTAL** (arm reaching forward):
```csv
time,joint1,joint2,joint3,joint4,joint5,joint6,description
0.0,0.0,3.2,0.0,0.0,0.0,0.0,Arm HORIZONTAL
```

**LEFT/RIGHT** (base rotation while arm up):
```csv
time,joint1,joint2,joint3,joint4,joint5,joint6,description
0.0,-2.4,1.63,-0.1,0.0,0.0,0.0,Left UP
0.0,+2.4,1.63,-0.1,0.0,0.0,0.0,Right UP
```

**DOWN-ISH** (arm angled toward floor):
```csv
time,joint1,joint2,joint3,joint4,joint5,joint6,description
0.0,0.0,0.5,0.0,0.0,0.0,0.0,Arm DOWN angle
```

## All Joint Reference Positions

| Joint | Neutral | Up/Forward | Down/Back | Left | Right |
|-------|---------|-----------|-----------|------|-------|
| J1 (base) | 0.0 | 0.0 | 0.0 | -2.4 | +2.4 |
| J2 (shoulder) | 3.2 (horiz) | **1.63** (up) | 0.5 (down) | - | - |
| J3 (elbow) | 0.0 | -0.1 | 0.0 | - | - |
| J4 (wrist roll) | 0.0 | 0.0 | 0.0 | - | - |
| J5 (wrist pitch) | 0.0 | 0.0 | 0.0 | - | - |
| J6 (wrist rotate) | 0.0 | 0.0 | 0.0 | - | - |

## Testing Your Choreography

1. Start with J2 = 1.63 rad for UP positions
2. Use J2 = 3.2 rad for horizontal/neutral
3. Combine with J1 for left (-) and right (+) rotation
4. Fine-tune with small J2 adjustments (±0.2 rad)

## Common Mistakes

❌ **WRONG:** `J2 = 3.2` expecting arm to point up
✅ **CORRECT:** `J2 = 1.63` for arm pointing up

❌ **WRONG:** `J2 = 0.0` expecting horizontal
✅ **CORRECT:** `J2 = 3.2` for horizontal

## Why Does This Matter?

- Simulation physics matches real robot kinematics
- DH parameters define the mathematical model
- Joint angles are relative to mechanical reference frames
- Your choreography must use these specific values

## Quick Conversion

If you're thinking in "intuitive" angles:
- "Straight UP" → Use J2 = 1.63 rad (93°)
- "Horizontal FORWARD" → Use J2 = 3.2 rad (183°)  
- "Angled DOWN" → Use J2 = 0.5 rad (29°)

Remember: **J2 is NOT intuitive!** Always reference this guide.
