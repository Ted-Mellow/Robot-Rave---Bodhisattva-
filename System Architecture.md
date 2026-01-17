# System Architecture

## Concept: "Digital Echo" - Human-Robot Mirror Dance

A person dances in front of a camera. Their movements are captured and split into two realities:
- **PIPER (left)**: Physical robot arm mirrors movements in real-time
- **UR12e (right)**: Acts as "shadow" - mirrored or interpretive variations
- **Projection**: Shows CV skeleton, particle effects, and abstract visualizations

The effect: When the person moves their right arm, PIPER mirrors it on the left, UR12e mirrors on the right, and the projection shows the digital skeleton between them - creating a tri-fold mirror effect.

## Hardware

- **AgileX PIPER arm** - Left side mirror
- **Universal Robots UR12e** - Right side shadow
- **XuanPad M8-F Projector** - Visual output
- **Webcam** (Logitech C920 or better, 60fps preferred)
- **Computer** - Ubuntu 20.04, 16GB RAM, dedicated GPU preferred
- **Network switch** - Gigabit ethernet for low latency
- **Emergency stop button** - Critical safety component

## Software Stack

- **OS**: Ubuntu 20.04 LTS
- **Framework**: ROS Noetic
- **Python**: 3.8+
- **Key Libraries**: OpenCV, MediaPipe, urx (UR12e), AgileX SDK, NumPy, Pygame

## System Flow

```
┌─────────────┐
│   Camera    │
│  (Person)   │
└──────┬──────┘
       │
       ↓
┌─────────────────────────┐
│  CV Processing Node     │
│  - MediaPipe Pose       │
│  - Extract keypoints    │
│  - Filter/smooth data   │
└──────┬──────────────────┘
       │
       ↓
┌─────────────────────────┐
│  Coordinate Transform   │
│  - Human → Robot space  │
│  - Mirror logic         │
│  - Safety limits        │
└──────┬─────────┬────────┘
       │         │
       ↓         ↓
  ┌────────┐ ┌─────────┐
  │ PIPER  │ │  UR12e  │
  │  Node  │ │  Node   │
  └────────┘ └─────────┘
       │
       ↓
┌─────────────────────────┐
│  Visualization Node     │
│  - Skeleton rendering   │
│  - Effects              │
└──────┬──────────────────┘
       │
       ↓
┌─────────────┐
│  Projector  │
└─────────────┘
```

## Component Details

### Input Layer
- **Camera**: Captures person's movements at 30-60fps for real-time pose estimation

### Processing Layer
- **CV Processing Node**: 
  - MediaPipe Pose detection (model complexity 1 for speed)
  - Extracts keypoints: shoulder, elbow, wrist (right arm focus)
  - Exponential moving average filter for smoothing
  - Visibility thresholding (>0.5) to ignore occluded poses
  - Target latency: <50ms

- **Coordinate Transform**:
  - Maps normalized camera coordinates (0-1) to robot workspace
  - PIPER: Mirrors X-axis, smaller workspace (~0.3m reach)
  - UR12e: Direct mapping, larger workspace (~1.3m reach)
  - Safety limits: Workspace boundaries, speed caps, joint limits
  - Separate scaling factors per axis for natural movement feel

### Output Layer
- **PIPER Node**: 
  - Controls left-side robot arm
  - Receives mirrored coordinates
  - Cartesian position control with orientation constraints

- **UR12e Node**: 
  - Controls right-side robot arm
  - Receives direct or interpretive coordinates
  - Can operate in mirror mode or shadow/interpretive mode

- **Visualization Node**: 
  - Renders skeleton overlay on black background
  - Particle trail effects following wrist movement
  - Color-coded joints and segments
  - Optional: Beat detection sync, tempo-based effects

- **Projector**: 
  - Displays visualization output
  - Casts on wall behind robot arms
  - Creates immersive tri-fold mirror effect

## Key Behaviors

- **Real-time mirroring**: <100ms end-to-end latency from camera to robot movement
- **Smooth motion**: Exponential smoothing prevents jitter while maintaining responsiveness
- **Safety-first**: Workspace limits, speed caps, and emergency stop always active
- **Dual modes**: Direct mirroring or interpretive variations for UR12e
