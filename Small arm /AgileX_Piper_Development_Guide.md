# AgileX Piper Robot Arm - Comprehensive Development Guide

## Table of Contents
1. [Technical Specifications](#technical-specifications)
2. [Programming Languages & Technology Stacks](#programming-languages--technology-stacks)
3. [Movement Speed Control](#movement-speed-control)
4. [Computer Vision Integration](#computer-vision-integration)
5. [Development in Cursor IDE](#development-in-cursor-ide)
6. [Getting Started](#getting-started)
7. [Advanced Applications](#advanced-applications)

---

## Technical Specifications

### Physical Characteristics
- **Weight**: 4.2 kg (lightweight design)
- **Payload Capacity**: 1.5 kg stable payload
- **Degrees of Freedom**: 6-axis (6-DOF)
- **Working Radius**: 626-627 mm
- **Repeatability**: ±0.1 mm (high precision)
- **Construction**: Aluminum alloy and polymer/resin materials
- **Operating Temperature**: -20°C to 50°C
- **Ingress Protection**: IP22 (standard model)

### Joint Specifications
- **J1 (Base)**: ±154° range
- **J2**: 0°-195° range
- **J3-J6**: Variable ranges with specific constraints
- **Maximum Joint Speeds**: Up to ~225°/s on several joints (varies by joint)
- **6 Integrated Joint Motors**: High-precision servos with advanced path planning

### Electrical Interface
- **Communication Protocol**: CAN bus (CAN-H and CAN-L terminals)
- **Baud Rate**: 1,000,000 (fixed)
- **Connection**: CAN-to-USB adapter required for PC connection
- **Power Supply**: External 12V power supply
- **Base Mounting**: Four M5 threaded holes, 70mm hole spacing

### End Effector
- **Flange**: Standard mounting for various tools
- **Included Gripper**: Two-finger gripper (optional accessory)
- **Teach Pendant**: Available as optional accessory

---

## Programming Languages & Technology Stacks

### 1. Python SDK (Primary Development)

The Piper SDK is the primary method for controlling the robot arm and is available on GitHub.

**Installation:**
```bash
git clone https://github.com/agilexrobotics/piper_sdk.git
cd piper_sdk
pip install -r requirements.txt --break-system-packages
```

**Basic Python Control Example:**
```python
import time
from piper_sdk import *

if __name__ == "__main__":
    # Initialize robot arm with CAN interface
    piper = Piper("can0")
    interface = piper.init()
    piper.connect()
    time.sleep(0.1)
    
    # Get current joint states
    joint_states = piper.get_joint_states()[0]
    print(f"Current joint angles: {joint_states}")
    
    # Control gripper (if attached)
    gripper_state = piper.get_gripper_states()[0][0]
    print(f"Gripper opening: {gripper_state}")
    
    # Switch to slave mode for control
    piper.MasterSlaveConfig(0xFC, 0, 0, 0)
```

**Key Python SDK Features:**
- Direct CAN bus communication
- Joint position control
- Gripper control
- Teaching mode integration
- Position recording and playback
- Inverse kinematics support
- Real-time feedback reading

### 2. ROS1 (Robot Operating System)

Full ROS1 support with pre-loaded packages.

**Installation:**
```bash
git clone https://github.com/agilexrobotics/piper_ros.git
cd piper_ros
catkin_make
source devel/setup.bash
```

**Launch Piper in ROS1:**
```bash
roslaunch piper_ros piper.launch
```

**ROS1 Capabilities:**
- MoveIt! motion planning integration
- RViz visualization
- TF transformations
- Joint state publishers
- Trajectory control
- Sensor integration via ROS topics

### 3. ROS2 (Latest ROS Version)

Native ROS2 support for modern robotics development.

**Key Features:**
- Real-time control with DDS middleware
- Better performance than ROS1
- Modern C++ and Python APIs
- Enhanced security features
- Lifecycle node management

**Basic ROS2 Launch:**
```bash
ros2 launch piper_ros2 piper.launch.py
```

### 4. C++ Development

Available through the UGV_SDK for low-level control.

**Advantages:**
- Higher performance
- Lower latency
- Direct hardware access
- Real-time capabilities

**Use Cases:**
- High-frequency control loops
- Embedded systems integration
- Performance-critical applications

### 5. Technology Stack for Cursor IDE Development

**Recommended Setup for Cursor:**

```python
# Project Structure for Cursor
project/
├── src/
│   ├── piper_control.py      # Main control logic
│   ├── vision_module.py      # Computer vision
│   ├── trajectory_planner.py # Motion planning
│   └── utils.py              # Helper functions
├── config/
│   ├── robot_params.yaml     # Robot configuration
│   └── vision_config.yaml    # Camera settings
├── tests/
│   └── test_movements.py
└── requirements.txt
```

**Essential Libraries:**
```txt
piper-sdk
opencv-python
opencv-contrib-python
numpy
scipy
pyrealsense2  # For Intel RealSense cameras
Open3D        # For point cloud processing
matplotlib    # For visualization
pandas        # For data handling
```

---

## Movement Speed Control

### 1. Joint Speed Configuration

The Piper robot allows control over joint movement speeds for different applications.

**Speed Control Methods:**

#### Method 1: Global Speed Rate Control
```python
from piper_sdk import Piper

piper = Piper("can0")
interface = piper.init()
piper.connect()

# Set movement speed rate (0-100%)
# 100 = maximum speed, 50 = half speed
move_speed_rate = 75  # 75% of maximum speed

# Apply to trajectory execution
interface.SetMoveSpeedRate(move_speed_rate)
```

#### Method 2: Individual Joint Speed Control
```python
# Control individual joint speeds
joint_speeds = [50, 60, 70, 80, 90, 100]  # Speed for each joint

for i, speed in enumerate(joint_speeds):
    interface.SetJointSpeed(joint_id=i, speed=speed)
```

#### Method 3: Speed Control in Position Recording/Playback
```python
#!/usr/bin/env python3
import time, csv
from piper_sdk import Piper

# Configuration
move_spd_rate_ctrl = 100  # Global speed control (0-100)
play_interval = 0.1       # Delay between waypoints (seconds)

piper = Piper("can0")
interface = piper.init()
piper.connect()

# Read recorded positions
with open('trajectory.csv', 'r') as f:
    waypoints = list(csv.reader(f))
    waypoints = [[float(j) for j in i] for i in waypoints]

# Execute trajectory with speed control
for position in waypoints:
    interface.MoveToPosition(position, speed_rate=move_spd_rate_ctrl)
    time.sleep(play_interval)
```

### 2. Trajectory Speed Profiles

**Smooth Acceleration/Deceleration:**
```python
def smooth_trajectory(start_pos, end_pos, duration, frequency=100):
    """
    Generate smooth trajectory with S-curve velocity profile
    
    Args:
        start_pos: Starting joint angles (list of 6 values)
        end_pos: Target joint angles (list of 6 values)
        duration: Movement duration in seconds
        frequency: Control frequency in Hz
    """
    import numpy as np
    
    num_points = int(duration * frequency)
    t = np.linspace(0, 1, num_points)
    
    # S-curve velocity profile (smoother than linear)
    s = 3*t**2 - 2*t**3
    
    trajectory = []
    for i in range(len(start_pos)):
        joint_traj = start_pos[i] + (end_pos[i] - start_pos[i]) * s
        trajectory.append(joint_traj)
    
    return np.array(trajectory).T

# Execute smooth trajectory
trajectory = smooth_trajectory(current_pos, target_pos, duration=2.0)

for waypoint in trajectory:
    interface.MoveToJointAngles(waypoint.tolist())
    time.sleep(1/100)  # 100 Hz control
```

### 3. Speed Modification for Different Tasks

**Fast Pick-and-Place:**
```python
# High-speed configuration for simple tasks
fast_config = {
    'move_speed_rate': 100,
    'acceleration': 'high',
    'trajectory_type': 'joint_space'  # Faster than Cartesian
}
```

**Precise Assembly:**
```python
# Slow, precise configuration for delicate tasks
precise_config = {
    'move_speed_rate': 25,
    'acceleration': 'low',
    'trajectory_type': 'cartesian',  # Straight-line motion
    'force_limit': 5  # Newton force threshold
}
```

**General Purpose:**
```python
# Balanced configuration
balanced_config = {
    'move_speed_rate': 60,
    'acceleration': 'medium',
    'trajectory_type': 'blended'  # Mix of joint and Cartesian
}
```

### 4. Real-Time Speed Adjustment

```python
class AdaptiveSpeedController:
    def __init__(self, piper_interface):
        self.interface = piper_interface
        self.current_speed = 50
        
    def adjust_speed_by_distance(self, distance_to_target):
        """
        Automatically adjust speed based on proximity to target
        """
        if distance_to_target > 0.1:  # Far from target
            self.current_speed = 100
        elif distance_to_target > 0.05:  # Medium distance
            self.current_speed = 60
        else:  # Close to target
            self.current_speed = 30
            
        self.interface.SetMoveSpeedRate(self.current_speed)
        
    def adjust_speed_by_load(self, payload_weight):
        """
        Adjust speed based on payload weight
        """
        max_payload = 1.5  # kg
        safety_factor = 0.8
        
        load_ratio = payload_weight / (max_payload * safety_factor)
        self.current_speed = int(100 * (1 - 0.5 * load_ratio))
        
        self.interface.SetMoveSpeedRate(self.current_speed)
```

---

## Computer Vision Integration

### 1. OpenCV Integration

The Piper robot arm integrates seamlessly with OpenCV for vision-guided manipulation.

**Basic Vision Setup:**
```python
import cv2
import numpy as np
from piper_sdk import Piper

class VisionGuidedPiper:
    def __init__(self, can_port="can0", camera_id=0):
        # Initialize robot
        self.piper = Piper(can_port)
        self.interface = self.piper.init()
        self.piper.connect()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Calibration matrix (to be calibrated)
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def detect_colored_object(self, frame, color_range):
        """
        Detect objects by color using HSV color space
        
        Args:
            frame: Input BGR image
            color_range: Tuple of (lower_hsv, upper_hsv)
        
        Returns:
            center_x, center_y, contour
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_bound, upper_bound = color_range
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy, largest_contour
        
        return None, None, None
    
    def pixel_to_robot_coordinates(self, pixel_x, pixel_y, depth):
        """
        Convert pixel coordinates to robot base frame coordinates
        
        This requires camera-robot calibration (homography or hand-eye)
        """
        # Simplified example - replace with actual calibration
        # Using homography transformation
        pixel_point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
        
        if self.homography_matrix is not None:
            robot_point = cv2.perspectiveTransform(
                pixel_point.reshape(-1, 1, 2), 
                self.homography_matrix
            )
            
            x = robot_point[0][0][0] / 1000  # Convert to meters
            y = robot_point[0][0][1] / 1000
            z = depth
            
            return x, y, z
        
        return None, None, None

# Example usage: Red object detection and tracking
vision_piper = VisionGuidedPiper()

# Define red color range in HSV
red_lower = np.array([0, 120, 70])
red_upper = np.array([10, 255, 255])
color_range = (red_lower, red_upper)

while True:
    ret, frame = vision_piper.cap.read()
    if not ret:
        break
    
    # Detect red object
    cx, cy, contour = vision_piper.detect_colored_object(frame, color_range)
    
    if cx is not None:
        # Draw detection
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        # Convert to robot coordinates and move
        x, y, z = vision_piper.pixel_to_robot_coordinates(cx, cy, depth=0.3)
        if x is not None:
            target_position = [x, y, z, 0, 0, 0]  # x, y, z, roll, pitch, yaw
            vision_piper.piper.MoveToCartesianPosition(target_position)
    
    cv2.imshow('Vision Guided Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vision_piper.cap.release()
cv2.destroyAllWindows()
```

### 2. Depth Camera Integration (Intel RealSense / Orbbec)

**Setup for Depth-based Pick and Place:**
```python
import pyrealsense2 as rs
import numpy as np
import cv2
from piper_sdk import Piper

class DepthVisionPiper:
    def __init__(self, can_port="can0"):
        # Initialize Piper
        self.piper = Piper(can_port)
        self.interface = self.piper.init()
        self.piper.connect()
        
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        self.profile = self.pipeline.start(config)
        
        # Get camera intrinsics
        depth_stream = self.profile.get_stream(rs.stream.depth)
        self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
        
    def get_3d_coordinates(self, pixel_x, pixel_y, depth_frame):
        """
        Convert 2D pixel + depth to 3D coordinates in camera frame
        """
        depth = depth_frame.get_distance(pixel_x, pixel_y)
        
        if depth > 0:
            # Deproject pixel to 3D point
            point_3d = rs.rs2_deproject_pixel_to_point(
                self.intrinsics, [pixel_x, pixel_y], depth
            )
            return point_3d  # Returns [x, y, z] in meters
        
        return None
    
    def camera_to_robot_transform(self, camera_coords):
        """
        Transform coordinates from camera frame to robot base frame
        
        This requires hand-eye calibration
        Replace with your calibrated transformation matrix
        """
        # Example transformation (replace with calibrated values)
        # Assuming camera mounted 0.5m above robot base, looking down
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        translation = np.array([0.0, 0.0, 0.5])  # meters
        
        camera_point = np.array(camera_coords)
        robot_point = rotation_matrix @ camera_point + translation
        
        return robot_point.tolist()
    
    def pick_object_at_position(self, x, y, z):
        """
        Move robot to pick up object at given coordinates
        """
        # Approach position (above object)
        approach_height = 0.1  # 10cm above object
        approach_pos = [x, y, z + approach_height, 0, 0, 0]
        
        # Move to approach position
        self.piper.MoveToCartesianPosition(approach_pos, speed_rate=60)
        time.sleep(1)
        
        # Open gripper
        self.piper.SetGripperPosition(100)  # Fully open
        time.sleep(0.5)
        
        # Move down to grasp position
        grasp_pos = [x, y, z, 0, 0, 0]
        self.piper.MoveToCartesianPosition(grasp_pos, speed_rate=30)
        time.sleep(1)
        
        # Close gripper
        self.piper.SetGripperPosition(0)  # Fully closed
        time.sleep(0.5)
        
        # Lift object
        lift_pos = [x, y, z + 0.15, 0, 0, 0]
        self.piper.MoveToCartesianPosition(lift_pos, speed_rate=40)
        time.sleep(1)

# Example: Pick up colored blocks using depth camera
depth_piper = DepthVisionPiper()

while True:
    # Get frames
    frames = depth_piper.pipeline.wait_for_frames()
    aligned_frames = depth_piper.align.process(frames)
    
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    
    if not color_frame or not depth_frame:
        continue
    
    # Convert to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    # Detect object (example: green block)
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Get 3D coordinates
            camera_coords = depth_piper.get_3d_coordinates(cx, cy, depth_frame)
            
            if camera_coords:
                # Transform to robot frame
                robot_coords = depth_piper.camera_to_robot_transform(camera_coords)
                
                # Display coordinates
                cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(color_image, f"Robot: {robot_coords}", 
                           (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                # Pick up object (uncomment to enable)
                # depth_piper.pick_object_at_position(*robot_coords)
                # break
    
    cv2.imshow('Depth Vision', color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

depth_piper.pipeline.stop()
cv2.destroyAllWindows()
```

### 3. Advanced Vision Applications

#### Object Tracking and Following
```python
class ObjectTracker:
    def __init__(self, piper_interface):
        self.piper = piper_interface
        self.tracker = cv2.TrackerKCF_create()  # Or TrackerCSRT
        self.tracking = False
        
    def initialize_tracker(self, frame, bbox):
        """
        Initialize tracker with bounding box
        bbox format: (x, y, width, height)
        """
        self.tracker.init(frame, bbox)
        self.tracking = True
        
    def track_and_follow(self, frame):
        """
        Track object and move robot to follow
        """
        if not self.tracking:
            return None
        
        success, bbox = self.tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in bbox]
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Visual feedback
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Move robot to follow (simplified)
            # Convert pixel coordinates to robot commands
            error_x = center_x - frame.shape[1] // 2
            error_y = center_y - frame.shape[0] // 2
            
            # Proportional control
            kp = 0.001
            delta_x = kp * error_x
            delta_y = kp * error_y
            
            # Send movement command
            current_pos = self.piper.get_cartesian_position()
            new_pos = current_pos.copy()
            new_pos[0] += delta_x  # X adjustment
            new_pos[1] += delta_y  # Y adjustment
            
            self.piper.MoveToCartesianPosition(new_pos, speed_rate=50)
            
            return (center_x, center_y)
        
        return None
```

#### ArUco Marker Detection for Precision Positioning
```python
import cv2
import cv2.aruco as aruco

class ArUcoGuidedPiper:
    def __init__(self, piper_interface, marker_size=0.05):
        """
        marker_size: Physical size of ArUco marker in meters
        """
        self.piper = piper_interface
        self.marker_size = marker_size
        
        # ArUco dictionary
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters()
        
    def detect_markers(self, frame, camera_matrix, dist_coeffs):
        """
        Detect ArUco markers and estimate their pose
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, rejected = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )
        
        if ids is not None:
            # Estimate pose of each marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, camera_matrix, dist_coeffs
            )
            
            results = []
            for i in range(len(ids)):
                # Draw marker and axis
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, 
                                 rvecs[i], tvecs[i], 0.03)
                
                results.append({
                    'id': ids[i][0],
                    'rvec': rvecs[i][0],
                    'tvec': tvecs[i][0]
                })
            
            return results
        
        return None
    
    def move_to_marker(self, marker_data):
        """
        Move robot to align with detected marker
        """
        # Transform marker pose to robot base frame
        # (requires camera-robot calibration)
        tvec = marker_data['tvec']
        rvec = marker_data['rvec']
        
        # Convert to robot coordinates
        robot_position = self.camera_to_robot_transform(tvec, rvec)
        
        # Move to position
        self.piper.MoveToCartesianPosition(robot_position, speed_rate=60)
```

### 4. Machine Learning Integration

```python
import torch
import torchvision

class MLVisionPiper:
    def __init__(self, piper_interface):
        self.piper = piper_interface
        
        # Load pre-trained object detection model (YOLO, Faster R-CNN, etc.)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )
        self.model.eval()
        
        # COCO class names
        self.COCO_CLASSES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle',
            'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            # ... add all 80 COCO classes
            'cup', 'bottle', 'bowl', 'banana', 'apple', 'sandwich'
        ]
        
    def detect_objects(self, frame, confidence_threshold=0.7):
        """
        Detect objects using deep learning model
        """
        # Prepare image
        image_tensor = torchvision.transforms.ToTensor()(frame)
        image_tensor = image_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Parse results
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        detected_objects = []
        for i, score in enumerate(scores):
            if score > confidence_threshold:
                box = boxes[i]
                label = self.COCO_CLASSES[labels[i]]
                
                detected_objects.append({
                    'label': label,
                    'confidence': score,
                    'box': box,
                    'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                })
        
        return detected_objects
    
    def sort_by_object_type(self, frame):
        """
        Example: Sort objects by type using ML vision
        """
        objects = self.detect_objects(frame)
        
        # Define sorting locations for different object types
        sorting_bins = {
            'cup': [0.3, 0.2, 0.1],
            'bottle': [0.3, -0.2, 0.1],
            'bowl': [0.4, 0.0, 0.1]
        }
        
        for obj in objects:
            if obj['label'] in sorting_bins:
                # Pick up object at detected location
                cx, cy = obj['center']
                # ... convert to robot coordinates ...
                
                # Move to sorting bin
                target_location = sorting_bins[obj['label']]
                self.piper.MoveToCartesianPosition(target_location)
```

---

## Development in Cursor IDE

### 1. Setting Up Piper Development in Cursor

**Step 1: Create Project Structure**
```bash
mkdir piper_project
cd piper_project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 2: Install Dependencies**
```bash
pip install piper-sdk opencv-python numpy scipy matplotlib --break-system-packages
```

**Step 3: Configure Cursor for Python Development**

In Cursor, create a `.cursor/settings.json`:
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "files.associations": {
    "*.yaml": "yaml"
  }
}
```

### 2. AI-Assisted Development with Cursor

**Use Cursor's AI features for:**

1. **Code Generation**: Ask Cursor to generate Piper control code
   ```
   Prompt: "Generate a function to move the Piper robot arm in a circular
   pattern with radius 0.1m at height 0.3m"
   ```

2. **Code Explanation**: Highlight complex SDK functions and ask Cursor to explain

3. **Debugging**: Use Cursor's AI to help debug CAN communication issues

4. **Refactoring**: Ask Cursor to optimize your trajectory planning algorithms

### 3. Example Complete Project in Cursor

**main.py:**
```python
"""
Piper Robot Arm - Vision Guided Pick and Place
Developed in Cursor IDE
"""

import cv2
import numpy as np
from piper_sdk import Piper
import time

class PiperVisionApp:
    def __init__(self):
        self.piper = Piper("can0")
        self.interface = self.piper.init()
        self.piper.connect()
        time.sleep(0.1)
        
        self.camera = cv2.VideoCapture(0)
        self.running = False
        
    def calibrate_camera_robot(self):
        """
        Perform hand-eye calibration
        """
        print("Starting calibration...")
        # Implement calibration routine
        pass
    
    def main_loop(self):
        """
        Main application loop
        """
        self.running = True
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            # Detect objects
            objects = self.detect_objects(frame)
            
            # Pick and place
            for obj in objects:
                self.pick_and_place(obj)
            
            cv2.imshow('Piper Vision', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
        
        self.cleanup()
    
    def detect_objects(self, frame):
        # Your detection logic
        pass
    
    def pick_and_place(self, object_data):
        # Your pick and place logic
        pass
    
    def cleanup(self):
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = PiperVisionApp()
    app.main_loop()
```

### 4. Debugging Tips for Cursor

**CAN Communication Debugging:**
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add debug prints
logger.debug(f"CAN interface: {can_interface}")
logger.debug(f"Joint states: {piper.get_joint_states()}")
```

**Visual Debugging:**
```python
def visualize_robot_state(piper, frame):
    """
    Overlay robot state on camera feed
    """
    joint_states = piper.get_joint_states()[0]
    cartesian_pos = piper.get_cartesian_position()
    
    cv2.putText(frame, f"Joints: {[f'{j:.2f}' for j in joint_states]}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Position: ({cartesian_pos[0]:.3f}, "
               f"{cartesian_pos[1]:.3f}, {cartesian_pos[2]:.3f})", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame
```

---

## Getting Started

### 1. Hardware Setup

1. **Mount the Robot**
   - Secure base using four M5 screws through 70mm spaced holes
   - Ensure stable mounting surface
   - Verify no obstructions in workspace

2. **Connect CAN Module**
   - Plug CAN-to-USB adapter into PC
   - Connect to robot arm CAN terminals (CAN-H and CAN-L)
   - Apply 12V power supply

3. **Configure CAN Interface**
   ```bash
   # Find USB port
   python find_can_port.py
   
   # Activate CAN (output shows: Interface can0 is connected to USB port 3-1.4:1.0)
   bash can_activate.sh can0 1000000 "3-1.4:1.0"
   
   # Verify activation
   ifconfig  # Should show can0 interface
   ```

### 2. First Program

```python
#!/usr/bin/env python3
"""
First Piper Program - Basic Movement Test
"""

import time
from piper_sdk import Piper

def main():
    # Initialize
    print("Connecting to Piper...")
    piper = Piper("can0")
    interface = piper.init()
    piper.connect()
    time.sleep(0.1)
    
    # Get firmware version
    firmware = interface.GetPiperFirmwareVersion()
    print(f"Firmware version: {firmware}")
    
    # Read current position
    joint_states = piper.get_joint_states()[0]
    print(f"Current joint angles: {joint_states}")
    
    # Enable robot (switch to slave mode for control)
    interface.MasterSlaveConfig(0xFC, 0, 0, 0)
    time.sleep(0.1)
    
    # Move to home position (all joints at 0)
    home_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print("Moving to home position...")
    interface.MoveToJointAngles(home_position, speed_rate=50)
    time.sleep(3)
    
    # Simple movement test
    print("Executing test movement...")
    test_position = [0.2, 0.3, -0.2, 0.0, 0.0, 0.0]
    interface.MoveToJointAngles(test_position, speed_rate=60)
    time.sleep(3)
    
    # Return to home
    print("Returning to home...")
    interface.MoveToJointAngles(home_position, speed_rate=50)
    time.sleep(3)
    
    print("Test complete!")

if __name__ == "__main__":
    main()
```

### 3. Teaching Mode

The Piper includes a teaching mode for easy position recording:

```python
#!/usr/bin/env python3
"""
Teaching Mode - Record and Replay Positions
"""

import time
import csv
from piper_sdk import Piper

def record_positions():
    piper = Piper("can0")
    interface = piper.init()
    piper.connect()
    time.sleep(0.1)
    
    positions = []
    
    print("Press the teach button on the robot to enter teaching mode")
    print("Waiting for teaching mode...")
    
    # Wait for teaching mode
    timeout = time.time() + 10
    while interface.GetArmStatus().arm_status.ctrl_mode != 2:
        if time.time() > timeout:
            print("Timeout waiting for teaching mode")
            return
        time.sleep(0.01)
    
    print("Teaching mode active! Manually move the robot.")
    print("Press Enter to record position, 'q' to quit")
    
    count = 1
    while input(f"Position {count} (Enter to record, 'q' to quit): ") != 'q':
        joint_state = piper.get_joint_states()[0]
        gripper_state = piper.get_gripper_states()[0][0]
        
        position = list(joint_state) + [gripper_state]
        positions.append(position)
        print(f"Recorded position {count}: {position}")
        count += 1
    
    # Save to CSV
    with open('recorded_positions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(positions)
    
    print(f"Saved {len(positions)} positions to recorded_positions.csv")

def replay_positions():
    piper = Piper("can0")
    interface = piper.init()
    piper.connect()
    time.sleep(0.1)
    
    # Load positions
    with open('recorded_positions.csv', 'r') as f:
        positions = list(csv.reader(f))
        positions = [[float(x) for x in row] for row in positions]
    
    # Enable control mode
    interface.MasterSlaveConfig(0xFC, 0, 0, 0)
    time.sleep(0.1)
    
    print(f"Replaying {len(positions)} positions...")
    
    for i, pos in enumerate(positions):
        print(f"Moving to position {i+1}/{len(positions)}")
        
        # Joint angles (first 6 values)
        joint_angles = pos[:6]
        interface.MoveToJointAngles(joint_angles, speed_rate=70)
        time.sleep(1)
        
        # Gripper (7th value if present)
        if len(pos) > 6:
            piper.SetGripperPosition(pos[6])
            time.sleep(0.5)
    
    print("Replay complete!")

if __name__ == "__main__":
    print("1. Record positions")
    print("2. Replay positions")
    choice = input("Choose option (1/2): ")
    
    if choice == '1':
        record_positions()
    elif choice == '2':
        replay_positions()
```

---

## Advanced Applications

### 1. Trajectory Planning with MoveIt

```python
import rospy
import moveit_commander
import geometry_msgs.msg

class MoveItPiperController:
    def __init__(self):
        moveit_commander.roscpp_initialize([])
        rospy.init_node('piper_moveit_controller', anonymous=True)
        
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("piper_arm")
        
        self.group.set_planner_id("RRTConnect")
        self.group.set_planning_time(5)
        
    def plan_cartesian_path(self, waypoints, eef_step=0.01):
        """
        Plan smooth Cartesian path through waypoints
        """
        (plan, fraction) = self.group.compute_cartesian_path(
            waypoints,
            eef_step,
            0.0  # jump_threshold
        )
        
        print(f"Path planning: {fraction*100:.1f}% successful")
        return plan
    
    def execute_trajectory(self, plan):
        """
        Execute planned trajectory
        """
        self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
```

### 2. Force Control and Compliance

```python
class ForceControlledPiper:
    def __init__(self, piper_interface):
        self.piper = piper_interface
        self.force_threshold = 5.0  # Newtons
        
    def compliant_insertion(self, target_position, max_force=5.0):
        """
        Perform compliant insertion with force feedback
        """
        approach_speed = 30  # Slow approach
        
        while True:
            current_force = self.read_force_sensor()
            
            if current_force > max_force:
                print("Force limit reached, stopping")
                self.piper.Stop()
                break
            
            # Continue moving
            self.piper.MoveToCartesianPosition(
                target_position, 
                speed_rate=approach_speed
            )
            
            time.sleep(0.01)
    
    def read_force_sensor(self):
        """
        Read force from external force/torque sensor
        """
        # Implement based on your force sensor
        pass
```

### 3. Multi-Robot Coordination

```python
class MultiPiperController:
    def __init__(self, robot_can_ports):
        self.robots = []
        for port in robot_can_ports:
            piper = Piper(port)
            piper.init()
            piper.connect()
            self.robots.append(piper)
    
    def synchronized_movement(self, positions_list):
        """
        Move multiple robots simultaneously
        """
        import threading
        
        def move_robot(robot, position):
            robot.MoveToJointAngles(position, speed_rate=60)
        
        threads = []
        for robot, position in zip(self.robots, positions_list):
            thread = threading.Thread(target=move_robot, args=(robot, position))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
```

### 4. Integration with Mobile Base

```python
class MobilePiperSystem:
    def __init__(self, robot_can="can0", base_port="/dev/ttyUSB0"):
        # Initialize arm
        self.piper = Piper(robot_can)
        self.piper.init()
        self.piper.connect()
        
        # Initialize mobile base (AgileX Scout/Ranger)
        from ugv_sdk import RobotBase
        self.mobile_base = RobotBase(base_port)
        
    def navigate_and_pick(self, navigation_goal, pick_location):
        """
        Navigate to location and pick up object
        """
        # Navigate mobile base
        self.mobile_base.move_to(navigation_goal)
        
        # Wait for arrival
        while not self.mobile_base.is_at_goal():
            time.sleep(0.1)
        
        # Pick up object with arm
        self.piper.MoveToCartesianPosition(pick_location)
        self.piper.SetGripperPosition(0)  # Close gripper
```

---

## Additional Resources

### Official Documentation
- **GitHub**: https://github.com/agilexrobotics/piper_sdk
- **ROS Driver**: https://github.com/agilexrobotics/piper_ros
- **Support**: support@agilex.ai
- **User Manual**: Available at Generation Robots and Roboworks websites

### Community Projects
- **Hackster.io**: Search for "AgileX Piper" for community projects
- **ROS Discourse**: Open Robotics forum discussions
- **GitHub Examples**: agilexrobotics/Agilex-College repository

### Development Tools
- **CAN Tools**: can-utils package for Linux
- **Simulation**: Gazebo with Piper URDF models
- **Visualization**: RViz for ROS-based development
- **Computer Vision**: OpenCV, PCL (Point Cloud Library)

### Tips for Success
1. Always perform hand-eye calibration for vision applications
2. Start with slow speeds when testing new movements
3. Monitor joint temperatures during extended use
4. Keep firmware updated for best performance
5. Use teaching mode to quickly prototype movements
6. Implement emergency stop routines in all applications

---

**Created for development in Cursor IDE**
**Version: 1.0**
**Last Updated: January 2026**
