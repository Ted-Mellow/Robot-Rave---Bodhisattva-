#!/usr/bin/env python3
"""
Piper Robot Arm - PyBullet Simulation (CORRECTED)
Loads the corrected URDF and provides proper visualization
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import os


class PiperSimulation:
    """PyBullet simulation for Piper robot arm with corrected URDF"""
    
    # Joint limits from physical robot hardware specs (in radians)
    # J1: ±154°, J2: 0→195°, J3: -175→0°, J4: ±106°, J5: ±75°, J6: ±100°
    JOINT_LIMITS = {
        'lower': [-2.68781, 0, -3.05433, -1.85005, -1.30900, -1.74533],
        'upper': [2.68781, 3.40339, 0, 1.85005, 1.30900, 1.74533]
    }
    
    JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    
    def __init__(self, urdf_path=None, gui=True, time_step=1./240.):
        """
        Initialize PyBullet simulation with URDF
        
        Args:
            urdf_path: Path to URDF file
            gui: If True, show GUI. If False, run headless
            time_step: Physics simulation time step
        """
        self.gui = gui
        self.time_step = time_step
        self.robot_id = None
        self.joint_indices = []
        
        # Auto-detect URDF if not provided
        if urdf_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)  # Go up one level from simulation/
            urdf_paths = [
                os.path.join(parent_dir, "robot_models", "piper_corrected.urdf"),
                os.path.join(parent_dir, "robot_models", "piper.urdf"),
                os.path.join(script_dir, "robot_models", "piper_corrected.urdf"),
                os.path.join(script_dir, "robot_models", "piper.urdf"),
                os.path.join(script_dir, "piper.urdf"),
                "piper.urdf"
            ]
            for path in urdf_paths:
                if os.path.exists(path):
                    urdf_path = path
                    break
            if urdf_path is None:
                raise FileNotFoundError("Could not find URDF file. Searched: " + str(urdf_paths))
        
        self.urdf_path = urdf_path
        
        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
            # Configure camera for better view
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.3, 0, 0.3]
            )
            # Disable unnecessary GUI elements
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(time_step)
        p.setGravity(0, 0, -9.81)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load robot from URDF
        self._load_robot()
        
        # Add coordinate frame
        self._add_coordinate_frame()
        
        # Additional stabilization: Allow more time for physics to fully settle
        # This ensures no visible flopping when GUI appears or control begins
        print("⏳ Stabilizing robot physics...")
        for _ in range(150):
            p.stepSimulation()
        
        print(f"✅ Piper simulation initialized and stabilized")
        print(f"   Robot ID: {self.robot_id}")
        print(f"   Num joints: {p.getNumJoints(self.robot_id)}")
        print(f"   Controllable joints: {self.joint_indices}")
    
    def _load_robot(self):
        """Load robot from URDF file"""
        # Check if URDF exists
        if not os.path.exists(self.urdf_path):
            print(f"❌ URDF not found: {self.urdf_path}")
            print(f"   Current directory: {os.getcwd()}")
            print(f"   Script directory: {os.path.dirname(os.path.abspath(__file__))}")
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
        
        # Load URDF
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        
        # Get joint information
        num_joints = p.getNumJoints(self.robot_id)
        print(f"\n📋 Joint Information:")
        
        self.joint_indices = []
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # Only track revolute joints (type 0)
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                print(f"   Joint {i}: {joint_name} (revolute)")
            elif joint_type == p.JOINT_PRISMATIC:
                print(f"   Joint {i}: {joint_name} (prismatic - gripper)")
            elif joint_type == p.JOINT_FIXED:
                print(f"   Joint {i}: {joint_name} (fixed)")
        
        # Enable force/torque sensor for joints (optional)
        for joint_idx in self.joint_indices:
            p.enableJointForceTorqueSensor(self.robot_id, joint_idx, True)
        
        print(f"\n✅ Loaded URDF with {len(self.joint_indices)} controllable joints")
        
        # IMMEDIATELY set joints to home position to prevent gravity-induced flopping
        # This must happen before any physics steps to hold the arm in place
        home_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=home_position[i],
                force=500,
                positionGain=0.3,
                velocityGain=0.1
            )
        
        # Run physics steps to stabilize (without rendering delay)
        # This allows the PID controllers to engage and hold the joints
        for _ in range(100):
            p.stepSimulation()
        
        print("✅ Joints stabilized at home position")
    
    def _add_coordinate_frame(self):
        """Add world coordinate frame visualization"""
        axis_length = 0.3
        
        # X-axis (Red)
        p.addUserDebugLine([0, 0, 0], [axis_length, 0, 0], 
                          lineColorRGB=[1, 0, 0], lineWidth=3)
        # Y-axis (Green)
        p.addUserDebugLine([0, 0, 0], [0, axis_length, 0], 
                          lineColorRGB=[0, 1, 0], lineWidth=3)
        # Z-axis (Blue)
        p.addUserDebugLine([0, 0, 0], [0, 0, axis_length], 
                          lineColorRGB=[0, 0, 1], lineWidth=3)
        
        # Labels
        p.addUserDebugText("X", [axis_length + 0.05, 0, 0], 
                          textColorRGB=[1, 0, 0], textSize=1.5)
        p.addUserDebugText("Y", [0, axis_length + 0.05, 0], 
                          textColorRGB=[0, 1, 0], textSize=1.5)
        p.addUserDebugText("Z", [0, 0, axis_length + 0.05], 
                          textColorRGB=[0, 0, 1], textSize=1.5)
    
    def set_joint_positions(self, joint_angles, max_force=None):
        """
        Set joint positions with clamping to limits
        
        Args:
            joint_angles: List of 6 joint angles in radians
            max_force: Maximum force to apply (None = use per-joint defaults)
        """
        if not p.isConnected(self.client):
            raise RuntimeError("GUI window was closed - simulation disconnected")
        
        if len(joint_angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")
        
        # Per-joint force settings (J2 and J3 need MASSIVE force to lift against gravity)
        if max_force is None:
            joint_forces = [
                2000,  # J1: Base rotation (high load when arm extended)
                8000,  # J2: Shoulder (EXTREME - must lift entire arm vertically!)
                6000,  # J3: Elbow (VERY HIGH - extends forearm weight)
                1000,  # J4: Wrist roll (moderate load)
                1000,  # J5: Wrist pitch (moderate load)
                800    # J6: Wrist rotate (low load)
            ]
        else:
            joint_forces = [max_force] * 6
        
        # Clamp to limits
        clamped_angles = []
        for i, angle in enumerate(joint_angles):
            clamped = np.clip(
                angle,
                self.JOINT_LIMITS['lower'][i],
                self.JOINT_LIMITS['upper'][i]
            )
            clamped_angles.append(clamped)
        
        # Set position control with per-joint force and tuned gains
        for i, joint_idx in enumerate(self.joint_indices):
            # VERY high gains for J2 and J3 to overcome gravity and reach targets
            if i in [1, 2]:  # J2 (shoulder) and J3 (elbow)
                pos_gain = 1.0  # Maximum stiffness
                vel_gain = 0.5  # High damping
            else:
                pos_gain = 0.5
                vel_gain = 0.2
            
            if not self.is_connected():
                raise RuntimeError("GUI window was closed - simulation disconnected")
                
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=clamped_angles[i],
                force=joint_forces[i],
                positionGain=pos_gain,
                velocityGain=vel_gain
            )
    
    def get_joint_positions(self):
        """Get current joint positions"""
        states = p.getJointStates(self.robot_id, self.joint_indices)
        return [state[0] for state in states]
    
    def get_joint_velocities(self):
        """Get current joint velocities"""
        states = p.getJointStates(self.robot_id, self.joint_indices)
        return [state[1] for state in states]
    
    def get_end_effector_pose(self):
        """Get end effector position and orientation"""
        # Get the end effector link state
        # Use the last controllable joint (joint6) as end effector
        # In PyBullet, link N is the child of joint N
        ee_link_idx = self.joint_indices[-1]  # Last controllable joint
        link_state = p.getLinkState(self.robot_id, ee_link_idx)
        position = link_state[0]
        orientation = link_state[1]
        return position, orientation
    
    def set_gripper(self, opening):
        """
        Set gripper opening
        
        Args:
            opening: Gripper opening (0.0 = closed, 0.085 = fully open)
        """
        opening = np.clip(opening, 0.0, 0.0425)  # Each finger moves 42.5mm
        
        # Find gripper joints
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            
            if 'finger' in joint_name:
                p.setJointMotorControl2(
                    self.robot_id,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=opening,
                    force=20
                )
    
    def step(self):
        """Step the simulation forward"""
        if not p.isConnected(self.client):
            raise RuntimeError("GUI window was closed - simulation disconnected")
        p.stepSimulation()
        if self.gui:
            time.sleep(self.time_step)
    
    def is_connected(self):
        """Check if PyBullet is still connected"""
        try:
            return p.isConnected(self.client)
        except:
            return False
    
    def reset_to_home(self):
        """Reset robot to home position"""
        home_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.set_joint_positions(home_position)
        for _ in range(240):  # Wait 1 second
            self.step()
        print("✅ Reset to home position")
    
    def demo_motion(self):
        """Demo: Move through various positions"""
        print("\n🎬 Starting demo motion...")
        
        positions = [
            ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Home"),
            ([1.0, 0.5, -0.5, 0.0, 0.5, 0.0], "Position 1"),
            ([0.0, 1.5, -1.5, 0.0, 1.0, 1.5], "Position 2"),
            ([-1.0, 1.0, -1.0, 0.5, 0.0, -1.0], "Position 3"),
            ([0.0, 2.0, -2.0, 0.0, 0.5, 0.0], "Reach forward"),
            ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Return home"),
        ]
        
        for joints, name in positions:
            print(f"   → {name}")
            self.set_joint_positions(joints)
            for _ in range(240):  # 1 second per position
                self.step()
            time.sleep(0.5)
        
        print("✅ Demo complete!")
    
    def close(self):
        """Close the simulation safely"""
        try:
            if p.isConnected(self.client):
                p.disconnect(self.client)
                print("🔴 Simulation closed")
            else:
                print("ℹ️  Simulation already closed")
        except Exception as e:
            print(f"ℹ️  Simulation cleanup: {e}")


def main():
    """Main demo program"""
    print("=" * 60)
    print("PIPER ROBOT ARM - PYBULLET SIMULATION")
    print("=" * 60)
    
    # Find URDF file (check robot_models/ first, then current directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Go up from simulation/
    urdf_paths = [
        os.path.join(parent_dir, "robot_models", "piper.urdf"),
        os.path.join(parent_dir, "robot_models", "piper_corrected.urdf"),
        os.path.join(script_dir, "robot_models", "piper.urdf"),
        os.path.join(script_dir, "robot_models", "piper_corrected.urdf"),
        os.path.join(script_dir, "piper.urdf"),
        "piper.urdf"
    ]
    
    urdf_path = None
    for path in urdf_paths:
        if os.path.exists(path):
            urdf_path = path
            break
    
    if not urdf_path:
        print(f"\n❌ URDF file not found!")
        print("\nSearched in:")
        for path in urdf_paths:
            print(f"   {path}")
        print(f"\nCurrent directory: {os.getcwd()}")
        return
    
    print(f"\n✅ Found URDF: {urdf_path}")
    
    # Create simulation
    sim = PiperSimulation(urdf_path=urdf_path, gui=True)
    
    try:
        # Reset to home
        sim.reset_to_home()
        time.sleep(1)
        
        # Get end effector pose
        pos, orn = sim.get_end_effector_pose()
        print(f"\n📍 End effector position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        reach = np.sqrt(pos[0]**2 + pos[1]**2)
        print(f"   Horizontal reach: {reach*1000:.1f} mm")
        
        # Demo gripper
        print("\n🤏 Testing gripper...")
        sim.set_gripper(0.0)  # Close
        for _ in range(60):
            sim.step()
        
        sim.set_gripper(0.0425)  # Open
        for _ in range(60):
            sim.step()
        
        # Demo motion
        sim.demo_motion()
        
        # Keep window open
        print("\n✅ Demo complete!")
        print("Press Ctrl+C to exit...")
        while True:
            sim.step()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
