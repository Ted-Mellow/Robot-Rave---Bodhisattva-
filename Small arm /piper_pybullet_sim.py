#!/usr/bin/env python3
"""
Piper Robot Arm PyBullet Simulation
Simplified simulation environment for testing Piper arm movements
"""

import pybullet as p
import pybullet_data
import time
import numpy as np


class PiperSimulation:
    """PyBullet simulation class for Piper robot arm"""
    
    # DH Parameters from piper_sdk/kinematics/piper_fk.py
    DH_PARAMS = {
        'a': [0, 0, 285.03, -21.98, 0, 0],  # mm
        'd': [123, 0, 0, 250.75, 0, 91],    # mm
        'alpha': [0, -np.pi/2, 0, np.pi/2, -np.pi/2, np.pi/2],
        'theta_offset': [0, -np.pi*172.22/180, -102.78/180*np.pi, 0, 0, 0]
    }
    
    # Joint limits (radians) from piper_sdk/piper_param/piper_param_manager.py
    JOINT_LIMITS = {
        'lower': [-2.6179, 0, -2.967, -1.745, -1.22, -2.09439],
        'upper': [2.6179, 3.14, 0, 1.745, 1.22, 2.09439]
    }
    
    # Joint names
    JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    
    def __init__(self, gui=True, time_step=1./240.):
        """
        Initialize PyBullet simulation
        
        Args:
            gui (bool): If True, show GUI. If False, run headless
            time_step (float): Physics simulation time step
        """
        self.gui = gui
        self.time_step = time_step
        self.robot_id = None
        self.joint_indices = []
        
        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(time_step)
        p.setGravity(0, 0, -9.81)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Create simplified Piper arm from primitives
        self._create_robot()
        
        print(f"✅ Piper PyBullet Simulation initialized (GUI: {gui})")
        print(f"   Robot ID: {self.robot_id}")
        print(f"   Joint indices: {self.joint_indices}")
    
    def _create_robot(self):
        """Create a simplified robot arm using basic shapes"""
        # Base position
        base_pos = [0, 0, 0.05]
        base_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Create multi-body with simplified geometry
        # This is a placeholder - ideally load from URDF
        # For now, create a simple 6-link chain
        
        link_masses = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
        link_lengths = [0.123, 0.285, 0.251, 0.091, 0.05, 0.05]  # meters (from DH params)
        
        # Create visual and collision shapes
        visual_shapes = []
        collision_shapes = []
        link_positions = []
        link_orientations = []
        link_inertial_positions = []
        link_inertial_orientations = []
        link_parent_indices = []
        link_joint_types = []
        link_joint_axes = []
        
        for i in range(6):
            # Create cylindrical links
            visual_shapes.append(
                p.createVisualShape(
                    shapeType=p.GEOM_CYLINDER,
                    radius=0.03,
                    length=link_lengths[i],
                    rgbaColor=[0.7, 0.7, 0.7, 1.0]
                )
            )
            collision_shapes.append(
                p.createCollisionShape(
                    shapeType=p.GEOM_CYLINDER,
                    radius=0.03,
                    height=link_lengths[i]
                )
            )
            
            # Link positions (simplified)
            if i == 0:
                link_positions.append([0, 0, link_lengths[i]/2])
            else:
                link_positions.append([0, 0, link_lengths[i]/2])
            
            link_orientations.append([0, 0, 0, 1])
            link_inertial_positions.append([0, 0, 0])
            link_inertial_orientations.append([0, 0, 0, 1])
            
            if i == 0:
                link_parent_indices.append(0)  # Connect to base
            else:
                link_parent_indices.append(i)
            
            link_joint_types.append(p.JOINT_REVOLUTE)
            
            # Alternate joint axes for more realistic motion
            if i % 2 == 0:
                link_joint_axes.append([0, 0, 1])  # Z-axis rotation
            else:
                link_joint_axes.append([0, 1, 0])  # Y-axis rotation
        
        # Create base
        base_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.08,
            length=0.1,
            rgbaColor=[0.3, 0.3, 0.3, 1.0]
        )
        base_collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.08,
            height=0.1
        )
        
        # Create multi-body
        self.robot_id = p.createMultiBody(
            baseMass=2.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=base_pos,
            baseOrientation=base_orientation,
            linkMasses=link_masses,
            linkCollisionShapeIndices=collision_shapes,
            linkVisualShapeIndices=visual_shapes,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_positions,
            linkInertialFrameOrientations=link_inertial_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes
        )
        
        # Get joint indices
        num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = list(range(num_joints))[:6]  # Use first 6 joints
        
        # Set joint limits
        for i, joint_idx in enumerate(self.joint_indices):
            p.changeDynamics(
                self.robot_id,
                joint_idx,
                jointLowerLimit=self.JOINT_LIMITS['lower'][i],
                jointUpperLimit=self.JOINT_LIMITS['upper'][i]
            )
    
    def set_joint_positions(self, joint_angles, control_mode='position'):
        """
        Set joint positions
        
        Args:
            joint_angles (list): List of 6 joint angles in radians
            control_mode (str): 'position' or 'velocity'
        """
        if len(joint_angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")
        
        # Clamp to joint limits
        clamped_angles = []
        for i, angle in enumerate(joint_angles):
            clamped = np.clip(
                angle,
                self.JOINT_LIMITS['lower'][i],
                self.JOINT_LIMITS['upper'][i]
            )
            clamped_angles.append(clamped)
        
        if control_mode == 'position':
            for i, joint_idx in enumerate(self.joint_indices):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=clamped_angles[i],
                    force=100
                )
        else:
            raise NotImplementedError(f"Control mode '{control_mode}' not implemented")
    
    def get_joint_positions(self):
        """Get current joint positions"""
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        return [state[0] for state in joint_states]
    
    def get_joint_velocities(self):
        """Get current joint velocities"""
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        return [state[1] for state in joint_states]
    
    def get_end_effector_pose(self):
        """Get end effector position and orientation"""
        link_state = p.getLinkState(self.robot_id, self.joint_indices[-1])
        position = link_state[0]
        orientation = link_state[1]
        return position, orientation
    
    def is_connected(self):
        """Check if still connected to physics server"""
        try:
            p.getConnectionInfo(self.client)
            return True
        except:
            return False
    
    def step(self):
        """Step the simulation forward"""
        if not self.is_connected():
            raise RuntimeError("Physics server disconnected (GUI may have been closed)")
        p.stepSimulation()
        if self.gui:
            time.sleep(self.time_step)
    
    def reset(self):
        """Reset robot to home position"""
        home_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.set_joint_positions(home_position)
        for _ in range(100):
            self.step()
    
    def close(self):
        """Close the simulation"""
        if self.is_connected():
            p.disconnect(self.client)
            print("🔴 Simulation closed")
        else:
            print("🔴 Simulation already closed")


def demo_simple_motion():
    """Demo: Simple joint motion"""
    print("=" * 50)
    print("Demo: Simple Joint Motion")
    print("=" * 50)
    
    sim = PiperSimulation(gui=True)
    
    try:
        # Reset to home
        sim.reset()
        print("✅ Reset to home position")
        time.sleep(1)
        
        # Move joint 1
        print("Moving joint 1...")
        target = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sim.set_joint_positions(target)
        for _ in range(240):
            sim.step()
        
        # Move joint 2
        print("Moving joint 2...")
        target = [1.0, 1.5, 0.0, 0.0, 0.0, 0.0]
        sim.set_joint_positions(target)
        for _ in range(240):
            sim.step()
        
        # Move multiple joints
        print("Moving multiple joints...")
        target = [0.5, 1.0, -1.0, 0.5, 0.5, 1.0]
        sim.set_joint_positions(target)
        for _ in range(240):
            sim.step()
        
        # Return home
        print("Returning home...")
        sim.reset()
        
        print("\n✅ Demo complete! Press Ctrl+C to exit")
        while True:
            sim.step()
            
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt received")
    finally:
        sim.close()


if __name__ == "__main__":
    demo_simple_motion()
