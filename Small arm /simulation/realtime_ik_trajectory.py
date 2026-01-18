#!/usr/bin/env python3
"""
Real-time WebSocket IK Controller for Piper Robot Arm
Receives Cartesian coordinates from keypoint detection and uses inverse kinematics
to control the robot simulation in real-time.

Usage:
    python realtime_ik_trajectory.py

Then connect your keypoint detector to ws://localhost:8000/ws

Message format:
    {"target": [x, y]}  - normalized shoulder-relative coordinates (0-1 range)
"""

import sys
import os
import json
import threading
import queue

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from piper_simultion_corrected import PiperSimulation

# Shared state
command_queue = queue.Queue()
running = True

# Robot shoulder position in simulation (meters) - from URDF joint1 position
SHOULDER_POS = [0.0, 0.0, 0.123]

# Arm reach: ~0.5m total (285mm upper arm + 250mm forearm + gripper)
ARM_REACH = 0.5


def keypoint_to_sim_coords(kp_x, kp_y):
    """
    Convert shoulder-relative keypoint (x,y) normalized 0-1 to simulation (x,y,z)
    
    New coordinate mapping:
    - kp_x: 0-1, where 0.5 = shoulder position, >0.5 = forward, <0.5 = backward
    - kp_y: 0-1, where 0.5 = shoulder level, >0.5 = up, <0.5 = down

    Args:
        kp_x: Normalized X (0-1), 0.5 = shoulder, >0.5 = forward, <0.5 = backward
        kp_y: Normalized Y (0-1), 0.5 = shoulder level, >0.5 = up, <0.5 = down

    Returns:
        [sim_x, sim_y, sim_z] in meters, on XZ plane
    """
    # Convert from [0,1] centered at 0.5 to [-1,1] centered at 0
    x_offset = (kp_x - 0.5) * 2.0  # Maps [0,1] -> [-1,1]
    y_offset = (kp_y - 0.5) * 2.0  # Maps [0,1] -> [-1,1]
    
    sim_x = SHOULDER_POS[0] + (x_offset * ARM_REACH * 0.5)  # Forward/backward
    sim_y = 0.0  # Fixed Y to keep robot on XZ plane
    sim_z = SHOULDER_POS[2] + (y_offset * ARM_REACH * 0.5)  # Up/down
    return [sim_x, sim_y, sim_z]


def simulation_loop():
    """Main thread function that runs the PyBullet simulation with IK"""
    global running

    print("🤖 Starting PyBullet simulation with IK...")
    sim = PiperSimulation(gui=True)
    sim.reset_to_home()
    print("✅ Simulation ready, waiting for WebSocket commands...")

    # Default target position (arm extended forward)
    current_target = [0.3, 0.0, 0.3]

    while running:
        # Get latest command (non-blocking), keep only most recent
        try:
            while True:
                current_target = command_queue.get_nowait()
        except queue.Empty:
            pass

        # Apply IK to reach target position
        try:
            # Check if simulation is still valid
            if sim.robot_id is None:
                print("⚠️  Robot disconnected")
                running = False
                break
            
            sim.set_end_effector_position(current_target)
            sim.step()
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "disconnected" in error_msg or "closed" in error_msg or "not connected" in error_msg:
                print("⚠️  GUI window closed or physics server disconnected")
                running = False
                break
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if "not connected" in error_msg or "disconnected" in error_msg:
                print("⚠️  Physics server disconnected")
                running = False
                break
            # Re-raise other exceptions
            raise
        except KeyboardInterrupt:
            running = False
            break

    print("🔴 Stopping simulation...")
    try:
        sim.close()
    except Exception:
        pass


# FastAPI app
app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for receiving Cartesian target commands"""
    await websocket.accept()
    print("📡 WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"📥 WS received: {data}")

            try:
                msg = json.loads(data)

                # Extract target coordinates and convert to simulation space
                if "target" in msg:
                    target = msg["target"]
                    if len(target) == 2:
                        kp_x, kp_y = target
                        sim_coords = keypoint_to_sim_coords(kp_x, kp_y)
                        command_queue.put(sim_coords)
                        await websocket.send_text(json.dumps({
                            "target_sim": sim_coords,
                            "input": target
                        }))
                    else:
                        await websocket.send_text("error: target array must have 2 elements [x, y]")
                else:
                    await websocket.send_text("error: missing 'target' key")

            except json.JSONDecodeError:
                await websocket.send_text("error: invalid JSON")
            except Exception as e:
                await websocket.send_text(f"error: {str(e)}")

    except WebSocketDisconnect:
        print("📡 WebSocket client disconnected")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "mode": "IK (Cartesian)",
        "message": "Connect via WebSocket at /ws",
        "format": '{"target": [x, y]} - normalized 0-1 shoulder-relative coords'
    }


def run_server():
    """Run uvicorn server in background thread"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    print("=" * 60)
    print("PIPER ROBOT - IK CARTESIAN CONTROLLER")
    print("=" * 60)
    print("\nStarting server on http://0.0.0.0:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print("\nExpected message format:")
    print('  {"target": [x, y]}')
    print("  x, y: normalized 0-1, shoulder-relative")
    print("  +x = forward, +y = up")
    print(f"\nShoulder position: {SHOULDER_POS}")
    print(f"Arm reach: {ARM_REACH}m")
    print("=" * 60)

    # Start uvicorn server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Run PyBullet simulation on main thread (required for macOS GUI)
    try:
        simulation_loop()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        running = False
