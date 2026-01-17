#!/usr/bin/env python3
"""
Real-time WebSocket Trajectory Controller for Piper Robot Arm
Receives pose angles from CV pipeline and controls robot simulation in real-time.

Usage:
    python realtime_trajectory.py

Then connect pose_app.py to ws://localhost:8000/ws
"""

import sys
import os
import json
import math
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


def deg_to_rad(deg):
    """Convert degrees to radians"""
    return deg * (math.pi / 180.0)


def map_pose_to_joints(left_angles):
    """
    Map pose angles to robot joint angles.

    Input: [0, shoulder_deg, elbow_deg, wrist_deg] (degrees)
    Output: [j1, j2, j3, j4, j5, j6] (radians)

    Mapping:
        - J1 (base rotation): fixed at 0
        - J2 (shoulder pitch): shoulder angle
        - J3 (elbow pitch): elbow angle (inverted, 180° straight = 0 rad)
        - J4 (wrist roll): fixed at 0
        - J5 (wrist pitch): wrist angle
        - J6 (wrist rotation): fixed at 0
    """
    _, shoulder, elbow, wrist = left_angles

    # Convert to radians and apply to joints
    j1 = 0.0                          # Base rotation - fixed
    j2 = deg_to_rad(shoulder - 90)    # Shoulder pitch (90° input = horizontal)
    j3 = deg_to_rad(elbow - 180)      # Elbow (inverted, 180° = straight = 0 rad)
    j4 = 0.0                          # Wrist roll - fixed
    j5 = deg_to_rad(0)      # Wrist pitch
    j6 = deg_to_rad(90)                        # Wrist rotation - fixed

    return [j1, j2, j3, j4, j5, j6]


def simulation_loop():
    """Main thread function that runs the PyBullet simulation"""
    global running

    print("🤖 Starting PyBullet simulation...")
    sim = PiperSimulation(gui=True)
    sim.reset_to_home()
    print("✅ Simulation ready, waiting for WebSocket commands...")

    current_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    while running:
        # Get latest command (non-blocking), keep only most recent
        try:
            while True:
                current_joints = command_queue.get_nowait()
        except queue.Empty:
            pass

        # Apply current joint positions
        try:
            sim.set_joint_positions(current_joints)
            sim.step()
        except RuntimeError as e:
            if "disconnected" in str(e).lower() or "closed" in str(e).lower():
                print("⚠️  GUI window closed")
                running = False
                break
            raise
        except KeyboardInterrupt:
            running = False
            break

    print("🔴 Stopping simulation...")
    try:
        sim.close()
    except Exception:
        pass


# FastAPI app (no lifespan needed since we manage threads differently)
app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for receiving pose commands"""
    await websocket.accept()
    print("📡 WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"📥 WS received: {data}")

            try:
                msg = json.loads(data)

                # Extract left arm angles and map to robot joints
                if "left" in msg:
                    left_angles = msg["left"]
                    if len(left_angles) == 4:
                        joints = map_pose_to_joints(left_angles)
                        command_queue.put(joints)
                        await websocket.send_text(json.dumps({"joints": joints}))
                    else:
                        await websocket.send_text("error: left array must have 4 elements")
                else:
                    await websocket.send_text("error: missing 'left' key")

            except json.JSONDecodeError:
                await websocket.send_text("error: invalid JSON")
            except Exception as e:
                await websocket.send_text(f"error: {str(e)}")

    except WebSocketDisconnect:
        print("📡 WebSocket client disconnected")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "running", "message": "Connect via WebSocket at /ws"}


def run_server():
    """Run uvicorn server in background thread"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    print("=" * 60)
    print("PIPER ROBOT - REAL-TIME WEBSOCKET CONTROLLER")
    print("=" * 60)
    print("\nStarting server on http://0.0.0.0:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print("\nExpected message format:")
    print('  {"left": [0, shoulder_deg, elbow_deg, wrist_deg]}')
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
