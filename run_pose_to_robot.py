#!/usr/bin/env python3
"""
Launcher script for the full pose-to-robot pipeline:
1. Webcam → MediaPipe pose detection (pose_cartesian_app)
2. WebSocket → Simulation with IK (realtime_ik_trajectory)
3. Simulation → (eventually) Real PiPER robot

Usage:
    python run_pose_to_robot.py

This script starts both the simulation server and the pose detection app.
Both need the main thread (Tkinter and PyBullet GUI), so they run in separate processes.
"""

import sys
import os
import time
import subprocess
import signal
import multiprocessing

# Add paths for imports
project_root = os.path.dirname(os.path.abspath(__file__))
nkosi_path = os.path.join(project_root, "nkosi")
# Handle directory name with trailing space
simulation_path = os.path.join(project_root, "Small arm ", "simulation")

# Global processes for cleanup
processes = []


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Shutting down all processes...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except:
            try:
                proc.kill()
            except:
                pass
    sys.exit(0)


def run_simulation_server():
    """Run the realtime_ik_trajectory server"""
    server_script = os.path.join(simulation_path, "realtime_ik_trajectory.py")
    
    if not os.path.exists(server_script):
        print(f"❌ Error: Server script not found at {server_script}")
        return None
    
    print("🚀 Starting simulation server...")
    proc = subprocess.Popen(
        [sys.executable, server_script],
        cwd=simulation_path
    )
    return proc


def run_pose_app():
    """Run the pose detection app"""
    pose_script = os.path.join(nkosi_path, "pose_cartesian_app.py")
    
    if not os.path.exists(pose_script):
        print(f"❌ Error: Pose app script not found at {pose_script}")
        return None
    
    print("📹 Starting pose detection app...")
    proc = subprocess.Popen(
        [sys.executable, pose_script],
        cwd=nkosi_path
    )
    return proc


def check_dependencies():
    """Check if required dependencies are installed"""
    global nkosi_path
    missing = []
    
    # Check MediaPipe
    try:
        import mediapipe as mp
        if not hasattr(mp, 'solutions'):
            missing.append("mediapipe (incomplete installation)")
    except ImportError:
        missing.append("mediapipe")
    
    # Check OpenCV
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    # Check other critical dependencies
    try:
        import websocket
    except ImportError:
        missing.append("websocket-client")
    
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\n💡 Install them with:")
        print(f"   pip install {' '.join(missing)}")
        print("\n   Or install all requirements:")
        nkosi_req = os.path.join(nkosi_path, "requirements.txt")
        if os.path.exists(nkosi_req):
            print(f"   pip install -r {nkosi_req}")
        return False
    
    return True


def main():
    """Main entry point"""
    global processes
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 70)
    print("🤖 PIPER ROBOT - POSE TO ROBOT PIPELINE")
    print("=" * 70)
    
    # Check dependencies first
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return
    
    print("✅ All dependencies found!")
    
    print("\nThis will start:")
    print("  1. Simulation server (WebSocket + PyBullet)")
    print("  2. Pose detection app (Webcam + MediaPipe)")
    print("\nPress Ctrl+C to stop both components.\n")
    print("=" * 70)
    
    # Give a moment for user to read
    time.sleep(2)
    
    # Start simulation server
    server_proc = run_simulation_server()
    if server_proc:
        processes.append(server_proc)
    
    # Wait for server to initialize
    print("\n⏳ Waiting for server to initialize...")
    time.sleep(4)
    
    # Check if server is still running
    if server_proc and server_proc.poll() is not None:
        print("❌ Server failed to start. Check errors above.")
        signal_handler(None, None)
        return
    
    print("✅ Server started successfully!\n")
    
    # Start pose app
    pose_proc = run_pose_app()
    if pose_proc:
        processes.append(pose_proc)
    
    print("\n✅ Both components started!")
    print("   - Simulation window should appear")
    print("   - Pose detection window should appear")
    print("\n📋 Instructions:")
    print("   1. In the pose app: Calibrate by holding arm straight & horizontal")
    print("   2. Click 'Connect' to connect to the simulation server")
    print("   3. Click 'Stream' to start sending pose data")
    print("   4. Move your arm to control the robot in simulation")
    print("\nPress Ctrl+C to stop everything.\n")
    
    # Wait for processes to finish
    try:
        while True:
            # Check if any process died
            for proc in processes:
                if proc.poll() is not None:
                    print(f"\n⚠️  Process {proc.pid} exited unexpectedly")
                    signal_handler(None, None)
                    return
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    multiprocessing.freeze_support()
    main()
