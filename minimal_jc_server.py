#!/usr/bin/env python3
"""Minimal Piper REST API - joint control"""

from fastapi import FastAPI
import uvicorn
import time

import time
import math
import re
from piper_sdk import C_PiperInterface, LogLevel

app = FastAPI()

CAN_NAME = "can0"
STEP_DEG = 10.0

# Conversion used by SDK examples: millidegrees
RAD_TO_COUNTS = (1000.0 * 180.0 / math.pi)

def deg_to_counts(deg):
    return int(round(math.radians(deg) * RAD_TO_COUNTS))

def extract_joints(joint_state_obj):
    s = str(joint_state_obj)
    pairs = re.findall(r"Joint\s+(\d+):\s*(-?\d+)", s)
    if len(pairs) < 6:
        raise RuntimeError("Failed to read joint feedback")
    joints = [0]*6
    for i, v in pairs:
        joints[int(i)-1] = int(v)
    return joints

piper = C_PiperInterface(
    can_name=CAN_NAME,
    judge_flag=False,
    can_auto_init=True,
    dh_is_offset=1,
    start_sdk_joint_limit=False,
    start_sdk_gripper_limit=False,
    logger_level=LogLevel.WARNING,
    log_to_file=False,
)

@app.post("/enable")
def enable():

    piper.ConnectPort()
    time.sleep(0.2)

    # IMPORTANT: force remote/slave control
    #piper.MasterSlaveConfig(0xFC, 0, 0, 0)
    #time.sleep(0.2)

    piper.EnableArm(7)
    time.sleep(0.5)

    # Read joints ONCE
    msg = piper.GetArmJointMsgs()
    joints = extract_joints(msg.joint_state)

    print("Current joints:", joints)

    # Target: ONLY joint 1 moves +5 deg
    target = joints.copy()
    target[0] += deg_to_counts(STEP_DEG)

    print("Commanding joint 1 to:", target[0])

    # Put controller into joint position mode (gentle speed)
    piper.MotionCtrl_2(0x01, 0x01, 20)

    # SEND ONCE — no loops
    piper.JointCtrl(
        int(target[0]),
        int(joints[1]),
        int(joints[2]),
        int(joints[3]),
        int(joints[4]),
        int(joints[5]),
    )

    print("Command sent. Do NOT resend.")

    # Wait so you can observe motion
    time.sleep(3.0)

    print("Done.")

@app.post("/move")
def move(joints: list[float]):
    piper.JointCtrl(
        int(joints[0]),
        int(joints[1]),
        int(joints[2]),
        int(joints[3]),
        int(joints[4]),
        int(joints[5]),
    )
    return {"ok": True}

@app.get("/joints")
def get_joints():
    msg = piper.GetArmJointMsgs()
    joints = extract_joints(msg.joint_state)
    return joints

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
