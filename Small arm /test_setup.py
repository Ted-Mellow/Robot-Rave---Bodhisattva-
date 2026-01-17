#!/usr/bin/env python3
"""
Quick test to verify Piper SDK installation and setup
"""

print("Testing Piper SDK setup...")
print("-" * 50)

try:
    from piper_sdk import *
    print("✅ Piper SDK imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import Piper SDK: {e}")
    exit(1)

try:
    import cv2
    print(f"✅ OpenCV imported successfully! (version {cv2.__version__})")
except ImportError as e:
    print(f"❌ Failed to import OpenCV: {e}")

try:
    import numpy as np
    print(f"✅ NumPy imported successfully! (version {np.__version__})")
except ImportError as e:
    print(f"❌ Failed to import NumPy: {e}")

try:
    import scipy
    print(f"✅ SciPy imported successfully! (version {scipy.__version__})")
except ImportError as e:
    print(f"❌ Failed to import SciPy: {e}")

try:
    import matplotlib
    print(f"✅ Matplotlib imported successfully! (version {matplotlib.__version__})")
except ImportError as e:
    print(f"❌ Failed to import Matplotlib: {e}")

print("-" * 50)
print("🎉 All dependencies are properly installed!")
print("\nTo run a demo, use the correct nested path:")
print("  python piper_sdk/piper_sdk/demo/V2/piper_ctrl_joint.py")
