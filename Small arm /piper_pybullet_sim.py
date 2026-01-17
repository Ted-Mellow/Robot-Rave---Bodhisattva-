#!/usr/bin/env python3
"""
Piper PyBullet Simulation - Compatibility Wrapper
This module provides backwards compatibility by importing from the corrected simulation
"""

# Import from the corrected simulation module
import sys
import os

# Add simulation directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation'))

from piper_simultion_corrected import PiperSimulation

__all__ = ['PiperSimulation']
