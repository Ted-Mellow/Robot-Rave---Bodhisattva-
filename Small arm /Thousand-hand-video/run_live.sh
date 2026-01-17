#!/bin/bash
# Wrapper script to run with correct Python version
# This ensures we use the system Python with MediaPipe 0.10.9

# Use absolute path to system Python
/Library/Developer/CommandLineTools/usr/bin/python3 run_pipeline.py "$@"
