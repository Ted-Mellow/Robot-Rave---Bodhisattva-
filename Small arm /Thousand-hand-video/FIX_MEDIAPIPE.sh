#!/bin/bash
# Quick fix for MediaPipe compatibility issue
# This installs the correct version in your conda environment

echo "🔧 Fixing MediaPipe compatibility..."
echo ""
echo "Installing MediaPipe 0.10.9 in conda environment..."
pip install --force-reinstall mediapipe==0.10.9 pybullet

echo ""
echo "✅ Done! Now run:"
echo "   python run_pipeline.py --live --end 10"
