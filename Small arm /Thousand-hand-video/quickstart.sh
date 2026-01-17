#!/bin/bash
# Quickstart script for Thousand-Hand Bodhisattva pose detection
# Runs the complete pipeline with recommended settings

set -e

echo "=========================================="
echo "Thousand-Hand Bodhisattva - Quick Start"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    echo "   Please install Python 3.7 or later"
    exit 1
fi

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
python3 -c "import cv2, mediapipe, numpy, pybullet" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing"
    echo ""
    read -p "Install dependencies now? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Installing dependencies..."
        pip install -r requirements.txt
    else
        echo "❌ Cannot proceed without dependencies"
        echo "   Run: pip install -r requirements.txt"
        exit 1
    fi
fi

echo "✅ Dependencies OK"
echo ""

# Check if video exists
VIDEO="Cropped_thousandhand.mp4"

if [ ! -f "$VIDEO" ]; then
    echo "❌ Error: Video file not found"
    echo "   Expected: $VIDEO"
    echo "   Please ensure the video is in the current directory"
    exit 1
fi

echo "✅ Video found: $VIDEO"
echo ""

# Ask user for options
echo "Quick Start Options:"
echo ""
echo "1. Quick test (5 seconds, greyscale)"
echo "2. Medium test (30 seconds, greyscale)"
echo "3. Full video (greyscale, may take a while)"
echo "4. Custom options"
echo ""

read -p "Choose option [1-4]: " -n 1 -r
echo ""
echo ""

case $REPLY in
    1)
        echo "🚀 Running quick test (5 seconds)..."
        python3 run_pipeline.py --end 5 --greyscale
        ;;
    2)
        echo "🚀 Running medium test (30 seconds)..."
        python3 run_pipeline.py --end 30 --greyscale
        ;;
    3)
        echo "🚀 Running full video processing..."
        echo "⚠️  This may take several minutes..."
        python3 run_pipeline.py --greyscale
        ;;
    4)
        echo "Running with custom options..."
        echo ""
        read -p "Start time (seconds) [0]: " START
        START=${START:-0}
        read -p "End time (seconds) [10]: " END
        END=${END:-10}
        read -p "Use greyscale? [Y/n]: " GREY

        GREY_FLAG=""
        if [[ ! $GREY =~ ^[Nn]$ ]]; then
            GREY_FLAG="--greyscale"
        fi

        echo ""
        echo "🚀 Running pipeline..."
        python3 run_pipeline.py --start $START --end $END $GREY_FLAG
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Done!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - pose_data.json (pose detection data)"
echo "  - bodhisattva_dance.csv (robot choreography)"
echo ""
echo "To replay the simulation:"
echo "  python3 simulate_dance.py bodhisattva_dance.csv"
echo ""
echo "To see all options:"
echo "  python3 run_pipeline.py --help"
echo ""
