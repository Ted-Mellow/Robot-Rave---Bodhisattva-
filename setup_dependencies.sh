#!/bin/bash
# Setup script to install dependencies for the pose-to-robot pipeline

echo "🔧 Setting up dependencies for Robot Rave - Bodhisattva"
echo "=================================================="
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected!"
    echo "   It's recommended to use a virtual environment."
    echo "   If you have one, activate it first:"
    echo "     source venv/bin/activate  # or your venv path"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Virtual environment detected: ${VIRTUAL_ENV:-${CONDA_DEFAULT_ENV}}"
fi

echo ""
echo "📦 Installing dependencies from nkosi/requirements.txt..."
if [ -f "nkosi/requirements.txt" ]; then
    pip install -r nkosi/requirements.txt
    echo "✅ Dependencies installed!"
else
    echo "❌ nkosi/requirements.txt not found!"
    echo "   Installing core dependencies manually..."
    pip install mediapipe opencv-python websocket-client python-dotenv
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "You can now run: python run_pose_to_robot.py"
