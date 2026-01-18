#!/bin/bash
# Send trajectory to Raspberry Pi
#
# Usage:
#   ./send_to_pi.sh trajectory.json
#   ./send_to_pi.sh trajectory.json --play
#   ./send_to_pi.sh trajectory.json --play --loop

set -e

# Configuration
PI_HOST="172.20.10.12"
PI_USER="robot"
SSH_KEY="$HOME/.ssh/piper_robot"
PI_TRAJ_DIR="~/piper_control/trajectories"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <trajectory.json> [--play] [--loop] [--speed 1.0]"
    echo ""
    echo "Examples:"
    echo "  $0 video_trajectory.json"
    echo "  $0 video_trajectory.json --play"
    echo "  $0 video_trajectory.json --play --loop"
    echo "  $0 video_trajectory.json --play --speed 0.5"
    exit 1
fi

TRAJECTORY_FILE="$1"
shift

# Check if file exists
if [ ! -f "$TRAJECTORY_FILE" ]; then
    echo -e "${RED}Error: Trajectory file not found: $TRAJECTORY_FILE${NC}"
    exit 1
fi

TRAJECTORY_NAME=$(basename "$TRAJECTORY_FILE")

# Validate trajectory before sending
echo -e "${BLUE}Validating trajectory...${NC}"
if python validate_trajectory.py "$TRAJECTORY_FILE"; then
    echo -e "${GREEN}✓ Validation passed${NC}"
else
    echo -e "${RED}✗ Validation failed - trajectory not safe to send${NC}"
    echo ""
    echo "Review errors above or run: python validate_trajectory.py $TRAJECTORY_FILE"
    exit 1
fi
echo ""

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}SENDING TRAJECTORY TO RASPBERRY PI${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "  File: $TRAJECTORY_FILE"
echo "  Pi: $PI_USER@$PI_HOST"
echo ""

# Create trajectories directory on Pi
echo -e "${GREEN}[1/3]${NC} Creating directory on Pi..."
ssh -i "$SSH_KEY" "$PI_USER@$PI_HOST" "mkdir -p $PI_TRAJ_DIR"

# Copy trajectory file
echo -e "${GREEN}[2/3]${NC} Copying trajectory file..."
scp -i "$SSH_KEY" "$TRAJECTORY_FILE" "$PI_USER@$PI_HOST:$PI_TRAJ_DIR/$TRAJECTORY_NAME"

# Show file info
FILE_SIZE=$(wc -c < "$TRAJECTORY_FILE" | tr -d ' ')
FILE_SIZE_KB=$((FILE_SIZE / 1024))
echo -e "${GREEN}✓${NC} Copied $FILE_SIZE_KB KB"

# Parse frames count
FRAMES=$(grep -o '"frames"' "$TRAJECTORY_FILE" | wc -l)
echo -e "${GREEN}✓${NC} Trajectory ready on Pi"

echo ""
echo -e "${BLUE}================================================${NC}"

# If --play flag is passed, play the trajectory
if [ "$1" == "--play" ]; then
    shift
    PLAY_ARGS="$@"
    
    echo -e "${GREEN}[3/3]${NC} Starting playback on Pi..."
    echo ""
    
    ssh -i "$SSH_KEY" -t "$PI_USER@$PI_HOST" \
        "cd ~/piper_control && \
         source venv/bin/activate && \
         python arm_control/trajectory_player.py $PI_TRAJ_DIR/$TRAJECTORY_NAME $PLAY_ARGS"
else
    echo ""
    echo "Trajectory copied successfully!"
    echo ""
    echo "To play on Pi, run:"
    echo "  ssh -i $SSH_KEY $PI_USER@$PI_HOST"
    echo "  cd ~/piper_control && source venv/bin/activate"
    echo "  python arm_control/trajectory_player.py $PI_TRAJ_DIR/$TRAJECTORY_NAME"
    echo ""
    echo "Or use:"
    echo "  ./send_to_pi.sh $TRAJECTORY_FILE --play"
fi
