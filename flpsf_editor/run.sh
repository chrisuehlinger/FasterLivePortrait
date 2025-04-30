#!/bin/bash

# Make script exit when any command fails
set -e

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Set script directory as working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Make entrypoint executable
chmod +x entrypoint.sh

# Build the Docker image if it doesn't exist or --build flag is provided
if [[ ! $(docker images -q flpsf-editor:latest 2> /dev/null) ]] || [[ "$*" == *"--build"* ]]; then
    echo "Building FLPSF Editor Docker image..."
    docker build -t flpsf-editor:latest .
fi

# Get arguments to pass to main.py
ARGS=""
for arg in "$@"; do
    if [[ "$arg" != "--build" ]]; then
        ARGS="$ARGS $arg"
    fi
done

# Create data directory if it doesn't exist
DATA_DIR="$HOME/flpsf_data"
mkdir -p "$DATA_DIR"

# Setup X11 forwarding for GUI applications
if [[ -z "${DISPLAY}" ]]; then
    echo "Error: DISPLAY environment variable not set. Make sure X server is running."
    exit 1
fi

# Improved X11 socket handling
xhost +local:docker || {
    echo "Warning: Could not set xhost permissions. If you experience display issues, run 'xhost +local:docker' manually."
}

# Run the container with improved X11 forwarding
echo "Starting FLPSF Editor container..."
docker run --rm -it \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$SCRIPT_DIR/../":/app \
    -v "$DATA_DIR":/data \
    -e DISPLAY="$DISPLAY" \
    -e XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
    --ipc=host \
    --net=host \
    --security-opt=apparmor:unconfined \
    flpsf-editor:latest $ARGS

echo "FLPSF Editor container stopped."