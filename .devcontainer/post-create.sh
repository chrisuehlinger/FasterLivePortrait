#!/bin/bash
set -e

echo "Running post-creation setup..."

WORKSPACE_ROOT=/workspaces/FasterLivePortrait

# Install project requirements
echo "Installing Python requirements..."
if [ -f "$WORKSPACE_ROOT/requirements.txt" ]; then
  pip install -r "$WORKSPACE_ROOT/requirements.txt"
else
  echo "Warning: requirements.txt not found."
fi

pushd /workspaces/FasterLivePortrait/src/models/XPose/models/UniPose/ops
python setup.py build install
popd

# Check for NVIDIA GPU
echo "Checking NVIDIA GPU..."
nvidia-smi || echo "Warning: NVIDIA GPU not detected or driver issue"

# Verify key dependencies
echo "Verifying Python dependencies..."
pip list | grep -E 'tensorrt|onnxruntime|torch'

# Test X11 support
echo "Testing X11 support..."
if [ -n "$DISPLAY" ]; then
  echo "DISPLAY is set to $DISPLAY"
  xeyes &
  sleep 3
  killall xeyes || true
else
  echo "Warning: DISPLAY environment variable not set. X11 forwarding may not work."
fi

# Test Docker-in-Docker
echo "Testing Docker-in-Docker..."
docker --version || echo "Warning: Docker not accessible. Make sure the Docker socket is properly mounted."

echo "Post-creation setup complete!"