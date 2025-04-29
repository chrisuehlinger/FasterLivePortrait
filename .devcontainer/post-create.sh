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

# Install TensorRT - mirror how the original Dockerfile handles this
echo "Installing TensorRT..."
if [ -f "$WORKSPACE_ROOT/downloads/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz" ]; then
  echo "Found TensorRT tarball, copying and extracting..."
  cp "$WORKSPACE_ROOT/downloads/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz" /tmp/TensorRT.tar
  tar -xf /tmp/TensorRT.tar -C /usr/local/
  mv /usr/local/TensorRT-8.6.1.6 /usr/local/tensorrt
  python --version
  pip3 install /usr/local/tensorrt/python/tensorrt-*-cp310-*.whl
  rm -rf /tmp/TensorRT.tar
  echo 'export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH' >> /etc/bash.bashrc
  export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH
  echo "TensorRT installed successfully."
else
  echo "Warning: TensorRT tarball not found at $WORKSPACE_ROOT/downloads/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz"
fi

# Copy grid-sample3d-trt-plugin - mirror how the original Dockerfile handles this
echo "Setting up grid-sample3d-trt-plugin..."
if [ -d "$WORKSPACE_ROOT/grid-sample3d-trt-plugin" ]; then
  echo "Found grid-sample3d-trt-plugin directory, copying..."
  cp -r "$WORKSPACE_ROOT/grid-sample3d-trt-plugin"/* /opt/grid-sample3d-trt-plugin/
  
  # Build the plugin
  echo "Building grid-sample3d-trt-plugin..."
  cd /opt/grid-sample3d-trt-plugin
  export C_INCLUDE_PATH=/usr/local/tensorrt/include:${C_INCLUDE_PATH}
  export CPP_INCLUDE_PATH=/usr/local/tensorrt/include:${CPP_INCLUDE_PATH}
  export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:${LD_LIBRARY_PATH}
  
  cd build
  cmake .. -DTensorRT_ROOT=/usr/local/tensorrt
  make -j$(nproc)
  
  # Create symlink if necessary
  if [ -f "libgrid_sample3d_plugin.so" ] && [ ! -f "libgrid_sample_3d_plugin.so" ]; then
    ln -s libgrid_sample3d_plugin.so libgrid_sample_3d_plugin.so
  fi
  
  echo "grid-sample3d-trt-plugin built successfully."
else
  echo "Warning: grid-sample3d-trt-plugin directory not found."
fi

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