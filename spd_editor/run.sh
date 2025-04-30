#!/bin/bash
set -e

# Configuration
IMAGE_NAME="spd-editor"
IMAGE_TAG="latest"
CONTAINER_NAME="spd-editor-container"
GUI_MODE=true
MOUNT_DIR="$(pwd)/data"

# Process command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --cli)
      GUI_MODE=false
      shift
      ;;
    --gui)
      GUI_MODE=true
      shift
      ;;
    --data=*)
      MOUNT_DIR="${1#*=}"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS] [-- COMMAND]"
      echo ""
      echo "Options:"
      echo "  --cli           Run in CLI mode (without GUI)"
      echo "  --gui           Run in GUI mode (default)"
      echo "  --data=DIR      Mount the specified directory as /data in the container"
      echo "  --              Pass remaining arguments as command to the container"
      echo ""
      echo "Examples:"
      echo "  $0                           # Run the GUI"
      echo "  $0 --cli -- spd-editor info myfile.spd   # Run CLI command"
      echo "  $0 --data=/path/to/files     # Mount custom directory"
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if nvidia-smi exists to determine if NVIDIA drivers are installed
if command -v nvidia-smi &> /dev/null; then
  echo "NVIDIA GPU detected, enabling GPU support"
  NVIDIA_OPTS="--gpus all"
else
  echo "Warning: No NVIDIA GPU detected, running in CPU-only mode"
  NVIDIA_OPTS=""
fi

# Create the data directory if it doesn't exist
mkdir -p "$MOUNT_DIR"
echo "Using data directory: $MOUNT_DIR"

# Build the Docker image
echo "Building the SPD Editor image..."
docker build -t "$IMAGE_NAME:$IMAGE_TAG" .

# Determine the command to run
if [ "$GUI_MODE" = true ] && [ $# -eq 0 ]; then
  # Default command for GUI mode
  CMD=("python" "-m" "spd_editor.gui")
  DOCKER_OPTS="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix"
elif [ $# -gt 0 ]; then
  # User-specified command
  CMD=("$@")
else
  # Default for CLI mode
  CMD=("python" "-m" "spd_editor.cli")
fi

# Check if container is already running and remove it
if docker ps -a | grep -q "$CONTAINER_NAME"; then
  echo "Removing existing container..."
  docker rm -f "$CONTAINER_NAME" > /dev/null
fi

# Run the container with NVIDIA GPU support when available
echo "Starting SPD Editor..."
docker run --rm -it \
  --name "$CONTAINER_NAME" \
  -v "$MOUNT_DIR:/data" \
  $DOCKER_OPTS \
  $NVIDIA_OPTS \
  "$IMAGE_NAME:$IMAGE_TAG" "${CMD[@]}"

echo "SPD Editor session ended."