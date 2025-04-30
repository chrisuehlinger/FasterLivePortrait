# FLPSF Editor: FasterLivePortrait Source Format Editor

A specialized editor for creating, editing, and manipulating FasterLivePortrait Source Format (FLPSF) files.

## Overview

FLPSF Editor is a dedicated tool for working with FLPSF files, which store all the facial parameters needed for deterministic animation in FasterLivePortrait. This editor enables you to create, modify, and fine-tune all aspects of the source images without relying on automatic detection models.

## Usage

### Direct Run

```bash
python main.py [--file FILE_PATH] [--image IMAGE_PATH] [--config CONFIG_PATH] [--debug]
```

### Using Docker (Recommended)

We provide a Docker-based setup for consistent environment across different systems:

```bash
# Make the script executable
chmod +x run.sh

# Build and start the editor (first run or after changes)
./run.sh --build

# Start editor with existing image
./run.sh

# Open a specific FLPSF file
./run.sh --file /data/your_file.flpsf

# Open with a source image
./run.sh --image /data/your_image.jpg
```

All files in the `/data` directory within the container are mounted from `$HOME/flpsf_data` on your host system.

## Features

- Load and save FLPSF files
- Import source images and create new FLPSF files
- Edit all facial parameters with real-time visualization
- Manual landmark placement and adjustment
- Direct integration with FasterLivePortrait
- Preview animation effects

## Requirements

### Native Installation
- Python 3.8+
- PyQt5/PySide6
- NumPy
- OpenCV
- PyTorch (optional, for preview features)

### Docker Installation
- Docker
- X11 server (for Linux GUI forwarding)
- XQuartz (for macOS)
- VcXsrv or similar (for Windows)

## License

Same as FasterLivePortrait