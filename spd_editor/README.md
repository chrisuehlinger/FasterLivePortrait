# SPD Editor

The Source Portrait Descriptor (SPD) Editor is a tool for creating, viewing, editing, and validating SPD files. SPD is a file format designed to store facial portrait data for use with facial animation systems like FasterLivePortrait.

## Features

- Create SPD files from source images or videos
- View and edit SPD file contents
- Validate SPD files for correctness and completeness
- Extract data from SPD files
- Integrate with FasterLivePortrait for seamless animation workflow

## Docker Installation

The easiest way to use the SPD Editor is through Docker, which packages all dependencies and provides both GUI and CLI interfaces.

### Requirements

- Docker
- X11 for GUI mode (on Linux and macOS)

### Running with Docker

The included `run.sh` script makes it easy to build and run the SPD Editor in a Docker container.

#### Basic Usage

```bash
# Run the GUI (default)
./run.sh

# Run in CLI mode
./run.sh --cli

# Get help
./run.sh --help
```

#### Examples

```bash
# Mount a specific directory containing your SPD files
./run.sh --data=/path/to/my/spd/files

# Run a specific CLI command
./run.sh --cli -- spd-editor info /data/my_portrait.spd

# Create a new SPD file
./run.sh --cli -- spd-editor create /data/input_image.jpg /data/output.spd
```

### Docker X11 Setup

For GUI mode, you need X11 forwarding:

#### Linux

```bash
xhost +local:docker
./run.sh
```

#### macOS

1. Install XQuartz
2. Allow connections from network clients in XQuartz preferences
3. Restart XQuartz
4. Run:
```bash
IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
xhost + $IP
./run.sh
```

#### Windows

Use WSL2 with an X11 server like VcXsrv or Xming.

## Manual Installation

If you prefer not to use Docker, you can install the SPD Editor directly:

```bash
# Clone the repository
git clone https://github.com/username/spd-editor.git
cd spd-editor

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Run the editor
python -m spd_editor.gui  # For GUI mode
python -m spd_editor.cli  # For CLI mode
```

## Integration with FasterLivePortrait

The SPD Editor integrates with FasterLivePortrait to provide:

1. Direct loading of SPD files as source inputs
2. Export of processed source data as SPD files
3. Performance improvements by skipping face detection and analysis

### Command-line options

```bash
# Using SPD file as source input
python run.py --src_spd face.spd --dri_video video.mp4

# Exporting source data as SPD file
python run.py --src_image image.jpg --export_spd face.spd
```

## Documentation

For more detailed documentation, including the SPD file format specification, API reference, and tutorials, please see the [documentation](docs/index.md).

## License

[License information here]