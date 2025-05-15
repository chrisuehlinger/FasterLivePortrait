# prepare_source CLI Documentation

This script provides a command-line interface for preprocessing a source image or video using the `FasterLivePortraitPipeline.prepare_source` method. It extracts all relevant per-frame and per-face information and saves it to a serialized file (`.fsp` or `.pkl`) for later use.

## Usage
```
prepare_source.py --src <SOURCE_PATH> --output <OUTPUT_PATH> --cfg <CONFIG_YAML> [--animal] [--realtime]
```

### Arguments
- `--src`: Path to the source image or video.
- `--output`: Path to the output serialized source data file (e.g. `source.fsp`).
- `--cfg`: Path to the inference configuration YAML file.
- `--animal`: Use animal model flags.
- `--realtime`: Enable realtime cropping of multiple faces.

### Output File Format
See `SRC_DATA_FORMAT.md` for detailed specification of the `.fsp` format.
