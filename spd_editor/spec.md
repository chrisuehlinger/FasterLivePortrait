# Source Portrait Descriptor (SPD) Editor Specification

## Overview
The Source Portrait Descriptor (SPD) Editor is a standalone tool for creating, viewing, and editing SPD files. SPD files store pre-processed facial data for use with the FasterLivePortrait system, eliminating the need for face detection and analysis during animation.

## Requirements

### Functional Requirements
1. Create SPD files from source images
2. Open and parse existing SPD files
3. Visualize source image with facial landmarks overlay
4. Edit key facial parameters
5. Save modified SPD files
6. Export original or processed images from SPD files

### Non-Functional Requirements
1. Minimal external dependencies beyond those required by FasterLivePortrait
2. Support for Windows, macOS, and Linux
3. Command-line interface (CLI) for automation
4. Simple graphical user interface (GUI) for interactive use
5. Clear error messages and validation

## SPD File Format

### Header Section
- **Magic Bytes**: "SPDV" (4 bytes) - identifies the file as an SPD file
- **Version**: 1 (4 bytes) - file format version
- **Creation Timestamp**: Unix timestamp (8 bytes) - when the SPD file was created
- **Flags**: (4 bytes) - bitmap of flags:
  - Bit 0: Has original image
  - Bit 1: Has cropped image
  - Bit 2: Has resized image (256x256)
  - Bit 3: Has landmark data
  - Bit 4: Has motion parameters
  - Bit 5: Has appearance features
  - Bit 6: Has transformation matrices
  - Bit 7: Has mask data
  - Bits 8-31: Reserved for future use

### Image Data Section
- **Original Image**:
  - Width (4 bytes)
  - Height (4 bytes)
  - Channels (1 byte)
  - Image data (width * height * channels bytes) - RGB format
- **Cropped Image**:
  - Width (4 bytes)
  - Height (4 bytes)
  - Channels (1 byte)
  - Image data (width * height * channels bytes) - RGB format
- **Resized Image (256x256)**:
  - Channels (1 byte)
  - Image data (256 * 256 * channels bytes) - RGB format

### Facial Landmarks Section
- **Number of landmarks**: (4 bytes) - typically 106 for the model
- **Landmark data**: (number of landmarks * 2 * 4 bytes) - XY coordinates as float32
- **Cropped landmark data**: (number of landmarks * 2 * 4 bytes) - landmarks in crop space
- **Resized landmark data**: (number of landmarks * 2 * 4 bytes) - landmarks in 256x256 space

### Motion Parameters Section
- **Pitch**: (4 bytes) - float32
- **Yaw**: (4 bytes) - float32
- **Roll**: (4 bytes) - float32
- **Translation**: (3 * 4 bytes) - float32 array [tx, ty, tz]
- **Expression**: (number of expression parameters * 4 bytes) - float32 array
- **Scale**: (4 bytes) - float32
- **Keypoints**: (number of keypoints * 3 * 4 bytes) - float32 array of 3D keypoints

### Appearance Features Section
- **Feature dimensions**: (4 bytes) - number of dimensions
- **Feature data**: (feature dimensions * 4 bytes) - extracted appearance features (float32 array)

### Transformation Matrices Section
- **Crop to original transformation matrix**: (3 * 3 * 4 bytes) - float32 3x3 matrix (M_c2o)
- **Original to crop transformation matrix**: (3 * 3 * 4 bytes) - float32 3x3 matrix (inverse of M_c2o)

### Mask Data Section
- **Mask dimensions**:
  - Width (4 bytes)
  - Height (4 bytes)
- **Mask data**: (width * height * 4 bytes) - float32 alpha mask for blending

### Additional Flags Section
- **Lips normalized flag**: (1 byte) - boolean
- **Eye ratio**: (4 bytes) - float32 eye close ratio
- **Lip ratio**: (4 bytes) - float32 lip close ratio
- **Lip delta data**: (optional, if lips normalized flag is true)
  - Size: (4 bytes)
  - Data: (size * 4 bytes) - float32 array for lip delta animation

## Architecture

### Directory Structure
```
spd_editor/
├── __init__.py
├── main.py                  # Main entry point
├── cli.py                   # Command line interface
├── gui.py                   # Graphical user interface
├── spd/
│   ├── __init__.py
│   ├── format.py            # SPD file format specification
│   ├── reader.py            # SPD file reader
│   ├── writer.py            # SPD file writer
│   └── validator.py         # SPD file validation
├── analysis/
│   ├── __init__.py
│   ├── face_detector.py     # Face detection wrapper
│   ├── landmark_extractor.py # Landmark extraction
│   └── feature_extractor.py # Feature extraction
└── utils/
    ├── __init__.py
    ├── visualization.py     # Visualization utilities
    └── transformation.py    # Matrix transformation utilities
```

### Component Descriptions

#### SPD Module
- **format.py**: Defines constants and structures for the SPD format
- **reader.py**: Handles reading and parsing SPD files
- **writer.py**: Handles creating and writing SPD files
- **validator.py**: Validates SPD file integrity

#### Analysis Module
- **face_detector.py**: Wrapper for face detection functionality
- **landmark_extractor.py**: Extracts facial landmarks from images
- **feature_extractor.py**: Extracts appearance features from images

#### Utils Module
- **visualization.py**: Functions for visualizing landmarks, meshes, etc.
- **transformation.py**: Functions for handling transformation matrices

#### Main Interface
- **main.py**: Entry point, argument parsing, and routing
- **cli.py**: Command-line interface implementation
- **gui.py**: Graphical user interface implementation

## Implementation Plan

### Phase 1: Core Library Implementation
1. Implement SPD format definition
2. Implement SPD reader and writer
3. Implement SPD validator
4. Create utility functions for data conversion
5. Implement face analysis wrappers using FasterLivePortrait components

### Phase 2: Command-line Interface
1. Implement command-line arguments parsing
2. Implement commands for:
   - Creating SPD files from images
   - Reading and validating SPD files
   - Extracting data from SPD files
   - Simple operations like image export

### Phase 3: Graphical Interface
1. Create a simple GUI with:
   - File operations (open, save, export)
   - Image display with landmark overlay
   - Basic parameter adjustment controls
   - Visualization options

### Phase 4: Testing and Documentation
1. Create unit tests for core components
2. Create integration tests for end-to-end workflows
3. Document API and user workflows
4. Create example usage scripts

## Error Handling Strategy

1. **Validation Checks**:
   - File format validation (magic bytes, version)
   - Data integrity validation (sizes, dimensions)
   - Image format validation
   - Parameter range validation

2. **Error Types**:
   - `SPDFormatError`: For issues with file format
   - `FaceDetectionError`: For face detection failures
   - `ParameterValidationError`: For invalid parameters
   - `IOError`: For file read/write issues

3. **Error Reporting**:
   - Detailed error messages with contextual information
   - Error codes for systematic handling
   - Logging of errors for debugging

## Testing Plan

### Unit Tests
1. SPD Format Tests:
   - Test reading valid SPD files
   - Test reading corrupt SPD files
   - Test writing SPD files
   - Test round-trip conversion

2. Face Analysis Tests:
   - Test face detection on sample images
   - Test landmark extraction
   - Test feature extraction

3. Utility Tests:
   - Test visualization functions
   - Test transformation functions

### Integration Tests
1. End-to-end tests:
   - Create SPD from image and validate content
   - Modify SPD and validate changes
   - Use SPD in FasterLivePortrait pipeline

### User Acceptance Tests
1. CLI functionality tests
2. GUI functionality tests
3. Cross-platform compatibility tests

## FasterLivePortrait Integration

### Changes to FasterLivePortrait
1. Add SPD file support to `prepare_source` method:
   - Detect SPD files based on extension
   - Use SPD reader to load pre-processed data
   - Skip face detection and analysis steps
   
2. Add option to export SPD after source preparation:
   - Add command-line flag for SPD export
   - Add option in web UI for SPD export

## CLI Usage Examples

```bash
# Create SPD file from source image
python -m spd_editor create /path/to/source.jpg -o source.spd

# Validate SPD file
python -m spd_editor validate source.spd

# Extract source image from SPD
python -m spd_editor extract source.spd -o extracted_image.jpg

# Show SPD info
python -m spd_editor info source.spd

# Launch GUI
python -m spd_editor gui
```

## GUI Wireframe

```
+---------------------------------------------+
|  SPD Editor                           [X]   |
+---------------------------------------------+
| File | Edit | View | Help                   |
+---------------------------------------------+
|                                             |
|  +-----------------------------------+      |
|  |                                   |      |
|  |                                   |      |
|  |          Source Image             |      |
|  |       with Landmark Overlay       |      |
|  |                                   |      |
|  |                                   |      |
|  +-----------------------------------+      |
|                                             |
|  Parameters:                                |
|  +-----------------------------------+      |
|  | Landmarks: [Show] [Hide]          |      |
|  | Original Image: [Show] [Hide]     |      |
|  |                                   |      |
|  | Expression:                       |      |
|  | [------------------O--------] 0.5 |      |
|  |                                   |      |
|  | Eye Ratio:                        |      |
|  | [-------O-------------------] 0.3 |      |
|  |                                   |      |
|  | Lip Ratio:                        |      |
|  | [----------------O----------] 0.7 |      |
|  +-----------------------------------+      |
|                                             |
|  [Save SPD]  [Export Image]  [Close]        |
|                                             |
+---------------------------------------------+
```

## Dependencies

1. Core Libraries:
   - NumPy: For numerical operations
   - OpenCV: For image processing
   - PyTorch: For neural network models (from FasterLivePortrait)
   
2. GUI Libraries:
   - tkinter: For basic GUI (minimal additional dependency)
   - Pillow: For image display in GUI

3. FasterLivePortrait Components:
   - Face detection models
   - Landmark extraction models
   - Feature extraction models

## Known Limitations

1. Initial version supports only human faces (no animal support)
2. Limited editing capabilities in first release
3. No support for batch processing in initial GUI
4. Requires the same model dependencies as FasterLivePortrait

## Future Enhancements

1. Support for animal faces
2. Advanced parameter editing
3. Batch processing
4. Preview animation using sample driving sequences
5. Standalone mode with minimal dependencies