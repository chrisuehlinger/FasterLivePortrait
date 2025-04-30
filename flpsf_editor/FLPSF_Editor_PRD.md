# FLPSF Editor Project Requirements Document

## 1. Introduction

### 1.1 Purpose
The FLPSF Editor is a specialized tool designed for creating and editing FasterLivePortrait Source Format (FLPSF) files. These files encapsulate all the data needed for FasterLivePortrait to deterministically process source images without relying on automatic face detection, landmark extraction, or parameter calculation.

### 1.2 Project Scope
The FLPSF Editor will provide a comprehensive GUI-based interface for manipulating all aspects of FLPSF files, including facial landmarks, motion parameters, appearance features, and pasteback information. This tool will allow users to create and fine-tune FLPSF files for characters that automatic systems struggle with, like cartoon characters, animals, and stylized faces.

### 1.3 Target Users
- Animation artists working with non-human or stylized characters
- VFX professionals requiring precise control over facial animations
- Developers extending FasterLivePortrait to new character types
- Researchers requiring deterministic facial parameter manipulation

## 2. FLPSF Format Overview

### 2.1 Format Structure
The FasterLivePortrait Source Format (.flpsf) is a binary file format based on NumPy's .npz format, containing:

1. **Source Image Properties**
   - Image path, dimensions, and optional embedded image
   - Image hash for verification

2. **Facial Landmarks**
   - 106-point landmark array (for human faces)
   - Customizable landmark configurations for non-human faces

3. **Motion Parameters**
   - Head pose (pitch, yaw, roll)
   - Translation vector
   - Expression parameters
   - Scale factors
   - 3D keypoints

4. **Derived Parameters**
   - Rotation matrices
   - Appearance feature vectors
   - Transformed keypoints
   - Canonical keypoints

5. **Normalization Parameters**
   - Lip delta values
   - Eye delta values
   - Flags for normalization

6. **Pasteback Information**
   - Transformation matrices
   - Mask data

7. **Metadata**
   - Format version
   - Creation date
   - Software information

### 2.2 Integration with FasterLivePortrait
FLPSF files can be directly used with FasterLivePortrait via:
- Command-line interface with the `--flpsf` parameter
- Programmatic API through the `prepare_source_from_flpsf()` method

## 3. Functional Requirements

### 3.1 File Management
| ID | Requirement | Priority |
|----|-------------|----------|
| F1.1 | Open and load existing FLPSF files | High |
| F1.2 | Save edited FLPSF files | High |
| F1.3 | Import source images to create new FLPSF files | High |
| F1.4 | Export subsets of FLPSF data (e.g., landmarks only) | Medium |
| F1.5 | Batch processing of multiple files | Low |
| F1.6 | File validation and integrity checking | Medium |

### 3.2 Image Processing
| ID | Requirement | Priority |
|----|-------------|----------|
| F2.1 | Display and zoom source images | High |
| F2.2 | Adjust image contrast/brightness for better visibility | Medium |
| F2.3 | Crop and resize images | Medium |
| F2.4 | Apply filters for edge detection to assist landmark placement | Medium |
| F2.5 | Toggle between original and processed image views | Medium |

### 3.3 Landmark Editing
| ID | Requirement | Priority |
|----|-------------|----------|
| F3.1 | Manually place and adjust 106 facial landmarks | High |
| F3.2 | Automatic landmark detection as starting point (using existing models) | Medium |
| F3.3 | Landmark templates for common character types | Medium |
| F3.4 | Group selection and manipulation of landmark points | High |
| F3.5 | Symmetry tools for landmark placement | Medium |
| F3.6 | Visualization of landmark connections | High |
| F3.7 | Undo/redo functionality for landmark editing | High |

### 3.4 Parameter Editing
| ID | Requirement | Priority |
|----|-------------|----------|
| F4.1 | Edit head pose parameters (pitch, yaw, roll) | High |
| F4.2 | Adjust translation vectors | High |
| F4.3 | Modify expression parameters with visualization | High |
| F4.4 | Edit scale factors | High |
| F4.5 | 3D visualization and editing of keypoints | Medium |
| F4.6 | Real-time parameter validation | Medium |
| F4.7 | Parameter presets for common expressions and poses | Medium |

### 3.5 Feature Extraction
| ID | Requirement | Priority |
|----|-------------|----------|
| F5.1 | Option to use FasterLivePortrait models for initial feature extraction | High |
| F5.2 | Manual adjustment of extracted features | Medium |
| F5.3 | Custom feature creation for non-human characters | Medium |
| F5.4 | Feature visualization tools | Low |
| F5.5 | Compare extracted vs. manually created features | Low |

### 3.6 Pasteback Configuration
| ID | Requirement | Priority |
|----|-------------|----------|
| F6.1 | Create and edit pasteback masks | High |
| F6.2 | Adjust transformation matrices | Medium |
| F6.3 | Preview pasteback results | High |
| F6.4 | Brush tools for refining masks | Medium |
| F6.5 | Automatic mask generation with manual refinement | Medium |

### 3.7 Preview and Testing
| ID | Requirement | Priority |
|----|-------------|----------|
| F7.1 | Preview animation with pre-recorded driving videos | High |
| F7.2 | Live preview with webcam input | Medium |
| F7.3 | A/B comparison of different parameter sets | Medium |
| F7.4 | Export preview animations | Low |
| F7.5 | Performance metrics for parameter quality | Low |

### 3.8 Integration
| ID | Requirement | Priority |
|----|-------------|----------|
| F8.1 | Direct integration with FasterLivePortrait | High |
| F8.2 | Command-line interface for batch operations | Medium |
| F8.3 | Parameter exchange with other animation tools | Low |
| F8.4 | Plugin system for custom processors | Low |

## 4. Non-Functional Requirements

### 4.1 Usability
| ID | Requirement | Priority |
|----|-------------|----------|
| NF1.1 | Intuitive user interface for artists without technical expertise | High |
| NF1.2 | Context-sensitive help and tooltips | Medium |
| NF1.3 | Customizable shortcuts for frequent operations | Medium |
| NF1.4 | Dark mode and theme options | Low |
| NF1.5 | Progress indicators for long-running operations | Medium |

### 4.2 Performance
| ID | Requirement | Priority |
|----|-------------|----------|
| NF2.1 | Responsive UI during parameter editing | High |
| NF2.2 | Efficient processing of high-resolution images | Medium |
| NF2.3 | GPU acceleration for preview features when available | Medium |
| NF2.4 | Handle files with multiple faces efficiently | Low |
| NF2.5 | Memory optimization for large files | Medium |

### 4.3 Compatibility
| ID | Requirement | Priority |
|----|-------------|----------|
| NF3.1 | Cross-platform compatibility (Windows, macOS, Linux) | High |
| NF3.2 | Support for multiple image formats (JPG, PNG, BMP, etc.) | High |
| NF3.3 | Forward compatibility with future FLPSF versions | Medium |
| NF3.4 | Export to other facial animation formats | Low |

### 4.4 Security
| ID | Requirement | Priority |
|----|-------------|----------|
| NF4.1 | Safe handling of embedded image data | Medium |
| NF4.2 | Proper file permissions for saved data | Medium |
| NF4.3 | Backup creation before overwriting files | Medium |

## 5. User Interface Requirements

### 5.1 Main Application Window
- Split-view design with image workspace and parameter panels
- Toolbar for common operations
- Status bar for feedback and context information
- Dockable panels for specialized functions

### 5.2 Image Workspace
- Zoomable, pannable image view
- Rulers and measurement tools
- Grid overlay option
- Multiple view modes (landmarks, wireframe, solid)

### 5.3 Parameter Editing Panels
- Grouped by parameter type (landmarks, motion, features, etc.)
- Slider controls for numeric parameters
- Direct numeric input option
- Visual feedback for parameter changes

### 5.4 Landmark Editor
- Point selection and dragging interface
- Visual indicators for point types
- Connection visualization
- Group selection tools

### 5.5 3D Preview Panel
- 3D visualization of extracted face model
- Rotation and zoom controls
- Toggle for wireframe/solid view
- Parameter effect visualization

## 6. Technical Requirements

### 6.1 Development Environment
- Python 3.8+ with PyQt5/PySide6 for UI
- NumPy for data handling
- OpenCV for image processing
- Optional PyTorch integration for model-based features
- Pytest for automated testing

### 6.2 Dependencies
- Access to FasterLivePortrait models for parameter extraction
- 3D visualization libraries
- Image processing utilities

### 6.3 Architecture
- Modular design with separation of UI and processing logic
- Plugin architecture for extensibility
- Clear API for integration with FasterLivePortrait

## 7. Implementation Phases

### 7.1 Phase 1: Core Functionality (2-4 weeks)
- Basic UI framework
- File loading and saving
- Image display
- Simple landmark editing
- Basic parameter visualization

### 7.2 Phase 2: Advanced Editing (4-6 weeks)
- Complete landmark editing suite
- Motion parameter editing
- Template system
- Basic preview functionality
- UI refinements

### 7.3 Phase 3: Integration and Testing (3-4 weeks)
- FasterLivePortrait integration
- Advanced preview features
- Performance optimization
- User testing and feedback
- Bug fixing and refinement

### 7.4 Phase 4: Advanced Features (Optional, 4-6 weeks)
- 3D visualization
- Advanced pasteback tools
- Batch processing
- Plugin system
- Export to other formats

## 8. Testing Strategy

### 8.1 Unit Testing
- Test framework for core functionality
- Parameter validation tests
- File format integrity tests

### 8.2 Integration Testing
- FasterLivePortrait compatibility tests
- End-to-end workflow tests
- Cross-platform testing

### 8.3 User Acceptance Testing
- Artist workflow testing
- Performance testing with various character types
- Usability testing with target users

## 9. Documentation Requirements

### 9.1 User Documentation
- Getting started guide
- Tutorial for complete workflow
- Reference manual for all features
- Troubleshooting guide

### 9.2 Technical Documentation
- Architecture overview
- API documentation
- Plugin development guide
- FLPSF format specification

## 10. Future Considerations

- Web-based version for browser access
- Cloud integration for collaborative editing
- Custom model training for specialized character types
- Real-time collaboration features
- Mobile companion app for quick edits

## Appendix A: FLPSF Format Specification

```
FasterLivePortrait Source Format (FLPSF) v1.0.0
------------------------------------------------

File Extension: .flpsf
Base Format: NumPy .npz (compressed)

Required Fields:
- version: string (format version, e.g., "1.0.0")
- image_path: string (original image path)
- landmarks: numpy.ndarray (Nx2, typically 106x2 for humans)
- pitch: numpy.ndarray (1x1, head pitch angle)
- yaw: numpy.ndarray (1x1, head yaw angle)
- roll: numpy.ndarray (1x1, head roll angle)
- t: numpy.ndarray (1x3, translation vector)
- exp: numpy.ndarray (1x21x3, expression parameters)
- scale: numpy.ndarray (1x1, scale factor)
- kp: numpy.ndarray (1x20x3, 3D keypoints)
- R_s: numpy.ndarray (1x3x3, rotation matrix)
- f_s: numpy.ndarray (1x256, appearance feature vector)
- x_s: numpy.ndarray (1x20x3, transformed keypoints)
- x_c_s: numpy.ndarray (1x20x3, canonical keypoints)

Optional Fields:
- creation_date: string (ISO format)
- image_hash: string (SHA-256 hash)
- image_dimensions: numpy.ndarray (width, height)
- embedded_image: bytes (compressed image data)
- is_animal: boolean
- lip_delta_before_animation: numpy.ndarray
- flag_lip_zero: boolean
- mask_ori_float: numpy.ndarray
- M: numpy.ndarray (3x3, transformation matrix)
- software: string (creator software info)
- config: dict (configuration used during extraction)
```

## Appendix B: User Personas

### B.1 Professional Animator (Primary Persona)
- **Name**: Alex
- **Role**: Character Animator at an animation studio
- **Goals**: Create precise facial animations for stylized characters
- **Challenges**: Automatic systems fail with non-photorealistic characters
- **Needs**: Precise control, efficiency, intuitive tools

### B.2 Independent Game Developer
- **Name**: Jamie
- **Role**: Solo game developer creating an indie title
- **Goals**: Animate a small cast of unique characters
- **Challenges**: Limited technical knowledge of facial animation systems
- **Needs**: Templates, presets, guided workflows

### B.3 VFX Technical Director
- **Name**: Sam
- **Role**: Technical Director at a VFX studio
- **Goals**: Integrate custom characters into an existing pipeline
- **Challenges**: Ensuring consistency and deterministic results
- **Needs**: Batch processing, command-line tools, technical control

### B.4 Researcher
- **Name**: Dr. Chen
- **Role**: Computer Vision Researcher
- **Goals**: Test facial animation algorithms with controlled parameters
- **Challenges**: Need for reproducible results and parameter isolation
- **Needs**: Parameter manipulation, data export, comparative analysis