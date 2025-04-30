# SPD Editor - Face Analysis Integration

## Background
Now that we have the core SPD file format functionality, we need to integrate with FasterLivePortrait's face analysis components to extract facial data from source images. This will allow us to create SPD files directly from source images.

## Your Task
Create wrapper modules for FasterLivePortrait's face detection and analysis components so we can leverage them to extract the data needed for SPD files.

## Requirements

1. Create a new directory `analysis` with:
   - `__init__.py` module
   - `face_detector.py` module for face detection functionality
   - `landmark_extractor.py` module for facial landmark extraction
   - `feature_extractor.py` module for appearance feature extraction

2. In `face_detector.py`, implement:
   - A `FaceDetector` class that wraps FasterLivePortrait's face detection functionality
   - Methods to detect faces in images and return bounding boxes and confidence scores
   - Support for different detection models/methods from FasterLivePortrait

3. In `landmark_extractor.py`, implement:
   - A `LandmarkExtractor` class that wraps facial landmark extraction functionality
   - Methods to extract landmarks from detected faces
   - Support for different landmark models/configurations

4. In `feature_extractor.py`, implement:
   - A `FeatureExtractor` class for extracting appearance features used by FasterLivePortrait
   - Methods to extract motion parameters (pitch, yaw, roll, etc.)
   - Methods to generate transformation matrices and other required data

5. Create test files for each module:
   - `tests/test_face_detector.py`
   - `tests/test_landmark_extractor.py`
   - `tests/test_feature_extractor.py`

## Implementation Notes

- Create clean abstractions over FasterLivePortrait's components
- Handle dependency management carefully (don't import everything from FasterLivePortrait)
- Provide fallback options for missing components
- Document the required model files and how to obtain them
- Include configuration options for adapting to different use cases

## Test Requirements
- Test with sample images that have known face characteristics
- Mock FasterLivePortrait components where appropriate for unit testing
- Test error handling for images without faces
- Test with different model configurations

## Expected Output
When running the tests, all face analysis tests should pass, confirming that:
- Face detection works correctly on sample images
- Landmarks can be extracted from detected faces
- Appearance features and motion parameters are correctly extracted
- All necessary data for SPD files can be obtained from source images

The wrapper modules should provide a clean, simplified interface to FasterLivePortrait's face analysis capabilities, focused specifically on the data needed for SPD files.

## Todo List Management
After completing this step, update the todo.md checklist in the spd_editor directory to mark off completed items in the "Face Analysis Integration" section.