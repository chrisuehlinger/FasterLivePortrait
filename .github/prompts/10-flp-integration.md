# SPD Editor - FasterLivePortrait Integration

## Background
Now that we have built the core SPD Editor functionality, we need to integrate it with the main FasterLivePortrait system to enable direct creation and use of SPD files within the FasterLivePortrait pipeline.

## Your Task
Create integration hooks that allow FasterLivePortrait to use SPD files as source inputs, and add functionality to export SPD files after source preparation.

## Requirements

1. Create a new file `/workspaces/FasterLivePortrait/src/utils/spd_utils.py` with:
   - Functions to load SPD files into the FasterLivePortrait pipeline
   - Functions to convert from FasterLivePortrait's internal format to SPD
   - Utility functions for determining if a file is an SPD file

2. Modify FasterLivePortrait's `prepare_source` method to:
   - Detect SPD files based on extension or magic bytes
   - Load pre-processed data directly from SPD files when available
   - Skip face detection and analysis steps when using SPD files

3. Add an export option to FasterLivePortrait that allows saving source data as SPD after processing

4. Create test cases that verify the integration works correctly

## Implementation Notes

- Make minimal changes to the FasterLivePortrait codebase
- Use clear documentation to explain the integration points
- Keep SPD functionality optional so FasterLivePortrait still works without it
- Add appropriate logging to track when SPD files are being used
- Consider performance implications and optimize accordingly
- Handle errors gracefully to avoid breaking the main pipeline

## Test Requirements
- Test loading SPD files into FasterLivePortrait
- Test exporting processed source data as SPD files
- Test that animations look identical whether using SPD files or direct source images
- Test performance comparisons between SPD and direct source processing

## Expected Output
When the integration is complete, users should be able to:
- Use SPD files directly as source inputs for FasterLivePortrait:
  ```
  python run.py --src_spd face.spd --dri_video video.mp4 --cfg configs/trt_infer.yaml
  ```
- Export SPD files after source processing:
  ```
  python run.py --src_image image.jpg --export_spd face.spd --cfg configs/trt_infer.yaml
  ```

The integration should be seamless, with SPD files providing identical results to direct source images but with faster startup times and less computational overhead.

## Todo List Management
After completing this step, update the todo.md checklist in the spd_editor directory to mark off completed items in the "FasterLivePortrait Integration" section.