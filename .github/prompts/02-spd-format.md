# SPD Editor - Format Definition

## Background
Now that we have the basic project structure in place, we need to implement the SPD file format definition. The SPD (Source Portrait Descriptor) is a binary file format that stores pre-processed facial data extracted from source images for use with FasterLivePortrait.

## Your Task
Define the SPD file format constants, structures, and basic validation functions in the `format.py` module. This will serve as the foundation for both reading and writing SPD files.

## Requirements

1. In `spd/format.py`, implement the following:

   - Define constants for the SPD format:
     - Magic bytes: "SPDV" (marks the beginning of an SPD file)
     - Current version: 1
     - Flag bit definitions for the header section
     - Section markers/identifiers

   - Create data structures to represent the SPD file sections:
     - Header section (magic bytes, version, timestamp, flags)
     - Image data section
     - Facial landmarks section
     - Motion parameters section
     - Appearance features section
     - Transformation matrices section
     - Mask data section
     - Additional flags section

   - Include detailed type annotations and docstrings throughout

2. Create utility functions for:
   - Validating an SPD header (verifying magic bytes and version)
   - Converting between bit flags and boolean values
   - Computing section sizes based on data dimensions

3. Update the tests in `tests/test_format.py` to verify:
   - Magic bytes and version constants are defined correctly
   - Flag bit operations work as expected
   - Basic validation functions behave correctly

## Test Requirements
- Write unit tests for each part of the SPD format definition
- Test all flag combinations
- Verify header validation correctly identifies valid and invalid headers
- Test section size calculations with various input dimensions

## Expected Output
When running the tests, all format definition tests should pass, confirming that:
- Constants are correctly defined
- Flag handling works as expected
- Header validation correctly identifies valid and invalid headers

The module should not yet attempt to read or write full SPD files - that will be implemented in the next step. This step focuses on defining the format and providing the foundation for future implementation.

## Todo List Management
After completing this step, update the todo.md checklist in the spd_editor directory to mark off completed items in the "SPD Format Definition" section.