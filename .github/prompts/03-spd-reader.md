# SPD Editor - SPD Reader Implementation

## Background
With the SPD format definition in place, we now need to implement the functionality to read SPD files. The reader will parse binary SPD files and convert them into usable Python data structures.

## Your Task
Create a reader module that can parse SPD files according to the format definition we created in the previous step.

## Requirements

1. Create a new file `spd/reader.py` with:

   - A `SPDReader` class that takes a file path or file-like object and provides methods to:
     - Read and validate the header
     - Read each section of the SPD file
     - Provide access to the contained data through properties or methods
   
   - Implement proper error handling for:
     - Invalid file format (wrong magic bytes, unsupported version)
     - Corrupted data
     - Missing sections
     - Invalid section sizes
   
   - The reader should lazily load sections as needed rather than loading everything at once,
     with options to eager-load when needed

2. Create a new file `tests/test_reader.py` with comprehensive tests for the reader:
   - Test reading valid SPD files
   - Test error handling with corrupted files
   - Test reading files with different combinations of sections

3. Create a few small test SPD files to use in your tests (you can generate them manually for now)

## Implementation Notes

- Use the format definitions from `spd/format.py` to determine how to parse each section
- Ensure proper type annotations and docstrings
- Follow a clear error handling pattern with custom exceptions
- Implement context manager support (`with` statement) for the reader
- Provide debug logging option to print information about the file being read

## Test Requirements
- Create mock SPD files for testing different scenarios
- Test reading different combinations of flags/sections
- Test error handling with various invalid inputs
- Test reading large files efficiently

## Expected Output
When running the tests, all reader tests should pass, confirming that:
- The reader can successfully parse valid SPD files
- The reader correctly handles errors with invalid files
- All sections can be properly extracted and accessed

The reader should be able to read an SPD file and provide access to its components like images, landmarks, and feature data.

## Todo List Management
After completing this step, update the todo.md checklist in the spd_editor directory to mark off completed items in the "SPD Reader Implementation" section.