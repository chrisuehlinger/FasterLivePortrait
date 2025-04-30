# SPD Editor - Project Setup

## Background
We're building a Source Portrait Descriptor (SPD) Editor tool for the FasterLivePortrait system. The SPD format is a binary file that stores pre-processed facial data to eliminate the need for face detection during animation.

## Your Task
Set up the initial project structure for the SPD Editor according to the specifications. This includes creating the basic directory structure, implementing essential plumbing code and foundational components.

## Requirements

1. Create the following directory structure:
```
spd_editor/
├── __init__.py
├── main.py                  # Main entry point
├── spd/
│   ├── __init__.py
│   ├── format.py            # Will hold format specification (empty for now)
├── tests/
│   ├── __init__.py
│   ├── test_format.py       # Will hold format tests (basic structure only)
```

2. In `__init__.py`, add version information:
```python
"""Source Portrait Descriptor Editor for FasterLivePortrait"""

__version__ = "0.1.0"
```

3. In `main.py`, implement a basic entry point that imports from the SPD module and prints a version and welcome message.

4. Create a simple README.md in the spd_editor directory explaining the tool's purpose.

5. Add a setup.py file with basic package configuration.

## Test Requirements
- Create a simple test that confirms imports are working properly
- Ensure the welcome message appears when the module is run
- Test that the version is correctly defined

## Expected Output
When running `python -m spd_editor`, you should see a welcome message and version information printed out.

```
$ python -m spd_editor
SPD Editor v0.1.0 - Source Portrait Descriptor Editor for FasterLivePortrait
Use --help to see available commands
```

Focus on creating a clean foundation that we can build upon in subsequent steps. Use docstrings and type hints throughout the code to ensure good documentation from the start.

## Todo List Management
After completing this step, update the todo.md checklist in the spd_editor directory to mark off completed items in the "Project Setup" section.