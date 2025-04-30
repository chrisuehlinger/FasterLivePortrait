# SPD Editor - Command-Line Interface

## Background
Now that we have the core functionality for working with SPD files, we need to make this functionality accessible to users via a command-line interface (CLI). This will allow users to create, validate, and manipulate SPD files without writing Python code.

## Your Task
Create a command-line interface for the SPD Editor that exposes its key functionality in a user-friendly way.

## Requirements

1. Create a new file `cli.py` with:
   - A command-line interface using a library like `argparse` or `click`
   - Commands for all core SPD operations
   - Proper help text and documentation
   - Error handling and user feedback

2. Implement the following commands:
   - `create`: Create an SPD file from a source image
   - `info`: Display information about an existing SPD file
   - `validate`: Validate an SPD file and report any issues
   - `extract`: Extract data (image, landmarks, etc.) from an SPD file
   - `convert`: Convert between different formats (e.g., extract landmarks to CSV)
   - `visualize`: Create visualizations of SPD data

3. Update `main.py` to properly route to these CLI commands

4. Create a test file `tests/test_cli.py` to verify CLI functionality

## Implementation Notes

- Choose a CLI framework that balances simplicity and power (argparse, click, etc.)
- Provide comprehensive help text for all commands and options
- Use colored output for better user experience where appropriate
- Handle errors gracefully with useful error messages
- Provide both simple commands for common tasks and advanced options for power users
- Include examples in help text
- Support batch processing where appropriate

## Test Requirements
- Test all CLI commands with various inputs
- Test error handling for invalid inputs
- Test help text and documentation
- Test output formats and file handling

## Expected Output
When running the tests, all CLI tests should pass, confirming that:
- All commands work correctly with valid inputs
- Error handling works as expected with invalid inputs
- Help text is comprehensive and accurate

Users should be able to run commands like:
```
python -m spd_editor create --source image.jpg --output face.spd
python -m spd_editor info face.spd
python -m spd_editor extract --image face.spd --output extracted_face.jpg
python -m spd_editor validate face.spd
```

The CLI should provide a complete interface to the SPD Editor functionality that is accessible to users who are not Python developers.

## Todo List Management
After completing this step, update the todo.md checklist in the spd_editor directory to mark off completed items in the "CLI Implementation" section.