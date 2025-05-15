"""
Type definitions for SPD Editor command-line interface.

This module provides TypedDict and other type definitions for CLI operations.
"""
from typing import Dict, List, Optional, Union, Any, Callable
from typing_extensions import TypedDict, Literal

# Command result types
ExitCode = int  # 0 for success, non-zero for errors

# CLI command option types
class CommandOptions(TypedDict, total=False):
    """TypedDict representing common command options."""
    verbose: bool
    quiet: bool
    output: str
    format: str


class CreateOptions(CommandOptions, total=False):
    """TypedDict representing options for 'create' command."""
    source_image: str
    output_file: str
    add_landmarks: bool
    add_appearance: bool
    add_mask: bool
    landmark_type: str
    quality: int
    compress: bool
    encrypt: bool
    password: str


class InspectOptions(CommandOptions, total=False):
    """TypedDict representing options for 'inspect' command."""
    input_file: str
    section: Optional[str]
    decrypt: bool
    password: str


class ExtractOptions(CommandOptions, total=False):
    """TypedDict representing options for 'extract' command."""
    input_file: str
    section: str
    output: str
    decrypt: bool
    password: str


class ConvertOptions(CommandOptions, total=False):
    """TypedDict representing options for 'convert' command."""
    input_file: str
    output_file: str
    format: str
    compress: bool
    encrypt: bool
    password: str


class ValidateOptions(CommandOptions, total=False):
    """TypedDict representing options for 'validate' command."""
    input_file: str
    level: str  # basic, normal, strict
    decrypt: bool
    password: str


class VisualizeOptions(CommandOptions, total=False):
    """TypedDict representing options for 'visualize' command."""
    input_file: str
    output: str
    show: bool
    landmark_type: str
    decrypt: bool
    password: str
