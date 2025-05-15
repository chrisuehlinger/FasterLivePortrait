"""
Type definitions for SPD Editor visualization utilities.

This module provides TypedDict and other type definitions for the visualization
utilities in the SPD Editor.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from typing_extensions import TypedDict, Literal, NotRequired


class LandmarkVisualizationOptions(TypedDict):
    """TypedDict representing options for visualizing facial landmarks."""
    point_color: Union[Tuple[int, int, int], str]  # RGB tuple or color name
    point_size: int
    line_color: Union[Tuple[int, int, int], str]  # RGB tuple or color name
    line_thickness: int
    text_color: Union[Tuple[int, int, int], str]  # RGB tuple or color name
    text_size: float
    show_points: bool
    show_lines: bool
    show_labels: bool
    show_confidence: bool
    highlighted_points: NotRequired[List[int]]  # Points to highlight


class VisualizationFormat(TypedDict):
    """TypedDict representing format options for saving visualizations."""
    dpi: int
    width: int
    height: int
    format: Literal['png', 'jpg', 'svg', 'pdf']
    quality: NotRequired[int]  # 0-100, for jpg
    transparent: NotRequired[bool]  # For png with transparency


class ComparisonOptions(TypedDict):
    """TypedDict representing options for comparison visualizations."""
    layout: Literal['horizontal', 'vertical', 'grid']
    titles: List[str]
    global_title: NotRequired[str]
    show_differences: bool
    difference_color: Union[Tuple[int, int, int], str]
    match_threshold: float  # 0-1.0 for point matching
