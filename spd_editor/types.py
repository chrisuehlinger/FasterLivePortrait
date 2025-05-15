"""
Central module for importing and re-exporting all type definitions in the SPD Editor.

This module provides a single import point for all TypedDict, Protocol, 
and other type definitions used throughout the SPD Editor.
"""
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from typing_extensions import TypedDict, Protocol, Literal, NotRequired

# Re-export types from all modules
from spd_editor.spd.types import (
    LandmarkPoint, FacialLandmarks, HeadPose, MotionParameters,
    AppearanceFeatures, TransformationMatrix, MaskData,
    SPDFileInfo, ProcessingOptions, ValidationResult
)

from spd_editor.analysis.face_detection_types import (
    FaceBoundingBox, FaceDetection, DetectionOptions
)

from spd_editor.analysis.protocols import (
    FaceDetectorProtocol, LandmarkExtractorProtocol,
    FeatureExtractorProtocol, SPDProcessorProtocol
)

from spd_editor.utils.types import (
    LandmarkVisualizationOptions, VisualizationFormat,
    ComparisonOptions
)

from spd_editor.cli_types import (
    ExitCode, CommandOptions, CreateOptions, InspectOptions,
    ExtractOptions, ConvertOptions, ValidateOptions, VisualizeOptions
)

# Additional common type aliases
PathLike = Union[str, 'Path']
FileOrPath = Union['BinaryIO', PathLike]
ErrorSeverity = Literal["error", "warning", "info"]

# Export all types
__all__ = [
    # From spd.types
    'LandmarkPoint', 'FacialLandmarks', 'HeadPose', 'MotionParameters',
    'AppearanceFeatures', 'TransformationMatrix', 'MaskData',
    'SPDFileInfo', 'ProcessingOptions', 'ValidationResult',
    
    # From analysis.face_detection_types
    'FaceBoundingBox', 'FaceDetection', 'DetectionOptions',
    
    # From analysis.protocols
    'FaceDetectorProtocol', 'LandmarkExtractorProtocol',
    'FeatureExtractorProtocol', 'SPDProcessorProtocol',
    
    # From utils.types
    'LandmarkVisualizationOptions', 'VisualizationFormat',
    'ComparisonOptions',
    
    # From cli_types
    'ExitCode', 'CommandOptions', 'CreateOptions', 'InspectOptions',
    'ExtractOptions', 'ConvertOptions', 'ValidateOptions', 'VisualizeOptions',
    
    # Additional common type aliases
    'PathLike', 'FileOrPath', 'ErrorSeverity'
]
