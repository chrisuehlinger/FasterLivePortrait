"""
SPD (Source Portrait Descriptor) format handling module.
"""
from .format import *  # noqa
from .types import (
    LandmarkPoint, FacialLandmarks, HeadPose, MotionParameters,
    AppearanceFeatures, TransformationMatrix, MaskData,
    SPDFileInfo, ProcessingOptions, ValidationResult
)

__all__ = [
    # Re-export from format.py
    'MAGIC_BYTES', 'CURRENT_VERSION', 'HeaderFlags', 'SectionMarker',
    'HeaderSection', 'ImageSection', 'LandmarksSection', 'MotionParamsSection',
    'AppearanceSection', 'TransformationSection', 'MaskSection',
    'AdditionalFlagsSection', 'flags_to_dict',
    
    # From types.py
    'LandmarkPoint', 'FacialLandmarks', 'HeadPose', 'MotionParameters',
    'AppearanceFeatures', 'TransformationMatrix', 'MaskData',
    'SPDFileInfo', 'ProcessingOptions', 'ValidationResult'
]