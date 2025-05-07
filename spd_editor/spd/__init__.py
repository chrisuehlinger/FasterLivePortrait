"""
SPD (Source Portrait Descriptor) format handling module.
"""
from .reader import SPDReader # Assuming reader.py exists or will be created
from .writer import SPDWriter
from .format import *

__all__ = [
    "SPDReader",
    "SPDWriter",
    # Add constants from format.py if they should be directly accessible
    "MAGIC_BYTES", "CURRENT_VERSION", # Changed from "SPD_MAGIC_BYTES", "SPD_VERSION"
    "FLAG_HAS_ORIGINAL_IMAGE", "FLAG_HAS_CROPPED_IMAGE", "FLAG_HAS_RESIZED_IMAGE",
    "FLAG_HAS_LANDMARK_DATA", "FLAG_HAS_MOTION_PARAMETERS", "FLAG_HAS_APPEARANCE_FEATURES",
    "FLAG_HAS_TRANSFORMATION_MATRICES", "FLAG_HAS_MASK_DATA",
    "RESIZED_IMAGE_WIDTH", "RESIZED_IMAGE_HEIGHT", "DEFAULT_NUM_LANDMARKS"
]