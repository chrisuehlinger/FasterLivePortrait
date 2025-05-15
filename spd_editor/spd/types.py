"""
Type definitions for SPD (Source Portrait Descriptor) format.

This module contains TypedDict definitions and other type annotations
for the SPD format used in FasterLivePortrait.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
from typing_extensions import TypedDict, Literal, NotRequired


class LandmarkPoint(TypedDict):
    """TypedDict representing a single 2D or 3D landmark point."""
    x: float
    y: float
    z: NotRequired[float]  # Optional for 3D landmarks
    confidence: NotRequired[float]  # Optional confidence score


class FacialLandmarks(TypedDict):
    """TypedDict representing a set of facial landmarks."""
    points: List[LandmarkPoint]
    format: str  # Format identifier (e.g., 'mediapipe', 'dlib68', etc.)
    version: NotRequired[str]  # Optional version information


class HeadPose(TypedDict):
    """TypedDict representing head pose parameters."""
    pitch: float
    yaw: float
    roll: float
    translation_x: NotRequired[float]
    translation_y: NotRequired[float]
    translation_z: NotRequired[float]


class MotionParameters(TypedDict):
    """TypedDict representing motion parameters for animation."""
    head_pose: HeadPose
    expression_weights: Dict[str, float]  # Facial expression blend shape weights
    eye_gaze: NotRequired[Dict[str, float]]  # Optional eye gaze parameters


class AppearanceFeatures(TypedDict):
    """TypedDict representing appearance features."""
    embedding: List[float]  # Facial embedding vector
    model: NotRequired[str]  # Optional model identifier


class TransformationMatrix(TypedDict):
    """TypedDict representing a transformation matrix."""
    matrix: List[List[float]]  # 3x3 or 4x4 transformation matrix
    type: str  # Type of transformation ('rigid', 'affine', 'perspective', etc.)


class MaskData(TypedDict):
    """TypedDict representing mask data."""
    width: int
    height: int
    data: bytes  # Raw mask data
    format: str  # Mask format descriptor


class SPDFileInfo(TypedDict):
    """TypedDict representing metadata for an SPD file."""
    version: int
    timestamp: float
    flags: int
    sections: List[str]  # List of section names present in the file
    has_image: bool
    has_landmarks: bool
    has_motion_params: bool
    has_appearance: bool
    has_transforms: bool
    has_mask: bool
    is_compressed: bool
    is_encrypted: bool
    additional_flags: NotRequired[Dict[str, Any]]


class ProcessingOptions(TypedDict):
    """TypedDict representing processing options for SPD creation/editing."""
    landmark_type: str  # e.g., "mediapipe", "dlib68"
    detect_face: bool
    extract_landmarks: bool
    extract_appearance: bool
    generate_mask: bool
    compute_transforms: bool
    compress: NotRequired[bool]
    encrypt: NotRequired[bool]
    quality: NotRequired[int]  # Image quality (1-100)


class ValidationResult(TypedDict):
    """TypedDict representing the result of validating an SPD file."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: SPDFileInfo
