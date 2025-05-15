"""
TypedDict definitions for face detection results.
"""
from typing import List, Dict, Tuple, Optional, Union, Any
from typing_extensions import TypedDict, NotRequired


class FaceBoundingBox(TypedDict):
    """TypedDict representing a face bounding box in an image."""
    x1: float  # Left coordinate
    y1: float  # Top coordinate  
    x2: float  # Right coordinate
    y2: float  # Bottom coordinate
    
    # Alternative format used in some implementations
    left: NotRequired[float]
    top: NotRequired[float]
    right: NotRequired[float]
    bottom: NotRequired[float]


class FaceDetection(TypedDict):
    """TypedDict representing a detected face with metadata."""
    bbox: Union[List[float], Tuple[float, float, float, float]]  # [x1, y1, x2, y2]
    confidence: float  # Detection confidence score (0-1)
    landmarks: NotRequired[Dict[str, List[float]]]  # Optional facial landmarks by name
    pose: NotRequired[Dict[str, float]]  # Optional head pose information
    embedding: NotRequired[List[float]]  # Optional face embedding vector


class DetectionOptions(TypedDict):
    """TypedDict representing options for face detection."""
    min_confidence: float  # Minimum confidence threshold (0-1)
    max_faces: NotRequired[int]  # Maximum number of faces to return
    return_landmarks: NotRequired[bool]  # Whether to return facial landmarks
    return_pose: NotRequired[bool]  # Whether to return head pose
    return_embedding: NotRequired[bool]  # Whether to return face embedding
