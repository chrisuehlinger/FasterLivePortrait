"""
SPD (Source Portrait Descriptor) format definition.

This module contains specifications and utilities for working with the SPD format,
which stores pre-processed facial data to eliminate the need for face detection
during animation in the FasterLivePortrait system.
"""
import struct
import time
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple, TypeVar, Union, Any, BinaryIO, Protocol, cast
from typing_extensions import TypedDict, Literal, NotRequired

# SPD Format Constants
MAGIC_BYTES = b"SPDV"
CURRENT_VERSION = 1

# Header Flags (bit definitions)
class HeaderFlags(IntEnum):
    """Header flag bits that indicate which sections are present in the SPD file."""
    HAS_IMAGE = 1 << 0           # File includes source image data
    HAS_LANDMARKS = 1 << 1       # File includes facial landmark data
    HAS_MOTION_PARAMS = 1 << 2   # File includes motion parameters
    HAS_APPEARANCE = 1 << 3      # File includes appearance features
    HAS_TRANSFORMS = 1 << 4      # File includes transformation matrices
    HAS_MASK = 1 << 5            # File includes mask data
    IS_COMPRESSED = 1 << 6       # Data sections are compressed
    IS_ENCRYPTED = 1 << 7        # Data sections are encrypted


# Section Markers (4-byte identifiers)
class SectionMarker(IntEnum):
    """Section markers that identify the beginning of each section in the SPD file."""
    HEADER = 0x48454144  # "HEAD" (in hex)
    IMAGE = 0x494D4147   # "IMAG" (in hex)
    LANDMARKS = 0x4C4D4B53  # "LMKS" (in hex)
    MOTION_PARAMS = 0x4D4F5450  # "MOTP" (in hex)
    APPEARANCE = 0x41505045  # "APPE" (in hex)
    TRANSFORMS = 0x5452414E  # "TRAN" (in hex)
    MASK = 0x4D41534B  # "MASK" (in hex)
    ADDITIONAL_FLAGS = 0x464C4753  # "FLGS" (in hex)
    END = 0x454E4421  # "END!" (in hex)


# TypedDict definitions for structured data
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


class MotionParameters(TypedDict):
    """TypedDict representing motion parameters for animation."""
    head_pose: Dict[str, float]  # Head pose parameters (pitch, yaw, roll)
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


@dataclass
class HeaderSection:
    """
    Represents the header section of an SPD file.
    
    Attributes:
        magic_bytes: Must be 'SPDV' to identify this as an SPD file.
        version: SPD format version.
        timestamp: Unix timestamp when the file was created.
        flags: Bit flags indicating which sections are present and file properties.
    """
    magic_bytes: bytes = MAGIC_BYTES
    version: int = CURRENT_VERSION
    timestamp: float = 0.0
    flags: int = 0
    
    def __post_init__(self):
        """Validate the header after initialization."""
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class ImageSection:
    """
    Represents the image data section of an SPD file.
    
    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        channels: Number of color channels.
        format: Image format identifier (e.g., 'RGB', 'BGR').
        data: Raw image byte data.
    """
    width: int = 0
    height: int = 0
    channels: int = 0
    format: str = ""  # e.g., "RGB", "BGR", "RGBA"
    data: bytes = b""


@dataclass
class LandmarksSection:
    """
    Represents the facial landmarks section of an SPD file.
    
    Attributes:
        count: Number of landmarks.
        dimensions: Dimensionality of the landmarks (2 for 2D, 3 for 3D).
        landmark_type: Type identifier for the landmark set.
        points: List of landmark coordinates.
    """
    count: int = 0
    dimensions: int = 0  # 2 for 2D points, 3 for 3D points
    landmark_type: str = ""  # e.g., "mediapipe", "dlib68"
    points: List[List[float]] = None
    
    def __post_init__(self):
        """Initialize empty points list if none provided."""
        if self.points is None:
            self.points = []


@dataclass
class MotionParamsSection:
    """
    Represents the motion parameters section of an SPD file.
    
    Attributes:
        num_params: Number of motion parameters.
        param_names: Names/identifiers for each parameter.
        values: Parameter values.
    """
    num_params: int = 0
    param_names: List[str] = None
    values: List[float] = None
    
    def __post_init__(self):
        """Initialize empty lists if none provided."""
        if self.param_names is None:
            self.param_names = []
        if self.values is None:
            self.values = []


@dataclass
class AppearanceSection:
    """
    Represents the appearance features section of an SPD file.
    
    Attributes:
        feature_dim: Dimensionality of the feature vector.
        feature_type: Type identifier for the features.
        features: Feature vector data.
    """
    feature_dim: int = 0
    feature_type: str = ""  # e.g., "embedding", "latent"
    features: List[float] = None
    
    def __post_init__(self):
        """Initialize empty features list if none provided."""
        if self.features is None:
            self.features = []


@dataclass
class TransformationSection:
    """
    Represents the transformation matrices section of an SPD file.
    
    Attributes:
        num_transforms: Number of transformation matrices.
        transform_types: Type identifiers for each transformation.
        matrices: List of transformation matrices (each as a flattened list).
    """
    num_transforms: int = 0
    transform_types: List[str] = None  # e.g., "face2world", "world2face"
    matrices: List[List[float]] = None
    
    def __post_init__(self):
        """Initialize empty lists if none provided."""
        if self.transform_types is None:
            self.transform_types = []
        if self.matrices is None:
            self.matrices = []


@dataclass
class MaskSection:
    """
    Represents the mask data section of an SPD file.
    
    Attributes:
        width: Mask width in pixels.
        height: Mask height in pixels.
        format: Mask format identifier.
        data: Raw mask byte data.
    """
    width: int = 0
    height: int = 0
    format: str = ""  # e.g., "binary", "alpha"
    data: bytes = b""


@dataclass
class AdditionalFlagsSection:
    """
    Represents additional flags/metadata section of an SPD file.
    
    Attributes:
        num_flags: Number of additional flag key-value pairs.
        flags: Dictionary of additional flags and their values.
    """
    num_flags: int = 0
    flags: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize empty flags dict if none provided."""
        if self.flags is None:
            self.flags = {}


# Utility Functions
def validate_header(header_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate the SPD file header.
    
    Args:
        header_bytes: First bytes from the file containing the header.
        
    Returns:
        Tuple of (is_valid, error_message).
        If valid, error_message is empty.
    """
    if len(header_bytes) < 12:  # 4 (magic) + 4 (version) + 4 (min struct size)
        return False, "Header too short"
    
    magic = header_bytes[:4]
    if magic != MAGIC_BYTES:
        return False, f"Invalid magic bytes: expected {MAGIC_BYTES!r}, got {magic!r}"
    
    version_bytes = header_bytes[4:8]
    try:
        version = struct.unpack("!I", version_bytes)[0]
        if version > CURRENT_VERSION:
            return False, f"Unsupported version: {version} (current: {CURRENT_VERSION})"
    except struct.error as e:
        return False, f"Failed to decode version: {e}"
        
    return True, ""


def flags_to_dict(flags: int) -> Dict[str, bool]:
    """
    Convert header flags bit field to a dictionary of boolean values.
    
    Args:
        flags: Integer containing bit flags.
        
    Returns:
        Dictionary mapping flag names to boolean values.
    """
    return {
        flag.name: bool(flags & flag.value)
        for flag in HeaderFlags
    }


def dict_to_flags(flag_dict: Dict[str, bool]) -> int:
    """
    Convert a dictionary of boolean flags to a bit field integer.
    
    Args:
        flag_dict: Dictionary mapping flag names to boolean values.
        
    Returns:
        Integer with appropriate bits set for each True flag.
    """
    flags = 0
    for flag_name, is_set in flag_dict.items():
        if is_set:
            try:
                flag_value = HeaderFlags[flag_name].value
                flags |= flag_value
            except KeyError:
                pass  # Ignore invalid flag names
    return flags


def compute_section_size(data_shape: Tuple[int, ...], item_size: int = 4) -> int:
    """
    Compute the byte size of a section based on data dimensions.
    
    Args:
        data_shape: Tuple of dimensions (e.g., (height, width, channels)).
        item_size: Size of each data item in bytes (default: 4 for float32).
        
    Returns:
        Total size in bytes.
    """
    total_elements = 1
    for dim in data_shape:
        total_elements *= dim
    return total_elements * item_size


def get_required_flags_for_section(section_marker: SectionMarker) -> Optional[HeaderFlags]:
    """
    Get the header flag that must be set for a given section to be present.
    
    Args:
        section_marker: The section marker to check.
        
    Returns:
        The corresponding HeaderFlags value, or None if no specific flag is required.
    """
    section_to_flag = {
        SectionMarker.IMAGE: HeaderFlags.HAS_IMAGE,
        SectionMarker.LANDMARKS: HeaderFlags.HAS_LANDMARKS,
        SectionMarker.MOTION_PARAMS: HeaderFlags.HAS_MOTION_PARAMS,
        SectionMarker.APPEARANCE: HeaderFlags.HAS_APPEARANCE,
        SectionMarker.TRANSFORMS: HeaderFlags.HAS_TRANSFORMS,
        SectionMarker.MASK: HeaderFlags.HAS_MASK,
    }
    return section_to_flag.get(section_marker)