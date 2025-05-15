"""
SPD (Source Portrait Descriptor) writer implementation.

This module provides functionality to create SPD files according to the format
defined in the format module.
"""
import os
import io
import struct
import time
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, BinaryIO, Tuple, Any, TypeVar, Type, cast, Callable, Iterator
from typing_extensions import TypedDict, Protocol, runtime_checkable

from .format import (
    MAGIC_BYTES, CURRENT_VERSION,
    HeaderFlags, SectionMarker,
    HeaderSection, ImageSection, LandmarksSection, 
    MotionParamsSection, AppearanceSection, TransformationSection,
    MaskSection, AdditionalFlagsSection,
    validate_header, flags_to_dict, dict_to_flags
)
from .types import (
    SPDFileInfo, ProcessingOptions, LandmarkPoint,
    FacialLandmarks, MotionParameters, AppearanceFeatures
)

# Set up logger for this module
logger = logging.getLogger(__name__)

# Type aliases
PathLike = Union[str, Path]
FileOrPath = Union[BinaryIO, PathLike]
T = TypeVar('T')


class SPDError(Exception):
    """Base class for all SPD-related exceptions."""
    pass


class SPDFormatError(SPDError):
    """Raised when the SPD file format is invalid."""
    pass


class SPDVersionError(SPDError):
    """Raised when the SPD file version is not supported."""
    pass


class SPDSectionError(SPDError):
    """Raised when there's an issue with an SPD section."""
    pass


class SPDValidationError(SPDError):
    """Raised when input data for a section fails validation."""
    pass


class SPDWriter:
    """
    Writer for SPD (Source Portrait Descriptor) files.
    
    Provides methods to create SPD files and add sections to them.
    
    Attributes:
        file_path: The path to the SPD file (if opened from a path).
    """
    
    def __init__(self, file_or_path: FileOrPath):
        """
        Initialize the SPD writer.
        
        Args:
            file_or_path: Path to the SPD file to create, or a file-like object to write to.
        """
        self._file: Optional[BinaryIO] = None
        self._file_path: Optional[str] = None
        self._owns_file: bool = False
        
        # Track which sections have been written
        self._sections_written: Dict[SectionMarker, bool] = {
            marker: False for marker in SectionMarker
        }
        
        # Initialize the header
        self._header = HeaderSection()
        self._header.timestamp = time.time()
        self._header.flags = 0  # No sections by default
        
        # Open the file
        self._open_file(file_or_path)
        
        # Write placeholder for header - we'll come back and update it with real flags
        # after all sections are written
        self._write_placeholder_header()
    
    def _open_file(self, file_or_path: FileOrPath) -> None:
        """
        Open the file for writing.
        
        Args:
            file_or_path: Path to the file or a file-like object.
        
        Raises:
            IOError: If the file cannot be opened for writing.
        """
        if isinstance(file_or_path, (str, Path)):
            # Convert Path to string if needed
            file_path = str(file_or_path) if isinstance(file_or_path, Path) else file_or_path
            try:
                self._file = open(file_path, "wb")
                self._file_path = file_path
                self._owns_file = True
                logger.debug(f"Opened file for writing: {self._file_path}")
            except IOError as e:
                raise IOError(f"Failed to open file for writing: {e}")
        else:
            # Use provided file-like object
            self._file = file_or_path
            self._file_path = getattr(file_or_path, "name", None)
            self._owns_file = False
    
    def _validate_file_is_open(self) -> None:
        """
        Validate that the file is open for writing.
        
        Raises:
            RuntimeError: If the file is not open.
        """
        if self._file is None:
            raise RuntimeError("File is not open")
    
    def _update_header_flags(self, section_marker: SectionMarker) -> None:
        """
        Update the header flags based on the section being added.
        
        Args:
            section_marker: The marker for the section being added.
        """
        flag_mapping = {
            SectionMarker.IMAGE: HeaderFlags.HAS_IMAGE,
            SectionMarker.LANDMARKS: HeaderFlags.HAS_LANDMARKS,
            SectionMarker.MOTION_PARAMS: HeaderFlags.HAS_MOTION_PARAMS,
            SectionMarker.APPEARANCE: HeaderFlags.HAS_APPEARANCE,
            SectionMarker.TRANSFORMS: HeaderFlags.HAS_TRANSFORMS,
            SectionMarker.MASK: HeaderFlags.HAS_MASK,
        }
        
        if section_marker in flag_mapping:
            self._header.flags |= flag_mapping[section_marker].value
    
    def _write_placeholder_header(self) -> None:
        """
        Write a placeholder header to the file. We'll update this with real flags later.
        
        Raises:
            RuntimeError: If the file is not open.
        """
        self._validate_file_is_open()
        
        try:
            # Write magic bytes
            self._file.write(self._header.magic_bytes)
            
            # Write version number
            self._file.write(struct.pack("!I", self._header.version))
            
            # Write timestamp
            self._file.write(struct.pack("!d", self._header.timestamp))
            
            # Write placeholder flags (will be updated later)
            self._file.write(struct.pack("!I", 0))
            
            # Mark header as written
            self._sections_written[SectionMarker.HEADER] = True
            logger.debug("Wrote placeholder SPD header")
        except (IOError, struct.error) as e:
            raise SPDFormatError(f"Failed to write SPD header: {e}")
    
    def _update_header_in_file(self) -> None:
        """
        Update the header flags in the file with the current flags.
        
        Raises:
            RuntimeError: If the file is not open.
        """
        self._validate_file_is_open()
        
        try:
            # Save current position
            current_pos = self._file.tell()
            
            # Go back to header flags position (16 bytes from start)
            self._file.seek(16)
            
            # Write updated flags
            self._file.write(struct.pack("!I", self._header.flags))
            
            # Go back to previous position
            self._file.seek(current_pos)
            
            logger.debug(f"Updated header flags: 0x{self._header.flags:x}")
        except (IOError, struct.error) as e:
            raise SPDFormatError(f"Failed to update SPD header flags: {e}")
    
    def _write_section_marker(self, marker: SectionMarker) -> None:
        """
        Write a section marker to the file.
        
        Args:
            marker: The section marker to write.
            
        Raises:
            RuntimeError: If the file is not open.
        """
        self._validate_file_is_open()
        
        try:
            self._file.write(struct.pack("!I", marker.value))
            logger.debug(f"Wrote section marker: {marker.name}")
        except (IOError, struct.error) as e:
            raise SPDFormatError(f"Failed to write section marker: {e}")
    
    def _write_string(self, string: str) -> None:
        """
        Write a string to the file.
        
        Format: 4 bytes length + UTF-8 encoded string data
        
        Args:
            string: The string to write.
            
        Raises:
            RuntimeError: If the file is not open.
        """
        self._validate_file_is_open()
        
        try:
            # Encode the string as UTF-8
            string_bytes = string.encode('utf-8')
            
            # Write the string length
            self._file.write(struct.pack("!I", len(string_bytes)))
            
            # Write the string data
            self._file.write(string_bytes)
        except (IOError, struct.error) as e:
            raise SPDFormatError(f"Failed to write string: {e}")
    
    def write_image_section(self, image: ImageSection) -> None:
        """
        Write an image section to the file.
        
        Args:
            image: The image data to write.
            
        Raises:
            SPDValidationError: If the image data is invalid.
            RuntimeError: If the file is not open.
        """
        # Validate image data
        if image.width <= 0 or image.height <= 0 or image.channels <= 0:
            raise SPDValidationError("Image dimensions must be positive")
        
        if not image.data:
            raise SPDValidationError("Image data cannot be empty")
        
        if not image.format:
            raise SPDValidationError("Image format must be specified")
        
        expected_size = image.width * image.height * image.channels
        if len(image.data) != expected_size:
            raise SPDValidationError(
                f"Image data size mismatch: expected {expected_size}, got {len(image.data)}"
            )
        
        # Update header flags
        self._update_header_flags(SectionMarker.IMAGE)
        
        # Write section marker
        self._write_section_marker(SectionMarker.IMAGE)
        
        try:
            # Calculate section size
            # 4 bytes width + 4 bytes height + 1 byte channels + 
            # 4 bytes format string length + len(format) bytes format string +
            # width * height * channels bytes image data
            format_bytes = image.format.encode('utf-8')
            section_size = 4 + 4 + 1 + 4 + len(format_bytes) + len(image.data)
            
            # Write section size
            self._file.write(struct.pack("!I", section_size))
            
            # Write image dimensions
            self._file.write(struct.pack("!I", image.width))
            self._file.write(struct.pack("!I", image.height))
            self._file.write(bytes([image.channels]))
            
            # Write format string
            self._write_string(image.format)
            
            # Write image data
            self._file.write(image.data)
            
            # Mark section as written
            self._sections_written[SectionMarker.IMAGE] = True
            logger.debug(f"Wrote image section: {image.width}x{image.height}x{image.channels}")
        except (IOError, struct.error) as e:
            raise SPDFormatError(f"Failed to write image section: {e}")
    
    def write_landmarks_section(self, landmarks: LandmarksSection) -> None:
        """
        Write a landmarks section to the file.
        
        Args:
            landmarks: The landmarks data to write.
            
        Raises:
            SPDValidationError: If the landmarks data is invalid.
            RuntimeError: If the file is not open.
        """
        # Validate landmarks data
        if landmarks.count <= 0:
            raise SPDValidationError("Landmark count must be positive")
        
        if landmarks.dimensions <= 0:
            raise SPDValidationError("Landmark dimensions must be positive")
        
        if not landmarks.landmark_type:
            raise SPDValidationError("Landmark type must be specified")
        
        if not landmarks.points:
            raise SPDValidationError("Landmarks points cannot be empty")
        
        if len(landmarks.points) != landmarks.count:
            raise SPDValidationError(
                f"Landmarks count mismatch: expected {landmarks.count}, got {len(landmarks.points)}"
            )
        
        for point in landmarks.points:
            if len(point) != landmarks.dimensions:
                raise SPDValidationError(
                    f"Landmark point dimensions mismatch: expected {landmarks.dimensions}, got {len(point)}"
                )
        
        # Update header flags
        self._update_header_flags(SectionMarker.LANDMARKS)
        
        # Write section marker
        self._write_section_marker(SectionMarker.LANDMARKS)
        
        try:
            # Calculate section size
            # 4 bytes count + 4 bytes dimensions + 
            # 4 bytes landmark type length + len(landmark_type) bytes landmark type +
            # count * dimensions * 4 bytes point data
            landmark_type_bytes = landmarks.landmark_type.encode('utf-8')
            section_size = 4 + 4 + 4 + len(landmark_type_bytes) + (landmarks.count * landmarks.dimensions * 4)
            
            # Write section size
            self._file.write(struct.pack("!I", section_size))
            
            # Write landmarks metadata
            self._file.write(struct.pack("!I", landmarks.count))
            self._file.write(struct.pack("!I", landmarks.dimensions))
            
            # Write landmark type
            self._write_string(landmarks.landmark_type)
            
            # Write landmark points
            for point in landmarks.points:
                for coord in point:
                    self._file.write(struct.pack("!f", coord))
            
            # Mark section as written
            self._sections_written[SectionMarker.LANDMARKS] = True
            logger.debug(f"Wrote landmarks section: {landmarks.count} {landmarks.landmark_type} landmarks")
        except (IOError, struct.error) as e:
            raise SPDFormatError(f"Failed to write landmarks section: {e}")
    
    def write_motion_params_section(self, motion_params: MotionParamsSection) -> None:
        """
        Write a motion parameters section to the file.
        
        Args:
            motion_params: The motion parameters data to write.
            
        Raises:
            SPDValidationError: If the motion parameters data is invalid.
            RuntimeError: If the file is not open.
        """
        # Validate motion parameters data
        if motion_params.num_params <= 0:
            raise SPDValidationError("Number of motion parameters must be positive")
        
        if not motion_params.param_names:
            raise SPDValidationError("Parameter names cannot be empty")
        
        if not motion_params.values:
            raise SPDValidationError("Parameter values cannot be empty")
        
        if len(motion_params.param_names) != motion_params.num_params:
            raise SPDValidationError(
                f"Parameter names count mismatch: expected {motion_params.num_params}, got {len(motion_params.param_names)}"
            )
        
        if len(motion_params.values) != motion_params.num_params:
            raise SPDValidationError(
                f"Parameter values count mismatch: expected {motion_params.num_params}, got {len(motion_params.values)}"
            )
        
        # Update header flags
        self._update_header_flags(SectionMarker.MOTION_PARAMS)
        
        # Write section marker
        self._write_section_marker(SectionMarker.MOTION_PARAMS)
        
        try:
            # Calculate section size
            # 4 bytes num_params + 
            # (4 bytes param name length + len(param_name) bytes param name) * num_params +
            # 4 bytes * num_params values
            section_size = 4
            for name in motion_params.param_names:
                name_bytes = name.encode('utf-8')
                section_size += 4 + len(name_bytes)
            section_size += motion_params.num_params * 4
            
            # Write section size
            self._file.write(struct.pack("!I", section_size))
            
            # Write number of parameters
            self._file.write(struct.pack("!I", motion_params.num_params))
            
            # Write parameter names
            for name in motion_params.param_names:
                self._write_string(name)
            
            # Write parameter values
            for value in motion_params.values:
                self._file.write(struct.pack("!f", value))
            
            # Mark section as written
            self._sections_written[SectionMarker.MOTION_PARAMS] = True
            logger.debug(f"Wrote motion parameters section: {motion_params.num_params} parameters")
        except (IOError, struct.error) as e:
            raise SPDFormatError(f"Failed to write motion parameters section: {e}")
    
    def write_appearance_section(self, appearance: AppearanceSection) -> None:
        """
        Write an appearance features section to the file.
        
        Args:
            appearance: The appearance features data to write.
            
        Raises:
            SPDValidationError: If the appearance features data is invalid.
            RuntimeError: If the file is not open.
        """
        # Validate appearance features data
        if appearance.feature_dim <= 0:
            raise SPDValidationError("Feature dimension must be positive")
        
        if not appearance.feature_type:
            raise SPDValidationError("Feature type must be specified")
        
        if not appearance.features:
            raise SPDValidationError("Features cannot be empty")
        
        if len(appearance.features) != appearance.feature_dim:
            raise SPDValidationError(
                f"Features count mismatch: expected {appearance.feature_dim}, got {len(appearance.features)}"
            )
        
        # Update header flags
        self._update_header_flags(SectionMarker.APPEARANCE)
        
        # Write section marker
        self._write_section_marker(SectionMarker.APPEARANCE)
        
        try:
            # Calculate section size
            # 4 bytes feature_dim + 
            # 4 bytes feature type length + len(feature_type) bytes feature type +
            # 4 bytes * feature_dim features
            feature_type_bytes = appearance.feature_type.encode('utf-8')
            section_size = 4 + 4 + len(feature_type_bytes) + (appearance.feature_dim * 4)
            
            # Write section size
            self._file.write(struct.pack("!I", section_size))
            
            # Write feature dimension
            self._file.write(struct.pack("!I", appearance.feature_dim))
            
            # Write feature type
            self._write_string(appearance.feature_type)
            
            # Write features
            for feature in appearance.features:
                self._file.write(struct.pack("!f", feature))
            
            # Mark section as written
            self._sections_written[SectionMarker.APPEARANCE] = True
            logger.debug(f"Wrote appearance section: {appearance.feature_dim}-dimensional {appearance.feature_type} features")
        except (IOError, struct.error) as e:
            raise SPDFormatError(f"Failed to write appearance section: {e}")
    
    def write_transformation_section(self, transforms: TransformationSection) -> None:
        """
        Write a transformation matrices section to the file.
        
        Args:
            transforms: The transformation matrices data to write.
            
        Raises:
            SPDValidationError: If the transformation matrices data is invalid.
            RuntimeError: If the file is not open.
        """
        # Validate transformation matrices data
        if transforms.num_transforms <= 0:
            raise SPDValidationError("Number of transformation matrices must be positive")
        
        if not transforms.transform_types:
            raise SPDValidationError("Transform types cannot be empty")
        
        if not transforms.matrices:
            raise SPDValidationError("Transformation matrices cannot be empty")
        
        if len(transforms.transform_types) != transforms.num_transforms:
            raise SPDValidationError(
                f"Transform types count mismatch: expected {transforms.num_transforms}, got {len(transforms.transform_types)}"
            )
        
        if len(transforms.matrices) != transforms.num_transforms:
            raise SPDValidationError(
                f"Matrices count mismatch: expected {transforms.num_transforms}, got {len(transforms.matrices)}"
            )
        
        # Each matrix is expected to be 3x3 = 9 elements
        for matrix in transforms.matrices:
            if len(matrix) != 9:
                raise SPDValidationError(f"Transformation matrix must have 9 elements (3x3), got {len(matrix)}")
        
        # Update header flags
        self._update_header_flags(SectionMarker.TRANSFORMS)
        
        # Write section marker
        self._write_section_marker(SectionMarker.TRANSFORMS)
        
        try:
            # Calculate section size
            # 4 bytes num_transforms + 
            # (4 bytes transform type length + len(transform_type) bytes transform type) * num_transforms +
            # (9 * 4 bytes) * num_transforms matrices
            section_size = 4
            for name in transforms.transform_types:
                name_bytes = name.encode('utf-8')
                section_size += 4 + len(name_bytes)
            section_size += transforms.num_transforms * 9 * 4
            
            # Write section size
            self._file.write(struct.pack("!I", section_size))
            
            # Write number of transforms
            self._file.write(struct.pack("!I", transforms.num_transforms))
            
            # Write transform types
            for name in transforms.transform_types:
                self._write_string(name)
            
            # Write matrices
            for matrix in transforms.matrices:
                for value in matrix:
                    self._file.write(struct.pack("!f", value))
            
            # Mark section as written
            self._sections_written[SectionMarker.TRANSFORMS] = True
            logger.debug(f"Wrote transformation section: {transforms.num_transforms} transformation matrices")
        except (IOError, struct.error) as e:
            raise SPDFormatError(f"Failed to write transformation section: {e}")
    
    def write_mask_section(self, mask: MaskSection) -> None:
        """
        Write a mask section to the file.
        
        Args:
            mask: The mask data to write.
            
        Raises:
            SPDValidationError: If the mask data is invalid.
            RuntimeError: If the file is not open.
        """
        # Validate mask data
        if mask.width <= 0 or mask.height <= 0:
            raise SPDValidationError("Mask dimensions must be positive")
        
        if not mask.format:
            raise SPDValidationError("Mask format must be specified")
        
        if not mask.data:
            raise SPDValidationError("Mask data cannot be empty")
        
        expected_size = mask.width * mask.height
        if mask.format.lower() == "binary" and len(mask.data) != expected_size:
            raise SPDValidationError(
                f"Binary mask data size mismatch: expected {expected_size}, got {len(mask.data)}"
            )
        
        # Update header flags
        self._update_header_flags(SectionMarker.MASK)
        
        # Write section marker
        self._write_section_marker(SectionMarker.MASK)
        
        try:
            # Calculate section size
            # 4 bytes width + 4 bytes height + 
            # 4 bytes format length + len(format) bytes format +
            # data size bytes mask data
            format_bytes = mask.format.encode('utf-8')
            section_size = 4 + 4 + 4 + len(format_bytes) + len(mask.data)
            
            # Write section size
            self._file.write(struct.pack("!I", section_size))
            
            # Write mask dimensions
            self._file.write(struct.pack("!I", mask.width))
            self._file.write(struct.pack("!I", mask.height))
            
            # Write mask format
            self._write_string(mask.format)
            
            # Write mask data
            self._file.write(mask.data)
            
            # Mark section as written
            self._sections_written[SectionMarker.MASK] = True
            logger.debug(f"Wrote mask section: {mask.width}x{mask.height} {mask.format} mask")
        except (IOError, struct.error) as e:
            raise SPDFormatError(f"Failed to write mask section: {e}")
    
    def write_additional_flags_section(self, flags_section: AdditionalFlagsSection) -> None:
        """
        Write an additional flags section to the file.
        
        Args:
            flags_section: The additional flags data to write.
            
        Raises:
            SPDValidationError: If the additional flags data is invalid.
            RuntimeError: If the file is not open.
        """
        # Validate additional flags data
        if flags_section.num_flags <= 0:
            raise SPDValidationError("Number of additional flags must be positive")
        
        if not flags_section.flags:
            raise SPDValidationError("Additional flags cannot be empty")
        
        if len(flags_section.flags) != flags_section.num_flags:
            raise SPDValidationError(
                f"Additional flags count mismatch: expected {flags_section.num_flags}, got {len(flags_section.flags)}"
            )
        
        # Validate flag types
        for key, value in flags_section.flags.items():
            if not isinstance(key, str) or not key:
                raise SPDValidationError(f"Flag key must be a non-empty string: {key}")
                
            if not isinstance(value, (bool, int, float, str)):
                raise SPDValidationError(f"Flag value must be bool, int, float, or str: {key}={value}")
        
        # Write section marker
        self._write_section_marker(SectionMarker.ADDITIONAL_FLAGS)
        
        try:
            # Calculate section size
            # 4 bytes num_flags + 
            # (4 bytes key length + key bytes + 1 byte type + type-specific size) * num_flags
            section_size = 4
            
            # Determine size of each flag
            for key, value in flags_section.flags.items():
                key_bytes = key.encode('utf-8')
                section_size += 4 + len(key_bytes) + 1  # 4 bytes length, key bytes, 1 byte type
                
                if isinstance(value, bool):
                    section_size += 1  # 1 byte for bool
                elif isinstance(value, int):
                    section_size += 4  # 4 bytes for int
                elif isinstance(value, float):
                    section_size += 4  # 4 bytes for float
                elif isinstance(value, str):
                    val_bytes = value.encode('utf-8')
                    section_size += 4 + len(val_bytes)  # 4 bytes length + string bytes
            
            # Write section size
            self._file.write(struct.pack("!I", section_size))
            
            # Write number of flags
            self._file.write(struct.pack("!I", flags_section.num_flags))
            
            # Write flags
            for key, value in flags_section.flags.items():
                # Write key
                self._write_string(key)
                
                # Write type code and value
                if isinstance(value, bool):
                    self._file.write(bytes([0]))  # Type code 0 = boolean
                    self._file.write(bytes([1 if value else 0]))  # Boolean value
                elif isinstance(value, int):
                    self._file.write(bytes([1]))  # Type code 1 = integer
                    self._file.write(struct.pack("!i", value))  # Integer value
                elif isinstance(value, float):
                    self._file.write(bytes([2]))  # Type code 2 = float
                    self._file.write(struct.pack("!f", value))  # Float value
                elif isinstance(value, str):
                    self._file.write(bytes([3]))  # Type code 3 = string
                    self._write_string(value)  # String value
            
            # Mark section as written
            self._sections_written[SectionMarker.ADDITIONAL_FLAGS] = True
            logger.debug(f"Wrote additional flags section: {flags_section.num_flags} flags")
        except (IOError, struct.error) as e:
            raise SPDFormatError(f"Failed to write additional flags section: {e}")
    
    def write_end_marker(self) -> None:
        """
        Write the END marker to the file.
        
        Raises:
            RuntimeError: If the file is not open.
        """
        # Write section marker
        self._write_section_marker(SectionMarker.END)
        
        try:
            # Write section size (always 0 for END marker)
            self._file.write(struct.pack("!I", 0))
            
            # Mark section as written
            self._sections_written[SectionMarker.END] = True
            logger.debug("Wrote END marker")
        except (IOError, struct.error) as e:
            raise SPDFormatError(f"Failed to write END marker: {e}")
    
    def finalize(self) -> None:
        """
        Finalize the SPD file by updating the header and writing the END marker if needed.
        
        Raises:
            RuntimeError: If the file is not open.
        """
        # Update the header with final flags
        self._update_header_in_file()
        
        # If END not written, write it now
        if not self._sections_written[SectionMarker.END]:
            self.write_end_marker()
    
    def close(self) -> None:
        """
        Close the SPD file if it was opened by this writer.
        """
        if self._file is not None and self._owns_file:
            # Finalize the file before closing
            try:
                self.finalize()
            except SPDError:
                pass  # Don't let finalize errors prevent file closure
                
            self._file.close()
            self._file = None
            logger.debug(f"Closed file: {self._file_path}")
    
    def __enter__(self) -> 'SPDWriter':
        """
        Context manager entry method.
        
        Returns:
            The SPDWriter instance.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit method. Closes the file.
        """
        self.close()