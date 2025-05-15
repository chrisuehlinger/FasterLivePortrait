"""
SPD (Source Portrait Descriptor) reader implementation.

This module provides functionality to read SPD files according to the format
defined in the format module.
"""
import os
import io
import struct
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, BinaryIO, Tuple, Any, TypeVar, Type, cast, Callable
from typing_extensions import TypedDict, Literal

from .format import (
    MAGIC_BYTES, CURRENT_VERSION,
    HeaderFlags, SectionMarker,
    HeaderSection, ImageSection, LandmarksSection, 
    MotionParamsSection, AppearanceSection, TransformationSection,
    MaskSection, AdditionalFlagsSection,
    validate_header, flags_to_dict
)
from .types import (
    SPDFileInfo, ValidationResult
)

# Set up logger for this module
logger = logging.getLogger(__name__)

# Type aliases
PathLike = Union[str, Path]
FileOrPath = Union[BinaryIO, PathLike]
T = TypeVar('T')
SectionType = TypeVar('SectionType')

# Error severity levels
ErrorSeverity = Literal["error", "warning", "info"]

# Custom TypedDict for error information
class ErrorInfo(TypedDict):
    """TypedDict representing detailed error information."""
    message: str
    section: Optional[str]
    code: str
    severity: ErrorSeverity
    details: Optional[Dict[str, Any]]


class SPDError(Exception):
    """Base class for all SPD-related exceptions."""
    
    def __init__(self, message: str, code: str = "general_error", 
                 severity: ErrorSeverity = "error", 
                 details: Optional[Dict[str, Any]] = None) -> None:
        self.message = message
        self.code = code
        self.severity = severity
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> ErrorInfo:
        """Convert error to dictionary format."""
        return {
            "message": self.message,
            "section": None,
            "code": self.code,
            "severity": self.severity,
            "details": self.details
        }


class SPDFormatError(SPDError):
    """Raised when the SPD file format is invalid."""
    
    def __init__(self, message: str, code: str = "format_error", 
                 severity: ErrorSeverity = "error",
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code, severity, details)


class SPDVersionError(SPDError):
    """Raised when the SPD file version is not supported."""
    
    def __init__(self, message: str, version: int, 
                 code: str = "version_error", 
                 severity: ErrorSeverity = "error") -> None:
        details = {"version": version, "supported_version": CURRENT_VERSION}
        super().__init__(message, code, severity, details)


class SPDSectionError(SPDError):
    """Raised when there's an issue with an SPD section."""
    pass


class SPDReader:
    """
    Reader for SPD (Source Portrait Descriptor) files.
    
    Provides methods to read and access data from SPD files.
    Supports lazy loading of sections for memory efficiency.
    
    Attributes:
        header: The header section of the SPD file.
        file_path: The path to the SPD file (if opened from a path).
    """
    
    def __init__(self, file_or_path: FileOrPath, eager_load: bool = False):
        """
        Initialize the SPD reader.
        
        Args:
            file_or_path: A file path or file-like object to read from.
            eager_load: If True, load all sections immediately. Otherwise, use lazy loading.
        
        Raises:
            SPDFormatError: If the file is not a valid SPD file.
            SPDVersionError: If the file version is not supported.
            IOError: If there's an issue opening or reading the file.
        """
        self._file: Optional[BinaryIO] = None
        self._owns_file = False
        self.file_path: Optional[str] = None
        
        # Initialize with empty sections, will be populated as needed
        self._header: Optional[HeaderSection] = None
        self._image: Optional[ImageSection] = None
        self._landmarks: Optional[LandmarksSection] = None
        self._motion_params: Optional[MotionParamsSection] = None
        self._appearance: Optional[AppearanceSection] = None
        self._transforms: Optional[TransformationSection] = None
        self._mask: Optional[MaskSection] = None
        self._additional_flags: Optional[AdditionalFlagsSection] = None
        
        # Track which sections have been read
        self._sections_read: Dict[SectionMarker, bool] = {
            marker: False for marker in SectionMarker
        }
        
        # Track section offsets for faster seeking
        self._section_offsets: Dict[SectionMarker, int] = {}
        
        # Open the file and read the header
        self._open_file(file_or_path)
        self._read_header()
        
        # If eager loading, read all sections now
        if eager_load:
            self._load_all_sections()
    
    def _open_file(self, file_or_path: FileOrPath) -> None:
        """
        Open the SPD file.
        
        Args:
            file_or_path: A file path or file-like object.
        
        Raises:
            IOError: If there's an issue opening the file.
        """
        if isinstance(file_or_path, (str, Path)):
            self.file_path = str(file_or_path)
            try:
                self._file = open(file_or_path, "rb")
                self._owns_file = True
            except IOError as e:
                raise IOError(f"Failed to open SPD file '{file_or_path}': {e}")
        else:
            # Assume it's already a file-like object
            self._file = file_or_path
            self._owns_file = False
            
            # Try to get the name if available
            if hasattr(file_or_path, 'name'):
                self.file_path = str(file_or_path.name)
    
    def _read_header(self) -> None:
        """
        Read and validate the header of the SPD file.
        
        Raises:
            SPDFormatError: If the file is not a valid SPD file.
            SPDVersionError: If the file version is not supported.
        """
        if self._file is None:
            raise RuntimeError("File is not open")
        
        # Read the header bytes (20 bytes)
        # 4 bytes magic + 4 bytes version + 8 bytes timestamp + 4 bytes flags
        self._file.seek(0)
        header_bytes = self._file.read(20)
        
        # Validate the header
        is_valid, error_msg = validate_header(header_bytes)
        if not is_valid:
            raise SPDFormatError(f"Invalid SPD header: {error_msg}")
        
        # Parse the header
        try:
            magic = header_bytes[:4]
            version = struct.unpack("!I", header_bytes[4:8])[0]
            timestamp = struct.unpack("!d", header_bytes[8:16])[0]
            flags = struct.unpack("!I", header_bytes[16:20])[0]
            
            self._header = HeaderSection(
                magic_bytes=magic,
                version=version,
                timestamp=timestamp,
                flags=flags
            )
            
            # Mark the header as read
            self._sections_read[SectionMarker.HEADER] = True
            self._section_offsets[SectionMarker.HEADER] = 0
            
            # Log header info
            logger.debug(f"Read SPD header: version {version}, flags {flags}")
        
        except struct.error as e:
            raise SPDFormatError(f"Failed to parse SPD header: {e}")
        
        # Scan for section markers and record their positions
        self._scan_section_offsets()
    
    def _scan_section_offsets(self) -> None:
        """
        Scan the file to find the offsets of each section.
        This allows for faster seeking when lazy loading sections.
        """
        if self._file is None:
            raise RuntimeError("File is not open")
            
        current_pos = self._file.tell()
        
        try:
            # Start from the end of the header
            self._file.seek(20)
            
            # Try to read markers sequentially instead of scanning byte-by-byte
            while True:
                pos = self._file.tell()
                marker_bytes = self._file.read(4)
                if not marker_bytes or len(marker_bytes) < 4:
                    break  # End of file
                    
                try:
                    marker = struct.unpack("!I", marker_bytes)[0]
                    
                    # Check if this is a valid section marker
                    try:
                        section_marker = SectionMarker(marker)
                        logger.debug(f"Found section marker {section_marker.name} at offset {pos}")
                        self._section_offsets[section_marker] = pos
                        
                        # If it's the END marker, we're done
                        if section_marker == SectionMarker.END:
                            break
                        
                        # Read section size and skip to the next section
                        size_bytes = self._file.read(4)
                        if not size_bytes or len(size_bytes) < 4:
                            break  # Unexpected end of file
                            
                        section_size = struct.unpack("!I", size_bytes)[0]
                        self._file.seek(section_size, os.SEEK_CUR)
                            
                    except ValueError:
                        # Not a valid section marker, try the next byte
                        self._file.seek(-3, os.SEEK_CUR)  # Move back 3 bytes to check the next byte
                        
                except struct.error:
                    # Invalid data, try the next byte
                    self._file.seek(-3, os.SEEK_CUR)  # Move back 3 bytes to check the next byte
        
        finally:
            # Restore the original file position
            self._file.seek(current_pos)
    
    def _read_section_marker(self) -> Optional[SectionMarker]:
        """
        Read a section marker from the current position in the file.
        
        Returns:
            A SectionMarker if found, None if end of file.
        
        Raises:
            SPDFormatError: If the marker is invalid.
        """
        if self._file is None:
            raise RuntimeError("File is not open")
            
        marker_bytes = self._file.read(4)
        if not marker_bytes or len(marker_bytes) < 4:
            return None  # End of file
        
        try:
            marker = struct.unpack("!I", marker_bytes)[0]
            
            try:
                return SectionMarker(marker)
            except ValueError:
                raise SPDFormatError(f"Invalid section marker: 0x{marker:08x}")
                
        except struct.error as e:
            raise SPDFormatError(f"Failed to parse section marker: {e}")
    
    def _seek_to_section(self, section_marker: SectionMarker) -> bool:
        """
        Seek to the beginning of a specific section.
        
        Args:
            section_marker: The section marker to seek to.
            
        Returns:
            True if the section was found, False otherwise.
        """
        if self._file is None:
            raise RuntimeError("File is not open")
            
        # If we know the offset, seek directly to it
        if section_marker in self._section_offsets:
            self._file.seek(self._section_offsets[section_marker])
            return True
        
        # Otherwise, scan the file for the section
        self._file.seek(20)  # Start after the header
        
        while True:
            try:
                marker = self._read_section_marker()
                if marker is None:
                    return False  # End of file
                
                if marker == section_marker:
                    return True
                
                # Skip this section by reading the size and seeking forward
                size_bytes = self._file.read(4)
                if not size_bytes or len(size_bytes) < 4:
                    return False  # Unexpected end of file
                    
                try:
                    size = struct.unpack("!I", size_bytes)[0]
                    self._file.seek(size, os.SEEK_CUR)
                except struct.error:
                    return False  # Invalid size
            except SPDFormatError:
                # Skip this invalid section marker and try to find a valid one
                # Move forward 1 byte and try again
                self._file.seek(-3, os.SEEK_CUR)
        
        return False
    
    def _read_string(self, max_length: int = 1024) -> str:
        """
        Read a null-terminated string from the current position in the file.
        
        Args:
            max_length: Maximum length of the string to read.
            
        Returns:
            The string read from the file.
            
        Raises:
            SPDFormatError: If the string is invalid.
        """
        if self._file is None:
            raise RuntimeError("File is not open")
            
        # Read string length (4 bytes)
        length_bytes = self._file.read(4)
        if not length_bytes or len(length_bytes) < 4:
            raise SPDFormatError("Unexpected end of file while reading string length")
            
        try:
            length = struct.unpack("!I", length_bytes)[0]
            if length > max_length:
                raise SPDFormatError(f"String too long: {length} > {max_length}")
                
            # Read the string data
            string_bytes = self._file.read(length)
            if len(string_bytes) < length:
                raise SPDFormatError("Unexpected end of file while reading string data")
                
            # Decode as UTF-8
            return string_bytes.decode('utf-8')
            
        except struct.error as e:
            raise SPDFormatError(f"Failed to parse string length: {e}")
        except UnicodeDecodeError as e:
            raise SPDFormatError(f"Failed to decode string as UTF-8: {e}")
    
    def _read_image_section(self) -> ImageSection:
        """
        Read the image section from the file.
        
        Returns:
            An ImageSection containing the image data.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._file is None:
            raise RuntimeError("File is not open")
            
        # Skip the section marker, which was already read
        # Read section size (4 bytes)
        size_bytes = self._file.read(4)
        if not size_bytes or len(size_bytes) < 4:
            raise SPDSectionError("Unexpected end of file while reading image section size")
            
        try:
            section_size = struct.unpack("!I", size_bytes)[0]
            section_start = self._file.tell()
            
            # Read image dimensions
            width_bytes = self._file.read(4)
            height_bytes = self._file.read(4)
            channels_bytes = self._file.read(1)
            
            if len(width_bytes) < 4 or len(height_bytes) < 4 or len(channels_bytes) < 1:
                raise SPDSectionError("Unexpected end of file while reading image dimensions")
                
            width = struct.unpack("!I", width_bytes)[0]
            height = struct.unpack("!I", height_bytes)[0]
            channels = channels_bytes[0]
            
            # Read image format
            format_str = self._read_string()
            
            # Calculate expected data size
            header_size = (self._file.tell() - section_start)
            data_size = section_size - header_size
            
            # Read image data
            data = self._file.read(data_size)
            if len(data) < data_size:
                raise SPDSectionError("Unexpected end of file while reading image data")
                
            return ImageSection(
                width=width,
                height=height,
                channels=channels,
                format=format_str,
                data=data
            )
            
        except struct.error as e:
            raise SPDSectionError(f"Failed to parse image section: {e}")
    
    def _read_landmarks_section(self) -> LandmarksSection:
        """
        Read the landmarks section from the file.
        
        Returns:
            A LandmarksSection containing the landmark data.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._file is None:
            raise RuntimeError("File is not open")
            
        # Skip the section marker, which was already read
        # Read section size (4 bytes)
        size_bytes = self._file.read(4)
        if not size_bytes or len(size_bytes) < 4:
            raise SPDSectionError("Unexpected end of file while reading landmarks section size")
            
        try:
            section_size = struct.unpack("!I", size_bytes)[0]
            section_start = self._file.tell()
            
            # Read landmark count and dimensions
            count_bytes = self._file.read(4)
            dimensions_bytes = self._file.read(4)
            
            if len(count_bytes) < 4 or len(dimensions_bytes) < 4:
                raise SPDSectionError("Unexpected end of file while reading landmark dimensions")
                
            count = struct.unpack("!I", count_bytes)[0]
            dimensions = struct.unpack("!I", dimensions_bytes)[0]
            
            # Read landmark type
            landmark_type = self._read_string()
            
            # Read landmark points
            points = []
            for _ in range(count):
                point = []
                for _ in range(dimensions):
                    coord_bytes = self._file.read(4)
                    if len(coord_bytes) < 4:
                        raise SPDSectionError("Unexpected end of file while reading landmark coordinates")
                    coord = struct.unpack("!f", coord_bytes)[0]
                    point.append(coord)
                points.append(point)
                
            return LandmarksSection(
                count=count,
                dimensions=dimensions,
                landmark_type=landmark_type,
                points=points
            )
            
        except struct.error as e:
            raise SPDSectionError(f"Failed to parse landmarks section: {e}")
    
    def _read_motion_params_section(self) -> MotionParamsSection:
        """
        Read the motion parameters section from the file.
        
        Returns:
            A MotionParamsSection containing the motion parameter data.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._file is None:
            raise RuntimeError("File is not open")
            
        # Skip the section marker, which was already read
        # Read section size (4 bytes)
        size_bytes = self._file.read(4)
        if not size_bytes or len(size_bytes) < 4:
            raise SPDSectionError("Unexpected end of file while reading motion params section size")
            
        try:
            section_size = struct.unpack("!I", size_bytes)[0]
            section_start = self._file.tell()
            
            # Read parameter count
            num_params_bytes = self._file.read(4)
            if len(num_params_bytes) < 4:
                raise SPDSectionError("Unexpected end of file while reading motion params count")
                
            num_params = struct.unpack("!I", num_params_bytes)[0]
            
            # Read parameter names
            param_names = []
            for _ in range(num_params):
                param_name = self._read_string()
                param_names.append(param_name)
            
            # Read parameter values
            values = []
            for _ in range(num_params):
                value_bytes = self._file.read(4)
                if len(value_bytes) < 4:
                    raise SPDSectionError("Unexpected end of file while reading motion param values")
                value = struct.unpack("!f", value_bytes)[0]
                values.append(value)
                
            return MotionParamsSection(
                num_params=num_params,
                param_names=param_names,
                values=values
            )
            
        except struct.error as e:
            raise SPDSectionError(f"Failed to parse motion params section: {e}")
    
    def _read_appearance_section(self) -> AppearanceSection:
        """
        Read the appearance features section from the file.
        
        Returns:
            An AppearanceSection containing the feature data.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._file is None:
            raise RuntimeError("File is not open")
            
        # Skip the section marker, which was already read
        # Read section size (4 bytes)
        size_bytes = self._file.read(4)
        if not size_bytes or len(size_bytes) < 4:
            raise SPDSectionError("Unexpected end of file while reading appearance section size")
            
        try:
            section_size = struct.unpack("!I", size_bytes)[0]
            section_start = self._file.tell()
            
            # Read feature dimensions
            feature_dim_bytes = self._file.read(4)
            if len(feature_dim_bytes) < 4:
                raise SPDSectionError("Unexpected end of file while reading feature dimensions")
                
            feature_dim = struct.unpack("!I", feature_dim_bytes)[0]
            
            # Read feature type
            feature_type = self._read_string()
            
            # Read feature values
            features = []
            for _ in range(feature_dim):
                feature_bytes = self._file.read(4)
                if len(feature_bytes) < 4:
                    raise SPDSectionError("Unexpected end of file while reading feature values")
                feature = struct.unpack("!f", feature_bytes)[0]
                features.append(feature)
                
            return AppearanceSection(
                feature_dim=feature_dim,
                feature_type=feature_type,
                features=features
            )
            
        except struct.error as e:
            raise SPDSectionError(f"Failed to parse appearance section: {e}")
    
    def _read_transformation_section(self) -> TransformationSection:
        """
        Read the transformation matrices section from the file.
        
        Returns:
            A TransformationSection containing the transformation data.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._file is None:
            raise RuntimeError("File is not open")
            
        # Skip the section marker, which was already read
        # Read section size (4 bytes)
        size_bytes = self._file.read(4)
        if not size_bytes or len(size_bytes) < 4:
            raise SPDSectionError("Unexpected end of file while reading transformation section size")
            
        try:
            section_size = struct.unpack("!I", size_bytes)[0]
            section_start = self._file.tell()
            
            # Read number of transformations
            num_transforms_bytes = self._file.read(4)
            if len(num_transforms_bytes) < 4:
                raise SPDSectionError("Unexpected end of file while reading number of transforms")
                
            num_transforms = struct.unpack("!I", num_transforms_bytes)[0]
            
            # Read transformation types
            transform_types = []
            for _ in range(num_transforms):
                transform_type = self._read_string()
                transform_types.append(transform_type)
            
            # Read transformation matrices
            # Each matrix is assumed to be 3x3 = 9 float values
            matrices = []
            for _ in range(num_transforms):
                matrix = []
                for _ in range(9):  # 3x3 matrix
                    value_bytes = self._file.read(4)
                    if len(value_bytes) < 4:
                        raise SPDSectionError("Unexpected end of file while reading transformation matrix")
                    value = struct.unpack("!f", value_bytes)[0]
                    matrix.append(value)
                matrices.append(matrix)
                
            return TransformationSection(
                num_transforms=num_transforms,
                transform_types=transform_types,
                matrices=matrices
            )
            
        except struct.error as e:
            raise SPDSectionError(f"Failed to parse transformation section: {e}")
    
    def _read_mask_section(self) -> MaskSection:
        """
        Read the mask section from the file.
        
        Returns:
            A MaskSection containing the mask data.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._file is None:
            raise RuntimeError("File is not open")
            
        # Skip the section marker, which was already read
        # Read section size (4 bytes)
        size_bytes = self._file.read(4)
        if not size_bytes or len(size_bytes) < 4:
            raise SPDSectionError("Unexpected end of file while reading mask section size")
            
        try:
            section_size = struct.unpack("!I", size_bytes)[0]
            section_start = self._file.tell()
            
            # Read mask dimensions
            width_bytes = self._file.read(4)
            height_bytes = self._file.read(4)
            
            if len(width_bytes) < 4 or len(height_bytes) < 4:
                raise SPDSectionError("Unexpected end of file while reading mask dimensions")
                
            width = struct.unpack("!I", width_bytes)[0]
            height = struct.unpack("!I", height_bytes)[0]
            
            # Read mask format
            format_str = self._read_string()
            
            # Calculate expected data size
            header_size = (self._file.tell() - section_start)
            data_size = section_size - header_size
            
            # Read mask data
            data = self._file.read(data_size)
            if len(data) < data_size:
                raise SPDSectionError("Unexpected end of file while reading mask data")
                
            return MaskSection(
                width=width,
                height=height,
                format=format_str,
                data=data
            )
            
        except struct.error as e:
            raise SPDSectionError(f"Failed to parse mask section: {e}")
    
    def _read_additional_flags_section(self) -> AdditionalFlagsSection:
        """
        Read the additional flags section from the file.
        
        Returns:
            An AdditionalFlagsSection containing the flags data.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._file is None:
            raise RuntimeError("File is not open")
            
        # Skip the section marker, which was already read
        # Read section size (4 bytes)
        size_bytes = self._file.read(4)
        if not size_bytes or len(size_bytes) < 4:
            raise SPDSectionError("Unexpected end of file while reading additional flags section size")
            
        try:
            section_size = struct.unpack("!I", size_bytes)[0]
            section_start = self._file.tell()
            
            # Read number of flags
            num_flags_bytes = self._file.read(4)
            if len(num_flags_bytes) < 4:
                raise SPDSectionError("Unexpected end of file while reading number of flags")
                
            num_flags = struct.unpack("!I", num_flags_bytes)[0]
            
            # Read flags as key-value pairs
            flags = {}
            for _ in range(num_flags):
                # Read flag key
                key = self._read_string()
                
                # Read flag value type (1 byte)
                type_byte = self._file.read(1)
                if len(type_byte) < 1:
                    raise SPDSectionError("Unexpected end of file while reading flag value type")
                    
                value_type = type_byte[0]
                
                # Read flag value based on type
                if value_type == 0:  # boolean
                    value_byte = self._file.read(1)
                    value = bool(value_byte[0])
                elif value_type == 1:  # integer
                    value_bytes = self._file.read(4)
                    value = struct.unpack("!i", value_bytes)[0]
                elif value_type == 2:  # float
                    value_bytes = self._file.read(4)
                    value = struct.unpack("!f", value_bytes)[0]
                elif value_type == 3:  # string
                    value = self._read_string()
                else:
                    raise SPDSectionError(f"Unknown flag value type: {value_type}")
                
                flags[key] = value
                
            return AdditionalFlagsSection(
                num_flags=num_flags,
                flags=flags
            )
            
        except struct.error as e:
            raise SPDSectionError(f"Failed to parse additional flags section: {e}")
    
    def _load_all_sections(self) -> None:
        """
        Load all sections from the file based on header flags.
        
        This method reads all available sections in the file, guided by
        the flags set in the header.
        
        Raises:
            SPDSectionError: If a required section is missing or corrupt.
        """
        if self._header is None:
            raise RuntimeError("Header has not been read yet")
            
        # Get flags as a dictionary for easier access
        flags = flags_to_dict(self._header.flags)
        
        # Load sections based on flags
        if flags.get("HAS_IMAGE", False):
            try:
                self.image
            except SPDSectionError:
                logger.warning("Failed to read image section despite HAS_IMAGE flag")
            
        if flags.get("HAS_LANDMARKS", False):
            try:
                self.landmarks
            except SPDSectionError:
                logger.warning("Failed to read landmarks section despite HAS_LANDMARKS flag")
            
        if flags.get("HAS_MOTION_PARAMS", False):
            try:
                self.motion_params
            except SPDSectionError:
                logger.warning("Failed to read motion parameters section despite HAS_MOTION_PARAMS flag")
            
        if flags.get("HAS_APPEARANCE", False):
            try:
                self.appearance
            except SPDSectionError:
                logger.warning("Failed to read appearance section despite HAS_APPEARANCE flag")
            
        if flags.get("HAS_TRANSFORMS", False):
            try:
                self.transforms
            except SPDSectionError:
                logger.warning("Failed to read transformations section despite HAS_TRANSFORMS flag")
            
        if flags.get("HAS_MASK", False):
            try:
                self.mask
            except SPDSectionError:
                logger.warning("Failed to read mask section despite HAS_MASK flag")
        
        # Always try to read additional flags if present
        try:
            self.additional_flags
        except SPDSectionError:
            pass  # Additional flags section is optional
    
    @property
    def header(self) -> HeaderSection:
        """
        Get the header section of the SPD file.
        
        Returns:
            The HeaderSection object.
        """
        if self._header is None:
            self._read_header()
        return self._header
    
    @property
    def image(self) -> ImageSection:
        """
        Get the image section of the SPD file.
        
        Returns:
            The ImageSection object.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._image is None:
            # Check if the section should exist
            if not (self._header and (self._header.flags & HeaderFlags.HAS_IMAGE.value)):
                raise SPDSectionError("SPD file does not contain an image section")
                
            # Try to read the section
            if self._seek_to_section(SectionMarker.IMAGE):
                marker = self._read_section_marker()
                if marker != SectionMarker.IMAGE:
                    raise SPDSectionError(f"Expected IMAGE section marker, got {marker}")
                    
                self._image = self._read_image_section()
                self._sections_read[SectionMarker.IMAGE] = True
            else:
                raise SPDSectionError("Failed to find image section in SPD file")
                
        return self._image
    
    @property
    def landmarks(self) -> LandmarksSection:
        """
        Get the landmarks section of the SPD file.
        
        Returns:
            The LandmarksSection object.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._landmarks is None:
            # Check if the section should exist
            if not (self._header and (self._header.flags & HeaderFlags.HAS_LANDMARKS.value)):
                raise SPDSectionError("SPD file does not contain a landmarks section")
                
            # Try to read the section
            if self._seek_to_section(SectionMarker.LANDMARKS):
                marker = self._read_section_marker()
                if marker != SectionMarker.LANDMARKS:
                    raise SPDSectionError(f"Expected LANDMARKS section marker, got {marker}")
                    
                self._landmarks = self._read_landmarks_section()
                self._sections_read[SectionMarker.LANDMARKS] = True
            else:
                raise SPDSectionError("Failed to find landmarks section in SPD file")
                
        return self._landmarks
    
    @property
    def motion_params(self) -> MotionParamsSection:
        """
        Get the motion parameters section of the SPD file.
        
        Returns:
            The MotionParamsSection object.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._motion_params is None:
            # Check if the section should exist
            if not (self._header and (self._header.flags & HeaderFlags.HAS_MOTION_PARAMS.value)):
                raise SPDSectionError("SPD file does not contain a motion parameters section")
                
            # Try to read the section
            if self._seek_to_section(SectionMarker.MOTION_PARAMS):
                marker = self._read_section_marker()
                if marker != SectionMarker.MOTION_PARAMS:
                    raise SPDSectionError(f"Expected MOTION_PARAMS section marker, got {marker}")
                    
                self._motion_params = self._read_motion_params_section()
                self._sections_read[SectionMarker.MOTION_PARAMS] = True
            else:
                raise SPDSectionError("Failed to find motion parameters section in SPD file")
                
        return self._motion_params
    
    @property
    def appearance(self) -> AppearanceSection:
        """
        Get the appearance features section of the SPD file.
        
        Returns:
            The AppearanceSection object.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._appearance is None:
            # Check if the section should exist
            if not (self._header and (self._header.flags & HeaderFlags.HAS_APPEARANCE.value)):
                raise SPDSectionError("SPD file does not contain an appearance section")
                
            # Try to read the section
            if self._seek_to_section(SectionMarker.APPEARANCE):
                marker = self._read_section_marker()
                if marker != SectionMarker.APPEARANCE:
                    raise SPDSectionError(f"Expected APPEARANCE section marker, got {marker}")
                    
                self._appearance = self._read_appearance_section()
                self._sections_read[SectionMarker.APPEARANCE] = True
            else:
                raise SPDSectionError("Failed to find appearance section in SPD file")
                
        return self._appearance
    
    @property
    def transforms(self) -> TransformationSection:
        """
        Get the transformation matrices section of the SPD file.
        
        Returns:
            The TransformationSection object.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._transforms is None:
            # Check if the section should exist
            if not (self._header and (self._header.flags & HeaderFlags.HAS_TRANSFORMS.value)):
                raise SPDSectionError("SPD file does not contain a transformation section")
                
            # Try to read the section
            if self._seek_to_section(SectionMarker.TRANSFORMS):
                marker = self._read_section_marker()
                if marker != SectionMarker.TRANSFORMS:
                    raise SPDSectionError(f"Expected TRANSFORMS section marker, got {marker}")
                    
                self._transforms = self._read_transformation_section()
                self._sections_read[SectionMarker.TRANSFORMS] = True
            else:
                raise SPDSectionError("Failed to find transformation section in SPD file")
                
        return self._transforms
    
    @property
    def mask(self) -> MaskSection:
        """
        Get the mask section of the SPD file.
        
        Returns:
            The MaskSection object.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._mask is None:
            # Check if the section should exist
            if not (self._header and (self._header.flags & HeaderFlags.HAS_MASK.value)):
                raise SPDSectionError("SPD file does not contain a mask section")
                
            # Try to read the section
            if self._seek_to_section(SectionMarker.MASK):
                marker = self._read_section_marker()
                if marker != SectionMarker.MASK:
                    raise SPDSectionError(f"Expected MASK section marker, got {marker}")
                    
                self._mask = self._read_mask_section()
                self._sections_read[SectionMarker.MASK] = True
            else:
                raise SPDSectionError("Failed to find mask section in SPD file")
                
        return self._mask
    
    @property
    def additional_flags(self) -> AdditionalFlagsSection:
        """
        Get the additional flags section of the SPD file.
        
        Returns:
            The AdditionalFlagsSection object.
            
        Raises:
            SPDSectionError: If the section is missing or corrupt.
        """
        if self._additional_flags is None:
            # Try to read the section
            if self._seek_to_section(SectionMarker.ADDITIONAL_FLAGS):
                marker = self._read_section_marker()
                if marker != SectionMarker.ADDITIONAL_FLAGS:
                    raise SPDSectionError(f"Expected ADDITIONAL_FLAGS section marker, got {marker}")
                    
                self._additional_flags = self._read_additional_flags_section()
                self._sections_read[SectionMarker.ADDITIONAL_FLAGS] = True
            else:
                # Additional flags section is optional
                self._additional_flags = AdditionalFlagsSection()
                
        return self._additional_flags
    
    def get_available_sections(self) -> List[str]:
        """
        Get a list of section names that are available in this SPD file.
        
        Returns:
            A list of section names as strings.
        """
        if self._header is None:
            self._read_header()
            
        flags = flags_to_dict(self._header.flags)
        available = []
        
        if flags.get("HAS_IMAGE", False):
            available.append("image")
        if flags.get("HAS_LANDMARKS", False):
            available.append("landmarks")
        if flags.get("HAS_MOTION_PARAMS", False):
            available.append("motion_params")
        if flags.get("HAS_APPEARANCE", False):
            available.append("appearance")
        if flags.get("HAS_TRANSFORMS", False):
            available.append("transforms")
        if flags.get("HAS_MASK", False):
            available.append("mask")
            
        # Additional flags section is not tied to a header flag
        if SectionMarker.ADDITIONAL_FLAGS in self._section_offsets:
            available.append("additional_flags")
            
        return available
    
    def has_section(self, section_name: str) -> bool:
        """
        Check if a specific section is available in the SPD file.
        
        Args:
            section_name: Name of the section to check for.
            
        Returns:
            True if the section is available, False otherwise.
        """
        return section_name in self.get_available_sections()
    
    def close(self) -> None:
        """
        Close the SPD file if it was opened by this reader.
        """
        if self._file and self._owns_file:
            self._file.close()
            self._file = None
    
    def __enter__(self) -> 'SPDReader':
        """
        Enter the context manager protocol.
        
        Returns:
            This SPDReader instance.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context manager protocol.
        
        Closes the file if it was opened by this reader.
        """
        self.close()