"""
SPD (Source Portrait Descriptor) validator implementation.

This module provides functionality to validate SPD files according to the format
defined in the format module, with various levels of strictness.
"""
import os
import logging
import time
import math
import struct
from enum import Enum, auto
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple, Set, BinaryIO, cast, Callable
from typing_extensions import TypedDict, Protocol, Literal

from .format import (
    MAGIC_BYTES, CURRENT_VERSION,
    HeaderFlags, SectionMarker,
    HeaderSection, ImageSection, LandmarksSection, 
    MotionParamsSection, AppearanceSection, TransformationSection,
    MaskSection, AdditionalFlagsSection,
    validate_header, flags_to_dict, get_required_flags_for_section
)
from .reader import (
    SPDReader, SPDError, SPDFormatError, SPDSectionError, PathLike, FileOrPath
)

# Set up logger for this module
logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels for SPD files."""
    BASIC = auto()    # Basic format validation - can the file be opened/parsed
    STANDARD = auto()  # Standard validation - are sections consistent with flags
    STRICT = auto()   # Strict validation - detailed data integrity checks


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = auto()     # Informational message, not an issue
    WARNING = auto()  # Warning, may cause issues but not critical
    ERROR = auto()    # Critical error that makes the file invalid


@dataclass
class ValidationIssue:
    """
    Represents a validation issue found in an SPD file.
    
    Attributes:
        section: The section where the issue was found.
        severity: The severity of the issue.
        message: A description of the issue.
        suggestion: Optional suggestion for fixing the issue.
    """
    section: str
    severity: ValidationSeverity
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """
    Report for SPD file validation containing issues and summary.
    
    Attributes:
        is_valid: Whether the file is considered valid.
        validation_level: The level of validation performed.
        issues: List of validation issues found.
        elapsed_time_ms: Time taken to validate the file in milliseconds.
        sections_checked: List of sections that were checked.
        sections_valid: List of sections that were found to be valid.
        sections_invalid: List of sections that were found to be invalid.
        file_path: Path to the validated file if available.
    """
    is_valid: bool = True
    validation_level: ValidationLevel = ValidationLevel.BASIC
    issues: List[ValidationIssue] = field(default_factory=list)
    elapsed_time_ms: float = 0.0
    sections_checked: List[str] = field(default_factory=list)
    sections_valid: List[str] = field(default_factory=list)
    sections_invalid: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    
    def add_issue(
        self, 
        section: str, 
        severity: ValidationSeverity, 
        message: str, 
        suggestion: Optional[str] = None
    ) -> None:
        """
        Add an issue to the validation report.
        
        Args:
            section: The section where the issue was found.
            severity: The severity of the issue.
            message: A description of the issue.
            suggestion: Optional suggestion for fixing the issue.
        """
        issue = ValidationIssue(
            section=section,
            severity=severity,
            message=message,
            suggestion=suggestion
        )
        self.issues.append(issue)
        
        # Mark the section as invalid if this is an error
        if severity == ValidationSeverity.ERROR:
            self.is_valid = False
            if section not in self.sections_invalid:
                self.sections_invalid.append(section)
                if section in self.sections_valid:
                    self.sections_valid.remove(section)
    
    def get_error_count(self) -> int:
        """Get the number of errors in the report."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)
    
    def get_warning_count(self) -> int:
        """Get the number of warnings in the report."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)
    
    def get_info_count(self) -> int:
        """Get the number of info messages in the report."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO)
    
    def summarize(self) -> str:
        """
        Generate a summary of the validation report.
        
        Returns:
            A string summary of the validation report.
        """
        status = "VALID" if self.is_valid else "INVALID"
        file_info = f" for {self.file_path}" if self.file_path else ""
        
        summary = [
            f"SPD Validation Report{file_info}",
            f"Status: {status}",
            f"Validation Level: {self.validation_level.name}",
            f"Errors: {self.get_error_count()}, Warnings: {self.get_warning_count()}, Info: {self.get_info_count()}",
            f"Sections Checked: {', '.join(self.sections_checked) if self.sections_checked else 'None'}",
            f"Invalid Sections: {', '.join(self.sections_invalid) if self.sections_invalid else 'None'}",
            f"Time Taken: {self.elapsed_time_ms:.2f} ms",
        ]
        
        if self.issues:
            summary.append("\nIssues:")
            for i, issue in enumerate(self.issues, 1):
                summary.append(f"{i}. [{issue.severity.name}] {issue.section}: {issue.message}")
                if issue.suggestion:
                    summary.append(f"   Suggestion: {issue.suggestion}")
        else:
            summary.append("\nNo issues found.")
            
        return "\n".join(summary)


class SPDValidationError(SPDError):
    """Raised when validation of an SPD file fails."""
    pass


class SPDValidator:
    """
    Validator for SPD (Source Portrait Descriptor) files.
    
    Provides methods to validate SPD files at different levels of strictness.
    
    Attributes:
        validation_level: The level of validation to perform.
    """
    
    def __init__(
        self, 
        file_or_reader: Union[FileOrPath, SPDReader],
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ):
        """
        Initialize the SPD validator.
        
        Args:
            file_or_reader: A file path, file-like object, or SPDReader instance.
            validation_level: The level of validation to perform.
        """
        self.validation_level = validation_level
        self._reader: Optional[SPDReader] = None
        self._owns_reader = False
        self._header_error = None
        
        # Initialize the reader
        if isinstance(file_or_reader, SPDReader):
            self._reader = file_or_reader
            self._owns_reader = False
        else:
            try:
                self._reader = SPDReader(file_or_reader)
                self._owns_reader = True
            except SPDFormatError as e:
                # For basic validation level, we should handle validation of files
                # with header errors in the validate method
                if validation_level == ValidationLevel.BASIC:
                    # Store the error to report it during validation
                    self._header_error = str(e)
                else:
                    raise SPDValidationError(f"Failed to initialize SPDReader: {e}")
            except Exception as e:
                raise SPDValidationError(f"Failed to initialize SPDReader: {e}")
    
    def close(self) -> None:
        """Close the reader if it was opened by this validator."""
        if self._reader and self._owns_reader:
            self._reader.close()
            self._reader = None
    
    def __enter__(self) -> 'SPDValidator':
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.close()
    
    def validate(self) -> ValidationReport:
        """
        Validate the SPD file at the selected validation level.
        
        Returns:
            A validation report containing any issues found.
        """
        start_time = time.time()
        
        # Create validation report
        report = ValidationReport(
            validation_level=self.validation_level,
            file_path=getattr(self._reader, 'file_path', None)
        )
        
        try:
            # If we have a stored header error (for BASIC validation level)
            if self._header_error is not None:
                report.sections_checked.append("header")
                report.add_issue(
                    section="header",
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid header: {self._header_error}",
                    suggestion="The file may be corrupt or not an SPD file."
                )
                report.elapsed_time_ms = (time.time() - start_time) * 1000
                return report
            
            # Basic validation - can we read the header
            self._validate_header(report)
            
            if self.validation_level in (ValidationLevel.STANDARD, ValidationLevel.STRICT):
                # Standard validation - are sections consistent with flags
                self._validate_sections_against_flags(report)
                
                # Validate individual sections that are present
                available_sections = self._reader.get_available_sections() if self._reader else []
                
                if "image" in available_sections:
                    self._validate_image_section(report)
                
                if "landmarks" in available_sections:
                    self._validate_landmarks_section(report)
                
                if "motion_params" in available_sections:
                    self._validate_motion_params_section(report)
                
                if "appearance" in available_sections:
                    self._validate_appearance_section(report)
                
                if "transforms" in available_sections:
                    self._validate_transforms_section(report)
                
                if "mask" in available_sections:
                    self._validate_mask_section(report)
                
                if "additional_flags" in available_sections:
                    self._validate_additional_flags_section(report)
            
            if self.validation_level == ValidationLevel.STRICT:
                # Strict validation - detailed data integrity checks
                self._validate_cross_section_consistency(report)
                
                # Additional detailed validation for each section
                available_sections = self._reader.get_available_sections() if self._reader else []
                
                if "landmarks" in available_sections:
                    self._validate_landmarks_quality(report)
                
                if "transforms" in available_sections:
                    self._validate_transforms_quality(report)
            
        except Exception as e:
            # If any exception occurs during validation, log it and add it to the report
            logger.error(f"Exception during validation: {e}", exc_info=True)
            report.add_issue(
                section="general",
                severity=ValidationSeverity.ERROR,
                message=f"Exception during validation: {e}",
                suggestion="The SPD file may be corrupt or in an unsupported format."
            )
            report.is_valid = False
        
        # Calculate elapsed time
        report.elapsed_time_ms = (time.time() - start_time) * 1000
        
        return report
    
    def _validate_header(self, report: ValidationReport) -> None:
        """
        Validate the SPD file header.
        
        Args:
            report: The validation report to update.
        """
        report.sections_checked.append("header")
        
        try:
            # Access the header - this will validate magic bytes and version
            header = self._reader.header if self._reader else None
            if header is None:
                report.add_issue(
                    section="header",
                    severity=ValidationSeverity.ERROR,
                    message="Could not read header.",
                    suggestion="The file may be corrupt or not an SPD file."
                )
                return
            
            # Validate the header content
            if header.magic_bytes != MAGIC_BYTES:
                report.add_issue(
                    section="header",
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid magic bytes: expected {MAGIC_BYTES!r}, got {header.magic_bytes!r}.",
                    suggestion="The file is not a valid SPD file."
                )
            
            if header.version > CURRENT_VERSION:
                report.add_issue(
                    section="header",
                    severity=ValidationSeverity.ERROR,
                    message=f"Unsupported version: {header.version} (current: {CURRENT_VERSION}).",
                    suggestion="Use a newer version of the SPD editor."
                )
            elif header.version < CURRENT_VERSION:
                report.add_issue(
                    section="header",
                    severity=ValidationSeverity.WARNING,
                    message=f"Old version: {header.version} (current: {CURRENT_VERSION}).",
                    suggestion="Consider upgrading the SPD file to the latest version."
                )
            
            # Check timestamp - warn if in the future
            if header.timestamp > time.time() + 86400:  # More than a day in the future
                report.add_issue(
                    section="header",
                    severity=ValidationSeverity.WARNING,
                    message=f"Timestamp is in the future: {time.ctime(header.timestamp)}.",
                    suggestion="Check if the timestamp was set correctly."
                )
            
            # Add info about the header
            report.add_issue(
                section="header",
                severity=ValidationSeverity.INFO,
                message=f"Header version: {header.version}, timestamp: {time.ctime(header.timestamp)}."
            )
            
            # Add info about the flags
            flags_dict = flags_to_dict(header.flags)
            present_flags = [name for name, is_set in flags_dict.items() if is_set]
            if present_flags:
                flags_str = ", ".join(present_flags)
                report.add_issue(
                    section="header",
                    severity=ValidationSeverity.INFO,
                    message=f"Header flags: {flags_str}."
                )
            else:
                report.add_issue(
                    section="header",
                    severity=ValidationSeverity.WARNING,
                    message="No header flags are set.",
                    suggestion="The file may be empty or invalid."
                )
            
            report.sections_valid.append("header")
            
        except (SPDFormatError, SPDSectionError) as e:
            report.add_issue(
                section="header",
                severity=ValidationSeverity.ERROR,
                message=f"Invalid header: {e}.",
                suggestion="The file may be corrupt or not an SPD file."
            )
    
    def _validate_sections_against_flags(self, report: ValidationReport) -> None:
        """
        Validate that the sections in the file are consistent with the header flags.
        
        Args:
            report: The validation report to update.
        """
        if not self._reader or not self._reader.header:
            return
        
        report.sections_checked.append("flags")
        
        # Get flags and available sections
        flags_dict = flags_to_dict(self._reader.header.flags)
        available_sections = self._reader.get_available_sections()
        
        # Check for missing sections based on flags
        for flag_name, is_set in flags_dict.items():
            if not is_set:
                continue
            
            section_missing = False
            section_name = ""
            
            if flag_name == "HAS_IMAGE" and "image" not in available_sections:
                section_missing = True
                section_name = "image"
            elif flag_name == "HAS_LANDMARKS" and "landmarks" not in available_sections:
                section_missing = True
                section_name = "landmarks"
            elif flag_name == "HAS_MOTION_PARAMS" and "motion_params" not in available_sections:
                section_missing = True
                section_name = "motion_params"
            elif flag_name == "HAS_APPEARANCE" and "appearance" not in available_sections:
                section_missing = True
                section_name = "appearance"
            elif flag_name == "HAS_TRANSFORMS" and "transforms" not in available_sections:
                section_missing = True
                section_name = "transforms"
            elif flag_name == "HAS_MASK" and "mask" not in available_sections:
                section_missing = True
                section_name = "mask"
            
            if section_missing:
                report.add_issue(
                    section="flags",
                    severity=ValidationSeverity.ERROR,
                    message=f"Flag {flag_name} is set but {section_name} section is missing.",
                    suggestion=f"Remove the {flag_name} flag or add the {section_name} section."
                )
        
        # Check for sections that are present but not flagged
        for section_name in available_sections:
            if section_name == "additional_flags":
                continue  # Additional flags section doesn't have a header flag
            
            flag_name = None
            
            if section_name == "image":
                flag_name = "HAS_IMAGE"
            elif section_name == "landmarks":
                flag_name = "HAS_LANDMARKS"
            elif section_name == "motion_params":
                flag_name = "HAS_MOTION_PARAMS"
            elif section_name == "appearance":
                flag_name = "HAS_APPEARANCE"
            elif section_name == "transforms":
                flag_name = "HAS_TRANSFORMS"
            elif section_name == "mask":
                flag_name = "HAS_MASK"
            
            if flag_name and not flags_dict.get(flag_name, False):
                report.add_issue(
                    section="flags",
                    severity=ValidationSeverity.ERROR,
                    message=f"{section_name} section is present but flag {flag_name} is not set.",
                    suggestion=f"Set the {flag_name} flag or remove the {section_name} section."
                )
        
        # Check for encrypted or compressed data which we can't validate properly
        if flags_dict.get("IS_ENCRYPTED", False):
            report.add_issue(
                section="flags",
                severity=ValidationSeverity.WARNING,
                message="File is marked as encrypted, validation of encrypted content is limited.",
                suggestion="Decrypt the file for full validation."
            )
        
        if flags_dict.get("IS_COMPRESSED", False):
            report.add_issue(
                section="flags",
                severity=ValidationSeverity.INFO,
                message="File is marked as compressed."
            )
        
        # Only mark the section as valid if there are no errors
        if not any(i.section == "flags" and i.severity == ValidationSeverity.ERROR for i in report.issues):
            report.sections_valid.append("flags")
    
    def _validate_transforms_quality(self, report: ValidationReport) -> None:
        """
        Perform more detailed quality checks on transformation matrices.
        
        Args:
            report: The validation report to update.
        """
        if not self._reader:
            return
        
        try:
            transforms = self._reader.transforms
            if not transforms or not transforms.matrices:
                return
            
            for i in range(len(transforms.matrices)):
                matrix = transforms.matrices[i]
                # Matrix should have 9 elements (3x3)
                if len(matrix) != 9:
                    continue
                
                # Check for non-invertible matrices (determinant close to zero)
                # For a 3x3 matrix: det = a(ei-fh) - b(di-fg) + c(dh-eg)
                a, b, c = matrix[0], matrix[1], matrix[2]
                d, e, f = matrix[3], matrix[4], matrix[5]
                g, h, i_val = matrix[6], matrix[7], matrix[8]
                
                det = a * (e * i_val - f * h) - b * (d * i_val - f * g) + c * (d * h - e * g)
                
                if abs(det) < 1e-6:
                    transform_type = transforms.transform_types[i] if i < len(transforms.transform_types) else f"transform {i}"
                    report.add_issue(
                        section="transforms_quality",
                        severity=ValidationSeverity.ERROR,
                        message=f"Matrix for {transform_type} is non-invertible (determinant: {det}).",
                        suggestion="Transformation matrices should be invertible."
                    )
                    
                # Check for scaling issues - extremely large or small scaling factors
                # Estimate scaling by looking at the magnitude of the first two rows
                row1_scale = math.sqrt(a * a + b * b + c * c)
                row2_scale = math.sqrt(d * d + e * e + f * f)
                
                if row1_scale > 100 or row2_scale > 100:
                    transform_type = transforms.transform_types[i] if i < len(transforms.transform_types) else f"transform {i}"
                    report.add_issue(
                        section="transforms_quality",
                        severity=ValidationSeverity.WARNING,
                        message=f"Matrix for {transform_type} has very large scaling factors.",
                        suggestion="This may cause numerical issues during animation."
                    )
                    
                if row1_scale < 0.01 or row2_scale < 0.01:
                    transform_type = transforms.transform_types[i] if i < len(transforms.transform_types) else f"transform {i}"
                    report.add_issue(
                        section="transforms_quality",
                        severity=ValidationSeverity.WARNING,
                        message=f"Matrix for {transform_type} has very small scaling factors.",
                        suggestion="This may cause numerical issues during animation."
                    )
        
        except (SPDFormatError, SPDSectionError):
            # If there's an error reading the sections, skip this check
            pass

    def _validate_image_section(self, report: ValidationReport) -> None:
        """
        Validate the image section of the SPD file.
        
        Args:
            report: The validation report to update.
        """
        if not self._reader:
            return
            
        report.sections_checked.append("image")
        
        try:
            # Get the image section
            image = self._reader.image
            if not image:
                # Image section not present
                return
                
            # Check image dimensions
            if image.width <= 0 or image.height <= 0:
                report.add_issue(
                    section="image",
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid image dimensions: {image.width}x{image.height}",
                    suggestion="Image dimensions must be positive values."
                )
                
            # Check channels
            if image.channels <= 0 or image.channels > 4:
                report.add_issue(
                    section="image",
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid number of channels: {image.channels}",
                    suggestion="Image channels should be between 1 and 4 (grayscale, RGB, RGBA)."
                )
                
            # Check format string
            valid_formats = {"GRAY", "RGB", "RGBA", "BGR", "BGRA"}
            if image.format not in valid_formats:
                report.add_issue(
                    section="image",
                    severity=ValidationSeverity.WARNING,
                    message=f"Unrecognized image format: {image.format}",
                    suggestion=f"Common formats are: {', '.join(valid_formats)}."
                )
                
            # Check data size consistency
            expected_size = image.width * image.height * image.channels
            if len(image.data) != expected_size:
                report.add_issue(
                    section="image",
                    severity=ValidationSeverity.ERROR,
                    message=f"Image data size mismatch: expected {expected_size} bytes, got {len(image.data)}",
                    suggestion="Image data size should match width * height * channels."
                )
                
            # Mark as valid if no errors
            if not any(i.section == "image" and i.severity == ValidationSeverity.ERROR for i in report.issues):
                report.sections_valid.append("image")
                
        except (SPDFormatError, SPDSectionError) as e:
            report.add_issue(
                section="image",
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate image section: {e}",
                suggestion="The image section may be corrupt."
            )
    
    def _validate_landmarks_section(self, report: ValidationReport) -> None:
        """
        Validate the landmarks section of the SPD file.
        
        Args:
            report: The validation report to update.
        """
        if not self._reader:
            return
            
        report.sections_checked.append("landmarks")
        
        try:
            # Get the landmarks section
            landmarks = self._reader.landmarks
            if not landmarks:
                # Landmarks section not present
                return
                
            # Check landmark count
            if landmarks.count <= 0:
                report.add_issue(
                    section="landmarks",
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid landmark count: {landmarks.count}",
                    suggestion="Landmark count must be a positive value."
                )
                
            # Check dimensions
            if landmarks.dimensions <= 0:
                report.add_issue(
                    section="landmarks",
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid landmark dimensions: {landmarks.dimensions}",
                    suggestion="Landmark dimensions must be positive values."
                )
            elif landmarks.dimensions > 3:
                report.add_issue(
                    section="landmarks",
                    severity=ValidationSeverity.WARNING,
                    message=f"Unusual landmark dimensions: {landmarks.dimensions}",
                    suggestion="Most landmarks use 2D or 3D coordinates."
                )
                
            # Check points array size
            if len(landmarks.points) != landmarks.count:
                report.add_issue(
                    section="landmarks",
                    severity=ValidationSeverity.ERROR,
                    message=f"Landmark points count mismatch: expected {landmarks.count}, got {len(landmarks.points)}",
                    suggestion="The number of points should match the landmark count."
                )
                
            # Check individual points dimensions
            for i, point in enumerate(landmarks.points):
                if len(point) != landmarks.dimensions:
                    report.add_issue(
                        section="landmarks",
                        severity=ValidationSeverity.ERROR,
                        message=f"Point {i} has incorrect dimensions: expected {landmarks.dimensions}, got {len(point)}",
                        suggestion="All landmark points should have the same number of dimensions."
                    )
                    break
            
            # Mark as valid if no errors
            if not any(i.section == "landmarks" and i.severity == ValidationSeverity.ERROR for i in report.issues):
                report.sections_valid.append("landmarks")
                
        except (SPDFormatError, SPDSectionError) as e:
            report.add_issue(
                section="landmarks",
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate landmarks section: {e}",
                suggestion="The landmarks section may be corrupt."
            )
    
    def _validate_motion_params_section(self, report: ValidationReport) -> None:
        """
        Validate the motion parameters section of the SPD file.
        
        Args:
            report: The validation report to update.
        """
        if not self._reader:
            return
            
        report.sections_checked.append("motion_params")
        
        try:
            # Get the motion parameters section
            motion_params = self._reader.motion_params
            if not motion_params:
                # Motion parameters section not present
                return
                
            # Check parameter count
            if motion_params.num_params <= 0:
                report.add_issue(
                    section="motion_params",
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid parameter count: {motion_params.num_params}",
                    suggestion="Parameter count must be a positive value."
                )
                
            # Check parameter names
            if len(motion_params.param_names) != motion_params.num_params:
                report.add_issue(
                    section="motion_params",
                    severity=ValidationSeverity.ERROR,
                    message=f"Parameter names count mismatch: expected {motion_params.num_params}, got {len(motion_params.param_names)}",
                    suggestion="The number of parameter names should match the parameter count."
                )
                
            # Check values
            if len(motion_params.values) != motion_params.num_params:
                report.add_issue(
                    section="motion_params",
                    severity=ValidationSeverity.ERROR,
                    message=f"Parameter values count mismatch: expected {motion_params.num_params}, got {len(motion_params.values)}",
                    suggestion="The number of parameter values should match the parameter count."
                )
            
            # Mark as valid if no errors
            if not any(i.section == "motion_params" and i.severity == ValidationSeverity.ERROR for i in report.issues):
                report.sections_valid.append("motion_params")
                
        except (SPDFormatError, SPDSectionError) as e:
            report.add_issue(
                section="motion_params",
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate motion parameters section: {e}",
                suggestion="The motion parameters section may be corrupt."
            )
    
    def _validate_appearance_section(self, report: ValidationReport) -> None:
        """
        Validate the appearance section of the SPD file.
        
        Args:
            report: The validation report to update.
        """
        if not self._reader:
            return
            
        report.sections_checked.append("appearance")
        
        try:
            # Get the appearance section
            appearance = self._reader.appearance
            if not appearance:
                # Appearance section not present
                return
                
            # Check feature dimension
            if appearance.feature_dim <= 0:
                report.add_issue(
                    section="appearance",
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid feature dimension: {appearance.feature_dim}",
                    suggestion="Feature dimension must be a positive value."
                )
                
            # Check features length
            if len(appearance.features) != appearance.feature_dim:
                report.add_issue(
                    section="appearance",
                    severity=ValidationSeverity.ERROR,
                    message=f"Features length mismatch: expected {appearance.feature_dim}, got {len(appearance.features)}",
                    suggestion="The number of features should match the feature dimension."
                )
            
            # Mark as valid if no errors
            if not any(i.section == "appearance" and i.severity == ValidationSeverity.ERROR for i in report.issues):
                report.sections_valid.append("appearance")
                
        except (SPDFormatError, SPDSectionError) as e:
            report.add_issue(
                section="appearance",
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate appearance section: {e}",
                suggestion="The appearance section may be corrupt."
            )
    
    def _validate_transforms_section(self, report: ValidationReport) -> None:
        """
        Validate the transformations section of the SPD file.
        
        Args:
            report: The validation report to update.
        """
        if not self._reader:
            return
            
        report.sections_checked.append("transforms")
        
        try:
            # Get the transformations section
            transforms = self._reader.transforms
            if not transforms:
                # Transformations section not present
                return
                
            # Check transform count
            if transforms.num_transforms <= 0:
                report.add_issue(
                    section="transforms",
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid transform count: {transforms.num_transforms}",
                    suggestion="Transform count must be a positive value."
                )
                
            # Check transform types
            if len(transforms.transform_types) != transforms.num_transforms:
                report.add_issue(
                    section="transforms",
                    severity=ValidationSeverity.ERROR,
                    message=f"Transform types count mismatch: expected {transforms.num_transforms}, got {len(transforms.transform_types)}",
                    suggestion="The number of transform types should match the transform count."
                )
                
            # Check matrices
            if len(transforms.matrices) != transforms.num_transforms:
                report.add_issue(
                    section="transforms",
                    severity=ValidationSeverity.ERROR,
                    message=f"Transform matrices count mismatch: expected {transforms.num_transforms}, got {len(transforms.matrices)}",
                    suggestion="The number of matrices should match the transform count."
                )
            
            # Check individual matrices
            for i, matrix in enumerate(transforms.matrices):
                if len(matrix) != 9:  # 3x3 matrix flattened
                    report.add_issue(
                        section="transforms",
                        severity=ValidationSeverity.ERROR,
                        message=f"Matrix {i} has incorrect size: expected 9 elements, got {len(matrix)}",
                        suggestion="All transformation matrices should be 3x3 (9 elements)."
                    )
            
            # Mark as valid if no errors
            if not any(i.section == "transforms" and i.severity == ValidationSeverity.ERROR for i in report.issues):
                report.sections_valid.append("transforms")
                
        except (SPDFormatError, SPDSectionError) as e:
            report.add_issue(
                section="transforms",
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate transformations section: {e}",
                suggestion="The transformations section may be corrupt."
            )
    
    def _validate_mask_section(self, report: ValidationReport) -> None:
        """
        Validate the mask section of the SPD file.
        
        Args:
            report: The validation report to update.
        """
        if not self._reader:
            return
            
        report.sections_checked.append("mask")
        
        try:
            # Get the mask section
            mask = self._reader.mask
            if not mask:
                # Mask section not present
                return
                
            # Check mask dimensions
            if mask.width <= 0 or mask.height <= 0:
                report.add_issue(
                    section="mask",
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid mask dimensions: {mask.width}x{mask.height}",
                    suggestion="Mask dimensions must be positive values."
                )
                
            # Check mask format
            valid_formats = {"binary", "grayscale", "alpha"}
            if mask.format.lower() not in valid_formats:
                report.add_issue(
                    section="mask",
                    severity=ValidationSeverity.WARNING,
                    message=f"Unrecognized mask format: {mask.format}",
                    suggestion=f"Common formats are: {', '.join(valid_formats)}."
                )
                
            # Check mask data size for binary format (1 bit per pixel packed into bytes)
            if mask.format.lower() == "binary":
                expected_size = (mask.width * mask.height + 7) // 8  # Ceiling division to bytes
                if len(mask.data) != expected_size:
                    report.add_issue(
                        section="mask",
                        severity=ValidationSeverity.ERROR,
                        message=f"Binary mask data size mismatch: expected {expected_size} bytes, got {len(mask.data)}",
                        suggestion="Binary mask data size should be (width * height + 7) // 8 bytes."
                    )
            
            # Check mask data size for grayscale/alpha format (1 byte per pixel)
            elif mask.format.lower() in {"grayscale", "alpha"}:
                expected_size = mask.width * mask.height
                if len(mask.data) != expected_size:
                    report.add_issue(
                        section="mask",
                        severity=ValidationSeverity.ERROR,
                        message=f"Mask data size mismatch: expected {expected_size} bytes, got {len(mask.data)}",
                        suggestion=f"Mask data size should be width * height bytes for {mask.format} format."
                    )
            
            # Mark as valid if no errors
            if not any(i.section == "mask" and i.severity == ValidationSeverity.ERROR for i in report.issues):
                report.sections_valid.append("mask")
                
        except (SPDFormatError, SPDSectionError) as e:
            report.add_issue(
                section="mask",
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate mask section: {e}",
                suggestion="The mask section may be corrupt."
            )
    
    def _validate_additional_flags_section(self, report: ValidationReport) -> None:
        """
        Validate the additional flags section of the SPD file.
        
        Args:
            report: The validation report to update.
        """
        if not self._reader:
            return
            
        report.sections_checked.append("additional_flags")
        
        try:
            # Get the additional flags section
            additional_flags = self._reader.additional_flags
            if not additional_flags:
                # Additional flags section not present
                return
                
            # Check flag count
            if additional_flags.num_flags != len(additional_flags.flags):
                report.add_issue(
                    section="additional_flags",
                    severity=ValidationSeverity.ERROR,
                    message=f"Flag count mismatch: expected {additional_flags.num_flags}, got {len(additional_flags.flags)}",
                    suggestion="The number of flags should match the flag count."
                )
            
            # Check for required flags if this is a strict validation
            if self.validation_level == ValidationLevel.STRICT:
                # List of required flag names could be application-specific
                required_flags = {"source", "processed"}
                for flag_name in required_flags:
                    if flag_name not in additional_flags.flags:
                        report.add_issue(
                            section="additional_flags",
                            severity=ValidationSeverity.WARNING,
                            message=f"Missing recommended flag: {flag_name}",
                            suggestion=f"Consider adding the '{flag_name}' flag for better compatibility."
                        )
            
            # Mark as valid if no errors
            if not any(i.section == "additional_flags" and i.severity == ValidationSeverity.ERROR for i in report.issues):
                report.sections_valid.append("additional_flags")
                
        except (SPDFormatError, SPDSectionError) as e:
            report.add_issue(
                section="additional_flags",
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate additional flags section: {e}",
                suggestion="The additional flags section may be corrupt."
            )
    
    def _validate_landmarks_quality(self, report: ValidationReport) -> None:
        """
        Perform more detailed quality checks on landmarks.
        
        Args:
            report: The validation report to update.
        """
        if not self._reader:
            return
            
        try:
            landmarks = self._reader.landmarks
            if not landmarks or not landmarks.points:
                return
                
            # Check for NaN or Inf values
            has_invalid = False
            for i, point in enumerate(landmarks.points):
                for coord in point:
                    if math.isnan(coord) or math.isinf(coord):
                        has_invalid = True
                        report.add_issue(
                            section="landmarks_quality",
                            severity=ValidationSeverity.ERROR,
                            message=f"Landmark {i} contains NaN or Inf values",
                            suggestion="All landmark coordinates should be valid numbers."
                        )
                        break
                if has_invalid:
                    break
            
            # Check for extremely large values
            # Landmarks are typically normalized to image dimensions
            has_extreme_values = False
            for i, point in enumerate(landmarks.points):
                for coord in point:
                    if abs(coord) > 10000:  # Arbitrary large threshold
                        has_extreme_values = True
                        report.add_issue(
                            section="landmarks_quality",
                            severity=ValidationSeverity.WARNING,
                            message=f"Landmark {i} contains extremely large values",
                            suggestion="Landmark values may not be properly normalized."
                        )
                        break
                if has_extreme_values:
                    break
            
            # If we have an image section, check if landmarks are within image bounds
            # Only for 2D landmarks
            if landmarks.dimensions == 2 and self._reader.image:
                image = self._reader.image
                out_of_bounds = False
                for i, point in enumerate(landmarks.points):
                    x, y = point
                    if x < 0 or x > image.width or y < 0 or y > image.height:
                        out_of_bounds = True
                        report.add_issue(
                            section="landmarks_quality",
                            severity=ValidationSeverity.WARNING,
                            message=f"Landmark {i} is outside image bounds",
                            suggestion="Landmarks should typically be within the image dimensions."
                        )
                        break
            
        except (SPDFormatError, SPDSectionError):
            # If there's an error reading the sections, skip this check
            pass
    
    def _validate_cross_section_consistency(self, report: ValidationReport) -> None:
        """
        Validate consistency between different sections.
        
        Args:
            report: The validation report to update.
        """
        if not self._reader:
            return
            
        report.sections_checked.append("cross_section")
        
        try:
            # Check if image and mask dimensions match
            if self._reader.image and self._reader.mask:
                image = self._reader.image
                mask = self._reader.mask
                
                if image.width != mask.width or image.height != mask.height:
                    report.add_issue(
                        section="cross_section",
                        severity=ValidationSeverity.ERROR,
                        message=f"Image and mask dimensions don't match: image is {image.width}x{image.height}, mask is {mask.width}x{mask.height}",
                        suggestion="Image and mask should have the same dimensions."
                    )
            
            # Check if landmarks type makes sense with the motion params
            if self._reader.landmarks and self._reader.motion_params:
                landmarks = self._reader.landmarks
                motion_params = self._reader.motion_params
                
                # Example: if landmarks are "face" type, motion params should include typical face params
                if landmarks.landmark_type == "face" and not any(param in motion_params.param_names for param in ["yaw", "pitch", "roll"]):
                    report.add_issue(
                        section="cross_section",
                        severity=ValidationSeverity.WARNING,
                        message="Face landmarks present but no head pose parameters found",
                        suggestion="Face landmarks typically need head pose parameters (yaw, pitch, roll)."
                    )
            
            # Mark as valid if no errors
            if not any(i.section == "cross_section" and i.severity == ValidationSeverity.ERROR for i in report.issues):
                report.sections_valid.append("cross_section")
                
        except (SPDFormatError, SPDSectionError) as e:
            report.add_issue(
                section="cross_section",
                severity=ValidationSeverity.ERROR,
                message=f"Failed to validate cross-section consistency: {e}",
                suggestion="There may be inconsistencies between sections."
            )


def validate_spd_file(
    file_or_reader: Union[FileOrPath, SPDReader], 
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> Tuple[bool, ValidationReport]:
    """
    Validate an SPD file and return a boolean result and detailed report.
    
    This is a standalone function for simpler validation use cases.
    
    Args:
        file_or_reader: A file path, file-like object, or SPDReader instance.
        validation_level: The level of validation to perform.
        
    Returns:
        A tuple containing a boolean result (True if valid) and the detailed validation report.
    """
    validator = None
    try:
        try:
            validator = SPDValidator(file_or_reader, validation_level)
            report = validator.validate()
            return report.is_valid, report
        except SPDValidationError as e:
            # For basic validation, handle the error and return a report
            if validation_level == ValidationLevel.BASIC:
                report = ValidationReport(
                    is_valid=False,
                    validation_level=validation_level
                )
                report.add_issue(
                    section="header",
                    severity=ValidationSeverity.ERROR,
                    message=str(e),
                    suggestion="The file may be corrupt or not a valid SPD file."
                )
                return False, report
            else:
                # For other levels, just re-raise the exception
                raise
    finally:
        if validator and validator._owns_reader:
            validator.close()