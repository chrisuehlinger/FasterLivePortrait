"""
Tests for the SPDValidator class.
"""
import os
import time
import struct
import unittest
import tempfile
from pathlib import Path
from copy import deepcopy

from spd_editor.spd.format import (
    MAGIC_BYTES, CURRENT_VERSION,
    HeaderFlags, SectionMarker,
    ImageSection, LandmarksSection, AppearanceSection,
    MotionParamsSection, TransformationSection, MaskSection,
    AdditionalFlagsSection
)
from spd_editor.spd.reader import SPDReader, SPDFormatError, SPDSectionError
from spd_editor.spd.writer import SPDWriter, SPDValidationError as WriterValidationError
from spd_editor.spd.validator import (
    SPDValidator, ValidationLevel, ValidationSeverity, ValidationReport,
    validate_spd_file, SPDValidationError
)


class TestSPDValidatorBase(unittest.TestCase):
    """Base class for SPD validator tests."""

    def setUp(self) -> None:
        """Set up a temporary directory for test files."""
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.test_dir.cleanup()

    def create_valid_spd_file(self, path: str) -> str:
        """
        Create a valid SPD file with all sections for testing.
        
        Args:
            path: Path to create the file at.
            
        Returns:
            The path to the created file.
        """
        # Create all sections
        with SPDWriter(path) as writer:
            # Add image section
            image = ImageSection(
                width=64,
                height=64,
                channels=3,
                format="RGB",
                data=bytes([i % 256 for i in range(64 * 64 * 3)])
            )
            writer.write_image_section(image)
            
            # Add landmarks section
            landmarks = LandmarksSection(
                count=5,
                dimensions=2,
                landmark_type="test",
                points=[[float(i), float(i * 2)] for i in range(5)]
            )
            writer.write_landmarks_section(landmarks)
            
            # Add motion parameters section
            motion_params = MotionParamsSection(
                num_params=3,
                param_names=["yaw", "pitch", "roll"],
                values=[0.1, 0.2, 0.3]
            )
            writer.write_motion_params_section(motion_params)
            
            # Add appearance section
            appearance = AppearanceSection(
                feature_dim=4,
                feature_type="embedding",
                features=[0.1, 0.2, 0.3, 0.4]
            )
            writer.write_appearance_section(appearance)
            
            # Add transforms section
            transforms = TransformationSection(
                num_transforms=2,
                transform_types=["face2world", "world2face"],
                matrices=[
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                ]
            )
            writer.write_transformation_section(transforms)
            
            # Add mask section
            mask = MaskSection(
                width=64,
                height=64,
                format="binary",
                data=bytes([i % 2 for i in range(64 * 64)])
            )
            writer.write_mask_section(mask)
            
            # Add additional flags section
            additional_flags = AdditionalFlagsSection(
                num_flags=3,
                flags={
                    "source": "test.jpg",
                    "processed": True,
                    "quality": 0.95,
                }
            )
            writer.write_additional_flags_section(additional_flags)
        
        return path

    def create_invalid_header_file(self, path: str) -> str:
        """
        Create an SPD file with an invalid header.
        
        Args:
            path: Path to create the file at.
            
        Returns:
            The path to the created file.
        """
        # First create a valid file
        self.create_valid_spd_file(path)
        
        # Then corrupt the header
        with open(path, "r+b") as f:
            f.seek(0)
            f.write(b"XXXX")  # Invalid magic bytes
        
        return path

    def create_future_version_file(self, path: str) -> str:
        """
        Create an SPD file with a future version number.
        
        Args:
            path: Path to create the file at.
            
        Returns:
            The path to the created file.
        """
        # First create a valid file
        self.create_valid_spd_file(path)
        
        # Then modify the version number
        with open(path, "r+b") as f:
            f.seek(4)
            f.write(struct.pack("!I", CURRENT_VERSION + 1))  # Future version
        
        return path

    def create_missing_section_file(self, path: str) -> str:
        """
        Create an SPD file with a missing section but the flag set.
        
        Args:
            path: Path to create the file at.
            
        Returns:
            The path to the created file.
        """
        # Create file with most sections
        with SPDWriter(path) as writer:
            # Add image section
            image = ImageSection(
                width=64,
                height=64,
                channels=3,
                format="RGB",
                data=bytes([i % 256 for i in range(64 * 64 * 3)])
            )
            writer.write_image_section(image)
            
            # We're NOT adding landmarks, but will set the flag
            
            # Add motion parameters section
            motion_params = MotionParamsSection(
                num_params=3,
                param_names=["yaw", "pitch", "roll"],
                values=[0.1, 0.2, 0.3]
            )
            writer.write_motion_params_section(motion_params)
        
        # Manually set the HAS_LANDMARKS flag even though the section doesn't exist
        with open(path, "r+b") as f:
            f.seek(16)  # Flags position
            flags = HeaderFlags.HAS_IMAGE.value | HeaderFlags.HAS_MOTION_PARAMS.value | HeaderFlags.HAS_LANDMARKS.value
            f.write(struct.pack("!I", flags))
        
        return path

    def create_no_flag_section_file(self, path: str) -> str:
        """
        Create an SPD file with a section present but the flag not set.
        
        Args:
            path: Path to create the file at.
            
        Returns:
            The path to the created file.
        """
        # First create a valid file
        self.create_valid_spd_file(path)
        
        # Then clear the image flag (but leave the section)
        with open(path, "r+b") as f:
            f.seek(16)  # Flags position
            flags = (HeaderFlags.HAS_LANDMARKS.value | 
                    HeaderFlags.HAS_MOTION_PARAMS.value | 
                    HeaderFlags.HAS_APPEARANCE.value | 
                    HeaderFlags.HAS_TRANSFORMS.value | 
                    HeaderFlags.HAS_MASK.value)
            f.write(struct.pack("!I", flags))
        
        return path
        
    def create_corrupted_image_section_file(self, path: str) -> str:
        """
        Create an SPD file with corrupted image section data.
        
        Args:
            path: Path to create the file at.
            
        Returns:
            The path to the created file.
        """
        # Create a file with basic valid structure
        with SPDWriter(path) as writer:
            # Add image section with valid dimensions but wrong data size
            image = ImageSection(
                width=64,
                height=64,
                channels=3,
                format="RGB",
                data=bytes([i % 256 for i in range(64 * 64 * 2)])  # Intentionally wrong size
            )
            try:
                writer.write_image_section(image)
            except WriterValidationError:
                # If validation prevents writing invalid data, create valid file and then corrupt it
                writer.close()
                self.create_valid_spd_file(path)
                # Find the image section start and size and corrupt it
                with open(path, "r+b") as f:
                    # Simple corruption: truncate the file after image section starts
                    # Find the image section marker
                    f.seek(0, os.SEEK_END)
                    end_pos = f.tell()
                    f.seek(20)  # Skip header
                    while f.tell() < end_pos - 4:
                        marker_bytes = f.read(4)
                        if marker_bytes == struct.pack("!I", SectionMarker.IMAGE.value):
                            # We found the image section, now corrupt its size
                            f.write(struct.pack("!I", 10))  # Wrong size
                            break
                        f.seek(1, os.SEEK_CUR)  # Move one byte at a time
        
        return path
    
    def create_corrupted_landmarks_file(self, path: str) -> str:
        """
        Create an SPD file with corrupted landmarks data.
        
        Args:
            path: Path to create the file at.
            
        Returns:
            The path to the created file.
        """
        # Create a file with basic valid structure
        with SPDWriter(path) as writer:
            # Add image section
            image = ImageSection(
                width=64,
                height=64,
                channels=3,
                format="RGB",
                data=bytes([i % 256 for i in range(64 * 64 * 3)])
            )
            writer.write_image_section(image)

            # Create landmarks with mismatched count and points
            landmarks = LandmarksSection(
                count=5,
                dimensions=2,
                landmark_type="test",
                points=[[float(i), float(i * 2)] for i in range(5)]
            )
            writer.write_landmarks_section(landmarks)
        
        # Manually corrupt the landmarks count after writing
        with open(path, "r+b") as f:
            # Find landmarks section
            f.seek(0, os.SEEK_END)
            end_pos = f.tell()
            f.seek(20)  # Skip header
            while f.tell() < end_pos - 4:
                marker_bytes = f.read(4)
                if marker_bytes == struct.pack("!I", SectionMarker.LANDMARKS.value):
                    # Skip section size (4 bytes)
                    f.seek(4, os.SEEK_CUR)
                    # Change count from 5 to 10
                    f.write(struct.pack("!I", 10))
                    break
                f.seek(1, os.SEEK_CUR)
        
        return path
    
    def create_invalid_transforms_file(self, path: str) -> str:
        """
        Create an SPD file with invalid transformation matrices.
        
        Args:
            path: Path to create the file at.
            
        Returns:
            The path to the created file.
        """
        # Create a file with basic valid structure
        with SPDWriter(path) as writer:
            # Add image section
            image = ImageSection(
                width=64,
                height=64,
                channels=3,
                format="RGB",
                data=bytes([i % 256 for i in range(64 * 64 * 3)])
            )
            writer.write_image_section(image)
            
            # Add transforms section with a non-invertible matrix
            transforms = TransformationSection(
                num_transforms=1,
                transform_types=["broken_transform"],
                matrices=[
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Non-invertible matrix
                ]
            )
            
            # If validation prevents writing invalid data, use a nearly-singular matrix instead
            try:
                writer.write_transformation_section(transforms)
            except WriterValidationError:
                # Use a nearly-singular matrix
                transforms = TransformationSection(
                    num_transforms=1,
                    transform_types=["nearly_singular"],
                    matrices=[
                        [1e-10, 0.0, 0.0, 0.0, 1e-10, 0.0, 0.0, 0.0, 1e-10]  # Nearly singular
                    ]
                )
                writer.write_transformation_section(transforms)
        
        return path
    
    def create_cross_section_mismatch_file(self, path: str) -> str:
        """
        Create an SPD file with mismatches between sections (image and mask dimensions).
        
        Args:
            path: Path to create the file at.
            
        Returns:
            The path to the created file.
        """
        # Create a file with mismatched image and mask dimensions
        with SPDWriter(path) as writer:
            # Add image section
            image = ImageSection(
                width=64,
                height=64,
                channels=3,
                format="RGB",
                data=bytes([i % 256 for i in range(64 * 64 * 3)])
            )
            writer.write_image_section(image)
            
            # Add mask section with different dimensions
            mask = MaskSection(
                width=32,  # Different width
                height=32,  # Different height
                format="binary",
                data=bytes([i % 2 for i in range(32 * 32)])
            )
            writer.write_mask_section(mask)
        
        return path
    
    def create_landmarks_outside_bounds_file(self, path: str) -> str:
        """
        Create an SPD file with landmarks outside image bounds.
        
        Args:
            path: Path to create the file at.
            
        Returns:
            The path to the created file.
        """
        # Create a file with landmarks outside image bounds
        with SPDWriter(path) as writer:
            # Add image section
            image = ImageSection(
                width=64,
                height=64,
                channels=3,
                format="RGB",
                data=bytes([i % 256 for i in range(64 * 64 * 3)])
            )
            writer.write_image_section(image)
            
            # Add landmarks section with points outside the image
            landmarks = LandmarksSection(
                count=5,
                dimensions=2,
                landmark_type="test",
                points=[
                    [10.0, 10.0],
                    [20.0, 20.0],
                    [30.0, 30.0],
                    [100.0, 50.0],  # X outside bounds
                    [50.0, 100.0]   # Y outside bounds
                ]
            )
            writer.write_landmarks_section(landmarks)
        
        return path
    
    def create_duplicate_landmarks_file(self, path: str) -> str:
        """
        Create an SPD file with duplicate landmarks.
        
        Args:
            path: Path to create the file at.
            
        Returns:
            The path to the created file.
        """
        # Create a file with duplicate landmarks
        with SPDWriter(path) as writer:
            # Add image section
            image = ImageSection(
                width=64,
                height=64,
                channels=3,
                format="RGB",
                data=bytes([i % 256 for i in range(64 * 64 * 3)])
            )
            writer.write_image_section(image)
            
            # Add landmarks section with duplicate points
            landmarks = LandmarksSection(
                count=5,
                dimensions=2,
                landmark_type="test",
                points=[
                    [10.0, 10.0],
                    [20.0, 20.0],
                    [10.0, 10.0],  # Duplicate of first point
                    [20.0, 20.0],  # Duplicate of second point
                    [30.0, 30.0]
                ]
            )
            writer.write_landmarks_section(landmarks)
        
        return path


class TestSPDValidatorBasic(TestSPDValidatorBase):
    """Test basic validation functionality of the SPD validator."""

    def test_validate_valid_file(self) -> None:
        """Test validating a valid SPD file."""
        path = self.create_valid_spd_file(os.path.join(self.test_dir.name, "valid.spd"))
        
        with SPDValidator(path, ValidationLevel.BASIC) as validator:
            report = validator.validate()
        
        self.assertTrue(report.is_valid)
        self.assertEqual(report.validation_level, ValidationLevel.BASIC)
        self.assertEqual(len(report.sections_valid), 1)
        self.assertEqual(report.sections_valid[0], "header")
        self.assertEqual(len(report.sections_invalid), 0)
        
        # There should be some informational messages but no errors
        self.assertEqual(report.get_error_count(), 0)
        self.assertTrue(report.get_info_count() > 0)

    def test_validate_invalid_header(self) -> None:
        """Test validating a file with an invalid header."""
        path = self.create_invalid_header_file(os.path.join(self.test_dir.name, "invalid_header.spd"))
        
        # For basic validation, we should be able to get a validation report
        # even though the header is corrupt
        try:
            with open(path, 'rb') as f:
                # Call validate_spd_file instead of creating the validator directly
                is_valid, report = validate_spd_file(f, ValidationLevel.BASIC)
            
            self.assertFalse(is_valid)
            self.assertTrue(report.get_error_count() > 0)
            
            # Check for error messages about invalid header
            header_errors = [i for i in report.issues if i.section == "header" and 
                             i.severity == ValidationSeverity.ERROR]
            self.assertTrue(len(header_errors) > 0)
            # At least one error should mention magic bytes
            self.assertTrue(any("magic bytes" in issue.message.lower() 
                              for issue in header_errors))
            
        except SPDValidationError as e:
            self.fail(f"validate_spd_file should handle invalid headers in BASIC mode, got: {e}")
    
    def test_validate_future_version(self) -> None:
        """Test validating a file with a future version number."""
        path = self.create_future_version_file(os.path.join(self.test_dir.name, "future_version.spd"))
        
        # Similar to invalid header test
        try:
            with open(path, 'rb') as f:
                # Call validate_spd_file instead of creating the validator directly
                is_valid, report = validate_spd_file(f, ValidationLevel.BASIC)
            
            self.assertFalse(is_valid)
            self.assertTrue(report.get_error_count() > 0)
            
            # Check for error messages about version
            header_errors = [i for i in report.issues if i.section == "header" and 
                            i.severity == ValidationSeverity.ERROR]
            self.assertTrue(len(header_errors) > 0)
            # At least one error should mention version
            self.assertTrue(any("version" in issue.message.lower() 
                              for issue in header_errors))
            
        except SPDValidationError as e:
            self.fail(f"validate_spd_file should handle future versions in BASIC mode, got: {e}")
    
    def test_standalone_validation_function(self) -> None:
        """Test the standalone validation function."""
        path = self.create_valid_spd_file(os.path.join(self.test_dir.name, "valid.spd"))
        
        is_valid, report = validate_spd_file(path, ValidationLevel.BASIC)
        
        self.assertTrue(is_valid)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.validation_level, ValidationLevel.BASIC)


class TestSPDValidatorStandard(TestSPDValidatorBase):
    """Test standard validation functionality of the SPD validator."""
    
    def test_validate_valid_file_standard(self) -> None:
        """Test validating a valid SPD file with standard validation."""
        path = self.create_valid_spd_file(os.path.join(self.test_dir.name, "valid.spd"))
        
        with SPDValidator(path, ValidationLevel.STANDARD) as validator:
            report = validator.validate()
        
        self.assertTrue(report.is_valid)
        self.assertEqual(report.validation_level, ValidationLevel.STANDARD)
        
        # All sections should be checked and valid
        expected_sections = ["header", "flags", "image", "landmarks", "motion_params", 
                           "appearance", "transforms", "mask", "additional_flags"]
        for section in expected_sections:
            self.assertIn(section, report.sections_checked)
            self.assertIn(section, report.sections_valid)
        
        self.assertEqual(len(report.sections_invalid), 0)
        self.assertEqual(report.get_error_count(), 0)
    
    def test_missing_section_standard(self) -> None:
        """Test validating a file with a missing section but the flag set."""
        path = self.create_missing_section_file(os.path.join(self.test_dir.name, "missing_section.spd"))
        
        with SPDValidator(path, ValidationLevel.STANDARD) as validator:
            report = validator.validate()
        
        self.assertFalse(report.is_valid)
        self.assertEqual(report.validation_level, ValidationLevel.STANDARD)
        
        # Check for error about missing landmarks section
        flags_errors = [i for i in report.issues if i.severity == ValidationSeverity.ERROR and
                      "landmark" in i.message.lower() and "missing" in i.message.lower()]
        
        self.assertTrue(len(flags_errors) > 0)
    
    def test_no_flag_section_standard(self) -> None:
        """Test validating a file with a section present but the flag not set."""
        path = self.create_no_flag_section_file(os.path.join(self.test_dir.name, "no_flag_section.spd"))
        
        with SPDValidator(path, ValidationLevel.STANDARD) as validator:
            report = validator.validate()
        
        self.assertFalse(report.is_valid)
        self.assertEqual(report.validation_level, ValidationLevel.STANDARD)
        
        # Check for error message about missing flag
        image_flag_errors = [i for i in report.issues if i.severity == ValidationSeverity.ERROR and
                           "image" in i.message.lower() and "flag" in i.message.lower()]
        
        self.assertTrue(len(image_flag_errors) > 0)
    
    def test_corrupted_image_section(self) -> None:
        """Test validating a file with corrupted image data."""
        path = self.create_corrupted_image_section_file(os.path.join(self.test_dir.name, "corrupted_image.spd"))
        
        with SPDValidator(path, ValidationLevel.STANDARD) as validator:
            report = validator.validate()
            
        # A corrupted section should result in validation errors
        self.assertFalse(report.is_valid)
    
    def test_corrupted_landmarks_section(self) -> None:
        """Test validating a file with corrupted landmarks data."""
        path = self.create_corrupted_landmarks_file(os.path.join(self.test_dir.name, "corrupted_landmarks.spd"))
        
        with SPDValidator(path, ValidationLevel.STANDARD) as validator:
            report = validator.validate()
        
        self.assertFalse(report.is_valid)
        
        # Check for error message about landmarks count mismatch
        landmarks_errors = [i for i in report.issues if i.severity == ValidationSeverity.ERROR and
                           "landmarks" in i.section.lower() and "count" in i.message.lower()]
        
        self.assertTrue(len(landmarks_errors) > 0)


class TestSPDValidatorStrict(TestSPDValidatorBase):
    """Test strict validation functionality of the SPD validator."""
    
    def test_validate_valid_file_strict(self) -> None:
        """Test validating a valid SPD file with strict validation."""
        path = self.create_valid_spd_file(os.path.join(self.test_dir.name, "valid.spd"))
        
        with SPDValidator(path, ValidationLevel.STRICT) as validator:
            report = validator.validate()
        
        self.assertTrue(report.is_valid)
        self.assertEqual(report.validation_level, ValidationLevel.STRICT)
    
    def test_invalid_transforms_strict(self) -> None:
        """Test validating a file with invalid transformation matrices."""
        path = self.create_invalid_transforms_file(os.path.join(self.test_dir.name, "invalid_transforms.spd"))
        
        with SPDValidator(path, ValidationLevel.STRICT) as validator:
            report = validator.validate()
        
        # Poor quality matrices should be reported as warnings, not errors
        # The file could still be valid overall
        transform_quality_issues = [i for i in report.issues 
                                 if "transform" in i.section.lower() and 
                                 "singular" in i.message.lower() or 
                                 "invertible" in i.message.lower()]
        
        self.assertTrue(len(transform_quality_issues) > 0)
    
    def test_cross_section_mismatch_strict(self) -> None:
        """Test validating a file with mismatches between sections (image and mask dimensions)."""
        path = self.create_cross_section_mismatch_file(os.path.join(self.test_dir.name, "cross_section_mismatch.spd"))
        
        with SPDValidator(path, ValidationLevel.STRICT) as validator:
            report = validator.validate()
        
        # Dimensional mismatch should generate warnings
        cross_section_warnings = [i for i in report.issues 
                                if i.section == "cross_section" and
                                i.severity == ValidationSeverity.WARNING and
                                "dimension" in i.message.lower()]
        
        self.assertTrue(len(cross_section_warnings) > 0)
    
    def test_landmarks_outside_bounds_strict(self) -> None:
        """Test validating a file with landmarks outside image bounds."""
        path = self.create_landmarks_outside_bounds_file(os.path.join(self.test_dir.name, "landmarks_outside.spd"))
        
        with SPDValidator(path, ValidationLevel.STRICT) as validator:
            report = validator.validate()
        
        # Should generate warnings about landmarks outside bounds
        landmark_bounds_warnings = [i for i in report.issues 
                                 if i.section == "cross_section" and
                                 i.severity == ValidationSeverity.WARNING and
                                 "outside" in i.message.lower()]
        
        self.assertTrue(len(landmark_bounds_warnings) > 0)
    
    def test_duplicate_landmarks_strict(self) -> None:
        """Test validating a file with duplicate landmarks."""
        path = self.create_duplicate_landmarks_file(os.path.join(self.test_dir.name, "duplicate_landmarks.spd"))
        
        with SPDValidator(path, ValidationLevel.STRICT) as validator:
            report = validator.validate()
        
        # Should generate warnings about duplicate landmarks
        duplicate_warnings = [i for i in report.issues 
                           if i.section == "landmarks_quality" and
                           i.severity == ValidationSeverity.WARNING and
                           "duplicate" in i.message.lower()]
        
        self.assertTrue(len(duplicate_warnings) > 0)


class TestValidationReport(unittest.TestCase):
    """Test the ValidationReport class."""
    
    def test_report_summary(self) -> None:
        """Test generating a summary from a validation report."""
        report = ValidationReport(
            is_valid=False,
            validation_level=ValidationLevel.STANDARD,
            file_path="/test/file.spd",
            sections_checked=["header", "image", "landmarks"],
            sections_valid=["header", "landmarks"],
            sections_invalid=["image"],
            elapsed_time_ms=150.5
        )
        
        # Add some issues
        report.add_issue("header", ValidationSeverity.INFO, "This is a test header info")
        report.add_issue("image", ValidationSeverity.ERROR, "Invalid image data", "Fix the image data")
        report.add_issue("landmarks", ValidationSeverity.WARNING, "Suspicious landmarks", "Check landmarks")
        
        summary = report.summarize()
        
        # Check that the summary contains the expected information
        self.assertIn("INVALID", summary)
        self.assertIn("STANDARD", summary)
        self.assertIn("Errors: 1", summary)
        self.assertIn("Warnings: 1", summary)
        self.assertIn("Info: 1", summary)
        self.assertIn("header, image, landmarks", summary)
        self.assertIn("image", summary)
        self.assertIn("150.50 ms", summary)
        self.assertIn("[ERROR] image: Invalid image data", summary)
        self.assertIn("[WARNING] landmarks: Suspicious landmarks", summary)
        self.assertIn("[INFO] header: This is a test header info", summary)
        self.assertIn("Suggestion: Fix the image data", summary)
        self.assertIn("Suggestion: Check landmarks", summary)
    
    def test_count_methods(self) -> None:
        """Test the methods for counting issues by severity."""
        report = ValidationReport()
        
        # Initially should have no issues
        self.assertEqual(report.get_error_count(), 0)
        self.assertEqual(report.get_warning_count(), 0)
        self.assertEqual(report.get_info_count(), 0)
        
        # Add issues of each type
        report.add_issue("test", ValidationSeverity.ERROR, "Error")
        report.add_issue("test", ValidationSeverity.WARNING, "Warning")
        report.add_issue("test", ValidationSeverity.INFO, "Info")
        report.add_issue("test", ValidationSeverity.ERROR, "Another error")
        
        # Check counts
        self.assertEqual(report.get_error_count(), 2)
        self.assertEqual(report.get_warning_count(), 1)
        self.assertEqual(report.get_info_count(), 1)


if __name__ == '__main__':
    unittest.main()