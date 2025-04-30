"""
Tests for the SPD reader module.
"""
import unittest
import os
import struct
import io
import tempfile
import time
from pathlib import Path
import numpy as np

# Add the parent directory to sys.path for imports
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the modules we're testing
from spd.format import (
    MAGIC_BYTES, CURRENT_VERSION, 
    HeaderFlags, SectionMarker,
    HeaderSection, ImageSection, LandmarksSection,
    MotionParamsSection, AppearanceSection, TransformationSection,
    MaskSection, AdditionalFlagsSection
)
from spd.reader import (
    SPDReader, SPDError, SPDFormatError, SPDVersionError, SPDSectionError
)


class TestSPDReaderBasic(unittest.TestCase):
    """Test basic functionality of the SPD reader."""
    
    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
    
    def create_test_spd_file(self, path, include_sections=None):
        """
        Create a test SPD file with the specified sections.
        
        Args:
            path: Path to create the file at.
            include_sections: List of section markers to include, or None for default.
        
        Returns:
            The path to the created file.
        """
        if include_sections is None:
            include_sections = [SectionMarker.IMAGE, SectionMarker.LANDMARKS]
        
        # Calculate flags based on included sections
        flags = 0
        for section in include_sections:
            if section == SectionMarker.IMAGE:
                flags |= HeaderFlags.HAS_IMAGE.value
            elif section == SectionMarker.LANDMARKS:
                flags |= HeaderFlags.HAS_LANDMARKS.value
            elif section == SectionMarker.MOTION_PARAMS:
                flags |= HeaderFlags.HAS_MOTION_PARAMS.value
            elif section == SectionMarker.APPEARANCE:
                flags |= HeaderFlags.HAS_APPEARANCE.value
            elif section == SectionMarker.TRANSFORMS:
                flags |= HeaderFlags.HAS_TRANSFORMS.value
            elif section == SectionMarker.MASK:
                flags |= HeaderFlags.HAS_MASK.value
        
        with open(path, "wb") as f:
            # Write header
            f.write(MAGIC_BYTES)  # Magic bytes
            f.write(struct.pack("!I", CURRENT_VERSION))  # Version
            f.write(struct.pack("!d", time.time()))  # Timestamp
            f.write(struct.pack("!I", flags))  # Flags
            
            # Write image section if included
            if SectionMarker.IMAGE in include_sections:
                f.write(struct.pack("!I", SectionMarker.IMAGE.value))  # Section marker
                
                # Generate test image data (10x10 RGB)
                width, height, channels = 10, 10, 3
                image_data = bytes([i % 256 for i in range(width * height * channels)])
                
                # Calculate section size (metadata + data)
                # 4 bytes width + 4 bytes height + 1 byte channels + 
                # 4 bytes format string length + len("RGB") bytes format string +
                # width * height * channels bytes image data
                format_str = "RGB"
                format_str_bytes = format_str.encode('utf-8')
                section_size = 4 + 4 + 1 + 4 + len(format_str_bytes) + len(image_data)
                
                # Write section
                f.write(struct.pack("!I", section_size))  # Section size
                f.write(struct.pack("!I", width))  # Width
                f.write(struct.pack("!I", height))  # Height
                f.write(bytes([channels]))  # Channels
                f.write(struct.pack("!I", len(format_str_bytes)))  # Format string length
                f.write(format_str_bytes)  # Format string
                f.write(image_data)  # Image data
            
            # Write landmarks section if included
            if SectionMarker.LANDMARKS in include_sections:
                f.write(struct.pack("!I", SectionMarker.LANDMARKS.value))  # Section marker
                
                # Generate test landmark data (5 landmarks, 2D)
                count, dimensions = 5, 2
                landmark_type = "test"
                landmark_type_bytes = landmark_type.encode('utf-8')
                points = [(i/10.0, i/5.0) for i in range(count)]
                
                # Calculate section size
                # 4 bytes count + 4 bytes dimensions + 
                # 4 bytes landmark type length + len(landmark_type) bytes landmark type +
                # count * dimensions * 4 bytes point data
                section_size = 4 + 4 + 4 + len(landmark_type_bytes) + (count * dimensions * 4)
                
                # Write section
                f.write(struct.pack("!I", section_size))  # Section size
                f.write(struct.pack("!I", count))  # Count
                f.write(struct.pack("!I", dimensions))  # Dimensions
                f.write(struct.pack("!I", len(landmark_type_bytes)))  # Landmark type length
                f.write(landmark_type_bytes)  # Landmark type
                
                # Write points
                for point in points:
                    for coord in point:
                        f.write(struct.pack("!f", coord))  # Coordinate
            
            # Write motion parameters section if included
            if SectionMarker.MOTION_PARAMS in include_sections:
                f.write(struct.pack("!I", SectionMarker.MOTION_PARAMS.value))  # Section marker
                
                # Generate test motion parameter data
                num_params = 3
                param_names = ["yaw", "pitch", "roll"]
                values = [0.1, 0.2, 0.3]
                
                # Calculate section size
                # 4 bytes num_params + 
                # (4 bytes param name length + len(param_name) bytes param name) * num_params +
                # 4 bytes * num_params values
                section_size = 4
                for name in param_names:
                    name_bytes = name.encode('utf-8')
                    section_size += 4 + len(name_bytes)
                section_size += num_params * 4
                
                # Write section
                f.write(struct.pack("!I", section_size))  # Section size
                f.write(struct.pack("!I", num_params))  # Num params
                
                # Write param names
                for name in param_names:
                    name_bytes = name.encode('utf-8')
                    f.write(struct.pack("!I", len(name_bytes)))  # Name length
                    f.write(name_bytes)  # Name
                
                # Write param values
                for value in values:
                    f.write(struct.pack("!f", value))  # Value
            
            # Write appearance section if included
            if SectionMarker.APPEARANCE in include_sections:
                f.write(struct.pack("!I", SectionMarker.APPEARANCE.value))  # Section marker
                
                # Generate test appearance data
                feature_dim = 4
                feature_type = "embedding"
                feature_type_bytes = feature_type.encode('utf-8')
                features = [0.1, 0.2, 0.3, 0.4]
                
                # Calculate section size
                # 4 bytes feature_dim + 
                # 4 bytes feature type length + len(feature_type) bytes feature type +
                # 4 bytes * feature_dim features
                section_size = 4 + 4 + len(feature_type_bytes) + (feature_dim * 4)
                
                # Write section
                f.write(struct.pack("!I", section_size))  # Section size
                f.write(struct.pack("!I", feature_dim))  # Feature dim
                f.write(struct.pack("!I", len(feature_type_bytes)))  # Feature type length
                f.write(feature_type_bytes)  # Feature type
                
                # Write features
                for feature in features:
                    f.write(struct.pack("!f", feature))  # Feature
            
            # Write transforms section if included
            if SectionMarker.TRANSFORMS in include_sections:
                f.write(struct.pack("!I", SectionMarker.TRANSFORMS.value))  # Section marker
                
                # Generate test transform data
                num_transforms = 2
                transform_types = ["face2world", "world2face"]
                # Identity matrices
                matrices = [
                    [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 1]
                ]
                
                # Calculate section size
                # 4 bytes num_transforms + 
                # (4 bytes transform type length + len(transform_type) bytes transform type) * num_transforms +
                # (9 * 4 bytes) * num_transforms matrices
                section_size = 4
                for name in transform_types:
                    name_bytes = name.encode('utf-8')
                    section_size += 4 + len(name_bytes)
                section_size += num_transforms * 9 * 4
                
                # Write section
                f.write(struct.pack("!I", section_size))  # Section size
                f.write(struct.pack("!I", num_transforms))  # Num transforms
                
                # Write transform types
                for name in transform_types:
                    name_bytes = name.encode('utf-8')
                    f.write(struct.pack("!I", len(name_bytes)))  # Name length
                    f.write(name_bytes)  # Name
                
                # Write matrices
                for matrix in matrices:
                    for value in matrix:
                        f.write(struct.pack("!f", value))  # Value
            
            # Write mask section if included
            if SectionMarker.MASK in include_sections:
                f.write(struct.pack("!I", SectionMarker.MASK.value))  # Section marker
                
                # Generate test mask data
                width, height = 10, 10
                mask_format = "binary"
                mask_format_bytes = mask_format.encode('utf-8')
                mask_data = bytes([i % 2 for i in range(width * height)])
                
                # Calculate section size
                # 4 bytes width + 4 bytes height + 
                # 4 bytes format length + len(format) bytes format +
                # width * height bytes mask data
                section_size = 4 + 4 + 4 + len(mask_format_bytes) + len(mask_data)
                
                # Write section
                f.write(struct.pack("!I", section_size))  # Section size
                f.write(struct.pack("!I", width))  # Width
                f.write(struct.pack("!I", height))  # Height
                f.write(struct.pack("!I", len(mask_format_bytes)))  # Format length
                f.write(mask_format_bytes)  # Format
                f.write(mask_data)  # Mask data
            
            # Write additional flags section
            f.write(struct.pack("!I", SectionMarker.ADDITIONAL_FLAGS.value))  # Section marker
            
            # Generate test flags data
            num_flags = 3
            flag_data = [
                ("source", 3, "test.jpg"),  # string
                ("processed", 0, True),     # boolean
                ("quality", 2, 0.95),       # float
            ]
            
            # Calculate section size
            section_size = 4  # 4 bytes for num_flags
            for key, type_code, value in flag_data:
                key_bytes = key.encode('utf-8')
                section_size += 4 + len(key_bytes) + 1  # 4 bytes length, key bytes, 1 byte type
                
                if type_code == 0:  # boolean
                    section_size += 1
                elif type_code in (1, 2):  # int or float
                    section_size += 4
                elif type_code == 3:  # string
                    val_bytes = value.encode('utf-8')
                    section_size += 4 + len(val_bytes)
            
            # Write section
            f.write(struct.pack("!I", section_size))  # Section size
            f.write(struct.pack("!I", num_flags))  # Num flags
            
            # Write flags
            for key, type_code, value in flag_data:
                # Write key
                key_bytes = key.encode('utf-8')
                f.write(struct.pack("!I", len(key_bytes)))  # Key length
                f.write(key_bytes)  # Key
                
                # Write type
                f.write(bytes([type_code]))  # Type code
                
                # Write value based on type
                if type_code == 0:  # boolean
                    f.write(bytes([1 if value else 0]))  # Boolean value
                elif type_code == 1:  # int
                    f.write(struct.pack("!i", value))  # Integer value
                elif type_code == 2:  # float
                    f.write(struct.pack("!f", value))  # Float value
                elif type_code == 3:  # string
                    val_bytes = value.encode('utf-8')
                    f.write(struct.pack("!I", len(val_bytes)))  # String length
                    f.write(val_bytes)  # String value
            
            # Write end marker
            f.write(struct.pack("!I", SectionMarker.END.value))  # Section marker
            f.write(struct.pack("!I", 0))  # Section size (0 for end marker)
        
        return path
    
    def create_corrupted_spd_file(self, path, corruption_type):
        """
        Create a corrupted SPD file for testing error handling.
        
        Args:
            path: Path to create the file at.
            corruption_type: Type of corruption to apply.
        
        Returns:
            The path to the created file.
        """
        # First create a normal file
        self.create_test_spd_file(path)
        
        # Then corrupt it based on the corruption type
        with open(path, "r+b") as f:
            if corruption_type == "bad_magic":
                # Corrupt the magic bytes
                f.seek(0)
                f.write(b"XXXX")
            
            elif corruption_type == "future_version":
                # Set a future version
                f.seek(4)
                f.write(struct.pack("!I", CURRENT_VERSION + 1))
            
            elif corruption_type == "truncated_header":
                # Truncate the file after the magic bytes
                f.seek(4)
                f.truncate()
            
            elif corruption_type == "missing_section":
                # Corrupt a section marker
                f.seek(20)  # Skip header
                f.write(b"\xFF\xFF\xFF\xFF")  # Invalid section marker
            
            elif corruption_type == "bad_section_size":
                # Set an invalid section size
                f.seek(24)  # Skip header and section marker
                f.write(struct.pack("!I", 0xFFFFFFFF))  # Unreasonably large size
                
            elif corruption_type == "truncated_section":
                # Truncate the file in the middle of a section
                f.seek(30)
                f.truncate()
                
        return path
    
    def test_file_not_found(self):
        """Test that trying to open a non-existent file raises an error."""
        with self.assertRaises(IOError):
            SPDReader("nonexistent_file.spd")
    
    def test_invalid_magic_bytes(self):
        """Test that invalid magic bytes are detected."""
        path = os.path.join(self.test_dir.name, "bad_magic.spd")
        self.create_corrupted_spd_file(path, "bad_magic")
        
        with self.assertRaises(SPDFormatError):
            SPDReader(path)
    
    def test_future_version(self):
        """Test that a future version is rejected."""
        path = os.path.join(self.test_dir.name, "future_version.spd")
        self.create_corrupted_spd_file(path, "future_version")
        
        with self.assertRaises(SPDFormatError):
            SPDReader(path)
    
    def test_truncated_header(self):
        """Test handling of a truncated header."""
        path = os.path.join(self.test_dir.name, "truncated_header.spd")
        self.create_corrupted_spd_file(path, "truncated_header")
        
        with self.assertRaises(SPDFormatError):
            SPDReader(path)
    
    def test_missing_section(self):
        """Test handling of a missing section."""
        path = os.path.join(self.test_dir.name, "corrupt.spd")
        self.create_corrupted_spd_file(path, "missing_section")
        
        # Should be able to open the file, but reading the section should fail
        reader = SPDReader(path)
        with self.assertRaises(SPDSectionError):
            reader.image
    
    def test_bad_section_size(self):
        """Test handling of an invalid section size."""
        path = os.path.join(self.test_dir.name, "bad_section.spd")
        self.create_corrupted_spd_file(path, "bad_section_size")
        
        # Should be able to open the file, but reading the section should fail
        reader = SPDReader(path)
        with self.assertRaises(SPDSectionError):
            reader.image
    
    def test_truncated_section(self):
        """Test handling of a truncated section."""
        path = os.path.join(self.test_dir.name, "truncated_section.spd")
        self.create_corrupted_spd_file(path, "truncated_section")
        
        # Should be able to open the file, but reading the section should fail
        reader = SPDReader(path)
        with self.assertRaises(SPDSectionError):
            reader.image
    
    def test_missing_section_marker(self):
        """Test handling of a missing section that's indicated by flags."""
        path = os.path.join(self.test_dir.name, "test.spd")
        # Create a file with only a landmarks section but with the image flag set
        with open(path, "wb") as f:
            # Write header with HAS_IMAGE flag
            f.write(MAGIC_BYTES)  # Magic bytes
            f.write(struct.pack("!I", CURRENT_VERSION))  # Version
            f.write(struct.pack("!d", time.time()))  # Timestamp
            f.write(struct.pack("!I", HeaderFlags.HAS_IMAGE.value))  # Flags (only HAS_IMAGE)
            
            # Write end marker (no image section)
            f.write(struct.pack("!I", SectionMarker.END.value))  # End marker
            f.write(struct.pack("!I", 0))  # Section size (0 for end marker)
        
        # Should be able to open the file, but reading the image section should fail
        reader = SPDReader(path)
        with self.assertRaises(SPDSectionError):
            reader.image
    
    def test_read_from_buffer(self):
        """Test reading from a BytesIO buffer instead of a file."""
        path = os.path.join(self.test_dir.name, "test.spd")
        self.create_test_spd_file(path)
        
        # Read the file into memory
        with open(path, "rb") as f:
            data = f.read()
        
        # Create a reader from the buffer
        buffer = io.BytesIO(data)
        reader = SPDReader(buffer)
        
        # Should be able to read the sections
        self.assertIsNotNone(reader.header)
        self.assertIsNotNone(reader.image)
        self.assertIsNotNone(reader.landmarks)
    
    def test_context_manager(self):
        """Test using the reader as a context manager."""
        path = os.path.join(self.test_dir.name, "test.spd")
        self.create_test_spd_file(path)
        
        # Use the reader as a context manager
        with SPDReader(path) as reader:
            self.assertIsNotNone(reader.header)
            self.assertIsNotNone(reader.image)
            self.assertIsNotNone(reader.landmarks)
        
        # File should be closed after the context manager exits
        # This is hard to test directly, but we can check that the reader's _file attribute is None
        self.assertIsNone(reader._file)


class TestSPDReaderSections(unittest.TestCase):
    """Test reading specific sections from SPD files."""
    
    def setUp(self):
        """Create a temporary directory and SPD file with all sections."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        
        # Create a test file with all sections
        self.test_file = os.path.join(self.test_dir.name, "full.spd")
        self.reader_tester = TestSPDReaderBasic()
        self.reader_tester.create_test_spd_file(
            self.test_file, 
            include_sections=[
                SectionMarker.IMAGE, 
                SectionMarker.LANDMARKS,
                SectionMarker.MOTION_PARAMS,
                SectionMarker.APPEARANCE,
                SectionMarker.TRANSFORMS,
                SectionMarker.MASK
            ]
        )
        
        # Create a reader for the file
        self.reader = SPDReader(self.test_file)
    
    def test_read_header(self):
        """Test reading the header section."""
        header = self.reader.header
        self.assertEqual(header.magic_bytes, MAGIC_BYTES)
        self.assertEqual(header.version, CURRENT_VERSION)
        self.assertGreater(header.timestamp, 0)
        
        # All section flags should be set except IS_COMPRESSED and IS_ENCRYPTED
        expected_flags = (
            HeaderFlags.HAS_IMAGE.value | 
            HeaderFlags.HAS_LANDMARKS.value | 
            HeaderFlags.HAS_MOTION_PARAMS.value |
            HeaderFlags.HAS_APPEARANCE.value |
            HeaderFlags.HAS_TRANSFORMS.value |
            HeaderFlags.HAS_MASK.value
        )
        self.assertEqual(header.flags, expected_flags)
    
    def test_read_image(self):
        """Test reading the image section."""
        image = self.reader.image
        self.assertEqual(image.width, 10)
        self.assertEqual(image.height, 10)
        self.assertEqual(image.channels, 3)
        self.assertEqual(image.format, "RGB")
        self.assertEqual(len(image.data), 10 * 10 * 3)
    
    def test_read_landmarks(self):
        """Test reading the landmarks section."""
        landmarks = self.reader.landmarks
        self.assertEqual(landmarks.count, 5)
        self.assertEqual(landmarks.dimensions, 2)
        self.assertEqual(landmarks.landmark_type, "test")
        self.assertEqual(len(landmarks.points), 5)
        
        # Check the point values
        for i, point in enumerate(landmarks.points):
            self.assertEqual(len(point), 2)
            self.assertEqual(point[0], i/10.0)
            self.assertEqual(point[1], i/5.0)
    
    def test_read_motion_params(self):
        """Test reading the motion parameters section."""
        motion = self.reader.motion_params
        self.assertEqual(motion.num_params, 3)
        self.assertEqual(motion.param_names, ["yaw", "pitch", "roll"])
        self.assertEqual(motion.values, [0.1, 0.2, 0.3])
    
    def test_read_appearance(self):
        """Test reading the appearance section."""
        appearance = self.reader.appearance
        self.assertEqual(appearance.feature_dim, 4)
        self.assertEqual(appearance.feature_type, "embedding")
        self.assertEqual(appearance.features, [0.1, 0.2, 0.3, 0.4])
    
    def test_read_transforms(self):
        """Test reading the transforms section."""
        transforms = self.reader.transforms
        self.assertEqual(transforms.num_transforms, 2)
        self.assertEqual(transforms.transform_types, ["face2world", "world2face"])
        self.assertEqual(len(transforms.matrices), 2)
        
        # Check that the matrices are identity matrices
        for matrix in transforms.matrices:
            self.assertEqual(len(matrix), 9)
            self.assertEqual(matrix[0], 1.0)  # [0, 0]
            self.assertEqual(matrix[4], 1.0)  # [1, 1]
            self.assertEqual(matrix[8], 1.0)  # [2, 2]
    
    def test_read_mask(self):
        """Test reading the mask section."""
        mask = self.reader.mask
        self.assertEqual(mask.width, 10)
        self.assertEqual(mask.height, 10)
        self.assertEqual(mask.format, "binary")
        self.assertEqual(len(mask.data), 10 * 10)
    
    def test_read_additional_flags(self):
        """Test reading the additional flags section."""
        flags = self.reader.additional_flags
        self.assertEqual(flags.num_flags, 3)
        self.assertEqual(flags.flags["source"], "test.jpg")
        self.assertEqual(flags.flags["processed"], True)
        self.assertAlmostEqual(flags.flags["quality"], 0.95)
    
    def test_get_available_sections(self):
        """Test getting a list of available sections."""
        sections = self.reader.get_available_sections()
        
        # All sections should be available
        expected_sections = [
            "image", "landmarks", "motion_params", 
            "appearance", "transforms", "mask", "additional_flags"
        ]
        
        for section in expected_sections:
            self.assertIn(section, sections)
    
    def test_has_section(self):
        """Test checking if a section is available."""
        self.assertTrue(self.reader.has_section("image"))
        self.assertTrue(self.reader.has_section("landmarks"))
        self.assertTrue(self.reader.has_section("motion_params"))
        self.assertTrue(self.reader.has_section("appearance"))
        self.assertTrue(self.reader.has_section("transforms"))
        self.assertTrue(self.reader.has_section("mask"))
        self.assertTrue(self.reader.has_section("additional_flags"))
        
        self.assertFalse(self.reader.has_section("nonexistent"))


class TestSPDReaderLazyLoading(unittest.TestCase):
    """Test lazy loading of sections from SPD files."""
    
    def setUp(self):
        """Create a temporary directory and SPD file with all sections."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)
        
        # Create a test file with all sections
        self.test_file = os.path.join(self.test_dir.name, "full.spd")
        self.reader_tester = TestSPDReaderBasic()
        self.reader_tester.create_test_spd_file(
            self.test_file, 
            include_sections=[
                SectionMarker.IMAGE, 
                SectionMarker.LANDMARKS,
                SectionMarker.MOTION_PARAMS,
                SectionMarker.APPEARANCE,
                SectionMarker.TRANSFORMS,
                SectionMarker.MASK
            ]
        )
    
    def test_lazy_loading(self):
        """Test that sections are only loaded when accessed."""
        # Create a reader with lazy loading (default)
        reader = SPDReader(self.test_file)
        
        # Initially, only the header should be read
        self.assertIsNotNone(reader._header)
        self.assertIsNone(reader._image)
        self.assertIsNone(reader._landmarks)
        self.assertIsNone(reader._motion_params)
        self.assertIsNone(reader._appearance)
        self.assertIsNone(reader._transforms)
        self.assertIsNone(reader._mask)
        self.assertIsNone(reader._additional_flags)
        
        # Access the image section
        image = reader.image
        self.assertIsNotNone(reader._image)
        self.assertIsNone(reader._landmarks)  # Other sections still not loaded
        
        # Access the landmarks section
        landmarks = reader.landmarks
        self.assertIsNotNone(reader._image)
        self.assertIsNotNone(reader._landmarks)
        self.assertIsNone(reader._motion_params)  # Other sections still not loaded
    
    def test_eager_loading(self):
        """Test that all sections are loaded immediately with eager loading."""
        # Create a reader with eager loading
        reader = SPDReader(self.test_file, eager_load=True)
        
        # All sections should be loaded
        self.assertIsNotNone(reader._header)
        self.assertIsNotNone(reader._image)
        self.assertIsNotNone(reader._landmarks)
        self.assertIsNotNone(reader._motion_params)
        self.assertIsNotNone(reader._appearance)
        self.assertIsNotNone(reader._transforms)
        self.assertIsNotNone(reader._mask)
        self.assertIsNotNone(reader._additional_flags)


if __name__ == "__main__":
    unittest.main()