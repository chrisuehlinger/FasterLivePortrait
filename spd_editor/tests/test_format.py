"""
Tests for the SPD format module.
"""
import unittest
import struct
import time
import sys
import os
from typing import Dict, List

# Add the parent directory to sys.path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the module we're testing
from spd.format import (
    MAGIC_BYTES, CURRENT_VERSION, 
    HeaderFlags, SectionMarker,
    HeaderSection, ImageSection, LandmarksSection, 
    MotionParamsSection, AppearanceSection, TransformationSection,
    MaskSection, AdditionalFlagsSection,
    validate_header, flags_to_dict, dict_to_flags,
    compute_section_size, get_required_flags_for_section
)


class TestSPDFormat(unittest.TestCase):
    """Test suite for the SPD format module."""
    
    def test_module_imports(self):
        """Test that the module imports correctly."""
        self.assertIsNotNone(MAGIC_BYTES)
    
    def test_magic_bytes(self):
        """Test that magic bytes are correctly defined."""
        self.assertEqual(MAGIC_BYTES, b"SPDV")
    
    def test_current_version(self):
        """Test that current version is correctly defined."""
        self.assertEqual(CURRENT_VERSION, 1)
    
    def test_header_flags(self):
        """Test the header flags enum definition."""
        # Check that all expected flags are defined
        self.assertEqual(HeaderFlags.HAS_IMAGE.value, 1 << 0)
        self.assertEqual(HeaderFlags.HAS_LANDMARKS.value, 1 << 1)
        self.assertEqual(HeaderFlags.HAS_MOTION_PARAMS.value, 1 << 2)
        self.assertEqual(HeaderFlags.HAS_APPEARANCE.value, 1 << 3)
        self.assertEqual(HeaderFlags.HAS_TRANSFORMS.value, 1 << 4)
        self.assertEqual(HeaderFlags.HAS_MASK.value, 1 << 5)
        self.assertEqual(HeaderFlags.IS_COMPRESSED.value, 1 << 6)
        self.assertEqual(HeaderFlags.IS_ENCRYPTED.value, 1 << 7)
    
    def test_section_markers(self):
        """Test the section markers enum definition."""
        # Check that all expected section markers are defined
        self.assertEqual(SectionMarker.HEADER, 0x48454144)  # "HEAD"
        self.assertEqual(SectionMarker.IMAGE, 0x494D4147)   # "IMAG"
        self.assertEqual(SectionMarker.LANDMARKS, 0x4C4D4B53)  # "LMKS"
        self.assertEqual(SectionMarker.MOTION_PARAMS, 0x4D4F5450)  # "MOTP"
        self.assertEqual(SectionMarker.APPEARANCE, 0x41505045)  # "APPE"
        self.assertEqual(SectionMarker.TRANSFORMS, 0x5452414E)  # "TRAN"
        self.assertEqual(SectionMarker.MASK, 0x4D41534B)  # "MASK"
        self.assertEqual(SectionMarker.ADDITIONAL_FLAGS, 0x464C4753)  # "FLGS"
        self.assertEqual(SectionMarker.END, 0x454E4421)  # "END!"
    
    def test_header_section(self):
        """Test the HeaderSection data structure."""
        # Default initialization
        header = HeaderSection()
        self.assertEqual(header.magic_bytes, MAGIC_BYTES)
        self.assertEqual(header.version, CURRENT_VERSION)
        self.assertNotEqual(header.timestamp, 0.0)  # Should be set to current time
        self.assertEqual(header.flags, 0)
        
        # Custom initialization
        custom_time = time.time() - 3600  # 1 hour ago
        header = HeaderSection(
            magic_bytes=b"SPDV",
            version=1,
            timestamp=custom_time,
            flags=HeaderFlags.HAS_IMAGE.value | HeaderFlags.HAS_LANDMARKS.value
        )
        self.assertEqual(header.magic_bytes, b"SPDV")
        self.assertEqual(header.version, 1)
        self.assertEqual(header.timestamp, custom_time)
        self.assertEqual(header.flags, 3)  # 1 + 2 = 3
    
    def test_flags_to_dict(self):
        """Test converting bit flags to a dictionary."""
        # No flags set
        flag_dict = flags_to_dict(0)
        for name, value in flag_dict.items():
            self.assertFalse(value)
        
        # All flags set
        all_flags = sum(flag.value for flag in HeaderFlags)
        flag_dict = flags_to_dict(all_flags)
        for name, value in flag_dict.items():
            self.assertTrue(value)
        
        # Specific flags
        flags = HeaderFlags.HAS_IMAGE.value | HeaderFlags.HAS_LANDMARKS.value
        flag_dict = flags_to_dict(flags)
        self.assertTrue(flag_dict["HAS_IMAGE"])
        self.assertTrue(flag_dict["HAS_LANDMARKS"])
        self.assertFalse(flag_dict["HAS_MOTION_PARAMS"])
    
    def test_dict_to_flags(self):
        """Test converting a dictionary to bit flags."""
        # Empty dict
        flags = dict_to_flags({})
        self.assertEqual(flags, 0)
        
        # All flags True
        flag_dict = {flag.name: True for flag in HeaderFlags}
        flags = dict_to_flags(flag_dict)
        self.assertEqual(flags, sum(flag.value for flag in HeaderFlags))
        
        # Specific flags
        flag_dict = {
            "HAS_IMAGE": True,
            "HAS_LANDMARKS": True,
            "HAS_MOTION_PARAMS": False
        }
        flags = dict_to_flags(flag_dict)
        self.assertEqual(flags, HeaderFlags.HAS_IMAGE.value | HeaderFlags.HAS_LANDMARKS.value)
        
        # Invalid flag names should be ignored
        flag_dict = {
            "HAS_IMAGE": True,
            "INVALID_FLAG": True
        }
        flags = dict_to_flags(flag_dict)
        self.assertEqual(flags, HeaderFlags.HAS_IMAGE.value)
    
    def test_validate_header(self):
        """Test header validation."""
        # Valid header
        valid_header = (
            MAGIC_BYTES +
            struct.pack("!I", CURRENT_VERSION) +
            struct.pack("!d", time.time()) +
            struct.pack("!I", 0)
        )
        is_valid, error = validate_header(valid_header)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        
        # Invalid magic bytes
        invalid_magic = (
            b"XXXX" +
            struct.pack("!I", CURRENT_VERSION) +
            struct.pack("!d", time.time()) +
            struct.pack("!I", 0)
        )
        is_valid, error = validate_header(invalid_magic)
        self.assertFalse(is_valid)
        self.assertIn("Invalid magic bytes", error)
        
        # Future version
        future_version = (
            MAGIC_BYTES +
            struct.pack("!I", CURRENT_VERSION + 1) +
            struct.pack("!d", time.time()) +
            struct.pack("!I", 0)
        )
        is_valid, error = validate_header(future_version)
        self.assertFalse(is_valid)
        self.assertIn("Unsupported version", error)
        
        # Header too short
        short_header = MAGIC_BYTES + struct.pack("!I", CURRENT_VERSION)
        is_valid, error = validate_header(short_header)
        self.assertFalse(is_valid)
        self.assertEqual(error, "Header too short")
    
    def test_compute_section_size(self):
        """Test computing section sizes."""
        # 1D array: 10 elements, 4 bytes each = 40 bytes
        size = compute_section_size((10,))
        self.assertEqual(size, 40)
        
        # 2D array: 5x10 elements, 4 bytes each = 200 bytes
        size = compute_section_size((5, 10))
        self.assertEqual(size, 200)
        
        # 3D array: 3x4x5 elements, 4 bytes each = 240 bytes
        size = compute_section_size((3, 4, 5))
        self.assertEqual(size, 240)
        
        # Custom item size: 3x4x5 elements, 2 bytes each = 120 bytes
        size = compute_section_size((3, 4, 5), item_size=2)
        self.assertEqual(size, 120)
    
    def test_get_required_flags_for_section(self):
        """Test mapping sections to required header flags."""
        # Test all defined mappings
        self.assertEqual(
            get_required_flags_for_section(SectionMarker.IMAGE), 
            HeaderFlags.HAS_IMAGE
        )
        self.assertEqual(
            get_required_flags_for_section(SectionMarker.LANDMARKS), 
            HeaderFlags.HAS_LANDMARKS
        )
        self.assertEqual(
            get_required_flags_for_section(SectionMarker.MOTION_PARAMS), 
            HeaderFlags.HAS_MOTION_PARAMS
        )
        self.assertEqual(
            get_required_flags_for_section(SectionMarker.APPEARANCE), 
            HeaderFlags.HAS_APPEARANCE
        )
        self.assertEqual(
            get_required_flags_for_section(SectionMarker.TRANSFORMS), 
            HeaderFlags.HAS_TRANSFORMS
        )
        self.assertEqual(
            get_required_flags_for_section(SectionMarker.MASK), 
            HeaderFlags.HAS_MASK
        )
        
        # No flag required for header, end, or additional flags sections
        self.assertIsNone(get_required_flags_for_section(SectionMarker.HEADER))
        self.assertIsNone(get_required_flags_for_section(SectionMarker.END))
        self.assertIsNone(get_required_flags_for_section(SectionMarker.ADDITIONAL_FLAGS))
    
    def test_section_data_structures(self):
        """Test all data structures for SPD file sections."""
        # Test ImageSection
        img = ImageSection(width=640, height=480, channels=3, format="RGB", data=b"image_data")
        self.assertEqual(img.width, 640)
        self.assertEqual(img.height, 480)
        self.assertEqual(img.channels, 3)
        self.assertEqual(img.format, "RGB")
        self.assertEqual(img.data, b"image_data")
        
        # Test LandmarksSection
        landmarks = LandmarksSection(
            count=68, 
            dimensions=2, 
            landmark_type="dlib68",
            points=[[0.1, 0.2], [0.3, 0.4]]
        )
        self.assertEqual(landmarks.count, 68)
        self.assertEqual(landmarks.dimensions, 2)
        self.assertEqual(landmarks.landmark_type, "dlib68")
        self.assertEqual(len(landmarks.points), 2)
        
        # Test default initialization of points list
        landmarks = LandmarksSection()
        self.assertEqual(landmarks.points, [])
        
        # Test MotionParamsSection
        motion = MotionParamsSection(
            num_params=3,
            param_names=["yaw", "pitch", "roll"],
            values=[0.1, 0.2, 0.3]
        )
        self.assertEqual(motion.num_params, 3)
        self.assertEqual(motion.param_names, ["yaw", "pitch", "roll"])
        self.assertEqual(motion.values, [0.1, 0.2, 0.3])
        
        # Test default initialization of lists
        motion = MotionParamsSection()
        self.assertEqual(motion.param_names, [])
        self.assertEqual(motion.values, [])
        
        # Test AppearanceSection
        appearance = AppearanceSection(
            feature_dim=256,
            feature_type="embedding",
            features=[0.1] * 256
        )
        self.assertEqual(appearance.feature_dim, 256)
        self.assertEqual(appearance.feature_type, "embedding")
        self.assertEqual(len(appearance.features), 256)
        
        # Test default initialization of features list
        appearance = AppearanceSection()
        self.assertEqual(appearance.features, [])
        
        # Test TransformationSection
        transforms = TransformationSection(
            num_transforms=2,
            transform_types=["face2world", "world2face"],
            matrices=[[1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 0, 1]]
        )
        self.assertEqual(transforms.num_transforms, 2)
        self.assertEqual(transforms.transform_types, ["face2world", "world2face"])
        self.assertEqual(len(transforms.matrices), 2)
        
        # Test default initialization of lists
        transforms = TransformationSection()
        self.assertEqual(transforms.transform_types, [])
        self.assertEqual(transforms.matrices, [])
        
        # Test MaskSection
        mask = MaskSection(
            width=640,
            height=480,
            format="binary",
            data=b"mask_data"
        )
        self.assertEqual(mask.width, 640)
        self.assertEqual(mask.height, 480)
        self.assertEqual(mask.format, "binary")
        self.assertEqual(mask.data, b"mask_data")
        
        # Test AdditionalFlagsSection
        flags_section = AdditionalFlagsSection(
            num_flags=2,
            flags={"source": "person.jpg", "processed": True}
        )
        self.assertEqual(flags_section.num_flags, 2)
        self.assertEqual(flags_section.flags["source"], "person.jpg")
        self.assertTrue(flags_section.flags["processed"])
        
        # Test default initialization of flags dict
        flags_section = AdditionalFlagsSection()
        self.assertEqual(flags_section.flags, {})


if __name__ == "__main__":
    unittest.main()