"""
Tests for the SPD writer module.
"""
import unittest
import tempfile
import os
import struct
import time
import io
from pathlib import Path
import sys
from typing import Dict, List, Set

# Add the parent directory to sys.path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the modules we're testing
from spd.format import (
    MAGIC_BYTES, CURRENT_VERSION, 
    HeaderFlags, SectionMarker,
    HeaderSection, ImageSection, LandmarksSection, 
    MotionParamsSection, AppearanceSection, TransformationSection,
    MaskSection, AdditionalFlagsSection,
    flags_to_dict
)
from spd.writer import SPDWriter, SPDValidationError
from spd.reader import SPDReader


class TestSPDWriterBasic(unittest.TestCase):
    """Test basic functionality of the SPD writer."""
    
    def setUp(self):
        """Set up a temporary directory for test files."""
        self.test_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up temporary directory."""
        self.test_dir.cleanup()
    
    def test_create_empty_spd_file(self):
        """Test creating an empty SPD file with just a header."""
        path = os.path.join(self.test_dir.name, "empty.spd")
        
        # Create a new SPD file
        with SPDWriter(path) as writer:
            # Don't add any sections, just let it finalize
            pass
        
        # File should exist
        self.assertTrue(os.path.exists(path))
        
        # File size should be header (20 bytes) + END marker (8 bytes)
        self.assertEqual(os.path.getsize(path), 20 + 8)
        
        # Should be able to read it with SPDReader
        with SPDReader(path) as reader:
            # Header should have no flags set
            self.assertEqual(reader.header.flags, 0)
            # No sections should be available
            self.assertEqual(len(reader.get_available_sections()), 0)
    
    def test_write_image_section(self):
        """Test writing an image section."""
        path = os.path.join(self.test_dir.name, "image.spd")
        
        # Create test image data
        width, height, channels = 10, 10, 3
        image_data = bytes([i % 256 for i in range(width * height * channels)])
        
        # Create image section
        image = ImageSection(
            width=width,
            height=height,
            channels=channels,
            format="RGB",
            data=image_data
        )
        
        # Write SPD file with image section
        with SPDWriter(path) as writer:
            writer.write_image_section(image)
        
        # Read back the file
        with SPDReader(path) as reader:
            # Header should have HAS_IMAGE flag set
            self.assertTrue(flags_to_dict(reader.header.flags)["HAS_IMAGE"])
            # Image section should be available
            self.assertIn("image", reader.get_available_sections())
            # Image data should match what we wrote
            read_image = reader.image
            self.assertEqual(read_image.width, width)
            self.assertEqual(read_image.height, height)
            self.assertEqual(read_image.channels, channels)
            self.assertEqual(read_image.format, "RGB")
            self.assertEqual(read_image.data, image_data)
    
    def test_write_landmarks_section(self):
        """Test writing a landmarks section."""
        path = os.path.join(self.test_dir.name, "landmarks.spd")
        
        # Create test landmarks data
        count, dimensions = 5, 2
        landmark_type = "test"
        points = [(i/10.0, i/5.0) for i in range(count)]
        
        # Create landmarks section
        landmarks = LandmarksSection(
            count=count,
            dimensions=dimensions,
            landmark_type=landmark_type,
            points=points
        )
        
        # Write SPD file with landmarks section
        with SPDWriter(path) as writer:
            writer.write_landmarks_section(landmarks)
        
        # Read back the file
        with SPDReader(path) as reader:
            # Header should have HAS_LANDMARKS flag set
            self.assertTrue(flags_to_dict(reader.header.flags)["HAS_LANDMARKS"])
            # Landmarks section should be available
            self.assertIn("landmarks", reader.get_available_sections())
            # Landmarks data should match what we wrote
            read_landmarks = reader.landmarks
            self.assertEqual(read_landmarks.count, count)
            self.assertEqual(read_landmarks.dimensions, dimensions)
            self.assertEqual(read_landmarks.landmark_type, landmark_type)
            for i, point in enumerate(read_landmarks.points):
                self.assertAlmostEqual(point[0], points[i][0])
                self.assertAlmostEqual(point[1], points[i][1])
    
    def test_write_motion_params_section(self):
        """Test writing a motion parameters section."""
        path = os.path.join(self.test_dir.name, "motion_params.spd")
        
        # Create test motion parameters data
        num_params = 3
        param_names = ["yaw", "pitch", "roll"]
        values = [0.1, 0.2, 0.3]
        
        # Create motion parameters section
        motion_params = MotionParamsSection(
            num_params=num_params,
            param_names=param_names,
            values=values
        )
        
        # Write SPD file with motion parameters section
        with SPDWriter(path) as writer:
            writer.write_motion_params_section(motion_params)
        
        # Read back the file
        with SPDReader(path) as reader:
            # Header should have HAS_MOTION_PARAMS flag set
            self.assertTrue(flags_to_dict(reader.header.flags)["HAS_MOTION_PARAMS"])
            # Motion parameters section should be available
            self.assertIn("motion_params", reader.get_available_sections())
            # Motion parameters data should match what we wrote
            read_motion_params = reader.motion_params
            self.assertEqual(read_motion_params.num_params, num_params)
            self.assertEqual(read_motion_params.param_names, param_names)
            for i, value in enumerate(read_motion_params.values):
                self.assertAlmostEqual(value, values[i])
    
    def test_write_appearance_section(self):
        """Test writing an appearance features section."""
        path = os.path.join(self.test_dir.name, "appearance.spd")
        
        # Create test appearance data
        feature_dim = 4
        feature_type = "embedding"
        features = [0.1, 0.2, 0.3, 0.4]
        
        # Create appearance section
        appearance = AppearanceSection(
            feature_dim=feature_dim,
            feature_type=feature_type,
            features=features
        )
        
        # Write SPD file with appearance section
        with SPDWriter(path) as writer:
            writer.write_appearance_section(appearance)
        
        # Read back the file
        with SPDReader(path) as reader:
            # Header should have HAS_APPEARANCE flag set
            self.assertTrue(flags_to_dict(reader.header.flags)["HAS_APPEARANCE"])
            # Appearance section should be available
            self.assertIn("appearance", reader.get_available_sections())
            # Appearance data should match what we wrote
            read_appearance = reader.appearance
            self.assertEqual(read_appearance.feature_dim, feature_dim)
            self.assertEqual(read_appearance.feature_type, feature_type)
            for i, feature in enumerate(read_appearance.features):
                self.assertAlmostEqual(feature, features[i])
    
    def test_write_transformation_section(self):
        """Test writing a transformation matrices section."""
        path = os.path.join(self.test_dir.name, "transforms.spd")
        
        # Create test transformation data
        num_transforms = 2
        transform_types = ["face2world", "world2face"]
        # Identity matrices
        matrices = [
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1]
        ]
        
        # Create transformation section
        transforms = TransformationSection(
            num_transforms=num_transforms,
            transform_types=transform_types,
            matrices=matrices
        )
        
        # Write SPD file with transformation section
        with SPDWriter(path) as writer:
            writer.write_transformation_section(transforms)
        
        # Read back the file
        with SPDReader(path) as reader:
            # Header should have HAS_TRANSFORMS flag set
            self.assertTrue(flags_to_dict(reader.header.flags)["HAS_TRANSFORMS"])
            # Transformation section should be available
            self.assertIn("transforms", reader.get_available_sections())
            # Transformation data should match what we wrote
            read_transforms = reader.transforms
            self.assertEqual(read_transforms.num_transforms, num_transforms)
            self.assertEqual(read_transforms.transform_types, transform_types)
            for i, matrix in enumerate(read_transforms.matrices):
                for j, value in enumerate(matrix):
                    self.assertAlmostEqual(value, matrices[i][j])
    
    def test_write_mask_section(self):
        """Test writing a mask section."""
        path = os.path.join(self.test_dir.name, "mask.spd")
        
        # Create test mask data
        width, height = 10, 10
        mask_format = "binary"
        mask_data = bytes([i % 2 for i in range(width * height)])
        
        # Create mask section
        mask = MaskSection(
            width=width,
            height=height,
            format=mask_format,
            data=mask_data
        )
        
        # Write SPD file with mask section
        with SPDWriter(path) as writer:
            writer.write_mask_section(mask)
        
        # Read back the file
        with SPDReader(path) as reader:
            # Header should have HAS_MASK flag set
            self.assertTrue(flags_to_dict(reader.header.flags)["HAS_MASK"])
            # Mask section should be available
            self.assertIn("mask", reader.get_available_sections())
            # Mask data should match what we wrote
            read_mask = reader.mask
            self.assertEqual(read_mask.width, width)
            self.assertEqual(read_mask.height, height)
            self.assertEqual(read_mask.format, mask_format)
            self.assertEqual(read_mask.data, mask_data)
    
    def test_write_additional_flags_section(self):
        """Test writing an additional flags section."""
        path = os.path.join(self.test_dir.name, "additional_flags.spd")
        
        # Create test flags data
        num_flags = 3
        flags = {
            "source": "test.jpg",
            "processed": True,
            "quality": 0.95,
        }
        
        # Create additional flags section
        flags_section = AdditionalFlagsSection(
            num_flags=num_flags,
            flags=flags
        )
        
        # Write SPD file with additional flags section
        with SPDWriter(path) as writer:
            writer.write_additional_flags_section(flags_section)
        
        # Read back the file
        with SPDReader(path) as reader:
            # Additional flags section should be available
            self.assertIn("additional_flags", reader.get_available_sections())
            # Additional flags data should match what we wrote
            read_flags = reader.additional_flags
            self.assertEqual(read_flags.num_flags, num_flags)
            self.assertEqual(read_flags.flags["source"], "test.jpg")
            self.assertEqual(read_flags.flags["processed"], True)
            self.assertAlmostEqual(read_flags.flags["quality"], 0.95)
    
    def test_automatic_header_writing(self):
        """Test that header is automatically written if not explicitly written."""
        path = os.path.join(self.test_dir.name, "auto_header.spd")
        
        # Create test image data
        width, height, channels = 10, 10, 3
        image_data = bytes([i % 256 for i in range(width * height * channels)])
        
        # Create image section
        image = ImageSection(
            width=width,
            height=height,
            channels=channels,
            format="RGB",
            data=image_data
        )
        
        # Write SPD file with image section, without explicitly writing header
        with SPDWriter(path) as writer:
            writer.write_image_section(image)
        
        # Read back the file
        with SPDReader(path) as reader:
            # Header should be properly written with HAS_IMAGE flag set
            self.assertTrue(flags_to_dict(reader.header.flags)["HAS_IMAGE"])
            # Image section should be available
            self.assertIn("image", reader.get_available_sections())
    
    def test_multiple_sections(self):
        """Test writing multiple sections to the same file."""
        path = os.path.join(self.test_dir.name, "multiple.spd")
        
        # Create test data for multiple sections
        # Image section
        width, height, channels = 10, 10, 3
        image_data = bytes([i % 256 for i in range(width * height * channels)])
        image = ImageSection(width=width, height=height, channels=channels, format="RGB", data=image_data)
        
        # Landmarks section
        count, dimensions = 5, 2
        landmark_type = "test"
        points = [(i/10.0, i/5.0) for i in range(count)]
        landmarks = LandmarksSection(count=count, dimensions=dimensions, landmark_type=landmark_type, points=points)
        
        # Motion parameters section
        num_params = 3
        param_names = ["yaw", "pitch", "roll"]
        values = [0.1, 0.2, 0.3]
        motion_params = MotionParamsSection(num_params=num_params, param_names=param_names, values=values)
        
        # Write SPD file with multiple sections
        with SPDWriter(path) as writer:
            writer.write_image_section(image)
            writer.write_landmarks_section(landmarks)
            writer.write_motion_params_section(motion_params)
        
        # Read back the file
        with SPDReader(path) as reader:
            # Header should have all relevant flags set
            flags = flags_to_dict(reader.header.flags)
            self.assertTrue(flags["HAS_IMAGE"])
            self.assertTrue(flags["HAS_LANDMARKS"])
            self.assertTrue(flags["HAS_MOTION_PARAMS"])
            
            # All sections should be available
            sections = reader.get_available_sections()
            self.assertIn("image", sections)
            self.assertIn("landmarks", sections)
            self.assertIn("motion_params", sections)
            
            # Data should match what we wrote
            self.assertEqual(reader.image.width, width)
            self.assertEqual(reader.landmarks.count, count)
            self.assertEqual(reader.motion_params.param_names, param_names)


class TestSPDWriterValidation(unittest.TestCase):
    """Test validation functionality of the SPD writer."""
    
    def setUp(self):
        """Set up a temporary directory for test files."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.output = io.BytesIO()  # Use in-memory buffer for validation tests
    
    def tearDown(self):
        """Clean up temporary directory."""
        self.test_dir.cleanup()
    
    def test_invalid_image_data(self):
        """Test handling of invalid image data."""
        writer = SPDWriter(self.output)
        
        # Test invalid dimensions
        with self.assertRaises(SPDValidationError):
            invalid_image = ImageSection(width=0, height=10, channels=3, format="RGB", data=b"data")
            writer.write_image_section(invalid_image)
        
        with self.assertRaises(SPDValidationError):
            invalid_image = ImageSection(width=10, height=-1, channels=3, format="RGB", data=b"data")
            writer.write_image_section(invalid_image)
        
        with self.assertRaises(SPDValidationError):
            invalid_image = ImageSection(width=10, height=10, channels=0, format="RGB", data=b"data")
            writer.write_image_section(invalid_image)
        
        # Test empty data
        with self.assertRaises(SPDValidationError):
            invalid_image = ImageSection(width=10, height=10, channels=3, format="RGB", data=b"")
            writer.write_image_section(invalid_image)
        
        # Test empty format
        with self.assertRaises(SPDValidationError):
            invalid_image = ImageSection(width=10, height=10, channels=3, format="", data=b"data")
            writer.write_image_section(invalid_image)
        
        # Test data size mismatch
        with self.assertRaises(SPDValidationError):
            invalid_image = ImageSection(width=10, height=10, channels=3, format="RGB", data=b"not_enough_data")
            writer.write_image_section(invalid_image)
    
    def test_invalid_landmarks_data(self):
        """Test handling of invalid landmarks data."""
        writer = SPDWriter(self.output)
        
        # Test invalid count
        with self.assertRaises(SPDValidationError):
            invalid_landmarks = LandmarksSection(count=0, dimensions=2, landmark_type="test", points=[])
            writer.write_landmarks_section(invalid_landmarks)
        
        # Test invalid dimensions
        with self.assertRaises(SPDValidationError):
            invalid_landmarks = LandmarksSection(count=5, dimensions=0, landmark_type="test", points=[])
            writer.write_landmarks_section(invalid_landmarks)
        
        # Test empty landmark type
        with self.assertRaises(SPDValidationError):
            invalid_landmarks = LandmarksSection(count=5, dimensions=2, landmark_type="", points=[])
            writer.write_landmarks_section(invalid_landmarks)
        
        # Test empty points
        with self.assertRaises(SPDValidationError):
            invalid_landmarks = LandmarksSection(count=5, dimensions=2, landmark_type="test", points=[])
            writer.write_landmarks_section(invalid_landmarks)
        
        # Test count mismatch
        with self.assertRaises(SPDValidationError):
            invalid_landmarks = LandmarksSection(
                count=5, dimensions=2, landmark_type="test", 
                points=[(0.1, 0.2), (0.3, 0.4)]  # Only 2 points, but count is 5
            )
            writer.write_landmarks_section(invalid_landmarks)
        
        # Test dimensions mismatch
        with self.assertRaises(SPDValidationError):
            invalid_landmarks = LandmarksSection(
                count=2, dimensions=3, landmark_type="test", 
                points=[(0.1, 0.2), (0.3, 0.4)]  # 2D points, but dimensions is 3
            )
            writer.write_landmarks_section(invalid_landmarks)
    
    def test_invalid_motion_params_data(self):
        """Test handling of invalid motion parameters data."""
        writer = SPDWriter(self.output)
        
        # Test invalid num_params
        with self.assertRaises(SPDValidationError):
            invalid_params = MotionParamsSection(num_params=0, param_names=[], values=[])
            writer.write_motion_params_section(invalid_params)
        
        # Test empty param_names
        with self.assertRaises(SPDValidationError):
            invalid_params = MotionParamsSection(num_params=3, param_names=[], values=[0.1, 0.2, 0.3])
            writer.write_motion_params_section(invalid_params)
        
        # Test empty values
        with self.assertRaises(SPDValidationError):
            invalid_params = MotionParamsSection(num_params=3, param_names=["a", "b", "c"], values=[])
            writer.write_motion_params_section(invalid_params)
        
        # Test param_names count mismatch
        with self.assertRaises(SPDValidationError):
            invalid_params = MotionParamsSection(
                num_params=3, 
                param_names=["a", "b"],  # Only 2 names, but num_params is 3
                values=[0.1, 0.2, 0.3]
            )
            writer.write_motion_params_section(invalid_params)
        
        # Test values count mismatch
        with self.assertRaises(SPDValidationError):
            invalid_params = MotionParamsSection(
                num_params=3, 
                param_names=["a", "b", "c"],
                values=[0.1, 0.2]  # Only 2 values, but num_params is 3
            )
            writer.write_motion_params_section(invalid_params)
    
    def test_invalid_appearance_data(self):
        """Test handling of invalid appearance data."""
        writer = SPDWriter(self.output)
        
        # Test invalid feature_dim
        with self.assertRaises(SPDValidationError):
            invalid_appearance = AppearanceSection(feature_dim=0, feature_type="test", features=[])
            writer.write_appearance_section(invalid_appearance)
        
        # Test empty feature_type
        with self.assertRaises(SPDValidationError):
            invalid_appearance = AppearanceSection(feature_dim=4, feature_type="", features=[0.1, 0.2, 0.3, 0.4])
            writer.write_appearance_section(invalid_appearance)
        
        # Test empty features
        with self.assertRaises(SPDValidationError):
            invalid_appearance = AppearanceSection(feature_dim=4, feature_type="test", features=[])
            writer.write_appearance_section(invalid_appearance)
        
        # Test features count mismatch
        with self.assertRaises(SPDValidationError):
            invalid_appearance = AppearanceSection(
                feature_dim=4, 
                feature_type="test", 
                features=[0.1, 0.2]  # Only 2 features, but feature_dim is 4
            )
            writer.write_appearance_section(invalid_appearance)
    
    def test_invalid_transformation_data(self):
        """Test handling of invalid transformation data."""
        writer = SPDWriter(self.output)
        
        # Test invalid num_transforms
        with self.assertRaises(SPDValidationError):
            invalid_transforms = TransformationSection(num_transforms=0, transform_types=[], matrices=[])
            writer.write_transformation_section(invalid_transforms)
        
        # Test empty transform_types
        with self.assertRaises(SPDValidationError):
            invalid_transforms = TransformationSection(
                num_transforms=2, 
                transform_types=[], 
                matrices=[[1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1]]
            )
            writer.write_transformation_section(invalid_transforms)
        
        # Test empty matrices
        with self.assertRaises(SPDValidationError):
            invalid_transforms = TransformationSection(
                num_transforms=2, 
                transform_types=["a", "b"], 
                matrices=[]
            )
            writer.write_transformation_section(invalid_transforms)
        
        # Test transform_types count mismatch
        with self.assertRaises(SPDValidationError):
            invalid_transforms = TransformationSection(
                num_transforms=2, 
                transform_types=["a"],  # Only 1 type, but num_transforms is 2
                matrices=[[1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1]]
            )
            writer.write_transformation_section(invalid_transforms)
        
        # Test matrices count mismatch
        with self.assertRaises(SPDValidationError):
            invalid_transforms = TransformationSection(
                num_transforms=2, 
                transform_types=["a", "b"], 
                matrices=[[1, 0, 0, 0, 1, 0, 0, 0, 1]]  # Only 1 matrix, but num_transforms is 2
            )
            writer.write_transformation_section(invalid_transforms)
        
        # Test invalid matrix size
        with self.assertRaises(SPDValidationError):
            invalid_transforms = TransformationSection(
                num_transforms=1, 
                transform_types=["a"], 
                matrices=[[1, 0, 0, 0]]  # Only 4 elements, but should be 9
            )
            writer.write_transformation_section(invalid_transforms)
    
    def test_invalid_mask_data(self):
        """Test handling of invalid mask data."""
        writer = SPDWriter(self.output)
        
        # Test invalid dimensions
        with self.assertRaises(SPDValidationError):
            invalid_mask = MaskSection(width=0, height=10, format="binary", data=b"data")
            writer.write_mask_section(invalid_mask)
        
        with self.assertRaises(SPDValidationError):
            invalid_mask = MaskSection(width=10, height=-1, format="binary", data=b"data")
            writer.write_mask_section(invalid_mask)
        
        # Test empty format
        with self.assertRaises(SPDValidationError):
            invalid_mask = MaskSection(width=10, height=10, format="", data=b"data")
            writer.write_mask_section(invalid_mask)
        
        # Test empty data
        with self.assertRaises(SPDValidationError):
            invalid_mask = MaskSection(width=10, height=10, format="binary", data=b"")
            writer.write_mask_section(invalid_mask)
        
        # Test binary mask data size mismatch
        with self.assertRaises(SPDValidationError):
            invalid_mask = MaskSection(
                width=10, height=10, format="binary", 
                data=b"not_enough_data"  # Not enough data for 10x10 binary mask
            )
            writer.write_mask_section(invalid_mask)
    
    def test_invalid_additional_flags_data(self):
        """Test handling of invalid additional flags data."""
        writer = SPDWriter(self.output)
        
        # Test invalid num_flags
        with self.assertRaises(SPDValidationError):
            invalid_flags = AdditionalFlagsSection(num_flags=0, flags={})
            writer.write_additional_flags_section(invalid_flags)
        
        # Test empty flags
        with self.assertRaises(SPDValidationError):
            invalid_flags = AdditionalFlagsSection(num_flags=3, flags={})
            writer.write_additional_flags_section(invalid_flags)
        
        # Test flags count mismatch
        with self.assertRaises(SPDValidationError):
            invalid_flags = AdditionalFlagsSection(
                num_flags=3, 
                flags={"a": 1, "b": 2}  # Only 2 flags, but num_flags is 3
            )
            writer.write_additional_flags_section(invalid_flags)
        
        # Test invalid flag key
        with self.assertRaises(SPDValidationError):
            invalid_flags = AdditionalFlagsSection(
                num_flags=1, 
                flags={"": "empty_key"}  # Empty key is not allowed
            )
            writer.write_additional_flags_section(invalid_flags)
        
        # Test invalid flag value type
        with self.assertRaises(SPDValidationError):
            invalid_flags = AdditionalFlagsSection(
                num_flags=1, 
                flags={"key": [1, 2, 3]}  # List is not an allowed type
            )
            writer.write_additional_flags_section(invalid_flags)


class TestSPDWriterRoundTrip(unittest.TestCase):
    """Test round-trip functionality of the SPD writer."""
    
    def setUp(self):
        """Set up a temporary directory for test files."""
        self.test_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up temporary directory."""
        self.test_dir.cleanup()
    
    def test_round_trip_all_sections(self):
        """Test writing and then reading back all sections."""
        path = os.path.join(self.test_dir.name, "round_trip.spd")
        
        # Create test data for all sections
        # Image section
        width, height, channels = 10, 10, 3
        image_data = bytes([i % 256 for i in range(width * height * channels)])
        image = ImageSection(width=width, height=height, channels=channels, format="RGB", data=image_data)
        
        # Landmarks section
        count, dimensions = 5, 2
        landmark_type = "test"
        points = [(i/10.0, i/5.0) for i in range(count)]
        landmarks = LandmarksSection(count=count, dimensions=dimensions, landmark_type=landmark_type, points=points)
        
        # Motion parameters section
        num_params = 3
        param_names = ["yaw", "pitch", "roll"]
        values = [0.1, 0.2, 0.3]
        motion_params = MotionParamsSection(num_params=num_params, param_names=param_names, values=values)
        
        # Appearance section
        feature_dim = 4
        feature_type = "embedding"
        features = [0.1, 0.2, 0.3, 0.4]
        appearance = AppearanceSection(feature_dim=feature_dim, feature_type=feature_type, features=features)
        
        # Transformation section
        num_transforms = 2
        transform_types = ["face2world", "world2face"]
        matrices = [
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5]
        ]
        transforms = TransformationSection(num_transforms=num_transforms, transform_types=transform_types, matrices=matrices)
        
        # Mask section
        mask_width, mask_height = 10, 10
        mask_format = "binary"
        mask_data = bytes([i % 2 for i in range(mask_width * mask_height)])
        mask = MaskSection(width=mask_width, height=mask_height, format=mask_format, data=mask_data)
        
        # Additional flags section
        num_flags = 4
        flags = {
            "source": "test.jpg",
            "processed": True,
            "quality": 0.95,
            "count": 42
        }
        flags_section = AdditionalFlagsSection(num_flags=num_flags, flags=flags)
        
        # Write all sections to the SPD file
        with SPDWriter(path) as writer:
            writer.write_image_section(image)
            writer.write_landmarks_section(landmarks)
            writer.write_motion_params_section(motion_params)
            writer.write_appearance_section(appearance)
            writer.write_transformation_section(transforms)
            writer.write_mask_section(mask)
            writer.write_additional_flags_section(flags_section)
        
        # Read back the file and verify all sections
        with SPDReader(path) as reader:
            # Check header flags
            flags_dict = flags_to_dict(reader.header.flags)
            self.assertTrue(flags_dict["HAS_IMAGE"])
            self.assertTrue(flags_dict["HAS_LANDMARKS"])
            self.assertTrue(flags_dict["HAS_MOTION_PARAMS"])
            self.assertTrue(flags_dict["HAS_APPEARANCE"])
            self.assertTrue(flags_dict["HAS_TRANSFORMS"])
            self.assertTrue(flags_dict["HAS_MASK"])
            
            # Check available sections
            sections = reader.get_available_sections()
            self.assertEqual(set(sections), set(["image", "landmarks", "motion_params", "appearance", "transforms", "mask", "additional_flags"]))
            
            # Check image section
            read_image = reader.image
            self.assertEqual(read_image.width, width)
            self.assertEqual(read_image.height, height)
            self.assertEqual(read_image.channels, channels)
            self.assertEqual(read_image.format, "RGB")
            self.assertEqual(read_image.data, image_data)
            
            # Check landmarks section
            read_landmarks = reader.landmarks
            self.assertEqual(read_landmarks.count, count)
            self.assertEqual(read_landmarks.dimensions, dimensions)
            self.assertEqual(read_landmarks.landmark_type, landmark_type)
            for i, point in enumerate(read_landmarks.points):
                self.assertAlmostEqual(point[0], points[i][0])
                self.assertAlmostEqual(point[1], points[i][1])
            
            # Check motion parameters section
            read_motion_params = reader.motion_params
            self.assertEqual(read_motion_params.num_params, num_params)
            self.assertEqual(read_motion_params.param_names, param_names)
            for i, value in enumerate(read_motion_params.values):
                self.assertAlmostEqual(value, values[i])
            
            # Check appearance section
            read_appearance = reader.appearance
            self.assertEqual(read_appearance.feature_dim, feature_dim)
            self.assertEqual(read_appearance.feature_type, feature_type)
            for i, feature in enumerate(read_appearance.features):
                self.assertAlmostEqual(feature, features[i])
            
            # Check transformation section
            read_transforms = reader.transforms
            self.assertEqual(read_transforms.num_transforms, num_transforms)
            self.assertEqual(read_transforms.transform_types, transform_types)
            for i, matrix in enumerate(read_transforms.matrices):
                for j, value in enumerate(matrix):
                    self.assertAlmostEqual(value, matrices[i][j])
            
            # Check mask section
            read_mask = reader.mask
            self.assertEqual(read_mask.width, mask_width)
            self.assertEqual(read_mask.height, mask_height)
            self.assertEqual(read_mask.format, mask_format)
            self.assertEqual(read_mask.data, mask_data)
            
            # Check additional flags section
            read_flags = reader.additional_flags
            self.assertEqual(read_flags.num_flags, num_flags)
            self.assertEqual(read_flags.flags["source"], "test.jpg")
            self.assertEqual(read_flags.flags["processed"], True)
            self.assertAlmostEqual(read_flags.flags["quality"], 0.95)
            self.assertEqual(read_flags.flags["count"], 42)


if __name__ == "__main__":
    unittest.main()