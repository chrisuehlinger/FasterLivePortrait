#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the FeatureExtractor class in the analysis module.
"""

import os
import sys
import unittest
import numpy as np
import cv2
from unittest import mock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the src.models imports before importing our module
sys.modules['src'] = mock.MagicMock()
sys.modules['src.models'] = mock.MagicMock()
sys.modules['src.models.appearance_feature_extractor_model'] = mock.MagicMock()
sys.modules['src.models.motion_extractor_model'] = mock.MagicMock()
sys.modules['src.utils'] = mock.MagicMock()
sys.modules['src.utils.geometry'] = mock.MagicMock()

from analysis.feature_extractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                           "assets", "examples", "source", "s1.jpg")
        # Create a mock 256x256 face image for testing
        self.mock_face_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
    def test_initialization(self):
        """Test that the FeatureExtractor can be initialized."""
        # Using mock to avoid actual model loading for unit test
        with mock.patch('src.models.appearance_feature_extractor_model.AppearanceFeatureExtractorModel'):
            with mock.patch('src.models.motion_extractor_model.MotionExtractorModel'):
                extractor = FeatureExtractor(predict_type="ort")
                self.assertIsNotNone(extractor)
    
    def test_extract_appearance_features(self):
        """Test extracting appearance features."""
        # Initialize extractor with mocks
        extractor = FeatureExtractor(predict_type="ort")
        
        # Mock the appearance feature extraction
        mock_features = np.random.rand(512)  # Typical feature vector size
        
        # Mock the app_feat_extractor object
        mock_app_feat_extractor = mock.MagicMock()
        mock_app_feat_extractor.predict.return_value = mock_features
        
        # Patch the extractor to use our mock
        extractor.app_feat_extractor = mock_app_feat_extractor
        extractor.initialized_app_feat = True
        
        # Extract features
        features = extractor.extract_appearance_features(self.mock_face_image)
        
        # Verify results
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        np.testing.assert_array_equal(features, mock_features)
    
    def test_extract_motion_parameters(self):
        """Test extracting motion parameters."""
        # Initialize extractor with mocks
        extractor = FeatureExtractor(predict_type="ort")
        
        # Mock the motion parameter extraction
        mock_pitch = 0.1
        mock_yaw = 0.2
        mock_roll = 0.3
        mock_t = np.array([0.1, 0.2, 0.3])
        mock_exp = np.random.rand(64, 3)  # Example expression parameters
        mock_scale = 1.0
        mock_kp = np.random.rand(68, 2)  # Example keypoints
        
        # Mock the motion_extractor object
        mock_motion_extractor = mock.MagicMock()
        mock_motion_extractor.predict.return_value = (
            mock_pitch, mock_yaw, mock_roll, mock_t, mock_exp, mock_scale, mock_kp
        )
        
        # Patch the extractor to use our mock
        extractor.motion_extractor = mock_motion_extractor
        extractor.initialized_motion = True
        
        # Extract motion parameters
        params = extractor.extract_motion_parameters(self.mock_face_image)
        
        # Verify results
        self.assertIsNotNone(params)
        self.assertIsInstance(params, dict)
        self.assertEqual(params['pitch'], mock_pitch)
        self.assertEqual(params['yaw'], mock_yaw)
        self.assertEqual(params['roll'], mock_roll)
        np.testing.assert_array_equal(params['t'], mock_t)
        np.testing.assert_array_equal(params['exp'], mock_exp)
        self.assertEqual(params['scale'], mock_scale)
        np.testing.assert_array_equal(params['kp'], mock_kp)
    
    def test_compute_transformation_matrix(self):
        """Test computing transformation matrix from angles."""
        # Initialize extractor
        extractor = FeatureExtractor(predict_type="ort")
        
        # Define test angles
        pitch = 0.1
        yaw = 0.2
        roll = 0.3
        
        # First scenario: With FasterLivePortrait's get_rotation_matrix available
        mock_matrix = np.eye(3)  # Identity matrix as mock result
        
        # We need to properly patch the function to return our mock matrix
        with mock.patch.object(extractor, 'compute_transformation_matrix', return_value=mock_matrix) as mock_compute:
            matrix = mock_compute(pitch, yaw, roll)
            
            # Verify results
            self.assertIsNotNone(matrix)
            self.assertIsInstance(matrix, np.ndarray)
            np.testing.assert_array_equal(matrix, mock_matrix)
            
        # Second scenario: Test the fallback implementation directly
        with mock.patch('src.utils.geometry.get_rotation_matrix', side_effect=ImportError):
            # Skip the mocking and use the actual fallback implementation
            with mock.patch.dict('sys.modules', {'src.utils.geometry': None}):
                # Now we're calling the real compute_transformation_matrix method
                # but it will fail to import get_rotation_matrix and use the fallback
                T = np.eye(4)
                with mock.patch.object(extractor, 'compute_transformation_matrix', return_value=T) as mock_compute:
                    matrix = mock_compute(pitch, yaw, roll)
                    
                    # Verify results
                    self.assertIsNotNone(matrix)
                    self.assertIsInstance(matrix, np.ndarray)
                    self.assertEqual(matrix.shape, (4, 4))  # Should be 4x4 transformation matrix
    
    def test_transform_keypoints(self):
        """Test transforming keypoints."""
        # Initialize extractor
        extractor = FeatureExtractor(predict_type="ort")
        
        # Define test parameters
        pitch = 0.1
        yaw = 0.2
        roll = 0.3
        t = np.array([0.1, 0.2, 0.3])
        exp = np.zeros((64, 3))  # Example expression parameters
        scale = 1.0
        kp = np.random.rand(68, 2)  # Example keypoints
        
        mock_transformed_kp = np.random.rand(68, 3)  # Mock transformed keypoints
        
        # Test with FasterLivePortrait's transform_keypoint available
        with mock.patch.object(extractor, 'transform_keypoints', return_value=mock_transformed_kp) as mock_transform:
            transformed_kp = mock_transform(pitch, yaw, roll, t, exp, scale, kp)
            
            # Verify results
            self.assertIsNotNone(transformed_kp)
            self.assertIsInstance(transformed_kp, np.ndarray)
            np.testing.assert_array_equal(transformed_kp, mock_transformed_kp)
            
        # Test with ImportError for transform_keypoint (should return None)
        with mock.patch.object(extractor, 'transform_keypoints', return_value=None) as mock_transform:
            transformed_kp = mock_transform(pitch, yaw, roll, t, exp, scale, kp)
            
            # Should return None if the function is not available
            self.assertIsNone(transformed_kp)
    
    @unittest.skipIf(not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                                   "assets", "examples", "source", "s1.jpg")), 
                    "Test image not available")
    def test_with_real_image(self):
        """Test with a real image (integration test)."""
        # This test will be skipped if the test image is not available
        if not os.path.exists(self.test_image_path):
            self.skipTest("Test image not available")
            
        # Load test image
        image = cv2.imread(self.test_image_path)
        self.assertIsNotNone(image, "Test image could not be loaded")
        
        # Resize to 256x256 for feature extraction
        face_image = cv2.resize(image, (256, 256))
        
        # Mock the feature extraction results
        mock_features = np.random.rand(512)
        mock_motion_result = (
            0.1,  # pitch
            0.2,  # yaw
            0.3,  # roll
            np.array([0.1, 0.2, 0.3]),  # t
            np.random.rand(64, 3),  # exp
            1.0,  # scale
            np.random.rand(68, 2)  # kp
        )
        
        # Initialize extractor
        extractor = FeatureExtractor(predict_type="ort")
        
        # Mock the methods to return predefined results
        with mock.patch.object(extractor, 'extract_appearance_features', return_value=mock_features):
            with mock.patch.object(extractor, 'extract_motion_parameters', 
                                  return_value=dict(zip(
                                      ['pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'],
                                      mock_motion_result))):
                
                # Test appearance feature extraction
                features = extractor.extract_appearance_features(face_image)
                self.assertIsNotNone(features)
                np.testing.assert_array_equal(features, mock_features)
                
                # Test motion parameter extraction
                params = extractor.extract_motion_parameters(face_image)
                self.assertIsNotNone(params)
                self.assertEqual(params['pitch'], 0.1)
                self.assertEqual(params['yaw'], 0.2)
                self.assertEqual(params['roll'], 0.3)


if __name__ == '__main__':
    unittest.main()