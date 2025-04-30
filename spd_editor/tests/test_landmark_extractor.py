#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the LandmarkExtractor class in the analysis module.
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
sys.modules['src.models.landmark_model'] = mock.MagicMock()
sys.modules['src.models.mediapipe_face_model'] = mock.MagicMock()
sys.modules['src.models.face_analysis_model'] = mock.MagicMock()
sys.modules['src.models.predictor'] = mock.MagicMock()

from analysis.landmark_extractor import LandmarkExtractor


class TestLandmarkExtractor(unittest.TestCase):
    """Test cases for LandmarkExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                           "assets", "examples", "source", "s1.jpg")
        self.mock_face = {
            'bbox': np.array([100, 100, 300, 300]),
            'confidence': 0.98,
            'landmark': np.random.rand(106, 2) * 200 + 100  # Random landmarks in the face area
        }
        
    def test_initialization(self):
        """Test that the LandmarkExtractor can be initialized."""
        # Using mock to avoid actual model loading for unit test
        with mock.patch('src.models.landmark_model.LandmarkModel'):
            extractor = LandmarkExtractor(predict_type="ort")
            self.assertIsNotNone(extractor)
    
    @unittest.skipIf(not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                                   "assets", "examples", "source", "s1.jpg")), 
                    "Test image not available")
    def test_extract_landmarks_with_real_image(self):
        """Test landmark extraction with a real image (integration test)."""
        # This test will be skipped if the test image is not available
        if not os.path.exists(self.test_image_path):
            self.skipTest("Test image not available")
            
        # Load test image
        image = cv2.imread(self.test_image_path)
        self.assertIsNotNone(image, "Test image could not be loaded")
        
        # Initialize extractor
        extractor = LandmarkExtractor(predict_type="ort")
        
        # Create a mock result for the landmark extraction
        mock_landmarks = np.random.rand(106, 2) * 200 + 100  # Random landmarks
        
        # Mock the extract_landmarks method to return predefined landmarks
        with mock.patch.object(extractor, 'extract_landmarks', return_value=mock_landmarks):
            # Test with face detection result
            landmarks = extractor.extract_landmarks(image, self.mock_face)
            
            # Verify results
            self.assertIsNotNone(landmarks)
            self.assertIsInstance(landmarks, np.ndarray)
            self.assertEqual(landmarks.shape[1], 2)  # x,y coordinates
            
            # Test with existing landmarks for tracking
            landmarks = extractor.extract_landmarks(image, self.mock_face['landmark'])
            
            # Verify results
            self.assertIsNotNone(landmarks)
            self.assertIsInstance(landmarks, np.ndarray)
            self.assertEqual(landmarks.shape[1], 2)  # x,y coordinates
    
    def test_extract_landmarks_batch(self):
        """Test batch landmark extraction."""
        # Create a dummy image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock extractor
        extractor = LandmarkExtractor(predict_type="ort")
        
        # Create mock face detections
        mock_faces = [
            {
                'bbox': np.array([200, 200, 300, 300]),
                'confidence': 0.96,
                'landmark': np.zeros((106, 2))
            },
            {
                'bbox': np.array([100, 100, 200, 200]),
                'confidence': 0.92,
                'landmark': np.zeros((106, 2))
            }
        ]
        
        # Mock landmarks to return for each face
        mock_landmarks_1 = np.random.rand(106, 2) * 100 + 200
        mock_landmarks_2 = np.random.rand(106, 2) * 100 + 100
        
        # Setup side effect to return different landmarks for different faces
        with mock.patch.object(extractor, 'extract_landmarks', side_effect=[mock_landmarks_1, mock_landmarks_2]):
            # Extract landmarks for batch of faces
            landmarks_batch = extractor.extract_landmarks_batch(image, mock_faces)
            
            # Verify results
            self.assertEqual(len(landmarks_batch), len(mock_faces))
            np.testing.assert_array_equal(landmarks_batch[0], mock_landmarks_1)
            np.testing.assert_array_equal(landmarks_batch[1], mock_landmarks_2)
    
    def test_get_landmark_types(self):
        """Test getting landmark type information."""
        # Initialize extractor
        extractor = LandmarkExtractor(predict_type="ort")
        extractor.model = "landmark_model"  # Set the model type
        
        # Get landmark types
        landmark_types = extractor.get_landmark_types()
        
        # Verify results
        self.assertIsInstance(landmark_types, dict)
        self.assertIn('jaw', landmark_types)
        self.assertIn('right_eye', landmark_types)
        self.assertIn('left_eye', landmark_types)
        self.assertIn('nose_tip', landmark_types)
        self.assertIn('outer_lip', landmark_types)
        
        # Check if extended landmarks are included
        self.assertIn('face_contour_ext', landmark_types)
        
        # Verify some landmark indices
        jaw_indices = landmark_types['jaw']
        self.assertIsInstance(jaw_indices, list)
        self.assertTrue(all(0 <= idx < 17 for idx in jaw_indices))

    def test_empty_results(self):
        """Test behavior with no face detected."""
        # Create a dummy image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Initialize extractor
        extractor = LandmarkExtractor(predict_type="ort")
        
        # Mock the extract_landmarks method to return None
        with mock.patch.object(extractor, 'extract_landmarks', return_value=None):
            # Test with no face detection
            landmarks = extractor.extract_landmarks(image, None)
            self.assertIsNone(landmarks)
            
            # Test batch extraction with empty list
            landmarks_batch = extractor.extract_landmarks_batch(image, [])
            self.assertEqual(landmarks_batch, [])


if __name__ == '__main__':
    unittest.main()