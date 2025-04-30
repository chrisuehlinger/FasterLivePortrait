#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the transformation utility module.
"""

import unittest
import numpy as np
import math
from typing import List, Tuple, Dict

# Import the module to test
from spd_editor.utils.transformation import (
    apply_transformation, compose_transforms, invert_transform,
    convert_coordinates, get_face_aligned_transform, transform_to_scale_and_center
)


class TestTransformation(unittest.TestCase):
    """Test suite for the transformation utilities."""
    
    def test_apply_transformation(self):
        """Test applying a transformation matrix to points."""
        # Create test points
        points_2d = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        
        # Create a simple translation matrix
        translation = np.array([
            [1, 0, 10],  # translate x by 10
            [0, 1, 20],  # translate y by 20
            [0, 0, 1]
        ])
        
        # Apply transformation
        transformed = apply_transformation(points_2d, translation)
        
        # Check results
        self.assertEqual(len(transformed), len(points_2d))
        self.assertAlmostEqual(transformed[0][0], 11.0)  # 1 + 10
        self.assertAlmostEqual(transformed[0][1], 22.0)  # 2 + 20
        self.assertAlmostEqual(transformed[1][0], 13.0)  # 3 + 10
        self.assertAlmostEqual(transformed[1][1], 24.0)  # 4 + 20
        
        # Test with 3D points
        points_3d = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        transformed_3d = apply_transformation(points_3d, translation)
        
        # Check that Z coordinates are preserved
        self.assertEqual(len(transformed_3d), len(points_3d))
        self.assertAlmostEqual(transformed_3d[0][2], 3.0)
        self.assertAlmostEqual(transformed_3d[1][2], 6.0)
    
    def test_compose_transforms(self):
        """Test composing multiple transformations."""
        # Create transformations
        translation = np.array([
            [1, 0, 10],
            [0, 1, 20],
            [0, 0, 1]
        ])
        
        # 90-degree rotation
        rotation = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        # Scale by 2
        scale = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 1]
        ])
        
        # Compose transforms (transformations are applied from left to right)
        composed = compose_transforms([translation, rotation, scale])
        
        # Test result with a point
        point = (1.0, 2.0)
        
        # Apply composed transform
        result = apply_transformation([point], composed)[0]
        
        # Apply transforms individually in the correct order
        # The point is first transformed by translation, then rotation, then scale
        p = point
        p = apply_transformation([p], translation)[0]  # First transformation
        p = apply_transformation([p], rotation)[0]     # Second transformation
        p = apply_transformation([p], scale)[0]        # Third transformation
        
        # Check results
        self.assertAlmostEqual(result[0], p[0], places=5)
        self.assertAlmostEqual(result[1], p[1], places=5)
        
        # Check empty transforms list
        identity = compose_transforms([])
        self.assertTrue(np.array_equal(identity, np.eye(3)))
    
    def test_invert_transform(self):
        """Test inverting a transformation matrix."""
        # Create a transformation matrix
        transform = np.array([
            [2, 0, 10],
            [0, 3, 20],
            [0, 0, 1]
        ])
        
        # Invert it
        inverse = invert_transform(transform)
        
        # Verify inverse by multiplication
        identity = np.dot(transform, inverse)
        
        # Should be close to identity matrix
        np.testing.assert_array_almost_equal(identity, np.eye(3))
        
        # Test with a singular matrix (should raise ValueError)
        singular = np.array([
            [0, 0, 0],
            [0, 3, 20],
            [0, 0, 1]
        ])
        
        with self.assertRaises(ValueError):
            invert_transform(singular)
    
    def test_convert_coordinates(self):
        """Test converting coordinates between spaces."""
        # Create sample points
        points = [(1.0, 2.0), (3.0, 4.0)]
        
        # Create transformation dictionaries
        transforms = {
            'face2world': np.array([
                [2, 0, 10],  # scale x by 2, translate by 10
                [0, 2, 20],  # scale y by 2, translate by 20
                [0, 0, 1]
            ]),
            'world2screen': np.array([
                [1, 0, 5],   # translate x by 5
                [0, 1, 10],  # translate y by 10
                [0, 0, 1]
            ])
        }
        
        # Test direct conversion
        face_to_world = convert_coordinates(points, 'face', 'world', transforms)
        self.assertAlmostEqual(face_to_world[0][0], 12.0)  # 1*2 + 10
        self.assertAlmostEqual(face_to_world[0][1], 24.0)  # 2*2 + 20
        
        # Test inverse conversion
        world_to_face = convert_coordinates(face_to_world, 'world', 'face', transforms)
        self.assertAlmostEqual(world_to_face[0][0], points[0][0])
        self.assertAlmostEqual(world_to_face[0][1], points[0][1])
        
        # Test multi-step conversion
        face_to_screen = convert_coordinates(points, 'face', 'screen', transforms)
        
        # Compute expected (face -> world -> screen)
        expected = apply_transformation(face_to_world, transforms['world2screen'])
        self.assertAlmostEqual(face_to_screen[0][0], expected[0][0])
        self.assertAlmostEqual(face_to_screen[0][1], expected[0][1])
        
        # Test invalid conversion path
        with self.assertRaises(ValueError):
            convert_coordinates(points, 'face', 'unknown', transforms)
    
    def test_get_face_aligned_transform(self):
        """Test computing face alignment transform."""
        # Create landmarks for a face
        # Simple case: left eye, right eye, nose (tilted)
        landmarks = [
            (100, 100),  # landmark 0
            (100, 200),  # landmark 1
            (200, 220),  # landmark 2 - right eye (36 in dlib)
            (300, 180),  # landmark 3 - left eye (45 in dlib)
            (250, 250),  # landmark 4 - nose (30 in dlib)
            (100, 300),  # landmark 5
        ]
        
        # Use indices 2, 3, 4 (matching 36, 45, 30)
        ref_points = [2, 3, 4]
        
        # Get alignment transform
        transform = get_face_aligned_transform(landmarks, ref_points)
        
        # Apply transformation to landmarks
        aligned = apply_transformation(landmarks, transform)
        
        # Check that eyes are roughly horizontal now
        right_eye = aligned[2]
        left_eye = aligned[3]
        
        # The y-coordinates should be close
        delta_y = abs(right_eye[1] - left_eye[1])
        self.assertLess(delta_y, 5.0, "Eyes should be aligned horizontally")
        
        # Test with default reference points (would raise error with our small landmark set)
        with self.assertRaises(ValueError):
            get_face_aligned_transform(landmarks)

    def test_transform_to_scale_and_center(self):
        """Test scaling and centering transform."""
        # Test with equal aspect ratio
        transform = transform_to_scale_and_center(100, 100, (200, 200), padding=0)
        
        # This should double the size
        self.assertAlmostEqual(transform[0, 0], 2.0)  # x scale
        self.assertAlmostEqual(transform[1, 1], 2.0)  # y scale
        
        # No offset needed for equal aspect ratio without padding
        self.assertAlmostEqual(transform[0, 2], 0.0)  # x translation
        self.assertAlmostEqual(transform[1, 2], 0.0)  # y translation
        
        # Test with different aspect ratios
        transform = transform_to_scale_and_center(100, 200, (300, 400), padding=0.1)
        
        # In this case, scale_x = 300/100 = 3, scale_y = 400/200 = 2
        # The minimum scale is 2, and with 10% padding it's 2*(1-0.1) = 1.8
        expected_scale = 1.8  # min(3, 2) * 0.9
        
        # Check scale factors
        self.assertAlmostEqual(transform[0, 0], expected_scale)
        self.assertAlmostEqual(transform[1, 1], expected_scale)
        
        # Calculate expected translations
        expected_width = 100 * expected_scale  # 100 * 1.8 = 180
        expected_height = 200 * expected_scale  # 200 * 1.8 = 360
        expected_x_offset = (300 - expected_width) / 2  # (300 - 180) / 2 = 60
        expected_y_offset = (400 - expected_height) / 2  # (400 - 360) / 2 = 20
        
        # Check translations
        self.assertAlmostEqual(transform[0, 2], expected_x_offset)
        self.assertAlmostEqual(transform[1, 2], expected_y_offset)


if __name__ == '__main__':
    unittest.main()