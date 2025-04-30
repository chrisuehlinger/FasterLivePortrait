#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the visualization utility module.
"""

import unittest
import numpy as np
import cv2
import os
import tempfile
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import the module to test
from spd_editor.utils.visualization import (
    draw_landmarks, render_3d_mesh, draw_transformed_landmarks, 
    create_comparison, save_visualization, save_3d_visualization,
    visualize_spd_data, LandmarkVisualizationOptions
)
from spd_editor.utils.transformation import (
    transform_to_scale_and_center
)


class TestVisualization(unittest.TestCase):
    """Test suite for the visualization utilities."""
    
    def setUp(self):
        """Set up test data."""
        # Create a test image
        self.image = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(self.image, (50, 50), (350, 250), (0, 0, 255), -1)
        
        # Create test 2D landmarks
        self.landmarks_2d = [
            (100, 100),  # left eye
            (300, 100),  # right eye
            (200, 150),  # nose
            (150, 200),  # left mouth
            (250, 200),  # right mouth
        ]
        
        # Create test 3D landmarks
        self.landmarks_3d = [
            (100, 100, 0),    # left eye
            (300, 100, 0),    # right eye
            (200, 150, 20),   # nose
            (150, 200, 10),   # left mouth
            (250, 200, 10),   # right mouth
        ]
        
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.test_dir.name
    
    def tearDown(self):
        """Clean up after tests."""
        self.test_dir.cleanup()
    
    def test_draw_landmarks(self):
        """Test drawing landmarks on an image."""
        # Create visualization options
        options = LandmarkVisualizationOptions(
            point_color=(0, 255, 0),
            point_size=5,
            draw_connections=True
        )
        
        # Draw landmarks
        result = draw_landmarks(self.image, self.landmarks_2d, options)
        
        # Check that the result is an image
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.image.shape)
        
        # Check that the result is different from the original
        self.assertFalse(np.array_equal(result, self.image))
        
        # Test with showing labels
        options.show_labels = True
        result = draw_landmarks(self.image, self.landmarks_2d, options)
        self.assertIsInstance(result, np.ndarray)
    
    def test_render_3d_mesh(self):
        """Test rendering a 3D mesh."""
        # Create connections for the mesh
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        
        # Render mesh
        fig = render_3d_mesh(self.landmarks_3d, connections)
        
        # Check that the result is a matplotlib figure
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with different view angles
        fig = render_3d_mesh(
            self.landmarks_3d, 
            connections, 
            view_angle=(30, 45, 0)
        )
        self.assertIsInstance(fig, plt.Figure)
        
        # Close the figures to avoid memory leaks
        plt.close('all')
        
        # Test error handling for 2D landmarks
        with self.assertRaises(ValueError):
            render_3d_mesh(self.landmarks_2d)
    
    def test_draw_transformed_landmarks(self):
        """Test drawing transformed landmarks."""
        # Create a transformation matrix
        transform = np.array([
            [1, 0, 50],   # translate x by 50
            [0, 1, 25],   # translate y by 25
            [0, 0, 1]
        ])
        
        # Draw transformed landmarks
        result = draw_transformed_landmarks(self.image, self.landmarks_2d, transform)
        
        # Check result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.image.shape)
        
        # Verify transformation by checking one point
        original = self.landmarks_2d[0]
        expected_x = original[0] + 50
        expected_y = original[1] + 25
        
        # Check if a green point (landmark) exists near the expected position
        # Note: This is a simplified check since we can't easily extract the exact landmark positions
        region = result[expected_y-5:expected_y+5, expected_x-5:expected_x+5, :]
        # Check if there's any green in the region
        self.assertTrue(np.any(region[:, :, 1] > 200))
    
    def test_create_comparison(self):
        """Test creating a side-by-side comparison."""
        # Create a second test image
        image2 = self.image.copy()
        cv2.circle(image2, (200, 150), 50, (0, 255, 0), -1)
        
        # Create comparison
        result = create_comparison(self.image, image2, "Test Comparison", ("Before", "After"))
        
        # Check result
        self.assertIsInstance(result, np.ndarray)
        
        # Check that the result is wider than the original images
        self.assertEqual(result.shape[0], self.image.shape[0] + int(self.image.shape[0] * 0.05))
        self.assertEqual(result.shape[1], self.image.shape[1] * 2)
        self.assertEqual(result.shape[2], 3)
        
        # Test with different sized images
        image3 = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.rectangle(image3, (50, 50), (250, 150), (255, 0, 0), -1)
        
        result = create_comparison(self.image, image3)
        self.assertIsInstance(result, np.ndarray)
    
    def test_save_visualization(self):
        """Test saving a visualization to a file."""
        # Create output path
        output_jpg = os.path.join(self.output_dir, "test_output.jpg")
        output_png = os.path.join(self.output_dir, "test_output.png")
        output_no_ext = os.path.join(self.output_dir, "test_output")
        
        # Test saving as JPG
        save_visualization(self.image, output_jpg)
        self.assertTrue(os.path.exists(output_jpg))
        
        # Test saving as PNG
        save_visualization(self.image, output_png)
        self.assertTrue(os.path.exists(output_png))
        
        # Test saving with no extension
        save_visualization(self.image, output_no_ext)
        self.assertTrue(os.path.exists(output_no_ext + ".png"))  # should default to PNG
    
    def test_save_3d_visualization(self):
        """Test saving a 3D visualization figure."""
        # Create a figure
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        fig = render_3d_mesh(self.landmarks_3d, connections)
        
        # Save the figure
        output_path = os.path.join(self.output_dir, "test_3d.png")
        save_3d_visualization(fig, output_path)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(output_path))
    
    def test_visualize_spd_data(self):
        """Test the high-level SPD data visualization function."""
        # Create a transformation dictionary
        transforms = {
            'scale': transform_to_scale_and_center(400, 300, (800, 600)),
            'rotate': np.array([
                [np.cos(np.radians(30)), -np.sin(np.radians(30)), 0],
                [np.sin(np.radians(30)), np.cos(np.radians(30)), 0],
                [0, 0, 1]
            ])
        }
        
        # Test without saving files - set interactive_display to False to prevent hanging
        results = visualize_spd_data(
            self.image, 
            self.landmarks_2d,
            transforms=transforms,
            show_3d=False,
            show_transformed=True,
            output_path=None,
            interactive_display=False
        )
        
        # Check results
        self.assertIn('landmarks', results)
        self.assertIsInstance(results['landmarks'], np.ndarray)
        
        # Test with 3D and saving files
        results = visualize_spd_data(
            self.image,
            self.landmarks_3d,
            transforms=transforms,
            show_3d=True,
            show_transformed=True,
            output_path=self.output_dir,
            interactive_display=False
        )
        
        # Check results
        self.assertIn('landmarks', results)
        self.assertIn('landmarks_path', results)
        self.assertIn('3d_figure', results)
        self.assertIn('3d_path', results)
        self.assertIn('transformed_scale', results)
        self.assertIn('transformed_rotate', results)
        
        # Check that files were created
        self.assertTrue(os.path.exists(results['landmarks_path']))
        self.assertTrue(os.path.exists(results['3d_path']))
        self.assertTrue(os.path.exists(results['transformed_scale_path']))
        self.assertTrue(os.path.exists(results['transformed_rotate_path']))


# Create reference images for visual testing
def create_reference_images():
    """
    Create reference images for visual testing of visualization functions.
    This is not part of the unit tests but can be run separately.
    """
    import sys
    import os
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "reference_images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test image
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (700, 500), (0, 0, 255), -1)
    
    # Create landmarks for a face (simplified)
    landmarks_2d = []
    
    # Add basic facial landmarks
    # Eyes
    landmarks_2d.extend([
        (300, 250), (320, 245), (340, 250), (320, 255),  # left eye
        (500, 250), (480, 245), (460, 250), (480, 255),  # right eye
    ])
    
    # Eyebrows
    landmarks_2d.extend([
        (280, 220), (320, 210), (360, 220),  # left eyebrow
        (520, 220), (480, 210), (440, 220),  # right eyebrow
    ])
    
    # Nose
    landmarks_2d.extend([
        (400, 280), (400, 320), (380, 350), (400, 350), (420, 350)  # nose
    ])
    
    # Mouth
    landmarks_2d.extend([
        (320, 420), (400, 430), (480, 420),  # top lip
        (320, 450), (400, 440), (480, 450),  # bottom lip
    ])
    
    # Jaw line
    landmarks_2d.extend([
        (300, 200), (350, 180), (400, 170), (450, 180), (500, 200),  # upper jaw
        (540, 250), (550, 300), (540, 350), (500, 400),  # right jaw
        (400, 480), # chin
        (300, 400), (260, 350), (250, 300), (260, 250),  # left jaw
    ])
    
    # Convert to 3D landmarks with different Z values
    landmarks_3d = []
    for x, y in landmarks_2d:
        # Set Z based on position (just for visualization)
        # Nose tip and mouth slightly forward, eyes slightly back
        if 380 <= x <= 420 and 320 <= y <= 350:  # nose tip
            z = 30
        elif 320 <= x <= 480 and 420 <= y <= 450:  # mouth
            z = 20
        elif (280 <= x <= 360 and 240 <= y <= 260) or (440 <= x <= 520 and 240 <= y <= 260):  # eyes
            z = -10
        else:
            z = 0
        landmarks_3d.append((x, y, z))
    
    # Create visualization options
    options = LandmarkVisualizationOptions(
        point_color=(0, 255, 0),
        point_size=5,
        point_alpha=1.0,
        connection_color=(255, 255, 0),
        connection_thickness=1,
        show_labels=True
    )
    
    # Draw landmarks
    landmark_viz = draw_landmarks(image, landmarks_2d, options)
    save_visualization(landmark_viz, os.path.join(output_dir, "landmark_visualization.png"))
    
    # Create a transformation matrix (rotate face slightly)
    center_x, center_y = 400, 300
    angle = 15
    scale = 1.2
    
    # Build rotation matrix
    rotation = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
    # Convert to 3x3 for homogeneous coordinates
    rotation_3x3 = np.eye(3)
    rotation_3x3[:2, :] = rotation
    
    # Draw transformed landmarks
    transformed = draw_transformed_landmarks(image, landmarks_2d, rotation_3x3, options)
    save_visualization(transformed, os.path.join(output_dir, "transformed_landmarks.png"))
    
    # Create comparison
    comparison = create_comparison(image, landmark_viz, "Original vs Landmarks", ("Original", "With Landmarks"))
    save_visualization(comparison, os.path.join(output_dir, "comparison.png"))
    
    # Render 3D mesh
    connections = []
    # Connect the eyes
    for i in range(0, 3):
        connections.append((i, i+1))
    for i in range(4, 7):
        connections.append((i, i+1))
    # Connect the eyebrows
    for i in range(8, 10):
        connections.append((i, i+1))
    for i in range(11, 13):
        connections.append((i, i+1))
    # Connect the nose
    for i in range(14, 18):
        connections.append((i, i+1))
    # Connect the mouth
    for i in range(19, 21):
        connections.append((i, i+1))
    for i in range(22, 24):
        connections.append((i, i+1))
    # Connect jaw line
    for i in range(25, 38):
        connections.append((i, (i+1) % 14 + 25))
    
    # Create multiple views
    for angle in [(0, 0, 0), (30, 0, 0), (0, 30, 0), (30, 30, 0)]:
        fig = render_3d_mesh(
            landmarks_3d, 
            connections,
            view_angle=angle
        )
        save_3d_visualization(
            fig, 
            os.path.join(output_dir, f"3d_mesh_elevation{angle[0]}_azimuth{angle[1]}.png")
        )
    
    print(f"Reference images created in {output_dir}")


if __name__ == '__main__':
    # Run the tests
    unittest.main()
    
    # To create reference images, uncomment the line below
    # create_reference_images()