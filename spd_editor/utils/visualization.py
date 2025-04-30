#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization utility module for SPD Editor.

This module provides functions for visualizing SPD (Source Portrait Descriptor) data,
including facial landmarks, 3D meshes, and transformation effects.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
import tempfile

# Import local modules
from .transformation import apply_transformation, Point2D, Point3D, Points, TransformMatrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
ColorType = Union[Tuple[int, int, int], str]
ConnectionsList = List[Tuple[int, int]]  # Pairs of landmark indices to connect


@dataclass
class LandmarkVisualizationOptions:
    """Configuration options for landmark visualization."""
    
    # Point style
    point_color: ColorType = (0, 255, 0)  # BGR for OpenCV, or named color for matplotlib
    point_size: int = 2
    point_alpha: float = 0.8  # Transparency (0-1)
    
    # Connection style
    draw_connections: bool = True
    connection_color: ColorType = (255, 255, 0)  # BGR for OpenCV
    connection_thickness: int = 1
    connection_alpha: float = 0.7
    
    # Connection definitions for different landmark types
    # These define which points should be connected with lines
    connections: Dict[str, ConnectionsList] = field(default_factory=dict)
    
    # Labels
    show_labels: bool = False
    label_color: ColorType = (255, 255, 255)
    label_size: float = 0.3
    label_thickness: int = 1
    
    def __post_init__(self):
        """Initialize default connection maps if none provided."""
        if not self.connections:
            # Default connections for common landmark formats
            
            # 68-point landmarks (dlib format)
            dlib68_connections = []
            # Jaw line
            for i in range(0, 16):
                dlib68_connections.append((i, i+1))
            # Right eyebrow
            for i in range(17, 21):
                dlib68_connections.append((i, i+1))
            # Left eyebrow
            for i in range(22, 26):
                dlib68_connections.append((i, i+1))
            # Nose bridge
            for i in range(27, 30):
                dlib68_connections.append((i, i+1))
            # Lower nose
            dlib68_connections.append((30, 31))
            dlib68_connections.append((31, 32))
            dlib68_connections.append((32, 33))
            dlib68_connections.append((33, 34))
            dlib68_connections.append((34, 35))
            # Right eye
            for i in range(36, 41):
                dlib68_connections.append((i, i+1))
            dlib68_connections.append((41, 36))
            # Left eye
            for i in range(42, 47):
                dlib68_connections.append((i, i+1))
            dlib68_connections.append((47, 42))
            # Outer lip
            for i in range(48, 59):
                dlib68_connections.append((i, i+1))
            dlib68_connections.append((59, 48))
            # Inner lip
            for i in range(60, 67):
                dlib68_connections.append((i, i+1))
            dlib68_connections.append((67, 60))
            
            self.connections['dlib68'] = dlib68_connections
            
            # Simple 5-point landmarks
            self.connections['simple'] = [
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 0)
            ]
            
            # MediaPipe face mesh connections (simplified)
            # For the full set, refer to MediaPipe documentation
            mediapipe_connections = []
            # Add some basic connections for face outline, eyes, nose, and mouth
            # This is a simplified subset
            self.connections['mediapipe'] = mediapipe_connections


def draw_landmarks(image: np.ndarray, 
                  landmarks: Points,
                  options: Optional[LandmarkVisualizationOptions] = None) -> np.ndarray:
    """
    Draw facial landmarks on an image.
    
    Args:
        image: Input image as a NumPy array (H, W, C)
        landmarks: List of landmark points (2D or 3D)
        options: Visualization options
        
    Returns:
        Image with landmarks drawn on it
    """
    if options is None:
        options = LandmarkVisualizationOptions()
    
    # Make a copy of the image to avoid modifying the original
    vis_image = image.copy()
    
    # Get image dimensions
    height, width = vis_image.shape[:2]
    
    # Determine landmark type for connection patterns
    landmark_type = None
    num_landmarks = len(landmarks)
    if num_landmarks == 68:
        landmark_type = 'dlib68'
    elif num_landmarks == 5:
        landmark_type = 'simple'
    elif num_landmarks > 400:
        landmark_type = 'mediapipe'
    
    # Draw connections first so points appear on top
    if options.draw_connections and landmark_type in options.connections:
        for start_idx, end_idx in options.connections[landmark_type]:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                
                # Extract 2D coordinates
                start_x, start_y = int(start_point[0]), int(start_point[1])
                end_x, end_y = int(end_point[0]), int(end_point[1])
                
                # Check if points are within image bounds
                if (0 <= start_x < width and 0 <= start_y < height and
                    0 <= end_x < width and 0 <= end_y < height):
                    cv2.line(vis_image, 
                           (start_x, start_y), 
                           (end_x, end_y), 
                           options.connection_color, 
                           options.connection_thickness)
    
    # Draw landmark points
    for i, point in enumerate(landmarks):
        x, y = int(point[0]), int(point[1])
        
        # Check if point is within image bounds
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(vis_image, (x, y), options.point_size, options.point_color, -1)
            
            # Draw label if enabled
            if options.show_labels:
                cv2.putText(vis_image, 
                          str(i), 
                          (x + 2, y - 2),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          options.label_size, 
                          options.label_color, 
                          options.label_thickness)
    
    return vis_image


def render_3d_mesh(landmarks: Points, 
                  connections: Optional[ConnectionsList] = None,
                  texture: Optional[np.ndarray] = None,
                  view_angle: Tuple[float, float, float] = (0, 0, 0),
                  figure_size: Tuple[int, int] = (10, 10)) -> Figure:
    """
    Render a 3D face mesh from landmarks.
    
    Args:
        landmarks: List of 3D landmark points
        connections: List of index pairs to connect with edges
        texture: Optional texture image to map onto the mesh
        view_angle: (elevation, azimuth, roll) viewing angles in degrees
        figure_size: Size of the output figure in inches
        
    Returns:
        Matplotlib Figure object with the rendered 3D mesh
    """
    # Check if points are 3D
    if not landmarks or len(landmarks[0]) < 3:
        raise ValueError("3D landmarks are required for mesh rendering")
    
    # Create figure
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x = [p[0] for p in landmarks]
    y = [p[1] for p in landmarks]
    z = [p[2] for p in landmarks]
    
    # Plot landmarks as points
    ax.scatter(x, y, z, c='g', marker='o', s=10)
    
    # Draw connections if provided
    if connections:
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                ax.plot([x[start_idx], x[end_idx]],
                       [y[start_idx], y[end_idx]],
                       [z[start_idx], z[end_idx]], 'b-', linewidth=0.5)
    
    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1], roll=view_angle[2])
    
    # Texture mapping is more complex and would require triangulation
    # and proper UV mapping - simplified approach:
    if texture is not None:
        logger.warning("Full texture mapping not implemented; displaying wireframe only")
    
    # Configure axes and title
    ax.set_title("3D Face Mesh")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    return fig


def draw_transformed_landmarks(image: np.ndarray,
                              landmarks: Points,
                              transform_matrix: TransformMatrix,
                              options: Optional[LandmarkVisualizationOptions] = None) -> np.ndarray:
    """
    Apply a transformation to landmarks and draw them on an image.
    
    Args:
        image: Input image
        landmarks: Original landmark points
        transform_matrix: Transformation matrix to apply
        options: Visualization options
        
    Returns:
        Image with transformed landmarks drawn on it
    """
    from .transformation import apply_transformation
    
    # Transform the landmarks
    transformed_landmarks = apply_transformation(landmarks, transform_matrix)
    
    # Draw the transformed landmarks
    return draw_landmarks(image, transformed_landmarks, options)


def create_comparison(original_image: np.ndarray, 
                     modified_image: np.ndarray,
                     title: str = "Before/After Comparison",
                     labels: Tuple[str, str] = ("Original", "Modified")) -> np.ndarray:
    """
    Create a side-by-side comparison of two images.
    
    Args:
        original_image: First image (before)
        modified_image: Second image (after)
        title: Title for the comparison
        labels: Labels for the two images
        
    Returns:
        A new image with both images side by side
    """
    # Ensure images are the same height
    h1, w1 = original_image.shape[:2]
    h2, w2 = modified_image.shape[:2]
    
    # Make heights equal by padding the shorter image
    if h1 != h2:
        max_h = max(h1, h2)
        if h1 < max_h:
            pad = (max_h - h1) // 2
            original_image = cv2.copyMakeBorder(
                original_image, pad, max_h - h1 - pad, 0, 0, 
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        else:
            pad = (max_h - h2) // 2
            modified_image = cv2.copyMakeBorder(
                modified_image, pad, max_h - h2 - pad, 0, 0, 
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
    
    # Create the side-by-side image
    comparison = np.hstack((original_image, modified_image))
    
    # Add title and labels
    h, w = comparison.shape[:2]
    font_scale = min(h, w) * 0.001  # Scale font based on image size
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Add a header bar
    header_height = int(h * 0.05)
    header = np.zeros((header_height, w, 3), dtype=np.uint8)
    comparison = np.vstack((header, comparison))
    
    # Add title and labels
    cv2.putText(comparison, title, 
              (w // 2 - len(title) * 5, header_height - 10), 
              font, font_scale, (255, 255, 255), 1)
    
    cv2.putText(comparison, labels[0], 
              (w1 // 2 - len(labels[0]) * 5, h + header_height - 20), 
              font, font_scale, (255, 255, 255), 1)
    
    cv2.putText(comparison, labels[1], 
              (w1 + w2 // 2 - len(labels[1]) * 5, h + header_height - 20), 
              font, font_scale, (255, 255, 255), 1)
    
    # Add vertical separator
    cv2.line(comparison, (w1, header_height), (w1, h + header_height), (255, 255, 255), 1)
    
    return comparison


def save_visualization(image: np.ndarray, 
                      file_path: str,
                      quality: int = 95) -> None:
    """
    Save a visualization image to a file.
    
    Args:
        image: Image to save
        file_path: Path to save the image to
        quality: JPEG quality (0-100) if saving as JPEG
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Determine file extension
    ext = os.path.splitext(file_path)[1].lower()
    
    # Save based on extension
    if ext in ['.jpg', '.jpeg']:
        cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == '.png':
        cv2.imwrite(file_path, image)
    else:
        # Default to PNG
        if not ext:
            file_path += '.png'
        cv2.imwrite(file_path, image)
    
    logger.info(f"Saved visualization to {file_path}")


def show_interactive(image: np.ndarray, 
                    title: Optional[str] = None,
                    wait_key: bool = True) -> None:
    """
    Display an image in an interactive window.
    
    Args:
        image: Image to display
        title: Window title
        wait_key: Whether to wait for a key press
        
    Returns:
        None
    """
    window_title = title if title else "SPD Visualization"
    cv2.imshow(window_title, image)
    
    if wait_key:
        cv2.waitKey(0)
        cv2.destroyWindow(window_title)


def save_3d_visualization(figure: Figure, 
                         file_path: str,
                         dpi: int = 100,
                         close_figure: bool = True) -> None:
    """
    Save a 3D visualization figure to a file.
    
    Args:
        figure: Matplotlib Figure object
        file_path: Path to save the figure to
        dpi: Resolution in dots per inch
        close_figure: Whether to close the figure after saving
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Save figure
    figure.savefig(file_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved 3D visualization to {file_path}")
    
    if close_figure:
        plt.close(figure)


def visualize_spd_data(image: np.ndarray,
                      landmarks: Points,
                      transforms: Optional[Dict[str, TransformMatrix]] = None,
                      show_3d: bool = False,
                      show_transformed: bool = False,
                      output_path: Optional[str] = None,
                      interactive_display: bool = False) -> Dict[str, Any]:
    """
    High-level function to visualize SPD data with multiple views.
    
    Args:
        image: Source image
        landmarks: Landmark points
        transforms: Dictionary of transformation matrices
        show_3d: Whether to create 3D visualization
        show_transformed: Whether to show transformed landmarks
        output_path: Directory to save visualizations to
        interactive_display: Whether to show interactive display when output_path is None
        
    Returns:
        Dictionary of results including created visualizations
    """
    results = {}
    
    # Configure visualization options
    options = LandmarkVisualizationOptions(
        point_color=(0, 255, 0),
        point_size=3,
        connection_color=(255, 255, 0),
        connection_thickness=1
    )
    
    # Draw landmarks on image
    landmark_viz = draw_landmarks(image, landmarks, options)
    results['landmarks'] = landmark_viz
    
    # Show interactive display if requested and no output path
    if output_path is None:
        if interactive_display:
            show_interactive(landmark_viz, "Landmarks")
    else:
        landmark_path = os.path.join(output_path, "landmarks.jpg")
        save_visualization(landmark_viz, landmark_path)
        results['landmarks_path'] = landmark_path
    
    # Create 3D visualization if requested
    if show_3d and landmarks and len(landmarks[0]) >= 3:
        # Determine connections based on landmark count
        connections = None
        if len(landmarks) == 68:
            if 'dlib68' in options.connections:
                connections = options.connections['dlib68']
        
        # Create the 3D visualization
        fig = render_3d_mesh(landmarks, connections)
        results['3d_figure'] = fig
        
        if output_path:
            mesh_path = os.path.join(output_path, "mesh_3d.png")
            save_3d_visualization(fig, mesh_path)
            results['3d_path'] = mesh_path
        else:
            # Close figure to avoid memory leaks
            plt.close(fig)
    
    # Show transformed landmarks if requested
    if show_transformed and transforms:
        for name, transform in transforms.items():
            transformed = draw_transformed_landmarks(image, landmarks, transform, options)
            key = f"transformed_{name}"
            results[key] = transformed
            
            if output_path:
                transform_path = os.path.join(output_path, f"{key}.jpg")
                save_visualization(transformed, transform_path)
                results[f"{key}_path"] = transform_path
            elif interactive_display:
                show_interactive(transformed, f"Transformed: {name}")
    
    return results