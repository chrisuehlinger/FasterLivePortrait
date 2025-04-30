#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformation utility module for SPD Editor.

This module provides functions for manipulating transformation matrices
and working with coordinate systems for SPD (Source Portrait Descriptor) files.
"""

import numpy as np
import logging
from typing import List, Tuple, Union, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]
Points = List[Union[Point2D, Point3D]]
TransformMatrix = np.ndarray  # 3x3 matrix


def apply_transformation(points: Points, 
                        transform_matrix: TransformMatrix) -> Points:
    """
    Apply a transformation matrix to a list of points.
    
    Args:
        points: List of 2D or 3D points
        transform_matrix: 3x3 transformation matrix
        
    Returns:
        List of transformed points with same dimensions as input
    """
    if not isinstance(transform_matrix, np.ndarray):
        transform_matrix = np.array(transform_matrix).reshape(3, 3)
    
    # Check if we're working with 2D or 3D points
    is_2d = len(points[0]) == 2 if points else True
    
    result = []
    for point in points:
        if is_2d:
            # For 2D points, extend to homogeneous coordinates
            p_hom = np.array([point[0], point[1], 1.0])
            # Apply transformation
            transformed = np.dot(transform_matrix, p_hom)
            # Normalize by dividing by the homogeneous coordinate
            if transformed[2] != 0:
                transformed = transformed / transformed[2]
            result.append((transformed[0], transformed[1]))
        else:
            # For 3D points
            p_hom = np.array([point[0], point[1], point[2] if point[2] != 0 else 1.0])
            transformed = np.dot(transform_matrix, p_hom)
            if transformed[2] != 0:
                transformed = transformed / transformed[2]
            result.append((transformed[0], transformed[1], point[2]))  # Preserve Z
    
    return result


def compose_transforms(transforms: List[TransformMatrix]) -> TransformMatrix:
    """
    Compose multiple transformation matrices into a single transformation.
    Transformations are applied from left to right.
    
    Args:
        transforms: List of 3x3 transformation matrices
        
    Returns:
        A single 3x3 transformation matrix representing the composition
    """
    if not transforms:
        # Return identity matrix if no transforms provided
        return np.eye(3)
    
    # Transformations are applied from left to right in the list
    # This means for a sequence [A, B, C], we compute A * B * C
    # In matrix terms, the rightmost matrix is applied first to points
    result = np.array(transforms[0])
    
    for transform in transforms[1:]:
        result = np.dot(np.array(transform), result)
    
    return result


def invert_transform(transform_matrix: TransformMatrix) -> TransformMatrix:
    """
    Compute the inverse of a transformation matrix.
    
    Args:
        transform_matrix: 3x3 transformation matrix
        
    Returns:
        The inverse transformation matrix
        
    Raises:
        ValueError: If the matrix is not invertible
    """
    matrix = np.array(transform_matrix)
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError as e:
        logger.error(f"Matrix is not invertible: {e}")
        raise ValueError(f"Matrix is not invertible: {e}")


def convert_coordinates(points: Points, 
                       source_space: str, 
                       target_space: str,
                       transforms: Dict[str, TransformMatrix]) -> Points:
    """
    Convert points from one coordinate space to another using the provided transforms.
    
    Args:
        points: List of points to convert
        source_space: Name of the source coordinate space
        target_space: Name of the target coordinate space
        transforms: Dictionary of named transformations
        
    Returns:
        Converted points in the target coordinate space
        
    Raises:
        ValueError: If the required transforms aren't available
    """
    # Check if we have a direct transformation
    key = f"{source_space}2{target_space}"
    if key in transforms:
        return apply_transformation(points, transforms[key])
    
    # Check for inverse transformation
    inv_key = f"{target_space}2{source_space}"
    if inv_key in transforms:
        return apply_transformation(points, invert_transform(transforms[inv_key]))
    
    # Check if we need to go through an intermediate space
    # e.g., source→world→target
    world_source_key = f"{source_space}2world"
    world_target_key = f"world2{target_space}"
    
    if world_source_key in transforms and world_target_key in transforms:
        # Convert to world space, then to target space
        world_points = apply_transformation(points, transforms[world_source_key])
        return apply_transformation(world_points, transforms[world_target_key])
    
    raise ValueError(f"No transformation path found from {source_space} to {target_space}")


def get_face_aligned_transform(landmarks: Points, 
                              reference_points: Optional[List[int]] = None) -> TransformMatrix:
    """
    Compute a transformation that aligns a face based on landmark positions.
    
    Args:
        landmarks: List of facial landmark points
        reference_points: Indices of landmarks to use for alignment 
                         (defaults to eyes and nose tip)
        
    Returns:
        A transformation matrix that aligns the face
    """
    if reference_points is None:
        # Default reference points: typically left eye, right eye, nose tip
        # The exact indices depend on the landmark format being used
        # These are common choices for 68-point landmark set
        reference_points = [36, 45, 30]  # left eye, right eye, nose
    
    if len(landmarks) < max(reference_points) + 1:
        raise ValueError(f"Not enough landmarks for reference points {reference_points}")
    
    # Extract reference points
    points = [landmarks[i] for i in reference_points]
    
    # For simplicity, compute a transformation that aligns the face horizontally
    # based on eye positions
    if len(points) >= 2:
        # Get eye positions
        left_eye = np.array(points[0][:2])
        right_eye = np.array(points[1][:2])
        
        # Calculate angle to rotate eyes to horizontal
        eye_vector = right_eye - left_eye
        angle = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
        
        # Create rotation matrix to align eyes horizontally
        center = (left_eye + right_eye) / 2
        rotation = np.array([
            [np.cos(np.radians(-angle)), -np.sin(np.radians(-angle)), 0],
            [np.sin(np.radians(-angle)), np.cos(np.radians(-angle)), 0],
            [0, 0, 1]
        ])
        
        # Create translation matrices
        translate_to_origin = np.array([
            [1, 0, -center[0]],
            [0, 1, -center[1]],
            [0, 0, 1]
        ])
        translate_back = np.array([
            [1, 0, center[0]],
            [0, 1, center[1]],
            [0, 0, 1]
        ])
        
        # Compose transformation: translate to origin, rotate, translate back
        transform = compose_transforms([
            translate_back,
            rotation,
            translate_to_origin
        ])
        
        return transform
    
    # Return identity transform if not enough reference points
    return np.eye(3)


def transform_to_scale_and_center(width: int, 
                                 height: int, 
                                 target_size: Optional[Tuple[int, int]] = None,
                                 padding: float = 0.1) -> TransformMatrix:
    """
    Create a transformation matrix that scales and centers an image.
    
    Args:
        width: Original image width
        height: Original image height
        target_size: Target size as (width, height), defaults to same as original
        padding: Padding around the image as fraction of size (0.1 = 10% padding)
        
    Returns:
        Transformation matrix for scaling and centering
    """
    if target_size is None:
        target_size = (width, height)
    
    # Calculate scale factors
    scale_x = target_size[0] / width
    scale_y = target_size[1] / height
    
    # Use smaller scale factor to maintain aspect ratio
    scale = min(scale_x, scale_y) * (1 - padding)
    
    # Calculate translations to center the image
    tx = (target_size[0] - width * scale) / 2
    ty = (target_size[1] - height * scale) / 2
    
    # Create transformation matrix
    transform = np.array([
        [scale, 0, tx],
        [0, scale, ty],
        [0, 0, 1]
    ])
    
    return transform