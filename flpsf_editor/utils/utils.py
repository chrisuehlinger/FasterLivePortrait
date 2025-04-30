#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for FLPSF Editor
"""

import os
import sys
import logging
import numpy as np
import cv2
from typing import Dict, Any, Optional, List, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('flpsf_editor.utils')

def create_directory(directory_path: str) -> bool:
    """
    Create directory if it doesn't exist
    
    Args:
        directory_path: Path to directory
        
    Returns:
        bool: True if directory exists or was created, False otherwise
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logger.info(f"Created directory: {directory_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        return False
        
def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file
    
    Args:
        image_path: Path to image file
        
    Returns:
        numpy.ndarray: Loaded image or None if loading failed
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
            
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None
        
def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save image to file
    
    Args:
        image: Image to save
        output_path: Path to save image to
        
    Returns:
        bool: True if image was saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if not create_directory(output_dir):
            return False
            
        result = cv2.imwrite(output_path, image)
        if not result:
            logger.error(f"Failed to save image to {output_path}")
            return False
            
        logger.info(f"Image saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False
        
def resize_image(image: np.ndarray, width: int = None, height: int = None, 
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize image to specified dimensions
    
    Args:
        image: Image to resize
        width: Target width (None to calculate from height)
        height: Target height (None to calculate from width)
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        numpy.ndarray: Resized image
    """
    if image is None:
        return None
        
    if width is None and height is None:
        return image
        
    h, w = image.shape[:2]
    
    if keep_aspect_ratio:
        if width is None:
            # Calculate width based on height
            aspect_ratio = w / h
            width = int(height * aspect_ratio)
        elif height is None:
            # Calculate height based on width
            aspect_ratio = h / w
            height = int(width * aspect_ratio)
        else:
            # Both width and height provided, maintain aspect ratio
            # by calculating the smallest dimension that fits
            aspect_ratio = w / h
            target_aspect_ratio = width / height
            
            if aspect_ratio > target_aspect_ratio:
                # Image is wider than target, fit to width
                new_width = width
                new_height = int(width / aspect_ratio)
            else:
                # Image is taller than target, fit to height
                new_height = height
                new_width = int(height * aspect_ratio)
                
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Resize without keeping aspect ratio
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
def convert_cv_to_qt_image(cv_image: np.ndarray) -> Any:
    """
    Convert OpenCV image to Qt image
    
    Args:
        cv_image: OpenCV image
        
    Returns:
        QImage: Qt image
    """
    try:
        from PyQt5.QtGui import QImage
        
        if cv_image is None:
            return None
            
        # Convert BGR to RGB
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
        height, width = cv_image.shape[:2]
        
        if len(cv_image.shape) == 3:
            # Color image
            bytes_per_line = 3 * width
            return QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            # Grayscale image
            bytes_per_line = width
            return QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    except Exception as e:
        logger.error(f"Error converting OpenCV image to Qt image: {e}")
        return None
        
def normalize_landmarks(landmarks: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """
    Normalize landmarks coordinates to [0, 1] range
    
    Args:
        landmarks: Array of landmarks coordinates
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        numpy.ndarray: Normalized landmarks
    """
    if landmarks is None:
        return None
        
    normalized = landmarks.copy()
    normalized[:, 0] /= image_width
    normalized[:, 1] /= image_height
    return normalized
    
def denormalize_landmarks(landmarks: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """
    Denormalize landmarks from [0, 1] range to pixel coordinates
    
    Args:
        landmarks: Array of normalized landmarks
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        numpy.ndarray: Denormalized landmarks
    """
    if landmarks is None:
        return None
        
    denormalized = landmarks.copy()
    denormalized[:, 0] *= image_width
    denormalized[:, 1] *= image_height
    return denormalized
    
def calculate_derived_params(landmarks: np.ndarray) -> Dict[str, float]:
    """
    Calculate derived parameters from facial landmarks
    
    Args:
        landmarks: Array of facial landmarks
        
    Returns:
        Dict[str, float]: Dictionary of derived parameters
    """
    # This is a simplistic implementation that would need to be replaced
    # with actual facial geometry calculations
    if landmarks is None or len(landmarks) < 68:  # Assuming 68-point landmark model
        return {}
        
    # Calculate some example derived parameters
    # In a real implementation, these would be based on actual facial geometry
    # and would match the FasterLivePortrait model's requirements
    
    # Find eye landmarks (simplified)
    left_eye_landmarks = landmarks[36:42]  # Left eye landmarks
    right_eye_landmarks = landmarks[42:48]  # Right eye landmarks
    
    # Calculate eye centers
    left_eye_center = np.mean(left_eye_landmarks, axis=0)
    right_eye_center = np.mean(right_eye_landmarks, axis=0)
    
    # Calculate inter-eye distance
    eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
    
    # Find nose tip and chin landmarks
    nose_tip = landmarks[30]
    chin = landmarks[8]
    
    # Calculate face height
    face_height = np.linalg.norm(chin - left_eye_center)
    
    # Find mouth landmarks
    mouth_landmarks = landmarks[48:68]
    mouth_width = np.linalg.norm(mouth_landmarks[6] - mouth_landmarks[0])
    
    # Calculate some ratios
    eye_distance_ratio = eye_distance / face_height
    mouth_width_ratio = mouth_width / eye_distance
    
    # Return derived parameters
    return {
        'eye_distance': float(eye_distance),
        'face_height': float(face_height),
        'eye_distance_ratio': float(eye_distance_ratio),
        'mouth_width_ratio': float(mouth_width_ratio),
    }
    
def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, 
                  color: Tuple[int, int, int] = (0, 255, 0), 
                  radius: int = 2, 
                  thickness: int = -1,
                  landmark_indices: List[int] = None) -> np.ndarray:
    """
    Draw landmarks on an image
    
    Args:
        image: Image to draw on
        landmarks: Array of landmarks coordinates
        color: Color for landmarks
        radius: Radius of landmark circles
        thickness: Thickness of landmark circles (-1 for filled)
        landmark_indices: List of landmark indices to draw (None for all)
        
    Returns:
        numpy.ndarray: Image with landmarks drawn
    """
    if image is None or landmarks is None:
        return image
        
    result = image.copy()
    
    if landmark_indices is None:
        # Draw all landmarks
        for x, y in landmarks.astype(int):
            cv2.circle(result, (x, y), radius, color, thickness)
    else:
        # Draw only selected landmarks
        for i in landmark_indices:
            if 0 <= i < len(landmarks):
                x, y = landmarks[i].astype(int)
                cv2.circle(result, (x, y), radius, color, thickness)
                
    return result