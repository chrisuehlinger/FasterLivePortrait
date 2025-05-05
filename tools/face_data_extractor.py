#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Data Extractor for FasterLivePortrait

This script extracts facial data from an image and saves it in a JSON format
suitable for use with FasterLivePortrait.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
import importlib.util

# Make sure the FasterLivePortrait modules are in the path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('face_data_extractor')

try:
    from spd_editor.analysis.face_detector import FaceDetector
    from spd_editor.analysis.landmark_extractor import LandmarkExtractor
    from spd_editor.analysis.feature_extractor import FeatureExtractor
    ANALYSIS_MODULES_AVAILABLE = True
    
    # Try to import crop utilities
    try:
        from src.utils.crop import crop_image_by_bbox
        from src.utils.face_align import trans_points
        CROP_UTILS_AVAILABLE = True
    except ImportError:
        CROP_UTILS_AVAILABLE = False
        logger.warning("Crop utilities not available, will use simple cropping")
        
except ImportError as e:
    logger.warning(f"Could not import analysis modules: {e}")
    ANALYSIS_MODULES_AVAILABLE = False
    CROP_UTILS_AVAILABLE = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract facial data from an image')
    
    parser.add_argument('input_image', type=str, 
                        help='Path to the input image')
    
    parser.add_argument('-o', '--output', type=str, 
                        help='Path to the output JSON file (default: based on input filename)')
    
    parser.add_argument('-a', '--aligned-output', type=str, 
                        help='Path to save the aligned face image (default: based on output filename)')
    
    parser.add_argument('-m', '--mask-output', type=str,
                        help='Path to save the face mask image (default: based on output filename)')
    
    parser.add_argument('-s', '--size', type=int, default=512,
                        help='Size of the output aligned face image (default: 512)')
    
    parser.add_argument('-f', '--scale-factor', type=float, default=1.25,
                        help='Scale factor for face crop (default: 1.25)')
    
    parser.add_argument('-v', '--vertical-shift', type=float, default=-0.1,
                        help='Vertical shift for face crop (default: -0.1, negative = up)')
    
    parser.add_argument('--predict-type', type=str, default='ort',
                        choices=['ort', 'trt'],
                        help='Prediction engine to use (default: ort)')
    
    parser.add_argument('--no-flame', action='store_true',
                        help='Skip FLAME model parameter extraction')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    
    args = parser.parse_args()
    
    # Set default output paths if not specified
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input_image))[0]
        args.output = f"{base_name}_face_data.json"
    
    if not args.aligned_output:
        base_dir = os.path.dirname(args.output)
        base_name = os.path.splitext(os.path.basename(args.output))[0]
        args.aligned_output = os.path.join(base_dir, f"{base_name}_aligned.png")
    
    if not args.mask_output:
        base_dir = os.path.dirname(args.output)
        base_name = os.path.splitext(os.path.basename(args.output))[0]
        args.mask_output = os.path.join(base_dir, f"{base_name}_mask.png")
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    return args


def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image from file."""
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
            
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def detect_face(image: np.ndarray, predict_type: str = 'ort') -> Optional[Dict[str, Any]]:
    """Detect the largest face in an image."""
    if not ANALYSIS_MODULES_AVAILABLE:
        logger.error("Face detection modules not available")
        return None
    
    try:
        detector = FaceDetector(predict_type=predict_type)
        face = detector.detect_largest_face(image)
        return face
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return None


def extract_landmarks(image: np.ndarray, face: Dict[str, Any], 
                     predict_type: str = 'ort') -> Optional[np.ndarray]:
    """Extract facial landmarks from a face."""
    if not ANALYSIS_MODULES_AVAILABLE:
        logger.error("Landmark extraction modules not available")
        return None
    
    try:
        landmark_extractor = LandmarkExtractor(predict_type=predict_type)
        landmarks = landmark_extractor.extract_landmarks(image, face)
        return landmarks
    except Exception as e:
        logger.error(f"Error in landmark extraction: {e}")
        return None


def align_face(image: np.ndarray, face: Dict[str, Any], landmarks: np.ndarray,
              size: int = 512, scale_factor: float = 1.25, 
              vertical_shift: float = -0.1) -> Tuple[Optional[np.ndarray], Dict[str, Any], Optional[np.ndarray]]:
    """Align and crop the face."""
    try:
        # Get bbox from the face detection result
        bbox = face['bbox']
        
        # First try to use project's crop utility if available
        if CROP_UTILS_AVAILABLE:
            try:
                # Adjust bbox with scale factor and vertical shift
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Apply vertical shift
                center_y += vertical_shift * height
                
                # Apply scale factor
                new_width = width * scale_factor
                new_height = height * scale_factor
                
                # Calculate new bbox
                new_x1 = max(0, center_x - new_width / 2)
                new_y1 = max(0, center_y - new_height / 2)
                new_x2 = min(image.shape[1], center_x + new_width / 2)
                new_y2 = min(image.shape[0], center_y + new_height / 2)
                adjusted_bbox = np.array([new_x1, new_y1, new_x2, new_y2])
                
                # Crop and align the face
                crop_result = crop_image_by_bbox(
                    image, bbox=adjusted_bbox, 
                    out_size=size, face_ratio=1.0
                )
                
                if crop_result and 'image_crop' in crop_result:
                    aligned_face = crop_result['image_crop']
                    
                    # Transform landmarks if transformation matrix is available
                    landmarks_aligned = []
                    if 'M' in crop_result and landmarks is not None:
                        landmarks_aligned = trans_points(landmarks, crop_result['M'])
                    else:
                        # Fall back to simple transformation
                        for lm in landmarks:
                            # Adjust for crop and resize
                            lm_x = (lm[0] - new_x1) * size / (new_x2 - new_x1)
                            lm_y = (lm[1] - new_y1) * size / (new_y2 - new_y1)
                            landmarks_aligned.append([lm_x, lm_y])
                        landmarks_aligned = np.array(landmarks_aligned)
                    
                    crop_params = {
                        "scale_factor": scale_factor,
                        "vertical_shift": vertical_shift,
                        "rotation_degree": 0.0,
                        "output_size": [size, size]
                    }
                    
                    return aligned_face, crop_params, landmarks_aligned
                    
            except Exception as e:
                logger.warning(f"Error using project's crop utility: {e}, falling back to simple crop")
        
        # Fall back to simple crop if the project's utility isn't available
        x1, y1, x2, y2 = bbox
        
        # Calculate bbox center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate bbox dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Apply scale factor
        scaled_width = width * scale_factor
        scaled_height = height * scale_factor
        
        # Apply vertical shift
        center_y += vertical_shift * height
        
        # Calculate new bbox coordinates
        new_x1 = max(0, center_x - scaled_width / 2)
        new_y1 = max(0, center_y - scaled_height / 2)
        new_x2 = min(image.shape[1], center_x + scaled_width / 2)
        new_y2 = min(image.shape[0], center_y + scaled_height / 2)
        
        # Crop the image
        cropped = image[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]
        
        # Resize to target size
        aligned_face = cv2.resize(cropped, (size, size))
        
        # Calculate transformation parameters
        crop_params = {
            "scale_factor": scale_factor,
            "vertical_shift": vertical_shift,
            "rotation_degree": 0.0,  # No rotation in this simple alignment
            "output_size": [size, size]
        }
        
        # Transform landmarks to aligned image coordinates
        landmarks_aligned = []
        for lm in landmarks:
            # First adjust for crop
            lm_x = (lm[0] - new_x1) 
            lm_y = (lm[1] - new_y1)
            
            # Then adjust for resize
            lm_x = lm_x * size / (new_x2 - new_x1)
            lm_y = lm_y * size / (new_y2 - new_y1)
            
            landmarks_aligned.append([lm_x, lm_y])
        
        return aligned_face, crop_params, np.array(landmarks_aligned)
    
    except Exception as e:
        logger.error(f"Error in face alignment: {e}")
        return None, {}, None


def generate_face_mask(aligned_face: np.ndarray, landmarks_aligned: np.ndarray, 
                      size: int = 512) -> Optional[np.ndarray]:
    """Generate a face mask based on landmarks."""
    try:
        # Create an empty mask
        mask = np.zeros((size, size), dtype=np.uint8)
        
        # Convert landmarks to points for cv2.fillConvexPoly
        if len(landmarks_aligned) >= 17:  # Use face outline landmarks
            # For a standard 68-point landmark set, the face outline is points 0-16
            # For larger sets like MediaPipe's, we'd need different indices
            outline_points = landmarks_aligned[:17].astype(np.int32)
            cv2.fillConvexPoly(mask, outline_points, 255)
        else:
            # Use a simple elliptical mask as fallback
            center = (size // 2, size // 2)
            axes = (size // 3, size // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Dilate the mask slightly to ensure it covers the face completely
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Apply Gaussian blur to soften edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    except Exception as e:
        logger.error(f"Error generating face mask: {e}")
        return None


def extract_flame_parameters(aligned_face: np.ndarray, 
                           predict_type: str = 'ort') -> Optional[Dict[str, Any]]:
    """Extract FLAME model parameters from the aligned face."""
    if not ANALYSIS_MODULES_AVAILABLE:
        logger.error("Feature extraction modules not available")
        return None
    
    try:
        feature_extractor = FeatureExtractor(predict_type=predict_type)
        motion_params = feature_extractor.extract_motion_parameters(aligned_face)
        
        if motion_params is None:
            return None
        
        # Initialize FLAME parameter structure with available data
        flame_params = {
            "pose": {
                "yaw": float(motion_params.get('yaw', 0.0)),
                "pitch": float(motion_params.get('pitch', 0.0)),
                "roll": float(motion_params.get('roll', 0.0)),
                "jaw_open": 0.1  # Default value, might not be available
            },
            "camera": {
                "focal_length": 1200.0,  # Default value
                "principal_point": [0.0, 0.0]
            },
        }
        
        # Add translation if available
        if 't' in motion_params:
            t = motion_params['t']
            flame_params["translation"] = [float(t[0]), float(t[1]), float(t[2]) if len(t) > 2 else 300.0]
        else:
            flame_params["translation"] = [0.0, 0.0, 300.0]
        
        # Add expression parameters if available
        if 'exp' in motion_params:
            exp_array = motion_params['exp'].flatten()
            flame_params["expression"] = [float(x) for x in exp_array[:min(10, len(exp_array))]]
        else:
            flame_params["expression"] = [0.0] * 10
        
        # Add shape parameters (may not be available)
        flame_params["shape"] = [0.0] * 10
        
        return flame_params
    
    except Exception as e:
        logger.error(f"Error extracting FLAME parameters: {e}")
        return None


def save_json(data: Dict[str, Any], output_path: str) -> bool:
    """Save data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Face data saved to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON data: {e}")
        return False


def save_image(image: np.ndarray, output_path: str) -> bool:
    """Save an image to file."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, image)
        logger.info(f"Image saved to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return False


def main():
    """Main function."""
    args = parse_arguments()
    
    # Check if analysis modules are available
    if not ANALYSIS_MODULES_AVAILABLE:
        logger.error("Required analysis modules not available. Exiting.")
        return 1
    
    # Load the image
    logger.info(f"Loading image: {args.input_image}")
    image = load_image(args.input_image)
    if image is None:
        return 1
    
    # Get original image dimensions
    original_size = [image.shape[1], image.shape[0]]  # [width, height]
    
    # Detect the largest face
    logger.info("Detecting face...")
    face = detect_face(image, args.predict_type)
    if face is None:
        logger.error("No face detected in the image")
        return 1
    
    # Extract facial landmarks
    logger.info("Extracting landmarks...")
    landmarks = extract_landmarks(image, face, args.predict_type)
    if landmarks is None or len(landmarks) == 0:
        logger.error("Failed to extract landmarks")
        return 1
    
    # Align and crop the face
    logger.info("Aligning face...")
    aligned_face, crop_params, landmarks_aligned = align_face(
        image, face, landmarks, 
        args.size, args.scale_factor, args.vertical_shift
    )
    if aligned_face is None:
        logger.error("Failed to align face")
        return 1
    
    # Generate face mask
    logger.info("Generating face mask...")
    face_mask = generate_face_mask(aligned_face, landmarks_aligned, args.size)
    if face_mask is None:
        logger.warning("Failed to generate face mask, proceeding without mask")
    
    # Extract FLAME parameters if requested
    flame_params = None
    if not args.no_flame:
        logger.info("Extracting FLAME parameters...")
        flame_params = extract_flame_parameters(aligned_face, args.predict_type)
        if flame_params is None:
            logger.warning("Failed to extract FLAME parameters, proceeding without them")
    
    # Save aligned face image
    logger.info(f"Saving aligned face to: {args.aligned_output}")
    if not save_image(aligned_face, args.aligned_output):
        logger.warning("Failed to save aligned face image")
    
    # Save face mask if available
    if face_mask is not None:
        logger.info(f"Saving face mask to: {args.mask_output}")
        if not save_image(face_mask, args.mask_output):
            logger.warning("Failed to save face mask image")
    
    # Prepare the output data
    output_data = {
        "original_image": os.path.basename(args.input_image),
        "original_size": original_size,
        "detected_bbox": {
            "x": int(face['bbox'][0]),
            "y": int(face['bbox'][1]),
            "width": int(face['bbox'][2] - face['bbox'][0]),
            "height": int(face['bbox'][3] - face['bbox'][1]),
            "score": float(face['confidence'])
        },
        "crop_parameters": crop_params,
        "aligned_face_image": os.path.basename(args.aligned_output),
        "landmarks_aligned": landmarks_aligned.tolist() if landmarks_aligned is not None else [],
        "landmarks_orig": landmarks.tolist() if landmarks is not None else []
    }
    
    if flame_params:
        output_data["flame_model"] = flame_params
    
    if face_mask is not None:
        output_data["face_mask"] = os.path.basename(args.mask_output)
    
    # Save the JSON data
    logger.info(f"Saving face data to: {args.output}")
    if not save_json(output_data, args.output):
        logger.error("Failed to save face data")
        return 1
    
    logger.info("Face data extraction completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
