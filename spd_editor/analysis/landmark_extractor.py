#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Landmark Extractor module for SPD Editor.

This module provides a wrapper for FasterLivePortrait's facial landmark extraction functionality.
"""

import os
import logging
import numpy as np
import cv2
import sys
import importlib.util
from typing import List, Dict, Tuple, Optional, Union, Any

# Add the parent directory to the path for importing FasterLivePortrait modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if required modules are available
SRC_AVAILABLE = importlib.util.find_spec("src") is not None
if not SRC_AVAILABLE:
    logger.warning("'src' module not found. Landmark extraction will be limited.")

# Try to import MediaPipe as a fallback
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    logger.warning("MediaPipe not available for fallback landmark detection.")


class LandmarkExtractor:
    """
    Wrapper for FasterLivePortrait's facial landmark extraction functionality.
    
    This class provides methods to extract facial landmarks from detected faces.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 predict_type: str = "trt",
                 landmark_type: str = "2d",
                 crop_params: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize the landmark extractor.
        
        Args:
            model_path: Path to the landmark model file.
                If None, uses the default model path.
            predict_type: Type of prediction to use ('trt' for TensorRT, 'ort' for ONNX Runtime).
            landmark_type: Type of landmarks to extract ('2d' or '3d').
            crop_params: Dictionary of cropping parameters.
        """
        self.model_path = model_path
        self.predict_type = predict_type
        self.landmark_type = landmark_type
        self.crop_params = crop_params if crop_params is not None else {}
        self.model = None
        self.landmark_model = None
        self.mediapipe_model = None
        self.initialized = False
        self.mediapipe_available = MEDIAPIPE_AVAILABLE
        
        # Try to initialize model if possible
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the landmark extraction model.
        
        This is separated from __init__ to allow lazy loading.
        """
        if not SRC_AVAILABLE and not self.mediapipe_available:
            logger.error("Cannot initialize landmark extractor: neither 'src' nor 'mediapipe' are available")
            return
            
        try:
            if SRC_AVAILABLE:
                # Try to import FasterLivePortrait models
                from src.models.landmark_model import LandmarkModel
                
                # If no model path provided, use default
                if self.model_path is None:
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                    
                    if self.landmark_type == "2d":
                        # Use 2D landmark model
                        if self.predict_type == "trt":
                            model_path = os.path.join(base_dir, "checkpoints/liveportrait_onnx/face_2dpose_106_static.trt")
                            if not os.path.exists(model_path):
                                logger.warning("TensorRT model not found, falling back to ONNX")
                                model_path = os.path.join(base_dir, "checkpoints/liveportrait_onnx/face_2dpose_106_static.onnx")
                                self.predict_type = "ort"
                        else:
                            model_path = os.path.join(base_dir, "checkpoints/liveportrait_onnx/face_2dpose_106_static.onnx")
                    else:
                        # For 3D landmarks, could use different model
                        if self.predict_type == "trt":
                            model_path = os.path.join(base_dir, "checkpoints/liveportrait_onnx/face_2dpose_106_static.trt")
                            if not os.path.exists(model_path):
                                logger.warning("TensorRT model not found, falling back to ONNX")
                                model_path = os.path.join(base_dir, "checkpoints/liveportrait_onnx/face_2dpose_106_static.onnx")
                                self.predict_type = "ort"
                        else:
                            model_path = os.path.join(base_dir, "checkpoints/liveportrait_onnx/face_2dpose_106_static.onnx")
                else:
                    model_path = self.model_path
                
                # Check if model exists
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    # Will try MediaPipe below if available
                else:
                    # Try to create the landmark model
                    model_kwargs = {
                        "predict_type": self.predict_type,
                        "model_path": model_path,
                        "crop_params": self.crop_params, # Pass crop_params here
                    }
                    
                    try:
                        # Try to initialize the landmark model
                        self.landmark_model = LandmarkModel(**model_kwargs)
                        self.model = "landmark_model"
                        self.initialized = True
                        logger.info(f"Landmark extractor initialized with LandmarkModel using {self.predict_type} backend")
                        return
                    except Exception as e:
                        logger.warning(f"Failed to initialize LandmarkModel: {e}. Trying MediaPipe as fallback.")
            
            # Try MediaPipe as fallback if available
            if self.mediapipe_available:
                try:
                    # Using MediaPipe's FaceMesh for landmark extraction
                    mp_face_mesh = mp.solutions.face_mesh
                    self.mediapipe_model = mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    self.model = "mediapipe"
                    self.initialized = True
                    logger.info("Landmark extractor initialized with MediaPipe (fallback)")
                except Exception as e:
                    logger.error(f"Failed to initialize MediaPipe: {e}")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize landmark extractor: {e}")
    
    def extract_landmarks(self, 
                         image: np.ndarray, 
                         face_detection: Optional[Union[Dict[str, Any], np.ndarray]] = None) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from an image.
        
        Args:
            image: Input image (assumed BGR from cv2.imread)
            face_detection: Face detection result or pre-computed landmarks
                If dict, expected to have 'bbox' key with [x1, y1, x2, y2]
                If ndarray, expected to be initial landmarks for tracking
                If None, will attempt to detect a face first
                
        Returns:
            ndarray of landmark points or None if no landmarks detected
        """
        if not self.initialized:
            self._initialize_model()
            
        if not self.initialized:
            logger.error("Landmark extractor not initialized")
            return None
            
        try:
            # Handle different input formats for face_detection
            bbox = None
            initial_landmarks = None
            
            if face_detection is not None:
                if isinstance(face_detection, dict) and 'bbox' in face_detection:
                    # Face detection result from FaceDetector
                    bbox = face_detection['bbox']
                    if 'landmark' in face_detection:
                        initial_landmarks = face_detection['landmark']
                elif isinstance(face_detection, np.ndarray):
                    # Pre-computed landmarks (e.g., for tracking)
                    initial_landmarks = face_detection
            
            # Extract landmarks based on the available model
            if self.model == "landmark_model":
                # Ensure image is RGB for LandmarkModel
                if len(image.shape) == 3 and image.shape[2] == 3:
                    img_rgb_for_lmk_model = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif len(image.shape) == 2: # Grayscale
                    img_rgb_for_lmk_model = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb_for_lmk_model = image # Assume already suitable

                # If we have a LandmarkModel
                if initial_landmarks is not None:
                    # For tracking/refinement, use the initial landmarks
                    landmarks = self.landmark_model.predict(img_rgb_for_lmk_model, initial_landmarks)
                else:
                    # No initial landmarks, need to detect first (this path might be less common for spd_editor)
                    # This part might need FaceAnalysisModel which itself handles BGR->RGB
                    # For simplicity, assuming initial_landmarks are usually provided from FaceDetector
                    # which should now provide good landmarks due to its own BGR->RGB fix.
                    # If this path is taken, ensure FaceAnalysisModel is used correctly.
                    logger.warning("LandmarkExtractor called without initial_landmarks for LandmarkModel; "
                                   "this scenario might require FaceAnalysisModel for initial detection.")
                    # Fallback or error if direct prediction on full image without initial lmk is not desired
                    # For now, let's assume initial_landmarks are typically provided.
                    # If not, the behavior of self.landmark_model.predict(img_rgb_for_lmk_model)
                    # (i.e. LandmarkModel.input_process with only one arg) will do a simple resize.
                    landmarks = self.landmark_model.predict(img_rgb_for_lmk_model)

                return landmarks
                
            elif self.model == "mediapipe" and self.mediapipe_available:
                # If we're using MediaPipe as fallback
                # Convert to RGB for MediaPipe (already handled if image was BGR)
                if len(image.shape) == 3 and image.shape[2] == 3: # BGR
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif len(image.shape) == 2: # GRAY
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else: # Already RGB or other
                    rgb_image = image
                
                # Process the image to find face landmarks
                results = self.mediapipe_model.process(rgb_image)
                if not results.multi_face_landmarks:
                    return None
                
                # Convert MediaPipe landmarks to numpy array in image coordinates
                height, width = image.shape[:2]
                landmarks = []
                for landmark in results.multi_face_landmarks[0].landmark:
                    # Convert normalized coordinates to pixel coordinates
                    x = landmark.x * width
                    y = landmark.y * height
                    landmarks.append([x, y])
                
                return np.array(landmarks)
                
            logger.error("No valid landmark extraction method available")
            return None
            
        except Exception as e:
            logger.error(f"Error in landmark extraction: {e}")
            return None
    
    def extract_landmarks_batch(self, 
                              image: np.ndarray, 
                              face_detections: List[Dict[str, Any]]) -> List[Optional[np.ndarray]]:
        """
        Extract facial landmarks for multiple faces in an image.
        
        Args:
            image: Input image
            face_detections: List of face detection results
                
        Returns:
            List of landmark arrays (or None for faces where landmark extraction failed)
        """
        return [self.extract_landmarks(image, face_detection) 
                for face_detection in face_detections]
    
    def get_landmark_types(self) -> Dict[str, List[int]]:
        """
        Get information about available landmark types and their indices.
        
        Returns:
            Dictionary mapping landmark type names to their indices in the landmark array
        """
        # Standard landmark mappings for 106-point model
        landmark_types = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_tip': list(range(31, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'outer_lip': list(range(48, 60)),
            'inner_lip': list(range(60, 68)),
        }
        
        # Add extended landmarks if using 106-point model
        if self.model == "landmark_model":
            landmark_types.update({
                'face_contour_ext': list(range(68, 83)),
                'eyebrows_ext': list(range(83, 92)),
                'nose_ext': list(range(92, 106))
            })
            
        return landmark_types