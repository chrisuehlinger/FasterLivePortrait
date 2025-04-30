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
from typing import List, Dict, Tuple, Optional, Union, Any

# Add the parent directory to the path for importing FasterLivePortrait modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LandmarkExtractor:
    """
    Wrapper for FasterLivePortrait's facial landmark extraction functionality.
    
    This class provides methods to extract facial landmarks from detected faces.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 predict_type: str = "trt",
                 landmark_type: str = "2d",
                 **kwargs):
        """
        Initialize the landmark extractor.
        
        Args:
            model_path: Path to the landmark model file.
                If None, uses the default model path.
            predict_type: Type of prediction to use ('trt' for TensorRT, 'ort' for ONNX Runtime).
            landmark_type: Type of landmarks to extract ('2d' or '3d').
        """
        self.model_path = model_path
        self.predict_type = predict_type
        self.landmark_type = landmark_type
        self.model = None
        self.landmark_model = None
        self.initialized = False
        self.mediapipe_available = False
        
        # Try to initialize model if possible
        self._initialize_model()
        
    def _initialize_model(self):
        """
        Initialize the landmark extraction model.
        
        This is separated from __init__ to allow lazy loading.
        """
        try:
            # Try to import FasterLivePortrait models
            from src.models.landmark_model import LandmarkModel
            from src.models.mediapipe_face_model import MediaPipeFaceModel
            
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
            
            # Try to create the landmark model
            model_kwargs = {
                "predict_type": self.predict_type,
                "model_path": model_path,
            }
            
            try:
                # Try to initialize the landmark model
                self.landmark_model = LandmarkModel(**model_kwargs)
                self.model = "landmark_model"
                self.initialized = True
                logger.info(f"Landmark extractor initialized with LandmarkModel using {self.predict_type} backend")
            except Exception as e:
                logger.warning(f"Failed to initialize LandmarkModel: {e}. Trying MediaPipe as fallback.")
                
                # Try MediaPipe as fallback
                try:
                    import mediapipe as mp
                    self.mediapipe_model = MediaPipeFaceModel()
                    self.model = "mediapipe"
                    self.mediapipe_available = True
                    self.initialized = True
                    logger.info("Landmark extractor initialized with MediaPipe (fallback)")
                except ImportError:
                    logger.error("MediaPipe not available as fallback")
                    raise
            
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
            image: Input image (RGB format if using MediaPipe, BGR if using LandmarkModel)
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
                # If we have a LandmarkModel
                if initial_landmarks is not None:
                    # For tracking/refinement, use the initial landmarks
                    landmarks = self.landmark_model.predict(image, initial_landmarks)
                else:
                    # No initial landmarks, need to detect first
                    from src.models.face_analysis_model import FaceAnalysisModel
                    temp_detector = FaceAnalysisModel(
                        predict_type=self.predict_type,
                        model_path=[self.model_path]
                    )
                    faces = temp_detector.predict(image)
                    if not faces:
                        return None
                    landmarks = faces[0]  # Take the first face
                
                return landmarks
                
            elif self.model == "mediapipe" and self.mediapipe_available:
                # If we're using MediaPipe as fallback
                # Convert to RGB for MediaPipe
                if image.shape[2] == 3:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = image
                
                results = self.mediapipe_model.predict(rgb_image)
                if not results:
                    return None
                    
                return results[0]  # Return the first face's landmarks
            
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
    
    def get_landmark_types(self) -> Dict[str, int]:
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