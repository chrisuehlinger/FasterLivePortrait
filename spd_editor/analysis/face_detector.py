#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Detector module for SPD Editor.

This module provides a wrapper for FasterLivePortrait's face detection functionality.
"""

import os
import logging
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union, Any
import sys

# Add the parent directory to the path for importing FasterLivePortrait modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Wrapper for FasterLivePortrait's face detection functionality.
    
    This class provides methods to detect faces in images and return
    bounding boxes and confidence scores.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 predict_type: str = "trt",
                 det_thresh: float = 0.5,
                 nms_thresh: float = 0.4,
                 **kwargs):
        """
        Initialize the face detector.
        
        Args:
            model_path: Path to the face detection model file.
                If None, uses the default model path.
            predict_type: Type of prediction to use ('trt' for TensorRT, 'ort' for ONNX Runtime).
            det_thresh: Detection threshold for face confidence.
            nms_thresh: Non-maximum suppression threshold.
        """
        self.model_path = model_path
        self.predict_type = predict_type
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self.model = None
        self.initialized = False
        self.face_objects = []  # Store face objects for test access
        
        # Try to initialize model if possible
        self._initialize_model()
        
    def _initialize_model(self):
        """
        Initialize the face detection model.
        
        This is separated from __init__ to allow lazy loading.
        """
        try:
            from src.models.face_analysis_model import FaceAnalysisModel
            from src.models.predictor import get_predictor
            
            # If no model path provided, use default
            if self.model_path is None:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                model_paths = [
                    os.path.join(base_dir, "checkpoints/liveportrait_onnx/retinaface_det_static.trt"),
                    os.path.join(base_dir, "checkpoints/liveportrait_onnx/face_2dpose_106_static.trt")
                ]
                
                # Fall back to ONNX if TRT not available
                if not os.path.exists(model_paths[0]) and self.predict_type == "trt":
                    logger.warning("TensorRT model not found, falling back to ONNX")
                    model_paths = [
                        os.path.join(base_dir, "checkpoints/liveportrait_onnx/retinaface_det_static.onnx"),
                        os.path.join(base_dir, "checkpoints/liveportrait_onnx/face_2dpose_106_static.onnx")
                    ]
                    self.predict_type = "ort"
            else:
                # If custom model path is provided
                if isinstance(self.model_path, str):
                    model_paths = [self.model_path]
                else:
                    model_paths = self.model_path
            
            # Create model instance
            model_kwargs = {
                "predict_type": self.predict_type,
                "model_path": model_paths
            }
            self.model = FaceAnalysisModel(**model_kwargs)
            self.model.det_thresh = self.det_thresh
            self.model.nms_thresh = self.nms_thresh
            self.initialized = True
            logger.info(f"Face detector initialized successfully with {self.predict_type} backend")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image in BGR format (OpenCV format)
            
        Returns:
            List of dictionaries containing face detection results:
            [
                {
                    'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
                    'confidence': float,        # Detection confidence
                    'landmark': np.ndarray,     # Face landmarks if available
                },
                ...
            ]
        """
        if not self.initialized:
            self._initialize_model()
            
        if not self.initialized:
            logger.error("Face detector not initialized")
            return []
            
        try:
            # Ensure image is BGR (OpenCV format)
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                
            # Detect faces using the model
            landmarks_list = self.model.predict(image)
            
            # Store face objects for testing purposes
            if hasattr(self.model, 'face_objects'):
                self.face_objects = self.model.face_objects
            
            # Convert to standardized output format
            results = []
            if not landmarks_list:
                return results
                
            # Access internal face data from the model to get bounding boxes
            for i, landmark in enumerate(landmarks_list):
                # Try to get the underlying Face object that has bbox and confidence
                if hasattr(self.model, 'face_objects') and len(self.model.face_objects) > i:
                    face_obj = self.model.face_objects[i]
                    bbox = face_obj.bbox
                    confidence = face_obj.det_score
                else:
                    # If we don't have direct access to the Face objects,
                    # estimate a bounding box from landmarks
                    if landmark is not None and len(landmark) > 0:
                        x1 = np.min(landmark[:, 0])
                        y1 = np.min(landmark[:, 1])
                        x2 = np.max(landmark[:, 0])
                        y2 = np.max(landmark[:, 1])
                        bbox = np.array([x1, y1, x2, y2])
                        confidence = 1.0  # No confidence available
                    else:
                        continue
                        
                results.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'landmark': landmark
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
            
    def detect_largest_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect the largest face in an image.
        
        Args:
            image: Input image in BGR format (OpenCV format)
            
        Returns:
            Dictionary containing face detection results for the largest face,
            or None if no face is detected.
        """
        faces = self.detect(image)
        
        if not faces:
            return None
            
        # Find the face with the largest area
        largest_face = max(faces, key=lambda face: 
                          (face['bbox'][2] - face['bbox'][0]) * 
                          (face['bbox'][3] - face['bbox'][1]))
                          
        return largest_face