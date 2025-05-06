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
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the path for importing FasterLivePortrait modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Check if required modules are available
SRC_AVAILABLE = importlib.util.find_spec("src") is not None
if not SRC_AVAILABLE:
    logger.warning("'src' module not found. Using OpenCV fallback for face detection.")


class FaceDetector:
    """
    Wrapper for facial detection functionality with fallbacks.
    
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
        self.model_type = None  # Type of detection model being used
        self.opencv_face_detector = None
        self.opencv_landmark_detector = None
        
        # Try to initialize model if possible
        self._initialize_model()
        
    def _initialize_model(self):
        """
        Initialize the face detection model.
        
        This is separated from __init__ to allow lazy loading.
        """
        # Try to initialize FasterLivePortrait model if available
        if SRC_AVAILABLE:
            try:
                from src.models.face_analysis_model import FaceAnalysisModel
                
                # If no model path provided, use default
                if self.model_path is None:
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                    model_paths = [
                        os.path.join(base_dir, "checkpoints/liveportrait_onnx/retinaface_det_static.onnx"),
                        os.path.join(base_dir, "checkpoints/liveportrait_onnx/face_2dpose_106_static.onnx")
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
                
                # Check if models exist
                if not os.path.exists(model_paths[0]):
                    logger.error(f"Model file not found: {model_paths[0]}")
                    # Will fall back to OpenCV
                else:
                    # Create model instance
                    model_kwargs = {
                        "predict_type": self.predict_type,
                        "model_path": model_paths
                    }
                    self.model = FaceAnalysisModel(**model_kwargs)
                    self.model.det_thresh = self.det_thresh
                    self.model.nms_thresh = self.nms_thresh
                    self.initialized = True
                    self.model_type = "face_analysis_model"
                    logger.info(f"Face detector initialized successfully with {self.predict_type} backend")
                    return
            
            except ImportError as e:
                logger.error(f"Failed to import required modules: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize face detector: {e}")
        
        # Fall back to OpenCV's built-in face detection
        try:
            # Initialize OpenCV's built-in face detector
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            if os.path.exists(face_cascade_path):
                self.opencv_face_detector = cv2.CascadeClassifier(face_cascade_path)
                
                # Try to initialize facial landmark detector if available
                landmark_model_path = cv2.data.haarcascades + '../lbpcascades/lbpcascade_frontalface.xml'
                if os.path.exists(landmark_model_path):
                    self.opencv_landmark_detector = cv2.face.createFacemarkLBF()
                    self.opencv_landmark_detector.loadModel(landmark_model_path)
                
                self.initialized = True
                self.model_type = "opencv"
                logger.info("Face detector initialized with OpenCV fallback")
            else:
                logger.error(f"OpenCV Haar cascade file not found: {face_cascade_path}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV face detector: {e}")
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image in BGR format (OpenCV format)
            
        Returns:
            List of dictionaries containing face detection results
        """
        if not self.initialized:
            self._initialize_model()
            
        if not self.initialized:
            logger.error("Face detector not initialized")
            return []

        results: List[Dict[str, Any]] = []
            
        try:
            # Ensure image is BGR (OpenCV format)
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                
            if self.model_type == "face_analysis_model":
                # Detect faces using the FasterLivePortrait model
                landmarks_list_internal = self.model.predict(image)
                
                # Store face objects for testing purposes
                if hasattr(self.model, 'face_objects'):
                    self.face_objects = self.model.face_objects
                
                if landmarks_list_internal: # Only proceed if faces were detected
                    for i, landmark_points in enumerate(landmarks_list_internal):
                        # Try to get the underlying Face object that has bbox and confidence
                        if hasattr(self.model, 'face_objects') and self.model.face_objects and i < len(self.model.face_objects):
                            face_obj = self.model.face_objects[i]
                            bbox = face_obj.bbox
                            confidence = face_obj.det_score
                        else:
                            # If we don't have direct access to the Face objects,
                            # estimate a bounding box from landmarks
                            if landmark_points is not None and len(landmark_points) > 0:
                                x1 = np.min(landmark_points[:, 0])
                                y1 = np.min(landmark_points[:, 1])
                                x2 = np.max(landmark_points[:, 0])
                                y2 = np.max(landmark_points[:, 1])
                                bbox = np.array([x1, y1, x2, y2])
                                confidence = 1.0  # No confidence available
                            else:
                                continue # Skip if landmark_points is None or empty
                                
                        results.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'landmark': landmark_points
                        })
                
            elif self.model_type == "opencv":
                # Detect faces using OpenCV
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_rects_cv = self.opencv_face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                for (x, y, w, h) in face_rects_cv:
                    bbox = np.array([x, y, x + w, y + h])
                    landmark_points_cv = None
                    if self.opencv_landmark_detector:
                        # OpenCV landmark detector expects a list of rects
                        # The fit method expects rects as list of tuples/np.arrays [(x,y,w,h), ...]
                        face_roi_for_lmk = np.array([[x, y, w, h]], dtype=np.int32)
                        try:
                            ok, landmarks_fit = self.opencv_landmark_detector.fit(gray, face_roi_for_lmk)
                            if ok and landmarks_fit is not None and len(landmarks_fit) > 0:
                                landmark_points_cv = landmarks_fit[0][0] # Get landmarks for the current face
                        except cv2.error as e:
                            logger.warning(f"OpenCV landmark fitting failed: {e}")


                    results.append({
                        'bbox': bbox,
                        'confidence': 1.0,  # Cascade classifiers don't give confidence directly
                        'landmark': landmark_points_cv # This might be None
                    })
            return results
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return [] # Return empty list on error

    def detect_largest_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detects all faces in an image and returns the largest one.
        
        Args:
            image: Input image in BGR format (OpenCV format).
            
        Returns:
            A dictionary containing the largest face's detection results (bbox, confidence, landmark),
            or None if no faces are detected.
        """
        faces = self.detect(image)
        
        if not faces:
            return None
            
        # Filter for faces that have a valid bounding box for area calculation
        valid_faces = []
        for f in faces:
            if 'bbox' in f and f['bbox'] is not None and len(f['bbox']) == 4:
                # Ensure bbox elements are numbers for area calculation
                try:
                    # Check if width and height are positive
                    width = float(f['bbox'][2]) - float(f['bbox'][0])
                    height = float(f['bbox'][3]) - float(f['bbox'][1])
                    if width > 0 and height > 0:
                        valid_faces.append(f)
                except (TypeError, ValueError):
                    logger.warning(f"Invalid bbox format for area calculation: {f['bbox']}")
                    continue
        
        if not valid_faces:
            return None
            
        # Find the face with the largest area
        largest_face = max(valid_faces, key=lambda face: 
                          (face['bbox'][2] - face['bbox'][0]) * 
                          (face['bbox'][3] - face['bbox'][1]))
                          
        return largest_face