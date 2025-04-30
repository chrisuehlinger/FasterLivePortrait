#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature Extractor module for SPD Editor.

This module provides a wrapper for FasterLivePortrait's feature extraction functionality,
including appearance features and motion parameters.
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


class FeatureExtractor:
    """
    Wrapper for FasterLivePortrait's feature extraction functionality.
    
    This class provides methods to extract appearance features and motion
    parameters from facial images.
    """
    
    def __init__(self, 
                 app_feat_model_path: Optional[str] = None,
                 motion_model_path: Optional[str] = None,
                 predict_type: str = "trt",
                 **kwargs):
        """
        Initialize the feature extractor.
        
        Args:
            app_feat_model_path: Path to the appearance feature extraction model.
                If None, uses the default model path.
            motion_model_path: Path to the motion extraction model.
                If None, uses the default model path.
            predict_type: Type of prediction to use ('trt' for TensorRT, 'ort' for ONNX Runtime).
        """
        self.app_feat_model_path = app_feat_model_path
        self.motion_model_path = motion_model_path
        self.predict_type = predict_type
        self.app_feat_extractor = None
        self.motion_extractor = None
        self.initialized_app_feat = False
        self.initialized_motion = False
        
        # Try to initialize models if possible
        self._initialize_models()
        
    def _initialize_models(self):
        """
        Initialize the feature extraction models.
        
        This is separated from __init__ to allow lazy loading.
        """
        self._initialize_app_feat_extractor()
        self._initialize_motion_extractor()
    
    def _initialize_app_feat_extractor(self):
        """
        Initialize the appearance feature extraction model.
        """
        try:
            # Try to import FasterLivePortrait models
            from src.models.appearance_feature_extractor_model import AppearanceFeatureExtractorModel
            
            # If no model path provided, use default
            if self.app_feat_model_path is None:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                
                if self.predict_type == "trt":
                    model_path = os.path.join(base_dir, "checkpoints/liveportrait_onnx/appearance_feature_extractor_static.trt")
                    if not os.path.exists(model_path):
                        logger.warning("TensorRT appearance model not found, falling back to ONNX")
                        model_path = os.path.join(base_dir, "checkpoints/liveportrait_onnx/appearance_feature_extractor_static.onnx")
                        app_feat_predict_type = "ort"
                    else:
                        app_feat_predict_type = self.predict_type
                else:
                    model_path = os.path.join(base_dir, "checkpoints/liveportrait_onnx/appearance_feature_extractor_static.onnx")
                    app_feat_predict_type = self.predict_type
            else:
                model_path = self.app_feat_model_path
                app_feat_predict_type = self.predict_type
            
            # Create model instance
            model_kwargs = {
                "predict_type": app_feat_predict_type,
                "model_path": model_path
            }
            
            self.app_feat_extractor = AppearanceFeatureExtractorModel(**model_kwargs)
            self.initialized_app_feat = True
            logger.info(f"Appearance feature extractor initialized with {app_feat_predict_type} backend")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules for appearance feature extraction: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize appearance feature extractor: {e}")

    def _initialize_motion_extractor(self):
        """
        Initialize the motion extraction model.
        """
        try:
            # Try to import FasterLivePortrait models
            from src.models.motion_extractor_model import MotionExtractorModel
            
            # If no model path provided, use default
            if self.motion_model_path is None:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                
                if self.predict_type == "trt":
                    model_path = os.path.join(base_dir, "checkpoints/liveportrait_onnx/motion_extractor_static.trt")
                    if not os.path.exists(model_path):
                        logger.warning("TensorRT motion model not found, falling back to ONNX")
                        model_path = os.path.join(base_dir, "checkpoints/liveportrait_onnx/motion_extractor_static.onnx")
                        motion_predict_type = "ort"
                    else:
                        motion_predict_type = self.predict_type
                else:
                    model_path = os.path.join(base_dir, "checkpoints/liveportrait_onnx/motion_extractor_static.onnx")
                    motion_predict_type = self.predict_type
            else:
                model_path = self.motion_model_path
                motion_predict_type = self.predict_type
            
            # Create model instance
            model_kwargs = {
                "predict_type": motion_predict_type,
                "model_path": model_path
            }
            
            self.motion_extractor = MotionExtractorModel(**model_kwargs)
            self.initialized_motion = True
            logger.info(f"Motion extractor initialized with {motion_predict_type} backend")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules for motion extraction: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize motion extractor: {e}")
    
    def extract_appearance_features(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract appearance features from a face image.
        
        Args:
            face_image: Face image (expected to be cropped and aligned to 256x256)
                
        Returns:
            ndarray of appearance features or None if extraction failed
        """
        if not self.initialized_app_feat:
            self._initialize_app_feat_extractor()
            
        if not self.initialized_app_feat:
            logger.error("Appearance feature extractor not initialized")
            return None
            
        try:
            # Ensure image is in the right format (RGB)
            if len(face_image.shape) == 2:  # Grayscale
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 4:  # RGBA
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
            elif face_image.shape[2] == 3 and face_image.dtype == np.uint8:
                # If BGR, convert to RGB
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                
            # Resize to 256x256 if needed
            if face_image.shape[0] != 256 or face_image.shape[1] != 256:
                face_image = cv2.resize(face_image, (256, 256))
                
            # Extract features
            features = self.app_feat_extractor.predict(face_image)
            return features
            
        except Exception as e:
            logger.error(f"Error in appearance feature extraction: {e}")
            return None
    
    def extract_motion_parameters(self, face_image: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract motion parameters from a face image.
        
        Args:
            face_image: Face image (expected to be cropped and aligned to 256x256)
                
        Returns:
            Dictionary containing motion parameters:
            {
                'pitch': float,
                'yaw': float,
                'roll': float,
                't': ndarray,  # translation parameters
                'exp': ndarray,  # expression parameters
                'scale': float,
                'kp': ndarray,  # keypoints
            }
            or None if extraction failed
        """
        if not self.initialized_motion:
            self._initialize_motion_extractor()
            
        if not self.initialized_motion:
            logger.error("Motion extractor not initialized")
            return None
            
        try:
            # Ensure image is in the right format (RGB)
            if len(face_image.shape) == 2:  # Grayscale
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 4:  # RGBA
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
            elif face_image.shape[2] == 3 and face_image.dtype == np.uint8:
                # If BGR, convert to RGB
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                
            # Resize to 256x256 if needed
            if face_image.shape[0] != 256 or face_image.shape[1] != 256:
                face_image = cv2.resize(face_image, (256, 256))
                
            # Extract motion parameters
            pitch, yaw, roll, t, exp, scale, kp = self.motion_extractor.predict(face_image)
            
            # Return as dictionary for better clarity
            return {
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                't': t,
                'exp': exp,
                'scale': scale,
                'kp': kp
            }
            
        except Exception as e:
            logger.error(f"Error in motion parameter extraction: {e}")
            return None
    
    def compute_transformation_matrix(self, 
                                     pitch: float, 
                                     yaw: float, 
                                     roll: float) -> Optional[np.ndarray]:
        """
        Compute a transformation matrix from pose angles.
        
        Args:
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
            roll: Roll angle in radians
                
        Returns:
            4x4 transformation matrix or None if computation failed
        """
        try:
            from src.utils.geometry import get_rotation_matrix
            
            # Compute rotation matrix
            rotation_matrix = get_rotation_matrix(pitch, yaw, roll)
            return rotation_matrix
            
        except ImportError:
            # Fallback implementation if the original function is not available
            try:
                # Create rotation matrices for each axis
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]
                ])
                
                Ry = np.array([
                    [np.cos(yaw), 0, np.sin(yaw)],
                    [0, 1, 0],
                    [-np.sin(yaw), 0, np.cos(yaw)]
                ])
                
                Rz = np.array([
                    [np.cos(roll), -np.sin(roll), 0],
                    [np.sin(roll), np.cos(roll), 0],
                    [0, 0, 1]
                ])
                
                # Combine rotations
                R = Rz @ Ry @ Rx
                
                # Create full 4x4 transformation matrix
                T = np.eye(4)
                T[:3, :3] = R
                
                return T
                
            except Exception as e:
                logger.error(f"Error in transformation matrix computation: {e}")
                return None
        except Exception as e:
            logger.error(f"Error in transformation matrix computation: {e}")
            return None
            
    def transform_keypoints(self, 
                           pitch: float,
                           yaw: float,
                           roll: float,
                           t: np.ndarray,
                           exp: np.ndarray,
                           scale: Union[float, np.ndarray],
                           kp: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform 3D keypoints using motion parameters.
        
        Args:
            pitch: Pitch angle
            yaw: Yaw angle
            roll: Roll angle
            t: Translation vector
            exp: Expression parameters
            scale: Scale factor
            kp: Original keypoints
                
        Returns:
            Transformed keypoints or None if transformation failed
        """
        try:
            from src.utils.geometry import transform_keypoint
            
            # Transform keypoints using the original function
            transformed_kp = transform_keypoint(pitch, yaw, roll, t, exp, scale, kp)
            return transformed_kp
            
        except ImportError:
            logger.error("Could not import transform_keypoint function")
            return None
        except Exception as e:
            logger.error(f"Error in keypoint transformation: {e}")
            return None