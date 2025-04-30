#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Application Controller for FLPSF Editor
"""

import os
import sys
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Union, Any, Tuple

# Add parent directory to path for importing FasterLivePortrait modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from models.app_model import AppModel
from src.utils.flpsf import verify_flpsf_file, extract_src_infos

logger = logging.getLogger('flpsf_editor.controller')

class AppController:
    """
    Main application controller that handles user interactions
    """
    
    def __init__(self, model: AppModel):
        """
        Initialize the application controller
        
        Args:
            model (AppModel): Application model
        """
        self.model = model
        
    def open_file(self, file_path: str) -> bool:
        """
        Open a FLPSF file
        
        Args:
            file_path (str): Path to FLPSF file
            
        Returns:
            bool: True if file opened successfully, False otherwise
        """
        return self.model.load_flpsf(file_path)
        
    def save_file(self, file_path: str = None) -> bool:
        """
        Save current data to a FLPSF file
        
        Args:
            file_path (str, optional): Path to save the file. If None, uses current path.
            
        Returns:
            bool: True if file saved successfully, False otherwise
        """
        return self.model.save_flpsf(file_path)
        
    def load_image(self, image_path: str) -> bool:
        """
        Load a source image
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            bool: True if image loaded successfully, False otherwise
        """
        return self.model.load_image(image_path)
        
    def update_landmarks(self, landmarks: np.ndarray) -> bool:
        """
        Update facial landmarks
        
        Args:
            landmarks (np.ndarray): Array of landmark points
            
        Returns:
            bool: True if landmarks updated successfully, False otherwise
        """
        return self.model.set_landmarks(landmarks)
        
    def update_motion_param(self, param: str, value: np.ndarray) -> bool:
        """
        Update a motion parameter
        
        Args:
            param (str): Parameter name
            value (np.ndarray): Parameter value
            
        Returns:
            bool: True if parameter updated successfully, False otherwise
        """
        return self.model.set_motion_param(param, value)
        
    def update_derived_param(self, param: str, value: np.ndarray) -> bool:
        """
        Update a derived parameter
        
        Args:
            param (str): Parameter name
            value (np.ndarray): Parameter value
            
        Returns:
            bool: True if parameter updated successfully, False otherwise
        """
        return self.model.set_derived_param(param, value)
        
    def update_norm_param(self, param: str, value: Any) -> bool:
        """
        Update a normalization parameter
        
        Args:
            param (str): Parameter name
            value (Any): Parameter value
            
        Returns:
            bool: True if parameter updated successfully, False otherwise
        """
        return self.model.set_norm_param(param, value)
        
    def update_pasteback_info(self, param: str, value: np.ndarray) -> bool:
        """
        Update pasteback information
        
        Args:
            param (str): Parameter name
            value (np.ndarray): Parameter value
            
        Returns:
            bool: True if parameter updated successfully, False otherwise
        """
        return self.model.set_pasteback_info(param, value)
        
    def new_file(self) -> bool:
        """
        Create a new file
        
        Returns:
            bool: True if new file created successfully, False otherwise
        """
        self.model.reset()
        return True
        
    def undo(self) -> bool:
        """
        Undo the last action
        
        Returns:
            bool: True if undo successful, False otherwise
        """
        return self.model.undo()
        
    def redo(self) -> bool:
        """
        Redo the last undone action
        
        Returns:
            bool: True if redo successful, False otherwise
        """
        return self.model.redo()
        
    def detect_landmarks(self, model_type: str = 'insightface') -> bool:
        """
        Detect landmarks in the current image using specified model
        
        Args:
            model_type (str): Type of model to use ('insightface', 'xpose', etc.)
            
        Returns:
            bool: True if landmarks detected successfully, False otherwise
        """
        try:
            if self.model.image is None:
                logger.error("No image loaded")
                return False
                
            # Import necessary modules based on model type
            if model_type == 'insightface':
                from src.models.insightface_wrapper import InsightfaceWrapper
                from src.models.landmark_regressor import LandmarkRegressor
                
                # Get configuration
                face_model_path = self.model.config.get('models.face_detection.checkpoint')
                landmark_model_path = self.model.config.get('models.landmark_detection.checkpoint')
                
                # Load models
                face_model = InsightfaceWrapper(model_path=face_model_path)
                landmark_model = LandmarkRegressor(model_path=landmark_model_path)
                
                # Convert image to BGR for insightface
                img_bgr = cv2.cvtColor(self.model.image, cv2.COLOR_RGB2BGR)
                
                # Detect faces
                faces = face_model.predict(img_bgr)
                if len(faces) == 0:
                    logger.warning("No faces detected in image")
                    return False
                    
                # Get first face (or largest)
                face = faces[0]
                
                # Extract landmarks
                landmarks = landmark_model.predict(self.model.image, face)
                
                # Update model
                return self.model.set_landmarks(landmarks)
                
            elif model_type == 'xpose':
                # For animal faces using XPose
                from src.utils.animal_landmark_runner import XPoseRunner
                
                # Get configuration
                xpose_model_path = self.model.config.get('models.animal_detection.checkpoint')
                
                # Load model
                xpose_runner = XPoseRunner(
                    model_config_path="models/XPose/config_model/UniPose_SwinT.py",
                    model_checkpoint_path=xpose_model_path,
                    flag_use_half_precision=True
                )
                
                # Convert to PIL image for XPose
                from PIL import Image
                img_pil = Image.fromarray(self.model.image)
                
                # Detect landmarks
                landmarks = xpose_runner.run(img_pil, 'face', 'animal_face', 0, 0)
                
                if landmarks is None:
                    logger.warning("No animal face detected in image")
                    return False
                    
                # Update model
                return self.model.set_landmarks(landmarks)
                
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error detecting landmarks: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def compute_motion_parameters(self) -> bool:
        """
        Compute motion parameters from landmarks
        
        Returns:
            bool: True if parameters computed successfully, False otherwise
        """
        try:
            if self.model.image is None:
                logger.error("No image loaded")
                return False
                
            if self.model.landmarks is None:
                logger.error("No landmarks available")
                return False
                
            # Import motion extraction model
            from src.models.motion_extractor import MotionExtractor
            
            # Get configuration
            model_path = self.model.config.get('models.motion_extraction.checkpoint')
            
            # Create a 256x256 version of the image for the model
            img_256 = cv2.resize(self.model.image, (256, 256), interpolation=cv2.INTER_AREA)
            
            # Load model
            motion_extractor = MotionExtractor(model_path=model_path)
            
            # Extract motion parameters
            pitch, yaw, roll, t, exp, scale, kp = motion_extractor.predict(img_256)
            
            # Update model
            success = True
            success &= self.model.set_motion_param('pitch', pitch)
            success &= self.model.set_motion_param('yaw', yaw)
            success &= self.model.set_motion_param('roll', roll)
            success &= self.model.set_motion_param('t', t)
            success &= self.model.set_motion_param('exp', exp)
            success &= self.model.set_motion_param('scale', scale)
            success &= self.model.set_motion_param('kp', kp)
            
            return success
            
        except Exception as e:
            logger.error(f"Error computing motion parameters: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def compute_appearance_features(self) -> bool:
        """
        Compute appearance features from the image
        
        Returns:
            bool: True if features computed successfully, False otherwise
        """
        try:
            if self.model.image is None:
                logger.error("No image loaded")
                return False
                
            # Import appearance feature extractor
            from src.models.appearance_feature_extractor import AppearanceFeatureExtractor
            
            # Get configuration
            model_path = self.model.config.get('models.appearance_extraction.checkpoint')
            
            # Create a 256x256 version of the image for the model
            img_256 = cv2.resize(self.model.image, (256, 256), interpolation=cv2.INTER_AREA)
            
            # Load model
            app_feat_extractor = AppearanceFeatureExtractor(model_path=model_path)
            
            # Extract features
            f_s = app_feat_extractor.predict(img_256)
            
            # Update model
            return self.model.set_derived_param('f_s', f_s)
            
        except Exception as e:
            logger.error(f"Error computing appearance features: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def compute_derived_parameters(self) -> bool:
        """
        Compute derived parameters from motion parameters
        
        Returns:
            bool: True if parameters computed successfully, False otherwise
        """
        try:
            if self.model.motion_params['pitch'] is None or \
               self.model.motion_params['yaw'] is None or \
               self.model.motion_params['roll'] is None:
                logger.error("Motion parameters not available")
                return False
                
            # Import util functions
            from src.utils.utils import get_rotation_matrix, transform_keypoint
            
            # Get motion parameters
            pitch = self.model.motion_params['pitch']
            yaw = self.model.motion_params['yaw']
            roll = self.model.motion_params['roll']
            t = self.model.motion_params['t']
            exp = self.model.motion_params['exp']
            scale = self.model.motion_params['scale']
            kp = self.model.motion_params['kp']
            
            # Compute rotation matrix
            R_s = get_rotation_matrix(pitch, yaw, roll)
            
            # Compute transformed keypoints
            x_s = transform_keypoint(pitch, yaw, roll, t, exp, scale, kp)
            
            # Set derived parameters
            success = True
            success &= self.model.set_derived_param('R_s', R_s)
            success &= self.model.set_derived_param('x_s', x_s)
            success &= self.model.set_derived_param('x_c_s', kp.copy())
            
            return success
            
        except Exception as e:
            logger.error(f"Error computing derived parameters: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def generate_pasteback_mask(self) -> bool:
        """
        Generate pasteback mask for the image
        
        Returns:
            bool: True if mask generated successfully, False otherwise
        """
        try:
            if self.model.image is None:
                logger.error("No image loaded")
                return False
                
            # Import util functions
            from src.utils.utils import prepare_paste_back
            
            # Get mask template
            mask_crop_path = os.path.join(parent_dir, "assets", "mask_template.png")
            if not os.path.exists(mask_crop_path):
                logger.error(f"Mask template not found: {mask_crop_path}")
                return False
                
            mask_crop = cv2.imread(mask_crop_path, cv2.IMREAD_COLOR)
            
            # Create a simple transformation matrix if not available
            if self.model.pasteback_info['M'] is None:
                # Default identity transform
                M = np.array([[1.0, 0.0, 0.0], 
                              [0.0, 1.0, 0.0], 
                              [0.0, 0.0, 1.0]], dtype=np.float32)
                self.model.set_pasteback_info('M', M)
            else:
                M = self.model.pasteback_info['M']
            
            # Prepare mask
            h, w = self.model.image.shape[:2]
            mask_ori_float = prepare_paste_back(mask_crop, M, dsize=(w, h))
            
            # Update model
            return self.model.set_pasteback_info('mask_ori_float', mask_ori_float)
            
        except Exception as e:
            logger.error(f"Error generating pasteback mask: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def preview_animation(self, driving_path: str) -> Optional[np.ndarray]:
        """
        Preview animation using a driving video frame
        
        Args:
            driving_path (str): Path to driving video or frame
            
        Returns:
            np.ndarray or None: Animated frame if successful, None otherwise
        """
        try:
            # TODO: Implement preview animation
            logger.warning("Preview animation not implemented yet")
            return None
            
        except Exception as e:
            logger.error(f"Error previewing animation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def validate_flpsf(self, file_path: str = None) -> Dict:
        """
        Validate FLPSF file or current model data for pipeline compatibility
        
        This performs a comprehensive validation to ensure the FLPSF file
        contains all required data in the correct format for the
        FasterLivePortrait animation pipeline.
        
        Args:
            file_path (str, optional): Path to FLPSF file. If None, validates current model data.
            
        Returns:
            Dict: Dictionary containing validation results with fields:
                - valid (bool): Whether the data is valid
                - errors (List[str]): List of error messages
                - warnings (List[str]): List of warning messages
                - missing_fields (List[str]): List of missing required fields
                - invalid_fields (List[str]): List of fields with invalid format/shape
                - compatibility (str): One of "full", "partial", or "incompatible"
        """
        results = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'missing_fields': [],
            'invalid_fields': [],
            'compatibility': 'incompatible'
        }
        
        try:
            # If file path is provided, validate the file
            if file_path:
                # Use built-in verification function
                basic_verification = verify_flpsf_file(file_path)
                results['valid'] = basic_verification['valid']
                results['errors'].extend(basic_verification['errors'])
                results['warnings'].extend(basic_verification['warnings'])
                
                # Early return if basic verification failed
                if not basic_verification['valid']:
                    return results
                
                # Load the file data
                from src.utils.flpsf import load_flpsf
                data = load_flpsf(file_path)
                if data is None:
                    results['errors'].append("Failed to load FLPSF file")
                    return results
            else:
                # Use current model data
                data = {}
                
                # Copy image information
                data['image_path'] = self.model.image_path
                data['image_dimensions'] = self.model.image_dimensions
                
                # Copy landmarks
                data['landmarks'] = self.model.landmarks
                
                # Copy motion parameters
                for param, value in self.model.motion_params.items():
                    data[param] = value
                
                # Copy derived parameters
                for param, value in self.model.derived_params.items():
                    data[param] = value
                
                # Copy normalization parameters
                for param, value in self.model.norm_params.items():
                    data[param] = value
                
                # Copy pasteback information
                for param, value in self.model.pasteback_info.items():
                    data[param] = value
            
            # Define required fields with their expected shapes
            required_fields = {
                'landmarks': ('ndarray', (-1, 2)),  # Variable number of landmarks, 2D points
                'pitch': ('ndarray', (1, 1)),
                'yaw': ('ndarray', (1, 1)),
                'roll': ('ndarray', (1, 1)),
                't': ('ndarray', (1, 3)),
                'exp': ('ndarray', (1, 21, 3)),  # 21 expression parameters, 3D
                'scale': ('ndarray', (1, 1)),
                'kp': ('ndarray', (1, 20, 3)),  # 20 keypoints, 3D
                'R_s': ('ndarray', (1, 3, 3)),  # 3x3 rotation matrix
                'f_s': ('ndarray', (1, 256)),  # 256-dim feature vector
                'x_s': ('ndarray', (1, 20, 3)),  # 20 transformed keypoints, 3D
                'x_c_s': ('ndarray', (1, 20, 3)),  # 20 canonical keypoints, 3D
            }
            
            # Optional fields with expected shapes
            optional_fields = {
                'lip_delta_before_animation': ('ndarray', (-1,)),
                'flag_lip_zero': ('bool', None),
                'mask_ori_float': ('ndarray', (-1, -1, -1)),  # Height x Width x Channels
                'M': ('ndarray', (3, 3)),  # 3x3 transformation matrix
                'image_path': ('str', None),
                'image_dimensions': ('ndarray', (2,)),
                'is_animal': ('bool', None),
            }
            
            # Check required fields
            for field, (field_type, expected_shape) in required_fields.items():
                # Check if field exists
                if field not in data or data[field] is None:
                    results['missing_fields'].append(field)
                    continue
                
                # Check field type
                if field_type == 'ndarray' and not isinstance(data[field], np.ndarray):
                    results['invalid_fields'].append(f"{field} (expected {field_type}, got {type(data[field]).__name__})")
                    continue
                
                # Check shape for arrays
                if field_type == 'ndarray' and expected_shape is not None:
                    value_shape = data[field].shape
                    shape_mismatch = False
                    
                    if len(value_shape) != len(expected_shape):
                        shape_mismatch = True
                    else:
                        for i, dim in enumerate(expected_shape):
                            if dim != -1 and dim != value_shape[i]:
                                shape_mismatch = True
                                break
                                
                    if shape_mismatch:
                        results['invalid_fields'].append(f"{field} (expected shape {expected_shape}, got {value_shape})")
            
            # Check optional fields
            for field, (field_type, expected_shape) in optional_fields.items():
                # Skip if field doesn't exist
                if field not in data or data[field] is None:
                    continue
                
                # Check field type
                if field_type == 'ndarray' and not isinstance(data[field], np.ndarray):
                    results['warnings'].append(f"{field} has incorrect type (expected {field_type}, got {type(data[field]).__name__})")
                    continue
                    
                # Check shape for arrays
                if field_type == 'ndarray' and expected_shape is not None:
                    value_shape = data[field].shape
                    shape_mismatch = False
                    
                    if len(value_shape) != len(expected_shape):
                        shape_mismatch = True
                    else:
                        for i, dim in enumerate(expected_shape):
                            if dim != -1 and dim != value_shape[i]:
                                shape_mismatch = True
                                break
                                
                    if shape_mismatch:
                        results['warnings'].append(f"{field} has incorrect shape (expected {expected_shape}, got {value_shape})")
            
            # Try to extract src_infos to verify compatibility with pipeline
            try:
                from src.utils.flpsf import extract_src_infos
                src_infos = extract_src_infos(data)
                if src_infos is None:
                    results['errors'].append("Failed to extract src_infos structure from data")
                else:
                    # Check if src_infos has correct structure for pipeline
                    if len(src_infos) > 0 and len(src_infos[0]) > 0 and len(src_infos[0][0]) >= 10:
                        # Check key elements
                        first_face_info = src_infos[0][0]
                        if not isinstance(first_face_info[0], dict):
                            results['errors'].append("Invalid src_infos structure: first element should be a dictionary")
                        else:
                            x_s_info = first_face_info[0]
                            required_keys = ['pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp']
                            missing_keys = [k for k in required_keys if k not in x_s_info or x_s_info[k] is None]
                            if missing_keys:
                                results['errors'].append(f"x_s_info missing keys: {', '.join(missing_keys)}")
                    else:
                        results['errors'].append(f"Invalid src_infos structure: expected nested list with at least 10 elements")
                        
            except Exception as e:
                results['errors'].append(f"Error checking src_infos compatibility: {str(e)}")
            
            # Determine final validation result
            results['valid'] = len(results['errors']) == 0 and len(results['missing_fields']) == 0 and len(results['invalid_fields']) == 0
            
            # Determine compatibility level
            if results['valid']:
                results['compatibility'] = 'full'
            elif len(results['missing_fields']) <= 3 and len(results['invalid_fields']) == 0:
                results['compatibility'] = 'partial'
            else:
                results['compatibility'] = 'incompatible'
                
            return results
            
        except Exception as e:
            logger.error(f"Error validating FLPSF: {str(e)}")
            import traceback
            traceback.print_exc()
            results['errors'].append(f"Validation error: {str(e)}")
            return results
            
    def complete_flpsf_data(self) -> bool:
        """
        Attempt to auto-complete missing FLPSF data using models
        
        This method checks which data is missing and attempts to fill
        it using the appropriate models and computations.
        
        Returns:
            bool: True if data was completed successfully, False otherwise
        """
        try:
            if self.model.image is None:
                logger.error("No image loaded")
                return False
            
            # Check what data is missing
            validation = self.validate_flpsf()
            
            # Track overall success
            success = True
            
            # If landmarks are missing, detect them
            if 'landmarks' in validation['missing_fields'] and self.model.image is not None:
                is_animal = self.model.flpsf_data.get('is_animal', False)
                model_type = 'xpose' if is_animal else 'insightface'
                landmark_success = self.detect_landmarks(model_type)
                success &= landmark_success
                logger.info(f"Landmark detection {'succeeded' if landmark_success else 'failed'}")
            
            # If motion parameters are missing and landmarks exist, compute them
            motion_params = ['pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp']
            missing_motion = [p for p in motion_params if p in validation['missing_fields']]
            if missing_motion and self.model.landmarks is not None:
                motion_success = self.compute_motion_parameters()
                success &= motion_success
                logger.info(f"Motion parameter computation {'succeeded' if motion_success else 'failed'}")
            
            # If derived parameters are missing and motion parameters exist, compute them
            derived_params = ['R_s', 'x_s', 'x_c_s']
            missing_derived = [p for p in derived_params if p in validation['missing_fields']]
            if missing_derived and all(self.model.motion_params[p] is not None for p in motion_params):
                derived_success = self.compute_derived_parameters()
                success &= derived_success
                logger.info(f"Derived parameter computation {'succeeded' if derived_success else 'failed'}")
            
            # If appearance features are missing, compute them
            if 'f_s' in validation['missing_fields'] and self.model.image is not None:
                appearance_success = self.compute_appearance_features()
                success &= appearance_success
                logger.info(f"Appearance feature computation {'succeeded' if appearance_success else 'failed'}")
            
            # If pasteback mask is missing, generate it
            if 'mask_ori_float' in validation['missing_fields']:
                mask_success = self.generate_pasteback_mask()
                success &= mask_success
                logger.info(f"Pasteback mask generation {'succeeded' if mask_success else 'failed'}")
            
            # Return overall success
            return success
            
        except Exception as e:
            logger.error(f"Error completing FLPSF data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False