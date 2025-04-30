#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Application Model for FLPSF Editor
"""

import os
import sys
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image
import torch

# Add parent directory to path for importing FasterLivePortrait modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from src.utils.flpsf import load_flpsf, save_flpsf, verify_flpsf_file
from utils.config import Config

logger = logging.getLogger('flpsf_editor.model')

class AppModel:
    """
    Main application model that manages FLPSF data and operations
    """
    
    def __init__(self, config: Config):
        """
        Initialize the application model
        
        Args:
            config (Config): Application configuration
        """
        self.config = config
        self.views = []
        
        # FLPSF data and metadata
        self.flpsf_data = {}
        self.flpsf_path = None
        self.is_modified = False
        
        # Source image data
        self.image_path = None
        self.image = None
        self.image_hash = None
        self.image_dimensions = None
        
        # Landmarks
        self.landmarks = None
        
        # Motion parameters
        self.motion_params = {
            'pitch': None,
            'yaw': None,
            'roll': None,
            't': None,
            'exp': None,
            'scale': None,
            'kp': None,
        }
        
        # Derived parameters
        self.derived_params = {
            'R_s': None,
            'f_s': None,
            'x_s': None,
            'x_c_s': None,
        }
        
        # Normalization parameters
        self.norm_params = {
            'lip_delta_before_animation': None,
            'flag_lip_zero': False,
            'eye_delta_before_animation': None,
        }
        
        # Pasteback information
        self.pasteback_info = {
            'mask_ori_float': None,
            'M': None,
        }
        
        # Undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 50
        
    def register_view(self, view):
        """
        Register a view to be notified of model changes
        
        Args:
            view: View to register
        """
        if view not in self.views:
            self.views.append(view)
            
    def unregister_view(self, view):
        """
        Unregister a view
        
        Args:
            view: View to unregister
        """
        if view in self.views:
            self.views.remove(view)
            
    def notify_views(self, event_type: str = None, data: Any = None):
        """
        Notify all views of a model change
        
        Args:
            event_type (str): Type of event that occurred
            data (Any): Additional data related to the event
        """
        for view in self.views:
            view.update_view(event_type, data)
            
    def load_flpsf(self, file_path: str) -> bool:
        """
        Load a FLPSF file
        
        Args:
            file_path (str): Path to FLPSF file
            
        Returns:
            bool: True if file loaded successfully, False otherwise
        """
        try:
            # Verify the file first
            verification_result = verify_flpsf_file(file_path)
            if not verification_result['valid']:
                logger.error(f"Invalid FLPSF file: {', '.join(verification_result['errors'])}")
                return False
                
            # Load the file data
            self.flpsf_data = load_flpsf(file_path)
            if self.flpsf_data is None:
                logger.error(f"Failed to load FLPSF file: {file_path}")
                return False
                
            # Set file path
            self.flpsf_path = file_path
            
            # Extract image information
            self.image_path = self.flpsf_data.get('image_path', None)
            self.image_hash = self.flpsf_data.get('image_hash', None)
            self.image_dimensions = self.flpsf_data.get('image_dimensions', None)
            
            # Load embedded image if available
            if 'decoded_image' in self.flpsf_data:
                self.image = self.flpsf_data['decoded_image']
                
            # Otherwise load from disk
            elif self.image_path and os.path.exists(self.image_path):
                self.image = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
                
            # Extract landmarks
            self.landmarks = self.flpsf_data.get('landmarks', None)
            
            # Extract motion parameters
            for param in self.motion_params:
                if param in self.flpsf_data:
                    self.motion_params[param] = self.flpsf_data[param]
                    
            # Extract derived parameters
            for param in self.derived_params:
                if param in self.flpsf_data:
                    self.derived_params[param] = self.flpsf_data[param]
                    
            # Extract normalization parameters
            for param in self.norm_params:
                if param in self.flpsf_data:
                    self.norm_params[param] = self.flpsf_data[param]
                    
            # Extract pasteback information
            for param in self.pasteback_info:
                if param in self.flpsf_data:
                    self.pasteback_info[param] = self.flpsf_data[param]
            
            # Reset modification flag
            self.is_modified = False
            
            # Clear undo/redo stacks
            self.undo_stack = []
            self.redo_stack = []
            
            # Notify views
            self.notify_views('file_loaded', {'path': file_path})
            
            logger.info(f"FLPSF file loaded: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading FLPSF file: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def save_flpsf(self, file_path: str = None, embed_image: bool = True) -> bool:
        """
        Save data to a FLPSF file
        
        Args:
            file_path (str, optional): Path to save the file. If None, uses current path.
            embed_image (bool): Whether to embed the image data in the file
            
        Returns:
            bool: True if file saved successfully, False otherwise
        """
        try:
            save_path = file_path or self.flpsf_path
            
            if save_path is None:
                logger.error("No file path specified for saving FLPSF file")
                return False
                
            # Prepare data to save
            save_data = {}
            
            # Add image information
            save_data['image_path'] = self.image_path
            save_data['image_dimensions'] = self.image_dimensions
            
            # Add landmarks
            if self.landmarks is not None:
                save_data['landmarks'] = self.landmarks
                
            # Add motion parameters
            for param, value in self.motion_params.items():
                if value is not None:
                    save_data[param] = value
                    
            # Add derived parameters
            for param, value in self.derived_params.items():
                if value is not None:
                    save_data[param] = value
                    
            # Add normalization parameters
            for param, value in self.norm_params.items():
                if value is not None:
                    save_data[param] = value
                    
            # Add pasteback information
            for param, value in self.pasteback_info.items():
                if value is not None:
                    save_data[param] = value
            
            # Save the file
            if save_flpsf(save_path, save_data, embed_image):
                self.flpsf_path = save_path
                self.is_modified = False
                
                # Notify views
                self.notify_views('file_saved', {'path': save_path})
                
                logger.info(f"FLPSF file saved: {save_path}")
                return True
            else:
                logger.error(f"Failed to save FLPSF file: {save_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving FLPSF file: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def load_image(self, image_path: str) -> bool:
        """
        Load a source image
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            bool: True if image loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return False
                
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
                
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Set image data
            self.image = img_rgb
            self.image_path = image_path
            self.image_dimensions = np.array([img.shape[1], img.shape[0]])
            
            # Reset other data
            self.landmarks = None
            
            for param in self.motion_params:
                self.motion_params[param] = None
                
            for param in self.derived_params:
                self.derived_params[param] = None
                
            for param in self.norm_params:
                self.norm_params[param] = None
                
            for param in self.pasteback_info:
                self.pasteback_info[param] = None
                
            # Set modification flag
            self.is_modified = True
            
            # Clear undo/redo stacks
            self.undo_stack = []
            self.redo_stack = []
            
            # Notify views
            self.notify_views('image_loaded', {'path': image_path})
            
            logger.info(f"Image loaded: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def set_landmarks(self, landmarks: np.ndarray) -> bool:
        """
        Set facial landmarks
        
        Args:
            landmarks (np.ndarray): Array of landmark points
            
        Returns:
            bool: True if landmarks set successfully, False otherwise
        """
        try:
            # Save current state for undo
            self._push_undo_state('landmarks')
            
            self.landmarks = landmarks
            self.is_modified = True
            
            # Notify views
            self.notify_views('landmarks_updated', {'landmarks': landmarks})
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting landmarks: {str(e)}")
            return False
            
    def set_motion_param(self, param: str, value: np.ndarray) -> bool:
        """
        Set a motion parameter
        
        Args:
            param (str): Parameter name
            value (np.ndarray): Parameter value
            
        Returns:
            bool: True if parameter set successfully, False otherwise
        """
        try:
            if param not in self.motion_params:
                logger.error(f"Invalid motion parameter: {param}")
                return False
                
            # Save current state for undo
            self._push_undo_state('motion_param', {'param': param})
            
            self.motion_params[param] = value
            self.is_modified = True
            
            # Notify views
            self.notify_views('param_updated', {'type': 'motion', 'param': param, 'value': value})
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting motion parameter: {str(e)}")
            return False
            
    def set_derived_param(self, param: str, value: np.ndarray) -> bool:
        """
        Set a derived parameter
        
        Args:
            param (str): Parameter name
            value (np.ndarray): Parameter value
            
        Returns:
            bool: True if parameter set successfully, False otherwise
        """
        try:
            if param not in self.derived_params:
                logger.error(f"Invalid derived parameter: {param}")
                return False
                
            # Save current state for undo
            self._push_undo_state('derived_param', {'param': param})
            
            self.derived_params[param] = value
            self.is_modified = True
            
            # Notify views
            self.notify_views('param_updated', {'type': 'derived', 'param': param, 'value': value})
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting derived parameter: {str(e)}")
            return False
            
    def set_norm_param(self, param: str, value: Any) -> bool:
        """
        Set a normalization parameter
        
        Args:
            param (str): Parameter name
            value (Any): Parameter value
            
        Returns:
            bool: True if parameter set successfully, False otherwise
        """
        try:
            if param not in self.norm_params:
                logger.error(f"Invalid normalization parameter: {param}")
                return False
                
            # Save current state for undo
            self._push_undo_state('norm_param', {'param': param})
            
            self.norm_params[param] = value
            self.is_modified = True
            
            # Notify views
            self.notify_views('param_updated', {'type': 'norm', 'param': param, 'value': value})
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting normalization parameter: {str(e)}")
            return False
            
    def set_pasteback_info(self, param: str, value: np.ndarray) -> bool:
        """
        Set pasteback information
        
        Args:
            param (str): Parameter name
            value (np.ndarray): Parameter value
            
        Returns:
            bool: True if parameter set successfully, False otherwise
        """
        try:
            if param not in self.pasteback_info:
                logger.error(f"Invalid pasteback parameter: {param}")
                return False
                
            # Save current state for undo
            self._push_undo_state('pasteback_info', {'param': param})
            
            self.pasteback_info[param] = value
            self.is_modified = True
            
            # Notify views
            self.notify_views('param_updated', {'type': 'pasteback', 'param': param, 'value': value})
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting pasteback parameter: {str(e)}")
            return False
            
    def reset(self) -> None:
        """
        Reset all data
        """
        # Save current state for undo
        self._push_undo_state('reset')
        
        self.flpsf_data = {}
        self.flpsf_path = None
        
        self.image_path = None
        self.image = None
        self.image_hash = None
        self.image_dimensions = None
        
        self.landmarks = None
        
        for param in self.motion_params:
            self.motion_params[param] = None
            
        for param in self.derived_params:
            self.derived_params[param] = None
            
        for param in self.norm_params:
            self.norm_params[param] = None
            
        for param in self.pasteback_info:
            self.pasteback_info[param] = None
            
        self.is_modified = False
        
        # Clear undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []
        
        # Notify views
        self.notify_views('reset')
        
    def _push_undo_state(self, action_type: str, action_data: Dict = None) -> None:
        """
        Push current state to undo stack
        
        Args:
            action_type (str): Type of action
            action_data (Dict, optional): Additional action data
        """
        # Create state snapshot
        state = {
            'action_type': action_type,
            'action_data': action_data or {},
            'snapshot': self._create_state_snapshot(action_type, action_data),
        }
        
        # Add to undo stack
        self.undo_stack.append(state)
        
        # Limit stack size
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)
            
        # Clear redo stack (new action invalidates redo)
        self.redo_stack = []
        
    def _create_state_snapshot(self, action_type: str, action_data: Dict = None) -> Dict:
        """
        Create a snapshot of the current state for undo/redo
        
        Args:
            action_type (str): Type of action
            action_data (Dict, optional): Additional action data
            
        Returns:
            Dict: Snapshot of current state
        """
        snapshot = {}
        
        # Select what to snapshot based on action type
        if action_type == 'reset':
            # Full snapshot
            snapshot = {
                'flpsf_path': self.flpsf_path,
                'image': self.image.copy() if self.image is not None else None,
                'image_path': self.image_path,
                'image_dimensions': self.image_dimensions.copy() if self.image_dimensions is not None else None,
                'landmarks': self.landmarks.copy() if self.landmarks is not None else None,
                'motion_params': {k: v.copy() if v is not None else None for k, v in self.motion_params.items()},
                'derived_params': {k: v.copy() if v is not None else None for k, v in self.derived_params.items()},
                'norm_params': {k: v.copy() if isinstance(v, np.ndarray) and v is not None else v 
                               for k, v in self.norm_params.items()},
                'pasteback_info': {k: v.copy() if v is not None else None for k, v in self.pasteback_info.items()},
            }
        elif action_type == 'landmarks':
            snapshot['landmarks'] = self.landmarks.copy() if self.landmarks is not None else None
        elif action_type == 'motion_param':
            param = action_data.get('param')
            snapshot[param] = self.motion_params[param].copy() if self.motion_params[param] is not None else None
        elif action_type == 'derived_param':
            param = action_data.get('param')
            snapshot[param] = self.derived_params[param].copy() if self.derived_params[param] is not None else None
        elif action_type == 'norm_param':
            param = action_data.get('param')
            value = self.norm_params[param]
            snapshot[param] = value.copy() if isinstance(value, np.ndarray) and value is not None else value
        elif action_type == 'pasteback_info':
            param = action_data.get('param')
            snapshot[param] = self.pasteback_info[param].copy() if self.pasteback_info[param] is not None else None
        
        return snapshot
        
    def undo(self) -> bool:
        """
        Undo the last action
        
        Returns:
            bool: True if undo successful, False otherwise
        """
        if not self.undo_stack:
            logger.info("Nothing to undo")
            return False
            
        # Get the last state
        state = self.undo_stack.pop()
        
        # Create redo snapshot
        redo_state = {
            'action_type': state['action_type'],
            'action_data': state['action_data'],
            'snapshot': self._create_state_snapshot(state['action_type'], state['action_data']),
        }
        
        # Push to redo stack
        self.redo_stack.append(redo_state)
        
        # Restore state
        self._restore_state(state)
        
        # Notify views
        self.notify_views('undo', {'action_type': state['action_type']})
        
        return True
        
    def redo(self) -> bool:
        """
        Redo the last undone action
        
        Returns:
            bool: True if redo successful, False otherwise
        """
        if not self.redo_stack:
            logger.info("Nothing to redo")
            return False
            
        # Get the last state
        state = self.redo_stack.pop()
        
        # Create undo snapshot
        undo_state = {
            'action_type': state['action_type'],
            'action_data': state['action_data'],
            'snapshot': self._create_state_snapshot(state['action_type'], state['action_data']),
        }
        
        # Push to undo stack
        self.undo_stack.append(undo_state)
        
        # Restore state
        self._restore_state(state)
        
        # Notify views
        self.notify_views('redo', {'action_type': state['action_type']})
        
        return True
        
    def _restore_state(self, state: Dict) -> None:
        """
        Restore state from a snapshot
        
        Args:
            state (Dict): State to restore
        """
        action_type = state['action_type']
        action_data = state['action_data']
        snapshot = state['snapshot']
        
        if action_type == 'reset':
            # Restore everything
            self.flpsf_path = snapshot.get('flpsf_path')
            self.image = snapshot.get('image')
            self.image_path = snapshot.get('image_path')
            self.image_dimensions = snapshot.get('image_dimensions')
            self.landmarks = snapshot.get('landmarks')
            
            motion_params = snapshot.get('motion_params', {})
            for param, value in motion_params.items():
                self.motion_params[param] = value
                
            derived_params = snapshot.get('derived_params', {})
            for param, value in derived_params.items():
                self.derived_params[param] = value
                
            norm_params = snapshot.get('norm_params', {})
            for param, value in norm_params.items():
                self.norm_params[param] = value
                
            pasteback_info = snapshot.get('pasteback_info', {})
            for param, value in pasteback_info.items():
                self.pasteback_info[param] = value
                
        elif action_type == 'landmarks':
            self.landmarks = snapshot.get('landmarks')
        elif action_type == 'motion_param':
            param = action_data.get('param')
            if param:
                self.motion_params[param] = snapshot.get(param)
        elif action_type == 'derived_param':
            param = action_data.get('param')
            if param:
                self.derived_params[param] = snapshot.get(param)
        elif action_type == 'norm_param':
            param = action_data.get('param')
            if param:
                self.norm_params[param] = snapshot.get(param)
        elif action_type == 'pasteback_info':
            param = action_data.get('param')
            if param:
                self.pasteback_info[param] = snapshot.get(param)