#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FasterLivePortrait Source Format (FLPSF) utilities

This module provides functions for reading and writing FLPSF files,
which store all necessary data for deterministic source image processing
in FasterLivePortrait.
"""

import os
import io
import time
import hashlib
import datetime
import numpy as np
from PIL import Image
import cv2


# Current version of the FLPSF format
FLPSF_VERSION = "1.0.0"


def compute_image_hash(image_path):
    """
    Compute SHA-256 hash of an image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        str: Hexadecimal hash string
    """
    hash_obj = hashlib.sha256()
    with open(image_path, 'rb') as f:
        hash_obj.update(f.read())
    return hash_obj.hexdigest()


def encode_image(img_array):
    """
    Encode image array to compressed bytes.
    
    Args:
        img_array: NumPy array of image
        
    Returns:
        bytes: Encoded image data
    """
    # Convert to RGB if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_pil = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
    return None


def decode_image(img_bytes):
    """
    Decode image bytes to array.
    
    Args:
        img_bytes: Encoded image data
        
    Returns:
        np.ndarray: Decoded image array
    """
    if img_bytes is not None:
        buffer = io.BytesIO(img_bytes)
        img_pil = Image.open(buffer)
        return np.array(img_pil)
    return None


def save_flpsf(file_path, source_data, embed_image=True):
    """
    Save source data to FLPSF file.
    
    Args:
        file_path: Output file path (.flpsf)
        source_data: Dictionary containing source data
        embed_image: Whether to embed the image data in the file
        
    Returns:
        bool: True if successful
    """
    try:
        # Make a copy of the data to avoid modifying the original
        data_to_save = {}
        
        # Add metadata
        data_to_save['version'] = FLPSF_VERSION
        data_to_save['creation_date'] = datetime.datetime.now().isoformat()
        data_to_save['software'] = 'FasterLivePortrait'
        
        # Add source image information
        image_path = source_data.get('image_path', '')
        data_to_save['image_path'] = image_path
        
        # Compute image hash if possible
        if os.path.exists(image_path):
            data_to_save['image_hash'] = compute_image_hash(image_path)
            
            # Read and encode image dimensions
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                data_to_save['image_dimensions'] = np.array([w, h])
                
                # Optionally embed the image
                if embed_image:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    data_to_save['embedded_image'] = encode_image(img_rgb)
        
        # Copy all source data elements
        for key, value in source_data.items():
            if key not in data_to_save:
                # Skip None values and special handling for embedded images
                if value is not None and key != 'embedded_image':
                    if isinstance(value, np.ndarray):
                        data_to_save[key] = value
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        data_to_save[key] = np.array(value)
                    elif isinstance(value, (bool, int, float, str)):
                        data_to_save[key] = value
        
        # Save to compressed NumPy format
        np.savez_compressed(file_path, **data_to_save)
        
        # Rename to .flpsf if needed
        if not file_path.endswith('.flpsf'):
            os.rename(file_path + '.npz', file_path)
        else:
            os.rename(file_path + '.npz', file_path)
            
        return True
    
    except Exception as e:
        print(f"Error saving FLPSF file: {e}")
        return False


def load_flpsf(file_path):
    """
    Load source data from FLPSF file.
    
    Args:
        file_path: Path to FLPSF file
        
    Returns:
        dict: Dictionary containing source data
    """
    try:
        # Load from NumPy file
        with np.load(file_path, allow_pickle=True) as data:
            # Convert to dictionary
            source_data = dict(data)
            
            # Decode embedded image if present
            if 'embedded_image' in source_data:
                img_bytes = source_data['embedded_image']
                source_data['decoded_image'] = decode_image(img_bytes)
            
            # Convert any 0-d arrays to scalars
            for key, value in source_data.items():
                if isinstance(value, np.ndarray) and value.ndim == 0:
                    source_data[key] = value.item()
            
            return source_data
    
    except Exception as e:
        print(f"Error loading FLPSF file: {e}")
        return None


def extract_src_infos(source_data):
    """
    Extract src_infos structure from loaded FLPSF data.
    
    Args:
        source_data: Dictionary containing source data
        
    Returns:
        list: src_infos structure compatible with FasterLivePortrait pipeline
    """
    try:
        # Create a properly structured src_infos list
        src_infos = [[]]
        
        # X_s_info dictionary
        x_s_info = {
            'pitch': source_data.get('pitch', None),
            'yaw': source_data.get('yaw', None),
            'roll': source_data.get('roll', None),
            't': source_data.get('t', None),
            'exp': source_data.get('exp', None),
            'scale': source_data.get('scale', None),
            'kp': source_data.get('kp', None)
        }
        
        # Append all elements in the correct order
        src_infos[0].append(x_s_info)
        
        # Add remaining elements in correct order
        src_infos[0].append(source_data.get('landmarks', None))
        src_infos[0].append(source_data.get('R_s', None))
        src_infos[0].append(source_data.get('f_s', None))
        src_infos[0].append(source_data.get('x_s', None))
        src_infos[0].append(source_data.get('x_c_s', None))
        src_infos[0].append(source_data.get('lip_delta_before_animation', None))
        src_infos[0].append(source_data.get('flag_lip_zero', False))
        src_infos[0].append(source_data.get('mask_ori_float', None))
        src_infos[0].append(source_data.get('M', None))
        
        return [src_infos]
    
    except Exception as e:
        print(f"Error extracting src_infos: {e}")
        return None


def create_flpsf_from_pipeline(pipe, output_path, embed_image=True):
    """
    Create FLPSF file from a prepared FasterLivePortrait pipeline.
    
    Args:
        pipe: Prepared FasterLivePortraitPipeline instance
        output_path: Path for output FLPSF file
        embed_image: Whether to embed the source image
        
    Returns:
        bool: True if successful
    """
    try:
        # Collect all necessary data
        source_data = {
            'image_path': pipe.source_path,
            'is_animal': pipe.is_animal,
            'image_dimensions': np.array(pipe.src_imgs[0].shape[:2][::-1]) if len(pipe.src_imgs) > 0 else None
        }
        
        # Extract face information from src_infos
        if len(pipe.src_infos) > 0 and len(pipe.src_infos[0]) > 0:
            face_info = pipe.src_infos[0][0]
            
            # [x_s_info, source_lmk, R_s, f_s, x_s, x_c_s, lip_delta, flag_lip_zero, mask, M]
            x_s_info = face_info[0]
            source_data['landmarks'] = face_info[1]
            source_data['R_s'] = face_info[2]
            source_data['f_s'] = face_info[3]
            source_data['x_s'] = face_info[4]
            source_data['x_c_s'] = face_info[5]
            source_data['lip_delta_before_animation'] = face_info[6]
            source_data['flag_lip_zero'] = face_info[7]
            source_data['mask_ori_float'] = face_info[8]
            source_data['M'] = face_info[9]
            
            # Motion parameters
            source_data['pitch'] = x_s_info['pitch']
            source_data['yaw'] = x_s_info['yaw']
            source_data['roll'] = x_s_info['roll']
            source_data['t'] = x_s_info['t']
            source_data['exp'] = x_s_info['exp']
            source_data['scale'] = x_s_info['scale']
            source_data['kp'] = x_s_info['kp']
            
        # Optionally embed the source image
        if embed_image and len(pipe.src_imgs) > 0:
            source_data['embedded_image'] = encode_image(pipe.src_imgs[0])
        
        # Save to file
        return save_flpsf(output_path, source_data, embed_image=False)  # Already embedded if requested
        
    except Exception as e:
        print(f"Error creating FLPSF from pipeline: {e}")
        return False


def verify_flpsf_file(file_path):
    """
    Verify FLPSF file integrity and contents.
    
    Args:
        file_path: Path to FLPSF file
        
    Returns:
        dict: Verification results
    """
    results = {
        'valid': False,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Load the file
        data = load_flpsf(file_path)
        if data is None:
            results['errors'].append("Failed to load file")
            return results
        
        # Check version
        if 'version' not in data:
            results['warnings'].append("Missing version information")
        
        # Check critical fields
        required_fields = ['landmarks', 'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp', 'R_s', 'f_s', 'x_s']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            results['errors'].append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Check image consistency
        if 'image_path' in data and 'image_hash' in data and os.path.exists(data['image_path']):
            current_hash = compute_image_hash(data['image_path'])
            if current_hash != data['image_hash']:
                results['warnings'].append("Image file has changed since FLPSF creation")
        
        # Validate array shapes
        if 'landmarks' in data and (not isinstance(data['landmarks'], np.ndarray) or data['landmarks'].shape[1] != 2):
            results['errors'].append(f"Invalid landmarks shape: {data['landmarks'].shape}")
        
        if 'exp' in data and (not isinstance(data['exp'], np.ndarray) or data['exp'].shape[1] != 21):
            results['errors'].append(f"Invalid exp shape: {data['exp'].shape}")
        
        # Set valid flag
        results['valid'] = len(results['errors']) == 0
        
        return results
    
    except Exception as e:
        results['errors'].append(f"Verification error: {str(e)}")
        return results