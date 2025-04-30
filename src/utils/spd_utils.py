"""
SPD (Source Portrait Descriptor) utilities for FasterLivePortrait.

This module provides integration between FasterLivePortrait and the SPD format,
allowing direct loading of SPD files as source inputs and exporting processed
source data as SPD files.
"""

import os
import logging
import copy
import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union, Any

# Import SPD modules
from spd_editor.spd.reader import SPDReader
from spd_editor.spd.writer import SPDWriter
from spd_editor.spd.format import (
    MAGIC_BYTES, ImageSection, LandmarksSection, 
    MotionParamsSection, AppearanceSection,
    TransformationSection, MaskSection, AdditionalFlagsSection
)

# Set up logger for this module
logger = logging.getLogger(__name__)

def is_spd_file(file_path: str) -> bool:
    """
    Determine if a file is an SPD file based on extension or magic bytes.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file is an SPD file, False otherwise
    """
    # Check file extension
    if file_path.lower().endswith('.spd'):
        return True
        
    # Check magic bytes
    try:
        with open(file_path, 'rb') as f:
            magic = f.read(4)
            return magic == MAGIC_BYTES
    except Exception as e:
        logger.error(f"Error checking if file is SPD: {e}")
        return False

def load_spd_file(spd_path: str, device: torch.device, config: Any) -> Tuple[List[np.ndarray], List[List], bool]:
    """
    Load an SPD file and convert it to the FasterLivePortrait internal format.
    
    Args:
        spd_path: Path to the SPD file
        device: Torch device (CPU or GPU)
        config: Pipeline configuration
        
    Returns:
        Tuple of (src_imgs, src_infos, is_source_video)
    """
    logger.info(f"Loading SPD file: {spd_path}")
    
    try:
        with SPDReader(spd_path) as reader:
            # Extract basic information
            additional_flags = reader.additional_flags
            flags = additional_flags.flags
            is_source_video = flags.get("is_source_video", False)
            
            # Extract image
            image_section = reader.image
            img_width = image_section.width
            img_height = image_section.height
            img_channels = image_section.channels
            img_format = image_section.format
            
            # Convert image bytes to numpy array
            img_shape = (img_height, img_width, img_channels)
            img_data = np.frombuffer(image_section.data, dtype=np.uint8).reshape(img_shape)
            
            if img_format.upper() == 'BGR':
                img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img_data
                
            # Create source image list
            src_imgs = [img_rgb]
            
            # Get motion parameters
            motion_params = reader.motion_params
            param_dict = {name: value for name, value in zip(motion_params.param_names, motion_params.values)}
            
            # Extract necessary parameters
            pitch = np.array([[param_dict.get("pitch", 0.0)]], dtype=np.float32)
            yaw = np.array([[param_dict.get("yaw", 0.0)]], dtype=np.float32)
            roll = np.array([[param_dict.get("roll", 0.0)]], dtype=np.float32)
            
            # Extract translation (t), assumes t is stored as tx, ty, tz
            t = np.array([[
                param_dict.get("tx", 0.0),
                param_dict.get("ty", 0.0),
                param_dict.get("tz", 0.0)
            ]], dtype=np.float32)
            
            # Extract expression parameters
            if "exp_data" in flags:
                exp_data = np.frombuffer(flags["exp_data"], dtype=np.float32)
                exp_shape = flags.get("exp_shape", (1, 20, 3))  # Default shape if not specified
                exp = exp_data.reshape(exp_shape)
            else:
                # Default expression parameters if not available
                exp = np.zeros((1, 20, 3), dtype=np.float32)
            
            # Extract scale
            scale = np.array([[param_dict.get("scale", 1.0)]], dtype=np.float32)
            
            # Extract keypoints
            if "kp_data" in flags:
                kp_data = np.frombuffer(flags["kp_data"], dtype=np.float32)
                kp_shape = flags.get("kp_shape", (1, 20, 3))  # Default shape if not specified
                kp = kp_data.reshape(kp_shape)
            else:
                # Default keypoints if not available
                kp = np.zeros((1, 20, 3), dtype=np.float32)
            
            # Create motion info dictionary
            x_s_info = {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "t": t,
                "exp": exp,
                "scale": scale,
                "kp": kp
            }
            
            # Extract landmarks
            landmarks = reader.landmarks
            source_lmk = np.array(landmarks.points, dtype=np.float32)
            
            # Extract transformation matrix
            transforms = reader.transforms
            transform_types = transforms.transform_types
            matrices = transforms.matrices
            
            # Find R_s (typically stored as "rotation_matrix")
            R_s = None
            for i, t_type in enumerate(transform_types):
                if t_type == "rotation_matrix" or t_type == "R_s":
                    matrix = matrices[i]
                    R_s = np.array(matrix, dtype=np.float32).reshape(1, 3, 3)
                    break
            
            if R_s is None:
                # If no rotation matrix found, generate identity matrix
                R_s = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
            
            # Extract appearance features
            appearance = reader.appearance
            f_s = np.array(appearance.features, dtype=np.float32).reshape(1, -1)
            
            # Extract x_s (transformed keypoints)
            if "x_s_data" in flags:
                x_s_data = np.frombuffer(flags["x_s_data"], dtype=np.float32)
                x_s_shape = flags.get("x_s_shape", (1, 20, 3))  # Default shape if not specified
                x_s = x_s_data.reshape(x_s_shape)
            else:
                # If not available, use kp as fallback
                x_s = copy.deepcopy(kp)
            
            # Extract x_c_s (center keypoints)
            if "x_c_s_data" in flags:
                x_c_s_data = np.frombuffer(flags["x_c_s_data"], dtype=np.float32)
                x_c_s_shape = flags.get("x_c_s_shape", (1, 20, 3))  # Default shape if not specified
                x_c_s = x_c_s_data.reshape(x_c_s_shape)
            else:
                # If not available, use kp as fallback
                x_c_s = copy.deepcopy(kp)
            
            # Extract lip_delta_before_animation and flag_lip_zero
            lip_delta_before_animation = None
            flag_lip_zero = flags.get("flag_lip_zero", False)
            
            if "lip_delta_data" in flags:
                lip_delta_data = np.frombuffer(flags["lip_delta_data"], dtype=np.float32)
                lip_delta_shape = flags.get("lip_delta_shape", (1, 60))  # Default shape
                lip_delta_before_animation = lip_delta_data.reshape(lip_delta_shape)
            
            # Extract mask_ori_float and M (transformation matrix)
            mask_ori_float = None
            if "mask_data" in flags and flags.get("has_mask", False):
                mask_data = np.frombuffer(flags["mask_data"], dtype=np.float32)
                mask_shape = (flags.get("mask_height", img_height), 
                              flags.get("mask_width", img_width))
                mask_ori_float = torch.from_numpy(mask_data.reshape(mask_shape)).to(device)
            
            # Find crop-to-original transformation matrix
            M = None
            for i, t_type in enumerate(transform_types):
                if t_type == "crop_to_original" or t_type == "M_c2o":
                    matrix = matrices[i]
                    M = torch.tensor(matrix, dtype=torch.float32).reshape(3, 3).to(device)
                    break
            
            if M is None:
                # If no transformation matrix found, generate identity matrix
                M = torch.eye(3, dtype=torch.float32).to(device)
            
            # Create the src_info structure that FasterLivePortrait expects
            src_info = [
                copy.deepcopy(x_s_info),
                source_lmk.copy(),
                R_s.copy(),
                f_s.copy(),
                x_s.copy(),
                x_c_s.copy(),
                lip_delta_before_animation.copy() if lip_delta_before_animation is not None else None,
                flag_lip_zero,
                mask_ori_float,
                M
            ]
            
            src_infos = [[src_info]]
            
            logger.info(f"Successfully loaded SPD file: {spd_path}")
            return src_imgs, src_infos, is_source_video
            
    except Exception as e:
        logger.error(f"Error loading SPD file: {e}")
        raise

def save_spd_file(output_path: str, src_imgs: List[np.ndarray], src_infos: List[List], is_source_video: bool = False) -> str:
    """
    Save processed source data as an SPD file.
    
    Args:
        output_path: Path to save the SPD file
        src_imgs: List of source images
        src_infos: List of source information lists
        is_source_video: Whether the source is a video
        
    Returns:
        Path to the saved SPD file
    """
    logger.info(f"Saving SPD file to: {output_path}")
    
    try:
        # Ensure file has .spd extension
        if not output_path.lower().endswith('.spd'):
            output_path += '.spd'
            
        with SPDWriter(output_path) as writer:
            # Process the first image only (for simplicity)
            img_rgb = src_imgs[0]
            src_info = src_infos[0][0]  # First face, first frame
            
            # Unpack source info
            x_s_info, source_lmk, R_s, f_s, x_s, x_c_s, lip_delta, flag_lip_zero, mask_ori_float, M = src_info
            
            # Convert RGB to BGR for OpenCV compatibility
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Create and write image section
            image_section = ImageSection(
                width=img_bgr.shape[1],
                height=img_bgr.shape[0],
                channels=img_bgr.shape[2],
                format="BGR",
                data=img_bgr.tobytes()
            )
            writer.write_image_section(image_section)
            
            # Create and write landmarks section
            landmarks_section = LandmarksSection(
                count=len(source_lmk),
                dimensions=source_lmk.shape[1],
                landmark_type="mediapipe" if source_lmk.shape[1] == 2 else "3d",
                points=[point.tolist() for point in source_lmk]
            )
            writer.write_landmarks_section(landmarks_section)
            
            # Create and write motion parameters section
            param_names = ["pitch", "yaw", "roll", "tx", "ty", "tz", "scale"]
            values = [
                float(x_s_info["pitch"][0][0]),
                float(x_s_info["yaw"][0][0]),
                float(x_s_info["roll"][0][0]),
                float(x_s_info["t"][0][0]),
                float(x_s_info["t"][0][1]),
                float(x_s_info["t"][0][2]),
                float(x_s_info["scale"][0][0])
            ]
            
            motion_section = MotionParamsSection(
                num_params=len(param_names),
                param_names=param_names,
                values=values
            )
            writer.write_motion_params_section(motion_section)
            
            # Create and write appearance features section
            appearance_section = AppearanceSection(
                feature_dim=len(f_s[0]),
                feature_type="embedding",
                features=f_s[0].tolist()
            )
            writer.write_appearance_section(appearance_section)
            
            # Create and write transformation matrices section
            # Store rotation matrix and crop-to-original transformation
            transform_types = ["rotation_matrix", "crop_to_original"]
            matrices = [
                R_s[0].flatten().tolist(),
                M.cpu().numpy().flatten().tolist()
            ]
            
            transform_section = TransformationSection(
                num_transforms=len(transform_types),
                transform_types=transform_types,
                matrices=matrices
            )
            writer.write_transformation_section(transform_section)
            
            # Write mask section if available
            if mask_ori_float is not None:
                mask_data = mask_ori_float.cpu().numpy()
                mask_section = MaskSection(
                    width=mask_data.shape[1],
                    height=mask_data.shape[0],
                    format="grayscale",
                    data=mask_data.tobytes()
                )
                writer.write_mask_section(mask_section)
            
            # Create additional flags with important metadata
            flags = {
                "source": os.path.basename(output_path),
                "is_source_video": is_source_video,
                "flag_lip_zero": flag_lip_zero,
                "has_mask": mask_ori_float is not None,
                "version": "1.0"
            }
            
            # Store shape information as strings - we'll store the actual data elsewhere
            if x_s_info["exp"] is not None:
                shape = x_s_info["exp"].shape
                flags["exp_shape"] = f"{shape[0]},{shape[1]},{shape[2]}"
            
            if x_s is not None:
                shape = x_s.shape
                flags["x_s_shape"] = f"{shape[0]},{shape[1]},{shape[2]}"
            
            if x_c_s is not None:
                shape = x_c_s.shape
                flags["x_c_s_shape"] = f"{shape[0]},{shape[1]},{shape[2]}"
            
            if lip_delta is not None:
                shape = lip_delta.shape
                flags["lip_delta_shape"] = f"{shape[0]},{shape[1]}"
                
            # Create and write additional flags section
            flags_section = AdditionalFlagsSection(
                num_flags=len(flags),
                flags=flags
            )
            writer.write_additional_flags_section(flags_section)
            
        logger.info(f"Successfully saved SPD file to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving SPD file: {e}")
        raise