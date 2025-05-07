"""
SPD (Source Portrait Descriptor) writer implementation.

This module provides functionality to create SPD files according to the format
defined in the format module.
"""
import os
import io
import struct
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, BinaryIO, Tuple, Any, TypeVar, Type, cast

from .format import (
    MAGIC_BYTES, CURRENT_VERSION,
    FLAG_HAS_ORIGINAL_IMAGE, FLAG_HAS_CROPPED_IMAGE, FLAG_HAS_RESIZED_IMAGE,
    FLAG_HAS_LANDMARK_DATA, FLAG_HAS_MOTION_PARAMETERS, FLAG_HAS_APPEARANCE_FEATURES,
    FLAG_HAS_TRANSFORMATION_MATRICES, FLAG_HAS_MASK_DATA,
    RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT,
    SectionMarker
)

# Set up logger for this module
logger = logging.getLogger(__name__)

class SPDWriter:
    def __init__(self, filepath: str, data: dict):
        """
        Initializes the SPDWriter.

        Args:
            filepath (str): The path to the SPD file to be written.
            data (dict): A dictionary containing the data to write.
                         Expected keys match the sections in spec.md, e.g.,
                         'original_image', 'cropped_image', 'resized_image_256',
                         'landmarks_original', 'landmarks_cropped', 'landmarks_resized',
                         'motion_params', 'appearance_features', 'transform_matrices',
                         'mask_data', 'additional_flags'.
        """
        self.filepath = filepath
        self.data = data
        self.flags = 0

    def _set_flags(self):
        """Sets the internal flags based on the data provided."""
        if self.data.get('original_image') is not None:
            self.flags |= FLAG_HAS_ORIGINAL_IMAGE
        if self.data.get('cropped_image') is not None:
            self.flags |= FLAG_HAS_CROPPED_IMAGE
        if self.data.get('resized_image_256') is not None:
            self.flags |= FLAG_HAS_RESIZED_IMAGE
        
        # Check for any landmark data
        if (self.data.get('landmarks_original') is not None or
            self.data.get('landmarks_cropped') is not None or
            self.data.get('landmarks_resized') is not None):
            self.flags |= FLAG_HAS_LANDMARK_DATA
            
        if self.data.get('motion_params') is not None:
            self.flags |= FLAG_HAS_MOTION_PARAMETERS
        if self.data.get('appearance_features') is not None:
            self.flags |= FLAG_HAS_APPEARANCE_FEATURES
        if self.data.get('transform_matrices') is not None:
            self.flags |= FLAG_HAS_TRANSFORMATION_MATRICES
        if self.data.get('mask_data') is not None:
            self.flags |= FLAG_HAS_MASK_DATA

    def _pack_string_payload(self, s: str) -> bytes:
        """Packs a string into a length-prefixed byte sequence (UTF-8 encoded)."""
        s_bytes = s.encode('utf-8')
        return struct.pack('!I', len(s_bytes)) + s_bytes

    def _get_image_payload(self, image_data: np.ndarray, image_format_str: str) -> bytes:
        """Prepares the payload for an image section."""
        if image_data is None:
            return b''
        height, width, channels = image_data.shape
        
        payload = bytearray()
        payload.extend(struct.pack('!I', width))
        payload.extend(struct.pack('!I', height))
        payload.extend(struct.pack('!B', channels))
        payload.extend(self._pack_string_payload(image_format_str))
        payload.extend(image_data.tobytes())
        return bytes(payload)

    def _get_resized_image_payload(self, image_data: np.ndarray, image_format_str: str) -> bytes:
        """Prepares the payload for a resized image section."""
        if image_data is None:
            return b''
        height, width, channels = image_data.shape
        if height != RESIZED_IMAGE_HEIGHT or width != RESIZED_IMAGE_WIDTH:
            raise ValueError(f"Resized image dimensions must be {RESIZED_IMAGE_WIDTH}x{RESIZED_IMAGE_HEIGHT}")

        payload = bytearray()
        payload.extend(struct.pack('!I', width))
        payload.extend(struct.pack('!I', height))
        payload.extend(struct.pack('!B', channels))
        payload.extend(self._pack_string_payload(image_format_str))
        payload.extend(image_data.tobytes())
        return bytes(payload)

    def _get_landmarks_payload(self, landmarks_data: np.ndarray, landmark_type_str: str) -> bytes:
        """Prepares the payload for a landmarks section."""
        if landmarks_data is None:
            return b''
        count = landmarks_data.shape[0]
        dimensions = landmarks_data.shape[1] if landmarks_data.ndim > 1 else 0
        
        payload = bytearray()
        payload.extend(struct.pack('!I', count))
        payload.extend(struct.pack('!I', dimensions))
        payload.extend(self._pack_string_payload(landmark_type_str))
        payload.extend(landmarks_data.astype(np.float32).tobytes())
        return bytes(payload)

    def _get_motion_params_payload(self, motion_params_data: Dict[str, Any]) -> bytes:
        """
        Prepares the payload for a motion parameters section.
        Expects motion_params_data to be a dict with 'names': List[str] and 'values': List[float].
        """
        if not motion_params_data or 'names' not in motion_params_data or 'values' not in motion_params_data:
            return b''

        param_names = motion_params_data['names']
        param_values = motion_params_data['values']
        num_params = len(param_names)

        if num_params != len(param_values):
            raise ValueError("Mismatch between number of motion parameter names and values.")

        payload = bytearray()
        payload.extend(struct.pack('!I', num_params))
        for name in param_names:
            payload.extend(self._pack_string_payload(name))
        for value in param_values:
            payload.extend(struct.pack('!f', float(value)))
        return bytes(payload)

    def _get_mask_payload(self, mask_data: np.ndarray, mask_format_str: str) -> bytes:
        """Prepares the payload for a mask section."""
        if mask_data is None:
            return b''
        if mask_data.ndim == 3 and mask_data.shape[2] == 1:
            mask_data_2d = mask_data.squeeze(axis=-1)
        elif mask_data.ndim == 2:
            mask_data_2d = mask_data
        else:
            raise ValueError(f"Mask data has unexpected shape: {mask_data.shape}")

        height, width = mask_data_2d.shape
        
        payload = bytearray()
        payload.extend(struct.pack('!I', width))
        payload.extend(struct.pack('!I', height))
        payload.extend(self._pack_string_payload(mask_format_str))
        # Ensure mask data is uint8, as commonly expected for masks.
        payload.extend(mask_data_2d.astype(np.uint8).tobytes())
        return bytes(payload)

    def _write_section(self, f: BinaryIO, marker: SectionMarker, payload: bytes):
        """Writes a section with its marker, size, and payload."""
        if not payload: # Do not write section if payload is empty
            return
        f.write(struct.pack('!I', marker.value))
        f.write(struct.pack('!I', len(payload)))
        f.write(payload)

    def write(self):
        """Writes the SPD file."""
        self._set_flags()
        
        with open(self.filepath, 'wb') as f:
            # Header Section
            f.write(MAGIC_BYTES)
            f.write(struct.pack('!I', CURRENT_VERSION))
            f.write(struct.pack('!d', time.time()))
            f.write(struct.pack('!I', self.flags))

            # Image Data Section
            if self.flags & FLAG_HAS_ORIGINAL_IMAGE:
                img_data = self.data.get('original_image')
                img_format = self.data.get('original_image_format', "BGR")
                if img_data is not None:
                    payload = self._get_image_payload(img_data, img_format)
                    self._write_section(f, SectionMarker.IMAGE, payload)
            
            if self.flags & FLAG_HAS_CROPPED_IMAGE:
                pass

            if self.flags & FLAG_HAS_RESIZED_IMAGE:
                pass

            # Facial Landmarks Section
            if self.flags & FLAG_HAS_LANDMARK_DATA:
                lm_data = self.data.get('landmarks_original')
                lm_type = self.data.get('landmarks_original_type', "unknown")
                if lm_data is not None:
                    payload = self._get_landmarks_payload(lm_data, lm_type)
                    self._write_section(f, SectionMarker.LANDMARKS, payload)

            # Motion Parameters Section
            if self.flags & FLAG_HAS_MOTION_PARAMETERS:
                mp_data = self.data.get('motion_params')
                if mp_data:
                    try:
                        payload = self._get_motion_params_payload(mp_data)
                        self._write_section(f, SectionMarker.MOTION_PARAMS, payload)
                    except (KeyError, ValueError, NotImplementedError) as e:
                        logger.error(f"Could not write motion params due to data structure: {e}.")

            # Appearance Features Section
            if self.flags & FLAG_HAS_APPEARANCE_FEATURES:
                pass

            # Transformation Matrices Section
            if self.flags & FLAG_HAS_TRANSFORMATION_MATRICES:
                pass

            # Mask Data Section
            if self.flags & FLAG_HAS_MASK_DATA:
                mask_data_arr = self.data.get('mask_data')
                mask_format = self.data.get('mask_data_format', "alpha")
                if mask_data_arr is not None:
                    payload = self._get_mask_payload(mask_data_arr, mask_format)
                    self._write_section(f, SectionMarker.MASK, payload)
            
            f.write(struct.pack('!I', SectionMarker.END.value))
            f.write(struct.pack('!I', 0))

        logger.info(f"SPD file written to {self.filepath}")