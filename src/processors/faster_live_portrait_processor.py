#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FasterLivePortrait Processor
# Processes video frames with the FasterLivePortrait pipeline

import logging
import os
import cv2
import numpy as np
import time
import pickle
from typing import List, Dict, Any, Tuple, Optional, Union, cast
from typing_extensions import TypedDict
from omegaconf import OmegaConf, DictConfig

from src.processors.base_video_processor import BaseVideoProcessor
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faster_live_portrait_processor")

# Define typed structures for frame processing commands
class SourceSwitchCommand(TypedDict, total=False):
    """Command to switch to a different source image"""
    action: str  # "switch_source"
    index: int   # Index of the source to switch to

class ProcessorCommand(TypedDict, total=False):
    """Base command type for processor commands"""
    action: str  # Command action type

# Define source info type
class SourceInfo(TypedDict, total=False):
    """Information about a source image processed by the pipeline"""
    landmarks: np.ndarray  # Facial landmarks
    bbox: List[int]         # Bounding box
    crop_params: Dict[str, float]  # Cropping parameters
    motion_data: Dict[str, np.ndarray]  # Motion extraction data

class FasterLivePortraitProcessor(BaseVideoProcessor):
    def __init__(self, config_path: str, src_image_path: str, 
                 src_image_2_path: Optional[str]=None, 
                 src_image_3_path: Optional[str]=None, 
                 src_image_4_path: Optional[str]=None, 
                 is_animal: bool=False, 
                 debug: bool=False) -> None:
        """Initialize the processor with source images and config"""
        super().__init__()
            
        self.config: DictConfig = OmegaConf.load(config_path)
        self.config.infer_params.flag_pasteback = True
        
        # Set up multiple source images
        self.src_image_paths: List[str] = [src_image_path]
        if src_image_2_path:
            self.src_image_paths.append(src_image_2_path)
        if src_image_3_path:
            self.src_image_paths.append(src_image_3_path)
        if src_image_4_path:
            self.src_image_paths.append(src_image_4_path)
            
        self.current_source_index: int = 0
        self.src_images: List[np.ndarray] = []
        self.src_infos: List[SourceInfo] = []
        self.src_originals: List[np.ndarray] = []  # Store original images for display/reference
        self.is_animal: bool = is_animal
        self.debug: bool = debug
        self.input_size: Tuple[int, int] = (256, 256)
        self.pipeline: Optional[FasterLivePortraitPipeline] = None
        self.is_first_frame_after_switch: bool = True
        
        try:
            self.pipeline = FasterLivePortraitPipeline(cfg=self.config, is_animal=is_animal)
            
            # Process each source image path
            for src_path in self.src_image_paths:
                self._process_source(src_path)
                
            # Use dimensions from config
            self.input_size = (256, 256)
            logger.info(f"FasterLivePortrait initialized with {len(self.src_image_paths)} source images")
            
        except Exception as e:
            logger.error(f"Error initializing FasterLivePortrait: {e}")
            raise
    
    def _process_source(self, src_path: str) -> None:
        """Process a source image path, handling both regular images and FSP files"""
        # Check if this is an FSP file (by extension or content)
        if src_path.lower().endswith('.fsp') or src_path.lower().endswith('.pkl'):
            logger.info(f"Loading source data from FSP file: {src_path}")
            try:
                # Load serialized source data
                with open(src_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Verify version if present
                if 'version' in data and data['version'] != 1:
                    logger.warning(f"FSP file version mismatch: {data.get('version', 'unknown')}, expected 1")
                
                # Append the first image from the FSP data
                self.src_images.append(data['src_imgs'][0])
                self.src_originals.append(data['src_imgs'][0].copy())
                
                # Append the source info for the first frame and first face
                self.src_infos.append(data['src_infos'][0])
                
                # Set pipeline attributes required for proper operation
                self.pipeline.is_source_video = data.get('is_source_video', False)
                
                # We've loaded the preprocessed data, no need to run prepare_source
                logger.info(f"Successfully loaded FSP data from {src_path}")
            except Exception as e:
                logger.error(f"Error loading FSP file {src_path}: {e}")
                raise
        else:
            # Regular image processing path
            logger.info(f"Processing regular source image: {src_path}")
            
            # First, load source image and get its dimensions
            img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not load source image: {src_path}")
            
            # Handle transparency in source images
            if img.shape[-1] == 4:  # Image has alpha channel
                logger.info(f"Source image has transparency. Replacing transparent pixels with green.")
                green_background = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * np.array([0, 255, 0], dtype=np.uint8)
                alpha = img[:, :, 3] / 255.0
                rgb = img[:, :, :3]
                img = (rgb * alpha[:, :, np.newaxis] + green_background * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
            
            # Store original image
            self.src_originals.append(img.copy())
            
            # Prepare the source image with the pipeline
            temp_path: str = f"/tmp/normalized_source_{len(self.src_images)}.jpg"
            cv2.imwrite(temp_path, img)
            
            # Process with the pipeline
            success: bool = self.pipeline.prepare_source(temp_path, realtime=True)
            if not success:
                raise ValueError(f"Could not process source image: {src_path}")
            
            # Store processed source data
            self.src_images.append(self.pipeline.src_imgs[0])
            self.src_infos.append(self.pipeline.src_infos[0])
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
    def switch_source(self, index: Optional[int]=None) -> int:
        """Switch to a specific source image or to the next one if index is None"""
        if index is not None:
            if 0 <= index < len(self.src_image_paths):
                self.current_source_index = index
        else:
            # Cycle to next source
            self.current_source_index = (self.current_source_index + 1) % len(self.src_image_paths)
            
        # Reset pipeline state for clean transition
        self.pipeline.frame_id = 0
        self.pipeline.R_d_0 = None
        self.pipeline.x_d_0_info = None
        self.pipeline.src_lmk_pre = None
        self.is_first_frame_after_switch = True
        
        logger.info(f"Switched to source image {self.current_source_index + 1}: {self.src_image_paths[self.current_source_index]}")
        return self.current_source_index
    
    def process_frame(self, frame: np.ndarray, command: Optional[ProcessorCommand]=None) -> Optional[np.ndarray]:
        """
        Process a frame using FasterLivePortrait
        
        Parameters:
        -----------
        frame: np.ndarray
            Input video frame to process
        command: Optional[ProcessorCommand]
            Optional command to modify processing, such as switching source image
            
        Returns:
        --------
        Optional[np.ndarray]:
            Processed frame or None if processing failed
        """
        # Check for command to switch source image
        if command and command.get("action") == "switch_source":
            source_cmd = cast(SourceSwitchCommand, command)  # Cast to more specific type
            index = source_cmd.get("index")
            self.switch_source(index)
        
        # Call parent for FPS calculation
        super().process_frame(frame)
        
        try:
            # Convert frame format if needed
            if frame.ndim == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Get original frame dimensions
            original_height, original_width = frame.shape[:2]
            
            # Use current source image
            current_src_img = self.src_images[self.current_source_index]
            current_src_info = self.src_infos[self.current_source_index]
            
            # Process frame with FasterLivePortrait
            first_frame: bool = self.frame_counter == 0
            dri_crop, out_crop, out_org, dri_motion_info = self.pipeline.run(
                frame, 
                current_src_img,
                current_src_info,
                first_frame=first_frame
            )
            
            # Add FPS and metadata to output frame
            if out_org is not None:
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                self.is_first_frame_after_switch = False
                if self.debug:
                    # Create thumbnail of driving frame
                    thumbnail_height: int = int(out_org.shape[0] / 4)  # 1/4 of output height
                    thumbnail_width: int = int(thumbnail_height * frame.shape[1] / frame.shape[0])  # Maintain aspect ratio
                    thumbnail = cv2.resize(frame, (thumbnail_width, thumbnail_height))
                    
                    # Create a position for the thumbnail in the lower right corner with padding
                    padding: int = 10
                    y_offset: int = out_org.shape[0] - thumbnail_height - padding
                    x_offset: int = out_org.shape[1] - thumbnail_width - padding
                    
                    # Add border to thumbnail
                    border_color: Tuple[int, int, int] = (0, 255, 0)  # Green border
                    border_size: int = 2
                    thumbnail_with_border = cv2.copyMakeBorder(
                        thumbnail, 
                        border_size, border_size, border_size, border_size, 
                        cv2.BORDER_CONSTANT, 
                        value=border_color
                    )
                    
                    # Create a region of interest in the output image
                    roi_height, roi_width = thumbnail_with_border.shape[:2]
                    roi = out_org[
                        y_offset:y_offset + roi_height,
                        x_offset:x_offset + roi_width
                    ]
                    
                    # Calculate alpha blend mask to make thumbnail slightly transparent
                    alpha: float = 0.7  # 70% opacity
                    # Blend the thumbnail with the background
                    if roi.shape[:2] == thumbnail_with_border.shape[:2]:  # Ensure shapes match
                        blended_roi = cv2.addWeighted(thumbnail_with_border, alpha, roi, 1-alpha, 0)
                        out_org[y_offset:y_offset + roi_height, x_offset:x_offset + roi_width] = blended_roi
                    
                    # Add FPS and metadata text
                    info_text: str = f"FPS: {self.fps} | Frame: {self.frame_counter} | Input: {original_width}x{original_height}"
                    cv2.putText(
                        out_org,
                        info_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    # Add source image indicator
                    source_info: str = f"Source: {self.current_source_index + 1}/{len(self.src_image_paths)}"
                    cv2.putText(
                        out_org,
                        source_info,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    # Add source file name
                    source_name: str = os.path.basename(self.src_image_paths[self.current_source_index])
                    cv2.putText(
                        out_org,
                        source_name,
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    
                    # Add instructions for source switching
                    instruction_text: str = "Press 1/2/3 to switch source images" 
                    cv2.putText(
                        out_org,
                        instruction_text,
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )
                
                return out_org
            elif self.is_first_frame_after_switch:
                # If no output image is generated, return the original frame
                logger.warning("No output image generated. Returning original frame.")
                self.is_first_frame_after_switch = False
                return current_src_img
            return None
            
        except RuntimeError as e:
            logger.error(f"Processing error: {str(e)}")
            # Fall back to basic processor if FasterLivePortrait fails
            return BaseVideoProcessor.process_frame(self, frame)
