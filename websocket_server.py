#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WebSocket server for FasterLivePortrait
# This script enables live WebSocket streaming of actor video to animate a source image

import asyncio
import json
import logging
import ssl
import uuid
import os
import cv2
import numpy as np
import time
import base64
from pathlib import Path
from typing import Dict, Optional, List
from omegaconf import OmegaConf

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import uvicorn

# Import conditionally to handle potential errors
try:
    from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
    from src.utils.utils import video_has_audio
    FASTER_LIVE_PORTRAIT_AVAILABLE = True
except Exception as e:
    logging.warning(f"Could not import FasterLivePortrait: {e}")
    FASTER_LIVE_PORTRAIT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("websocket_server")

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.actor_connections: Dict[str, WebSocket] = {}
        self.viewer_connections: Dict[str, List[WebSocket]] = {}
        self.frame_queues: Dict[str, asyncio.Queue] = {}
        self.processed_frames: Dict[str, np.ndarray] = {}
        self.frame_processors: Dict[str, object] = {}
        self.is_processing: Dict[str, bool] = {}
        self.frames_received: Dict[str, int] = {}
        self.frames_processed: Dict[str, int] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.performance_metrics = {
            "receive_time": [],
            "decode_time": [],
            "process_time": [],
            "encode_time": [],
            "send_time": [],
            "total_time": [],
            "queue_time": [], 
            "frames_received": {},
            "frames_processed": {},
            "avg_metrics": {}
        }
        self.last_metrics_log = time.time()
        self.metrics_log_interval = 5.0  # Log metrics every 5 seconds
        
    async def connect_actor(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.actor_connections[session_id] = websocket
        self.frame_queues[session_id] = asyncio.Queue(maxsize=1)  # Only store 1 frame
        self.is_processing[session_id] = False
        self.frames_received[session_id] = 0
        self.frames_processed[session_id] = 0
        logger.info(f"Actor connected: {session_id}")
        
        # Start the processing task for this session
        self.processing_tasks[session_id] = asyncio.create_task(
            self._process_frames_loop(session_id)
        )
        
    async def connect_viewer(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        
        # Initialize list for this session if it doesn't exist
        if session_id not in self.viewer_connections:
            self.viewer_connections[session_id] = []
            
        # Add this viewer to the session's viewers
        self.viewer_connections[session_id].append(websocket)
        logger.info(f"Viewer connected to session: {session_id}, total viewers: {len(self.viewer_connections[session_id])}")
        
    def disconnect_actor(self, session_id: str):
        if session_id in self.actor_connections:
            del self.actor_connections[session_id]
            logger.info(f"Actor disconnected: {session_id}")
            
            # Cancel the processing task
            if session_id in self.processing_tasks:
                self.processing_tasks[session_id].cancel()
                del self.processing_tasks[session_id]
            
            # Clean up related resources
            if session_id in self.frame_queues:
                # Clear any remaining frames
                queue = self.frame_queues[session_id]
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                del self.frame_queues[session_id]
                
            if session_id in self.processed_frames:
                del self.processed_frames[session_id]
            if session_id in self.is_processing:
                del self.is_processing[session_id]
            if session_id in self.frames_received:
                del self.frames_received[session_id]
            if session_id in self.frames_processed:
                del self.frames_processed[session_id]
            
    def disconnect_viewer(self, session_id: str, websocket: WebSocket):
        if session_id in self.viewer_connections:
            try:
                self.viewer_connections[session_id].remove(websocket)
                logger.info(f"Viewer disconnected from session: {session_id}, remaining viewers: {len(self.viewer_connections[session_id])}")
                
                # Remove the session entry if no more viewers
                if not self.viewer_connections[session_id]:
                    del self.viewer_connections[session_id]
                    
            except ValueError:
                # WebSocket was not in the list
                pass
    
    async def receive_frame(self, session_id: str, frame: np.ndarray):
        """Handle a new frame from an actor"""
        if session_id not in self.frame_queues:
            return False
        
        self.frames_received[session_id] += 1
        
        # Add to queue, replacing any existing frame
        queue = self.frame_queues[session_id]
        
        # Clear the queue to make room for new frame
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Put the new frame
        try:
            await queue.put(frame)
            return True
        except Exception as e:
            logger.error(f"Error adding frame to queue for session {session_id}: {e}")
            return False
    
    async def _process_frames_loop(self, session_id: str):
        """Background task to process frames for a session"""
        try:
            while session_id in self.actor_connections:
                # Wait for a frame to be available
                if session_id not in self.frame_queues:
                    await asyncio.sleep(0.01)
                    continue
                
                queue = self.frame_queues[session_id]
                
                try:
                    # Start timing for queue wait
                    queue_start_time = time.time()
                    
                    # Get the next frame from the queue (will wait if queue is empty)
                    frame = await queue.get()
                    
                    # Measure queue wait time
                    queue_time = time.time() - queue_start_time
                    self._update_metric("queue_time", queue_time * 1000)  # Convert to ms
                    
                    # Start timing the entire processing pipeline
                    total_start_time = time.time()
                    
                    # Get the processor from the server
                    processor = self.frame_processors.get(session_id)
                    if not processor:
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Process the frame (this is the most time-consuming step)
                    self.is_processing[session_id] = True
                    process_start_time = time.time()
                    processed_frame = processor.process_frame(frame)
                    process_time = time.time() - process_start_time
                    self._update_metric("process_time", process_time * 1000)  # Convert to ms
                    
                    if processed_frame is not None:
                        # Store the processed frame
                        self.processed_frames[session_id] = processed_frame
                        
                        # Encode the frame to JPEG
                        encode_start_time = time.time()
                        success, encoded_img = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if not success:
                            continue
                        binary_img = encoded_img.tobytes()
                        encode_time = time.time() - encode_start_time
                        self._update_metric("encode_time", encode_time * 1000)  # Convert to ms
                        
                        # Send to viewers
                        send_start_time = time.time()
                        await self._send_bytes_to_viewers(binary_img, session_id)
                        send_time = time.time() - send_start_time
                        self._update_metric("send_time", send_time * 1000)  # Convert to ms
                        
                        # Total time
                        total_time = time.time() - total_start_time
                        self._update_metric("total_time", total_time * 1000)  # Convert to ms
                        
                        # Update stats
                        self.frames_processed[session_id] += 1
                        
                        # Log performance metrics periodically
                        current_time = time.time()
                        if current_time - self.last_metrics_log >= self.metrics_log_interval:
                            self.last_metrics_log = current_time
                            avg_metrics = self._calculate_avg_metrics()
                            
                            # Calculate average FPS based on processing time
                            avg_fps = 1000 / avg_metrics["total_time"] if avg_metrics["total_time"] > 0 else 0
                            
                            # Log detailed performance info
                            logger.info(f"Performance metrics - Session {session_id}:")
                            logger.info(f"  Queue Wait: {avg_metrics['queue_time']:.2f}ms")
                            logger.info(f"  Processing: {avg_metrics['process_time']:.2f}ms ({avg_metrics['process_time']/avg_metrics['total_time']*100:.1f}%)")
                            logger.info(f"  JPEG Encode: {avg_metrics['encode_time']:.2f}ms ({avg_metrics['encode_time']/avg_metrics['total_time']*100:.1f}%)")
                            logger.info(f"  WebSocket Send: {avg_metrics['send_time']:.2f}ms ({avg_metrics['send_time']/avg_metrics['total_time']*100:.1f}%)")
                            logger.info(f"  Total Time: {avg_metrics['total_time']:.2f}ms (Theoretical max FPS: {avg_fps:.1f})")
                            
                            # Log frame counts
                            dropped = self.frames_received[session_id] - self.frames_processed[session_id]
                            drop_rate = dropped / self.frames_received[session_id] if self.frames_received[session_id] > 0 else 0
                            logger.info(f"  Frames: Received {self.frames_received[session_id]}, Processed {self.frames_processed[session_id]}")
                            logger.info(f"  Dropped: {dropped} frames ({drop_rate:.1%})")
                    
                except asyncio.CancelledError:
                    # Task is being cancelled
                    break
                except Exception as e:
                    logger.error(f"Error processing frame for session {session_id}: {e}")
                finally:
                    self.is_processing[session_id] = False
                    
        except asyncio.CancelledError:
            # Task is being cancelled
            pass
        except Exception as e:
            logger.error(f"Error in process_frames_loop for session {session_id}: {e}")
            
    async def _send_bytes_to_viewers(self, binary_data: bytes, session_id: str):
        """Send binary data to all connected viewers for a specific session"""
        if session_id not in self.viewer_connections or not self.viewer_connections[session_id]:
            # Even if there are no viewers, still send back to the actor for preview
            if session_id in self.actor_connections:
                try:
                    await self.actor_connections[session_id].send_bytes(binary_data)
                except Exception as e:
                    logger.error(f"Error sending preview to actor in session {session_id}: {e}")
            return
            
        # Send to all viewers of this session
        disconnected_viewers = []
        for i, viewer_websocket in enumerate(self.viewer_connections[session_id]):
            try:
                await viewer_websocket.send_bytes(binary_data)
            except Exception as e:
                logger.error(f"Error sending to viewer in session {session_id}: {e}")
                disconnected_viewers.append(viewer_websocket)
        
        # Remove any disconnected viewers
        for websocket in disconnected_viewers:
            self.viewer_connections[session_id].remove(websocket)
            
        # Always send back to the actor for preview
        if session_id in self.actor_connections:
            try:
                await self.actor_connections[session_id].send_bytes(binary_data)
            except Exception as e:
                logger.error(f"Error sending preview to actor in session {session_id}: {e}")

    # New method to send notifications to viewers when source is switched
    async def notify_viewers_source_switched(self, session_id: str, source_index: int, source_name: str):
        """Notify all viewers that the source image has been switched"""
        if session_id not in self.viewer_connections:
            return
            
        message = {
            "status": "source_switched", 
            "current_source": source_index,
            "source_name": source_name
        }
        
        disconnected_viewers = []
        for viewer_websocket in self.viewer_connections[session_id]:
            try:
                await viewer_websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending source switch notification to viewer in session {session_id}: {e}")
                disconnected_viewers.append(viewer_websocket)
                
        # Remove any disconnected viewers
        for websocket in disconnected_viewers:
            self.viewer_connections[session_id].remove(websocket)

    def _update_metric(self, metric_name, value):
        """Update a performance metric, maintaining a rolling average"""
        if len(self.performance_metrics[metric_name]) >= 30:  # Keep last 30 values
            self.performance_metrics[metric_name].pop(0)
        self.performance_metrics[metric_name].append(value)
        
    def _calculate_avg_metrics(self):
        """Calculate average metrics for logging"""
        metrics = {}
        for key in ["receive_time", "decode_time", "process_time", "encode_time", "send_time", "total_time", "queue_time"]:
            values = self.performance_metrics[key]
            if values:
                metrics[key] = sum(values) / len(values)
            else:
                metrics[key] = 0
        return metrics

# Base Video processor class
class BaseVideoProcessor:
    def __init__(self):
        self.frame_counter = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        
    def process_frame(self, frame):
        """Process a single frame using simple transformations"""
        # Calculate FPS
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        # Get original frame dimensions
        height, width = frame.shape[:2]
        
        # Simple OpenCV transformation (rotation + edge detection) for testing
        if self.frame_counter % 100 < 50:
            # Simple rotation for half the time
            M = cv2.getRotationMatrix2D((width/2, height/2), self.frame_counter % 360, 1)
            processed_img = cv2.warpAffine(frame, M, (width, height))
        else:
            # Edge detection for the other half
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            processed_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Add FPS counter and frame number to output
        info_text = f"FPS: {self.fps} | Frame: {self.frame_counter} | Size: {width}x{height}"
        cv2.putText(
            processed_img,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        self.frame_counter += 1
        return processed_img

# FasterLivePortrait video processor
class FasterLivePortraitProcessor(BaseVideoProcessor):
    def __init__(self, config_path, src_image_path, src_image_2_path=None, src_image_3_path=None, src_image_4_path=None, is_animal=False, debug=False):
        """Initialize the processor with source images and config"""
        super().__init__()
        
        if not FASTER_LIVE_PORTRAIT_AVAILABLE:
            raise ImportError("FasterLivePortrait is not available")
            
        self.config = OmegaConf.load(config_path)
        self.config.infer_params.flag_pasteback = True
        
        # Set up multiple source images
        self.src_image_paths = [src_image_path]
        if src_image_2_path:
            self.src_image_paths.append(src_image_2_path)
        if src_image_3_path:
            self.src_image_paths.append(src_image_3_path)
        if src_image_4_path:
            self.src_image_paths.append(src_image_4_path)
            
        self.current_source_index = 0
        self.src_images = []
        self.src_infos = []
        self.src_originals = []  # Store original images for display/reference
        self.is_animal = is_animal
        self.debug = debug
        
        try:
            self.pipeline = FasterLivePortraitPipeline(cfg=self.config, is_animal=is_animal)
            
            # First, load all source images and get their dimensions
            max_width = 0
            max_height = 0
            original_images = []
            
            # First pass: determine the largest dimensions
            for idx, path in enumerate(self.src_image_paths):
                logger.info(f"Loading source image {idx + 1}: {path}")
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Could not load source image {idx + 1}: {path}")
                
                # Handle transparency in source images
                if img.shape[-1] == 4:  # Image has alpha channel
                    logger.info(f"Source image {idx + 1} has transparency. Replacing transparent pixels with green.")
                    green_background = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * np.array([0, 255, 0], dtype=np.uint8)
                    alpha = img[:, :, 3] / 255.0
                    rgb = img[:, :, :3]
                    img = (rgb * alpha[:, :, np.newaxis] + green_background * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
                
                original_images.append(img)
                h, w = img.shape[:2]
                max_width = max(max_width, w)
                max_height = max(max_height, h)
                
            logger.info(f"Detected max dimensions across all sources: {max_width}x{max_height}")
            
            # Second pass: resize and prepare all images
            for idx, img in enumerate(original_images):
                path = self.src_image_paths[idx]
                h, w = img.shape[:2]
                
                # Store original image
                self.src_originals.append(img.copy())
                
                # If image is not the maximum size, resize it
                if h != max_height or w != max_width:
                    logger.info(f"Resizing source {idx + 1} from {w}x{h} to {max_width}x{max_height}")
                    
                    # Determine resize scale to maintain aspect ratio
                    scale = max(max_width / w, max_height / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # Resize to fill target dimensions while maintaining aspect ratio
                    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    
                    # Center crop to target dimensions
                    y_center = new_h // 2
                    x_center = new_w // 2
                    y_start = max(0, y_center - max_height // 2)
                    x_start = max(0, x_center - max_width // 2)
                    
                    # Handle edge case where resized image is still smaller than target in one dimension
                    if new_h < max_height:
                        y_start = 0
                    if new_w < max_width:
                        x_start = 0
                    
                    # Create a blank canvas of target size
                    normalized_img = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                    
                    # Calculate region to paste the resized image
                    paste_h = min(max_height, new_h)
                    paste_w = min(max_width, new_w)
                    
                    # Paste the center portion of the resized image
                    normalized_img[0:paste_h, 0:paste_w] = resized[
                        y_start:y_start + paste_h, 
                        x_start:x_start + paste_w
                    ]
                    
                    # Use the normalized image
                    img = normalized_img
                
                # Prepare the source image with the pipeline
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Save a temporary file for the pipeline to process
                temp_path = f"/tmp/normalized_source_{idx}.jpg"
                cv2.imwrite(temp_path, img)
                
                # Process with the pipeline
                success = self.pipeline.prepare_source(temp_path, realtime=True)
                if not success:
                    raise ValueError(f"Could not process source image {idx + 1}: {path}")
                
                # Store processed source data
                self.src_images.append(self.pipeline.src_imgs[0])
                self.src_infos.append(self.pipeline.src_infos[0])
                
                # Clean up temporary file
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            # Use dimensions from config
            self.input_size = (256, 256)
            logger.info(f"FasterLivePortrait initialized with {len(self.src_image_paths)} normalized source images")
            
        except Exception as e:
            logger.error(f"Error initializing FasterLivePortrait: {e}")
            raise
            
    def switch_source(self, index=None):
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
        
        logger.info(f"Switched to source image {self.current_source_index + 1}: {self.src_image_paths[self.current_source_index]}")
        return self.current_source_index
    
    def process_frame(self, frame, command=None):
        """Process a frame using FasterLivePortrait"""
        # Check for command to switch source image
        if command and command.get("action") == "switch_source":
            index = command.get("index")
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
            first_frame = self.frame_counter == 0
            dri_crop, out_crop, out_org, dri_motion_info = self.pipeline.run(
                frame, 
                current_src_img,
                current_src_info,
                first_frame=first_frame
            )
            
            # Add FPS and metadata to output frame
            if out_org is not None:
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)

                if self.debug:
                
                    # Create thumbnail of driving frame
                    thumbnail_height = int(out_org.shape[0] / 4)  # 1/4 of output height
                    thumbnail_width = int(thumbnail_height * frame.shape[1] / frame.shape[0])  # Maintain aspect ratio
                    thumbnail = cv2.resize(frame, (thumbnail_width, thumbnail_height))
                    
                    # Create a position for the thumbnail in the lower right corner with padding
                    padding = 10
                    y_offset = out_org.shape[0] - thumbnail_height - padding
                    x_offset = out_org.shape[1] - thumbnail_width - padding
                    
                    # Add border to thumbnail
                    border_color = (0, 255, 0)  # Green border
                    border_size = 2
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
                    alpha = 0.7  # 70% opacity
                    # Blend the thumbnail with the background
                    if roi.shape[:2] == thumbnail_with_border.shape[:2]:  # Ensure shapes match
                        blended_roi = cv2.addWeighted(thumbnail_with_border, alpha, roi, 1-alpha, 0)
                        out_org[y_offset:y_offset + roi_height, x_offset:x_offset + roi_width] = blended_roi
                    
                    # Add FPS and metadata text
                    info_text = f"FPS: {self.fps} | Frame: {self.frame_counter} | Input: {original_width}x{original_height}"
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
                    source_info = f"Source: {self.current_source_index + 1}/{len(self.src_image_paths)}"
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
                    source_name = os.path.basename(self.src_image_paths[self.current_source_index])
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
                    instruction_text = "Press 1/2/3 to switch source images" 
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
            
            return None
            
        except RuntimeError as e:
            logger.error(f"Processing error: {str(e)}")
            # Fall back to basic processor if FasterLivePortrait fails
            return BaseVideoProcessor.process_frame(self, frame)

# Server application setup
class Server:
    def __init__(self, config):
        self.app = FastAPI()
        self.connection_manager = ConnectionManager()
        self.config = config
        
        self._setup_routes()
        self._setup_middleware()
        
        # Try to initialize FasterLivePortrait processor, fall back to basic if it fails
        try:
            if config.use_basic_processor:
                logger.info("Basic processor requested via command line argument")
                raise ValueError("Basic processor explicitly requested")
            
            if not FASTER_LIVE_PORTRAIT_AVAILABLE:
                logger.error("FasterLivePortrait modules couldn't be imported")
                raise ImportError("FasterLivePortrait is not available")
            
            # Check if config path exists
            if not os.path.isfile(config.config_path):
                logger.error(f"Config file {config.config_path} not found")
                raise FileNotFoundError(f"Config file not found: {config.config_path}")
            
            # Check if source image exists
            if not os.path.isfile(config.source_image):
                logger.error(f"Source image {config.source_image} not found")
                raise FileNotFoundError(f"Source image not found: {config.source_image}")
                
            # Log all available source images
            source_images = []
            source_images.append(config.source_image)
            
            # Check additional source images if provided
            if config.source_image_2 and os.path.isfile(config.source_image_2):
                source_images.append(config.source_image_2)
                logger.info(f"Found additional source image 2: {config.source_image_2}")
            elif config.source_image_2:
                logger.warning(f"Source image 2 not found: {config.source_image_2}")
                
            if config.source_image_3 and os.path.isfile(config.source_image_3):
                source_images.append(config.source_image_3)
                logger.info(f"Found additional source image 3: {config.source_image_3}")
            elif config.source_image_3:
                logger.warning(f"Source image 3 not found: {config.source_image_3}")
                
            if config.source_image_4 and os.path.isfile(config.source_image_4):
                source_images.append(config.source_image_4)
                logger.info(f"Found additional source image 4: {config.source_image_4}")
            elif config.source_image_4:
                logger.warning(f"Source image 4 not found: {config.source_image_4}")
                
            logger.info(f"Initializing FasterLivePortrait with {len(source_images)} source images")
            logger.info(f"Config: {config.config_path}")
            logger.info(f"Is animal model: {config.is_animal}")
            
            try:
                # Handle transparency in source images
                for i, path in enumerate(source_images):
                    # Check if the image has an alpha channel (transparency)
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if img is not None and img.shape[-1] == 4:
                        # Image has alpha channel
                        logger.info(f"Source image {i+1} has transparency. Replacing transparent pixels with green.")
                        # Create a green background (BGR format)
                        green_background = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * np.array([0, 255, 0], dtype=np.uint8)
                        # Extract alpha channel
                        alpha = img[:, :, 3] / 255.0
                        # Convert to 3 channels (drop alpha)
                        rgb = img[:, :, :3]
                        # Alpha blend with green background
                        result = (rgb * alpha[:, :, np.newaxis] + green_background * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
                        # Save back to the file
                        cv2.imwrite(path, result)
                        logger.info(f"Updated source image {i+1} with transparent pixels replaced by green")
                
                self.processor = FasterLivePortraitProcessor(
                    config_path=config.config_path,
                    src_image_path=config.source_image,
                    src_image_2_path=config.source_image_2 if config.source_image_2 and os.path.isfile(config.source_image_2) else None,
                    src_image_3_path=config.source_image_3 if config.source_image_3 and os.path.isfile(config.source_image_3) else None,
                    src_image_4_path=config.source_image_4 if config.source_image_4 and os.path.isfile(config.source_image_4) else None,
                    is_animal=config.is_animal,
                    debug=config.debug,
                )
                logger.info(f"Successfully initialized FasterLivePortrait processor with {len(source_images)} source images")
                
                # Determine if multiple sources are available
                self.has_multiple_sources = len(source_images) > 1
                if self.has_multiple_sources:
                    logger.info(f"Multiple source images available: {len(source_images)}")
                    
            except Exception as processor_error:
                logger.error(f"Error initializing FasterLivePortrait processor: {type(processor_error).__name__}: {processor_error}")
                # Try to provide more detailed error information
                if hasattr(processor_error, "__traceback__"):
                    import traceback
                    tb_str = "".join(traceback.format_exception(None, processor_error, processor_error.__traceback__))
                    logger.error(f"Traceback:\n{tb_str}")
                raise
            
        except Exception as e:
            logger.error(f"Failed to initialize FasterLivePortrait processor: {e}")
            exit(1)
        
    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self):
        @self.app.get("/")
        async def get_index():
            return {"message": "FasterLivePortrait WebSocket Server"}
        
        @self.app.post("/create_session")
        async def create_session(request: dict = Body(...)):
            """Create a new session ID or validate an existing one"""
            # Allow custom session ID if provided, otherwise generate one
            session_id = request.get("session_id", str(uuid.uuid4()))
            return {
                "session_id": session_id,
                "status": "success"
            }
            
        @self.app.websocket("/ws/actor/{session_id}")
        async def actor_websocket(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for actors to stream video frames"""
            try:
                await self.connection_manager.connect_actor(session_id, websocket)
                
                # Register the processor for this session
                self.connection_manager.frame_processors[session_id] = self.processor
                
                # Send a confirmation to the actor
                response = {
                    "status": "connected", 
                    "session_id": session_id
                }
                
                # Add info about multiple sources if available
                if hasattr(self, 'has_multiple_sources') and self.has_multiple_sources:
                    source_count = len(self.processor.src_image_paths)
                    response["has_multiple_sources"] = True
                    response["source_count"] = source_count
                    response["sources"] = [os.path.basename(path) for path in self.processor.src_image_paths]
                    response["current_source"] = 0
                
                await websocket.send_json(response)
                
                # Setup last key press time to prevent too frequent source switching
                last_key_press_time = 0
                key_press_cooldown = 0.5  # seconds
                
                while True:
                    # Receive data from actor - could be a frame or a command
                    message = await websocket.receive()
                    
                    # Check if this is a text message (command) or binary (frame)
                    if "text" in message:
                        try:
                            # Parse text as JSON command
                            command = json.loads(message["text"])
                            current_time = time.time()
                            
                            # Handle source switching commands
                            if command.get("action") == "switch_source" and hasattr(self, 'has_multiple_sources') and self.has_multiple_sources:
                                # Check if we should process this key press (prevent too frequent switching)
                                if current_time - last_key_press_time >= key_press_cooldown:
                                    last_key_press_time = current_time
                                    
                                    # Get the requested source index
                                    index = command.get("index")
                                    if index is None:
                                        # Switch to next source if no specific index
                                        new_index = self.processor.switch_source()
                                    elif 0 <= index < len(self.processor.src_image_paths):
                                        # Switch to the requested source
                                        new_index = self.processor.switch_source(index)
                                    
                                    # Send confirmation back to actor
                                    await websocket.send_json({
                                        "status": "source_switched",
                                        "current_source": new_index,
                                        "source_name": os.path.basename(self.processor.src_image_paths[new_index])
                                    })
                                    
                                    # Notify viewers about the source switch
                                    await self.connection_manager.notify_viewers_source_switched(
                                        session_id, new_index, os.path.basename(self.processor.src_image_paths[new_index])
                                    )
                                    
                            # Handle key press events for source switching
                            elif command.get("action") == "key_press" and hasattr(self, 'has_multiple_sources') and self.has_multiple_sources:
                                key = command.get("key")
                                # Check if we should process this key press (prevent too frequent switching)
                                if current_time - last_key_press_time >= key_press_cooldown:
                                    last_key_press_time = current_time
                                    
                                    # Number keys 1-3 for source switching
                                    if key in ["1", "2", "3", "4"]:
                                        index = int(key) - 1
                                        if 0 <= index < len(self.processor.src_image_paths):
                                            new_index = self.processor.switch_source(index)
                                            
                                            # Send confirmation back to actor
                                            await websocket.send_json({
                                                "status": "source_switched",
                                                "current_source": new_index,
                                                "source_name": os.path.basename(self.processor.src_image_paths[new_index])
                                            })
                                            
                                            # Notify viewers about the source switch
                                            await self.connection_manager.notify_viewers_source_switched(
                                                session_id, new_index, os.path.basename(self.processor.src_image_paths[new_index])
                                            )
                                            
                        except json.JSONDecodeError:
                            logger.error(f"Received invalid JSON command: {message['text']}")
                        except Exception as e:
                            logger.error(f"Error processing command: {e}")
                            
                    elif "bytes" in message:
                        # Process binary data as video frame
                        frame_data = message["bytes"]
                        
                        # Decode image from binary data
                        nparr = np.frombuffer(frame_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # Add frame to processing queue (will replace any pending frame)
                            await self.connection_manager.receive_frame(session_id, frame)
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect_actor(session_id)
                logger.info(f"Actor disconnected: {session_id}")
            except Exception as e:
                logger.error(f"Error in actor websocket: {e}")
                self.connection_manager.disconnect_actor(session_id)
            
        @self.app.websocket("/ws/viewer/{session_id}")
        async def viewer_websocket(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for viewers to receive processed frames"""
            try:
                await self.connection_manager.connect_viewer(session_id, websocket)
                
                # Send confirmation to the viewer
                await websocket.send_json({
                    "status": "connected", 
                    "session_id": session_id,
                    "message": "Connected to stream. Waiting for video..."
                })
                
                # Keep the connection open, wait for heartbeats from the client
                while True:
                    # This will wait for any message from the client (like heartbeats)
                    message = await websocket.receive_text()
                    
                    # If it's a heartbeat, respond
                    if message == "heartbeat":
                        await websocket.send_json({"status": "heartbeat_ack"})
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect_viewer(session_id, websocket)
                logger.info(f"Viewer disconnected from session: {session_id}")
            except Exception as e:
                logger.error(f"Error in viewer websocket: {e}")
                self.connection_manager.disconnect_viewer(session_id, websocket)
        
        # Serve static files
        self.app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
        
        # Serve production-ready built frontend files
        if os.path.exists("frontend/dist"):
            self.app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="dist")
            logger.info("Serving production frontend from frontend/dist")
        elif os.path.exists("static"):
            # Fallback to the symbolic link created during Docker build
            self.app.mount("/", StaticFiles(directory="static", html=True), name="static")
            logger.info("Serving production frontend from static directory")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FasterLivePortrait WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run server on")
    parser.add_argument("--config-path", default="configs/trt_infer.yaml", help="Path to FasterLivePortrait config")
    parser.add_argument("--source-image", default="assets/examples/source/s2.jpg", help="Path to primary source image to animate")
    parser.add_argument("--source-image-2", help="Path to second source image to animate")
    parser.add_argument("--source-image-3", help="Path to third source image to animate")
    parser.add_argument("--source-image-4", help="Path to fourth source image to animate")
    parser.add_argument("--debug", action="store_true", help="Show Debug stats")
    parser.add_argument("--is-animal", action="store_true", help="Use animal model")
    parser.add_argument("--use-basic-processor", action="store_true", help="Use basic OpenCV processor instead of FasterLivePortrait")
    parser.add_argument("--ssl-cert", help="SSL certificate file")
    parser.add_argument("--ssl-key", help="SSL key file")
    
    args = parser.parse_args()
    
    server = Server(args)
    
    # Use SSL if certificates provided
    if args.ssl_cert and args.ssl_key:
        uvicorn.run(
            server.app, 
            host=args.host, 
            port=args.port,
            ssl_certfile=args.ssl_cert,
            ssl_keyfile=args.ssl_key
        )
    else:
        uvicorn.run(server.app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()