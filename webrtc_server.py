#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WebRTC server for FasterLivePortrait
# This script enables live WebRTC streaming of actor video to animate a source image

import asyncio
import json
import logging
import ssl
import uuid
import os
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, List
from omegaconf import OmegaConf

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame
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
logger = logging.getLogger("webrtc_server")

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.viewer_connections: Dict[str, WebSocket] = {}
        self.latest_frames: Dict[str, np.ndarray] = {}
        self.processed_frames: Dict[str, np.ndarray] = {}
        
    async def connect_actor(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"Actor connected: {session_id}")
        
    async def connect_viewer(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.viewer_connections[session_id] = websocket
        logger.info(f"Viewer connected: {session_id}")
        
    def disconnect_actor(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"Actor disconnected: {session_id}")
            
    def disconnect_viewer(self, session_id: str):
        if session_id in self.viewer_connections:
            del self.viewer_connections[session_id]
            logger.info(f"Viewer disconnected: {session_id}")
    
    async def send_frame_to_viewers(self, frame: np.ndarray, session_id: str):
        """Send processed frame to all connected viewers for a specific session"""
        if not self.viewer_connections:
            return
            
        # Convert frame to JPEG for efficient transmission
        success, encoded_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return
            
        binary_img = encoded_img.tobytes()
        
        # Send to all viewers of this session
        for viewer_id, connection in list(self.viewer_connections.items()):
            if viewer_id.startswith(f"{session_id}-"):
                try:
                    await connection.send_bytes(binary_img)
                except Exception as e:
                    logger.error(f"Error sending to viewer {viewer_id}: {e}")
                    self.disconnect_viewer(viewer_id)

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
    def __init__(self, config_path, src_image_path, is_animal=False):
        """Initialize the processor with source image and config"""
        super().__init__()
        
        if not FASTER_LIVE_PORTRAIT_AVAILABLE:
            raise ImportError("FasterLivePortrait is not available")
            
        self.config = OmegaConf.load(config_path)
        self.config.infer_params.flag_pasteback = True
        
        try:
            self.pipeline = FasterLivePortraitPipeline(cfg=self.config, is_animal=is_animal)
            success = self.pipeline.prepare_source(src_image_path, realtime=True)
            if not success:
                raise ValueError(f"Could not process source image: {src_image_path}")
                
            self.src_image = cv2.imread(src_image_path)
            self.src_rgb = cv2.cvtColor(self.src_image, cv2.COLOR_BGR2RGB)
            
            # Use dimensions from config
            self.input_size = (256, 256)
            logger.info(f"FasterLivePortrait initialized with input size: {self.input_size}")
            
        except Exception as e:
            logger.error(f"Error initializing FasterLivePortrait: {e}")
            raise
    
    def process_frame(self, frame):
        """Process a frame using FasterLivePortrait"""
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
            
            # Process frame with FasterLivePortrait
            first_frame = self.frame_counter == 0
            dri_crop, out_crop, out_org, dri_motion_info = self.pipeline.run(
                frame, 
                self.pipeline.src_imgs[0], 
                self.pipeline.src_infos[0],
                first_frame=first_frame
            )
            
            # Add FPS and metadata to output frame
            if out_org is not None:
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
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
                return out_org
            
            return None
            
        except RuntimeError as e:
            logger.error(f"Processing error: {str(e)}")
            # Fall back to basic processor if FasterLivePortrait fails
            return BaseVideoProcessor.process_frame(self, frame)

# WebRTC video track for processing frames
class RTCVideoProcessor(MediaStreamTrack):
    kind = "video"
    
    def __init__(self, track, processor, connection_manager, session_id):
        super().__init__()
        self.track = track
        self.processor = processor
        self.connection_manager = connection_manager
        self.session_id = session_id
        self.processing = False
        self.latest_frame = None
        self.dropped_frames = 0
        self.total_frames = 0
        self.last_stats_time = time.time()
        
    async def recv(self):
        self.total_frames += 1
        
        # Get frame from the original track
        frame = await self.track.recv()
        
        # If we're currently processing a frame, store this one as latest and return original
        if self.processing:
            self.latest_frame = frame
            self.dropped_frames += 1
            return frame
            
        # Start processing
        self.processing = True
        
        try:
            # Check if we have a newer frame that came in while getting this one
            process_frame = self.latest_frame if self.latest_frame else frame
            self.latest_frame = None
            
            # Convert to OpenCV format
            img = process_frame.to_ndarray(format="bgr24")
            
            # Process the frame
            processed_img = self.processor.process_frame(img)
            
            # If processing failed, use original frame
            if processed_img is None:
                processed_img = img
            
            # Log stats periodically
            current_time = time.time()
            if current_time - self.last_stats_time > 5.0:  # Every 5 seconds
                drop_percentage = (self.dropped_frames / max(1, self.total_frames)) * 100
                logger.info(f"Frame stats - Processed: {self.total_frames - self.dropped_frames}, " +
                          f"Dropped: {self.dropped_frames} ({drop_percentage:.1f}%), " +
                          f"Total: {self.total_frames}")
                self.dropped_frames = 0
                self.total_frames = 0
                self.last_stats_time = current_time
            
            # Send the processed frame to all viewers via websocket
            await self.connection_manager.send_frame_to_viewers(processed_img, self.session_id)
            
            # Convert back to VideoFrame
            new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
            new_frame.pts = process_frame.pts
            new_frame.time_base = process_frame.time_base
            return new_frame
        finally:
            # End processing state
            self.processing = False

# Server application setup
class Server:
    def __init__(self, config):
        self.app = FastAPI()
        self.connection_manager = ConnectionManager()
        self.config = config
        self.peer_connections = {}
        
        # Try to initialize FasterLivePortrait processor, fall back to basic if it fails
        try:
            if config.use_basic_processor:
                raise ValueError("Basic processor requested")
                
            self.processor = FasterLivePortraitProcessor(
                config_path=config.config_path,
                src_image_path=config.source_image,
                is_animal=config.is_animal
            )
            logger.info("Using FasterLivePortrait processor")
        except Exception as e:
            logger.warning(f"Failed to initialize FasterLivePortrait processor: {e}")
            logger.info("Falling back to basic video processor")
            self.processor = BaseVideoProcessor()
        
        self._setup_routes()
        self._setup_middleware()
        
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
            return {"message": "FasterLivePortrait WebRTC Server"}
            
        @self.app.post("/offer")
        async def receive_offer(request: dict = Body(...)):
            offer = RTCSessionDescription(sdp=request["sdp"], type=request["type"])
            pc = RTCPeerConnection()
            
            # Allow custom session ID if provided, otherwise generate one
            session_id = request.get("session_id", str(uuid.uuid4()))
            logger.info(f"New connection with session ID: {session_id}")
            
            self.peer_connections[session_id] = pc
            
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                if pc.connectionState == "failed" or pc.connectionState == "closed":
                    if session_id in self.peer_connections:
                        del self.peer_connections[session_id]
            
            @pc.on("track")
            def on_track(track):
                if track.kind == "video":
                    local_video = RTCVideoProcessor(
                        track=track,
                        processor=self.processor,
                        connection_manager=self.connection_manager,
                        session_id=session_id
                    )
                    pc.addTrack(local_video)
                
            # Set the remote description
            await pc.setRemoteDescription(offer)
            
            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            return {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "session_id": session_id
            }
            
        @self.app.websocket("/ws/viewer/{session_id}")
        async def viewer_websocket(websocket: WebSocket, session_id: str):
            try:
                # Modify session_id to include a unique viewer ID
                viewer_id = f"{session_id}-viewer-{uuid.uuid4()}"
                await self.connection_manager.connect_viewer(viewer_id, websocket)
                while True:
                    # Keep connection alive and wait for frames
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.connection_manager.disconnect_viewer(viewer_id)
                
        # Serve static files
        self.app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FasterLivePortrait WebRTC Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run server on")
    parser.add_argument("--config-path", default="configs/onnx_infer.yaml", help="Path to FasterLivePortrait config")
    parser.add_argument("--source-image", default="assets/examples/source/s2.jpg", help="Path to source image to animate")
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