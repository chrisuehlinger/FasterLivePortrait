#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# WebSocket server for FasterLivePortrait
# This script enables live WebSocket streaming of actor video to animate a source image

import json
import logging
import uuid
import os
import cv2
import numpy as np
import time
from omegaconf import OmegaConf

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import uvicorn
from src.processors.base_video_processor import BaseVideoProcessor
from src.processors.faster_live_portrait_processor import FasterLivePortraitProcessor
from src.connection_manager import ConnectionManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("websocket_server")

# Connection manager for WebSockets

# Server application setup
class Server:
    def __init__(self, config):
        self.app = FastAPI()
        self.connection_manager = ConnectionManager()
        self.config = config
        
        self._setup_routes()
        self._setup_middleware()
        
        # Save proxy configuration
        self.proxy_target = getattr(config, "proxy_target", None)
        self.proxy_sources = []
        
        if getattr(config, "proxy_sources", None):
            try:
                self.proxy_sources = [int(idx) for idx in config.proxy_sources.split(',')]
                logger.info(f"Configured sources {self.proxy_sources} to be proxied to {self.proxy_target}")
                
                    
            except ValueError:
                logger.error(f"Invalid proxy sources format: {config.proxy_sources}. Should be comma-separated integers.")
        
        # Try to initialize FasterLivePortrait processor, fall back to basic if it fails
        try:
            if config.use_basic_processor:
                logger.info("Basic processor requested via command line argument")
                raise ValueError("Basic processor explicitly requested")
            
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
        
        @self.app.post("/switch_source/{session_id}/{index}")
        async def switch_source(request: dict = Body(...)):
            # Allow custom session ID if provided, otherwise generate one
            session_id = request.get("session_id", "performer1")
            index = int(request.get("index", 0))
            old_index = self.processor.current_source_index
            
            # Update proxy status if needed
            await self.connection_manager.handle_proxy_switch(session_id, old_index, index)
            
            self.processor.switch_source(index)
                                        
            # Notify viewers about the source switch
            await self.connection_manager.notify_viewers_source_switched(
                session_id, index, os.path.basename(self.processor.src_image_paths[index])
            )
            
            # Notify actors about the source switch
            await self.connection_manager.notify_actors_source_switched(
                session_id, index, os.path.basename(self.processor.src_image_paths[index])
            )

            return {
                "session_id": session_id,
                "status": "success"
            }
        
        # Add a new route to toggle animation pause state
        @self.app.post("/toggle_pause/{session_id}")
        async def toggle_pause(session_id: str, request: Request):
            data = await request.json()
            paused = data.get("paused", False)
            
            if session_id in self.connection_manager.frame_processors:
                processor = self.connection_manager.frame_processors[session_id]
                
                # Set pause animation flag in the pipeline
                if hasattr(processor, 'pipeline') and processor.pipeline:
                    processor.pipeline.pause_animation = paused
                    
                    # Broadcast pause state to all viewers
                    await self.connection_manager.notify_viewers_source_switched(
                        session_id, processor.current_source_index, os.path.basename(processor.src_image_paths[processor.current_source_index])
                    )
                    
                    return {"success": True, "session_id": session_id, "paused": paused}
            
            return {"success": False, "error": "Session not found"}
            
        @self.app.websocket("/ws/actor/{session_id}")
        async def actor_websocket(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for actors to stream video frames"""
            try:
                await self.connection_manager.connect_actor(session_id, websocket)
                
                # Register the processor for this session
                self.connection_manager.frame_processors[session_id] = self.processor
                
                # Configure proxy sources for this session if available
                if hasattr(self, 'proxy_sources') and self.proxy_sources and hasattr(self, 'proxy_target') and self.proxy_target:
                    self.connection_manager.set_proxy_sources(session_id, self.proxy_sources, self.proxy_target)
                
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
                    
                    # Add info about which sources are proxied
                    if hasattr(self, 'proxy_sources') and self.proxy_sources:
                        response["proxy_sources"] = self.proxy_sources
                        response["proxy_target"] = self.proxy_target
                
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
                                    old_index = self.processor.current_source_index
                                    
                                    if index is None:
                                        # Switch to next source if no specific index
                                        new_index = self.processor.switch_source()
                                    elif 0 <= index < len(self.processor.src_image_paths):
                                        # Switch to the requested source
                                        new_index = self.processor.switch_source(index)
                                    
                                    # Handle proxy switching if needed
                                    await self.connection_manager.handle_proxy_switch(session_id, old_index, new_index)
                                    
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
                                            old_index = self.processor.current_source_index
                                            new_index = self.processor.switch_source(index)
                                            
                                            # Handle proxy switching if needed
                                            await self.connection_manager.handle_proxy_switch(session_id, old_index, new_index)
                                            
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
                    
        @self.app.websocket("/ws/director/{session_id}")
        async def director_websocket(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for directors to send"""
            try:
                await self.connection_manager.connect_director(session_id, websocket)
                
                # Send confirmation to the director
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
                self.connection_manager.disconnect_director(session_id, websocket)
                logger.info(f"Director disconnected from session: {session_id}")
            except Exception as e:
                logger.error(f"Error in director websocket: {e}")
                self.connection_manager.disconnect_director(session_id, websocket)

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
    
    # Add proxy arguments
    parser.add_argument("--proxy-target", help="IP:port of remote WebSocket server for proxying (e.g. '192.168.1.100:8080')")
    parser.add_argument("--proxy-sources", help="Comma-separated list of source indices to proxy (0-based, e.g. '1,2,3')")
    
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