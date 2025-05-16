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
from typing import Dict, List, Optional, Any, Union, Protocol, Mapping
from typing_extensions import TypedDict
from omegaconf import DictConfig
import argparse
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig

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

# Define types for server configuration
@dataclass
class ServerConfig:
    """Strongly typed server configuration parsed from command line arguments"""
    host: str
    port: int
    config_path: str
    source_image: str
    source_image_2: Optional[str]
    source_image_3: Optional[str]
    source_image_4: Optional[str]
    debug: bool
    is_animal: bool
    use_basic_processor: bool
    ssl_cert: Optional[str]
    ssl_key: Optional[str]
    proxy_target: Optional[str]
    proxy_sources: Optional[str]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ServerConfig':
        """Create a ServerConfig from parsed command line arguments"""
        return cls(
            host=args.host,
            port=args.port,
            config_path=args.config_path,
            source_image=args.source_image,
            source_image_2=args.source_image_2,
            source_image_3=args.source_image_3,
            source_image_4=args.source_image_4,
            debug=args.debug,
            is_animal=args.is_animal,
            use_basic_processor=args.use_basic_processor,
            ssl_cert=args.ssl_cert,
            ssl_key=args.ssl_key,
            proxy_target=args.proxy_target,
            proxy_sources=args.proxy_sources
        )

# Server application setup
class Server:
    def __init__(self, config: ServerConfig) -> None:
        self.app: FastAPI = FastAPI()
        self.connection_manager: ConnectionManager = ConnectionManager()
        self.config: ServerConfig = config
        
        self._setup_routes()
        self._setup_middleware()
        
        # Save proxy configuration
        self.proxy_target: Optional[str] = config.proxy_target
        self.proxy_sources: List[int] = []
        
        if config.proxy_sources:
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
            source_images: List[str] = []
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
                # Check for FSP files and log appropriately
                for i, path in enumerate(source_images):
                    if path.lower().endswith('.fsp') or path.lower().endswith('.pkl'):
                        logger.info(f"Source image {i+1} is an FSP file: {path}")
                    else:
                        # Handle transparency in regular image files
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
                
                self.processor: FasterLivePortraitProcessor = FasterLivePortraitProcessor(
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
                self.has_multiple_sources: bool = len(source_images) > 1
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
        
    def _setup_middleware(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self) -> None:
        @self.app.get("/")
        async def get_index() -> Dict[str, str]:
            return {"message": "FasterLivePortrait WebSocket Server"}
        
        # Define request and response types
        class SessionRequest(TypedDict, total=False):
            session_id: Optional[str]
            
        class SessionResponse(TypedDict):
            session_id: str
            status: str
            
        @self.app.post("/create_session")
        async def create_session(request: SessionRequest = Body(...)) -> SessionResponse:
            """Create a new session ID or validate an existing one"""
            # Allow custom session ID if provided, otherwise generate one
            session_id: str = str(request.get("session_id", str(uuid.uuid4())))
            return {
                "session_id": session_id,
                "status": "success"
            }
        
        # Define switch source request and response types
        class SwitchSourceRequest(TypedDict, total=False):
            session_id: str
            index: int
            
        class SwitchSourceResponse(TypedDict):
            session_id: str
            status: str
            
        @self.app.post("/switch_source/{session_id}/{index}")
        async def switch_source(request: SwitchSourceRequest = Body(...)) -> SwitchSourceResponse:
            # Allow custom session ID if provided, otherwise generate one
            session_id_param: str = str(request.get("session_id", "performer1"))
            index_param: int = int(request.get("index", 0))
            source_old_index: int = self.processor.current_source_index
            
            # Update proxy status if needed
            await self.connection_manager.handle_proxy_switch(session_id_param, source_old_index, index_param)
            
            self.processor.switch_source(index_param)
                                        
            # Notify viewers about the source switch
            await self.connection_manager.notify_viewers_source_switched(
                session_id_param, index_param, os.path.basename(self.processor.src_image_paths[index_param])
            )
            
            # Notify actors about the source switch
            await self.connection_manager.notify_actors_source_switched(
                session_id_param, index_param, os.path.basename(self.processor.src_image_paths[index_param])
            )

            return {
                "session_id": session_id_param,
                "status": "success"
            }
        
        # Define request and response types for toggling pause state
        class PauseRequest(TypedDict, total=False):
            paused: bool
            
        class PauseResponse(TypedDict):
            session_id: str
            status: str
            paused: bool
            
        # Add a new route to toggle animation pause state
        @self.app.post("/toggle_pause/{session_id}")
        async def toggle_pause(session_id: str, request: Request) -> PauseResponse:
            data: PauseRequest = await request.json()
            paused: bool = data.get("paused", False)
            
            if session_id in self.connection_manager.frame_processors:
                processor = self.connection_manager.frame_processors[session_id]
                
                # Set pause animation flag in the pipeline
                if hasattr(processor, 'pipeline') and processor.pipeline:
                    processor.pipeline.pause_animation = paused
                    
                    # Broadcast pause state to all viewers
                    await self.connection_manager.notify_viewers_source_switched(
                        session_id, processor.current_source_index, os.path.basename(processor.src_image_paths[processor.current_source_index])
                    )
                    
                    return {"status": "success", "session_id": session_id, "paused": paused}
            
            return {"success": False, "error": "Session not found"}
            
        @self.app.websocket("/ws/actor/{session_id}")
        async def actor_websocket(websocket: WebSocket, session_id: str) -> None:
            """WebSocket endpoint for actors to stream video frames"""
            try:
                await self.connection_manager.connect_actor(session_id, websocket)
                
                # Register the processor for this session
                self.connection_manager.frame_processors[session_id] = self.processor
                
                # Configure proxy sources for this session if available
                if hasattr(self, 'proxy_sources') and self.proxy_sources and hasattr(self, 'proxy_target') and self.proxy_target:
                    self.connection_manager.set_proxy_sources(session_id, self.proxy_sources, self.proxy_target)
                
                # Define strongly typed actor response
                class ActorResponse(TypedDict, total=False):
                    status: str
                    session_id: str
                    has_multiple_sources: bool
                    source_count: int
                    sources: List[str]
                    current_source: int
                    proxy_sources: List[int]
                    proxy_target: Optional[str]

                # Prepare connection confirmation message
                response: ActorResponse = {
                    "action": "connected", 
                    "session_id": session_id,
                    "client_type": "actor"
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
                
                # Send to the actor directly for immediate feedback, then broadcast to all
                await websocket.send_json(response)
                await self.connection_manager.broadcast_message(session_id, response, exclude_websocket=websocket)
                
                # Setup last key press time to prevent too frequent source switching
                last_key_press_time: float = 0.0
                key_press_cooldown: float = 0.5  # seconds
                
                # Define message structure for WebSocket messages
                class WebSocketMessage(TypedDict, total=False):
                    type: str  # "websocket.receive", "websocket.disconnect", etc.
                    text: Optional[str]  # For text messages (commands)
                    bytes: Optional[bytes]  # For binary messages (frames)
                    
                while True:
                    # Receive data from actor - could be a frame or a command
                    message: WebSocketMessage = await websocket.receive()
                    
                    # Check if this is a text message (command) or binary (frame)
                    if "text" in message:
                        try:
                            # Define command structure
                            class Command(TypedDict, total=False):
                                action: str
                                index: Optional[int]
                                key: Optional[str]
                                value: Optional[float]
                                paused: Optional[bool]
                            
                            # Parse text as JSON command
                            command: Command = json.loads(message["text"])
                            current_time: float = time.time()
                            
                            # Handle source switching commands
                            if command.get("action") == "switch_source" and hasattr(self, 'has_multiple_sources') and self.has_multiple_sources:
                                # Check if we should process this key press (prevent too frequent switching)
                                if current_time - last_key_press_time >= key_press_cooldown:
                                    last_key_press_time = current_time
                                    
                                    # Get the requested source index
                                    index = command.get("index")
                                    old_index: int = self.processor.current_source_index
                                    
                                    if index is None:
                                        # Switch to next source if no specific index
                                        source_new_index: int = self.processor.switch_source()
                                    elif 0 <= index < len(self.processor.src_image_paths):
                                        # Switch to the requested source
                                        source_new_index: int = self.processor.switch_source(index)
                                    
                                    # Handle proxy switching if needed
                                    await self.connection_manager.handle_proxy_switch(session_id, old_index, source_new_index)
                                    
                                    # Create source switch message
                                    source_switch_message = {
                                        "action": "source_switched",
                                        "current_source": source_new_index,
                                        "source_name": os.path.basename(self.processor.src_image_paths[source_new_index]),
                                        "initiated_by": "actor"
                                    }
                                    
                                    # Send to actor directly for immediate feedback, then broadcast to all
                                    await websocket.send_json(source_switch_message)
                                    await self.connection_manager.broadcast_message(session_id, source_switch_message, exclude_websocket=websocket)
                                    
                            # Handle key press events for source switching
                            elif command.get("action") == "key_press" and hasattr(self, 'has_multiple_sources') and self.has_multiple_sources:
                                key: str = command.get("key")
                                # Check if we should process this key press (prevent too frequent switching)
                                if current_time - last_key_press_time >= key_press_cooldown:
                                    last_key_press_time = current_time
                                    
                                    # Number keys 1-3 for source switching
                                    if key in ["1", "2", "3", "4"]:
                                        index = int(key) - 1
                                        if 0 <= index < len(self.processor.src_image_paths):
                                            source_old_index: int = self.processor.current_source_index
                                            source_new_index: int = self.processor.switch_source(index)
                                            
                                            # Handle proxy switching if needed
                                            await self.connection_manager.handle_proxy_switch(session_id, source_old_index, source_new_index)
                                            
                                            # Create source switch message
                                            source_switch_message = {
                                                "action": "source_switched",
                                                "current_source": source_new_index,
                                                "source_name": os.path.basename(self.processor.src_image_paths[source_new_index]),
                                                "initiated_by": "actor",
                                                "key_pressed": key
                                            }
                                            
                                            # Send to actor directly for immediate feedback, then broadcast to all
                                            await websocket.send_json(source_switch_message)
                                            await self.connection_manager.broadcast_message(session_id, source_switch_message, exclude_websocket=websocket)
                                            
                        except json.JSONDecodeError:
                            logger.error(f"Received invalid JSON command: {message['text']}")
                        except Exception as e:
                            logger.error(f"Error processing command: {e}")
                            
                    elif "bytes" in message:
                        # Process binary data as video frame
                        frame_data: bytes = message["bytes"]
                        
                        # Decode image from binary data
                        nparr = np.frombuffer(frame_data, np.uint8)
                        frame: np.ndarray = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
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
        async def director_websocket(websocket: WebSocket, session_id: str) -> None:
            """WebSocket endpoint for directors to send commands and receive updates"""
            try:
                await self.connection_manager.connect_director(session_id, websocket)
                
                # Define strongly typed director response
                class DirectorResponse(TypedDict, total=False):
                    status: str
                    session_id: str
                    message: str
                    has_multiple_sources: bool
                    source_count: int
                    sources: List[str]
                    current_source: int

                # Prepare confirmation message for the director
                response: DirectorResponse = {
                    "action": "connected", 
                    "session_id": session_id,
                    "message": "Connected to stream. Waiting for video...",
                    "client_type": "director"
                }
                
                # Add info about multiple sources if available
                if hasattr(self, 'has_multiple_sources') and self.has_multiple_sources and session_id in self.connection_manager.frame_processors:
                    processor = self.connection_manager.frame_processors[session_id]
                    source_count = len(processor.src_image_paths)
                    response["has_multiple_sources"] = True
                    response["source_count"] = source_count
                    response["sources"] = [os.path.basename(path) for path in processor.src_image_paths]
                    response["current_source"] = processor.current_source_index
                
                # Send to the director directly for immediate feedback, then broadcast to all
                await websocket.send_json(response)
                await self.connection_manager.broadcast_message(session_id, response, exclude_websocket=websocket)
                
                # Send the current intensity if available
                if session_id in self.connection_manager.animation_intensity:
                    intensity_message = {
                        "action": "intensity_update",
                        "value": self.connection_manager.animation_intensity[session_id],
                        "client_type": "director"
                    }
                    await websocket.send_json(intensity_message)
                
                while True:
                    # This will wait for any message from the client (like heartbeats or commands)
                    message: str = await websocket.receive_text()
                    
                    # If it's a heartbeat, respond only to the sender
                    if message == "heartbeat":
                        await websocket.send_json({"action": "heartbeat_ack"})
                    else:
                        # Define intensity update command structure
                        class IntensityCommand(TypedDict, total=False):
                            action: str
                            value: float
                        
                        # Try to parse as JSON command
                        try:
                            data: IntensityCommand = json.loads(message)
                            
                            # Handle intensity update
                            if data.get("action") == "intensity_update":
                                intensity: float = float(data.get("value", 1.0))
                                await self.connection_manager.handle_intensity_update(session_id, intensity, websocket)
                            
                            # Handle set_motal messages
                            elif data.get("action") == "set_motal":
                                motal_value = data.get("value")
                                logger.info(f"Director set motal to {motal_value} for session {session_id}")
                                # Broadcast to all viewers and directors
                                await self.connection_manager.broadcast_message(
                                    session_id, 
                                    {
                                        "action": "set_motal",
                                        "value": motal_value,
                                        "initiated_by": "director"
                                    },
                                    exclude_websocket=websocket
                                )
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Received invalid JSON from director: {message}")
                        except Exception as e:
                            logger.error(f"Error processing director message: {e}")
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect_director(session_id, websocket)
                logger.info(f"Director disconnected from session: {session_id}")
            except Exception as e:
                logger.error(f"Error in director websocket: {e}")
                self.connection_manager.disconnect_director(session_id, websocket)

        @self.app.websocket("/ws/viewer/{session_id}")
        async def viewer_websocket(websocket: WebSocket, session_id: str) -> None:
            """WebSocket endpoint for viewers to receive processed frames"""
            try:
                await self.connection_manager.connect_viewer(session_id, websocket)
                
                # Define strongly typed viewer response
                class ViewerResponse(TypedDict, total=False):
                    status: str
                    session_id: str
                    message: str
                    current_source: int
                    source_name: str
                
                # Prepare confirmation message for the viewer
                response: ViewerResponse = {
                    "action": "connected", 
                    "session_id": session_id,
                    "message": "Connected to stream. Waiting for video...",
                    "client_type": "viewer"
                }
                
                # Add info about current source if available
                if hasattr(self, 'has_multiple_sources') and self.has_multiple_sources and session_id in self.connection_manager.frame_processors:
                    processor = self.connection_manager.frame_processors[session_id]
                    response["current_source"] = processor.current_source_index
                    response["source_name"] = os.path.basename(processor.src_image_paths[processor.current_source_index])
                
                # Send to the viewer directly for immediate feedback, then broadcast to all
                await websocket.send_json(response)
                await self.connection_manager.broadcast_message(session_id, response, exclude_websocket=websocket)
                
                # Send the current intensity if available
                if session_id in self.connection_manager.animation_intensity:
                    intensity_message = {
                        "action": "intensity_update",
                        "value": self.connection_manager.animation_intensity[session_id],
                        "client_type": "viewer"
                    }
                    await websocket.send_json(intensity_message)
                
                while True:
                    # This will wait for any message from the client (like heartbeats or viewer commands)
                    message: str = await websocket.receive_text()
                    
                    # If it's a heartbeat, respond only to the sender
                    if message == "heartbeat":
                        await websocket.send_json({"action": "heartbeat_ack"})
                    else:
                        # Define intensity update command structure
                        class IntensityCommand(TypedDict, total=False):
                            action: str
                            value: float
                        
                        # Try to parse as JSON command
                        try:
                            data: IntensityCommand = json.loads(message)
                            
                            # Handle intensity update (viewers might also send updates)
                            if data.get("action") == "intensity_update":
                                intensity: float = float(data.get("value", 1.0))
                                await self.connection_manager.handle_intensity_update(session_id, intensity, websocket)
                                
                        except json.JSONDecodeError:
                            pass  # Silently ignore invalid JSON from viewers
                        except Exception as e:
                            logger.error(f"Error processing viewer message: {e}")
                    
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

def main() -> None:
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
    
    # Create strongly typed server config
    config = ServerConfig.from_args(args)
    
    server = Server(config)
    
    # Use SSL if certificates provided
    if config.ssl_cert and config.ssl_key:
        uvicorn.run(
            server.app, 
            host=config.host, 
            port=config.port,
            ssl_certfile=config.ssl_cert,
            ssl_keyfile=config.ssl_key
        )
    else:
        uvicorn.run(server.app, host=config.host, port=config.port)

if __name__ == "__main__":
    main()