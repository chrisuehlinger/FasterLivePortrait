import asyncio
import cv2
import numpy as np
import time
from typing import Dict, List, Set
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("connection_manager")

from fastapi import WebSocket
from .proxy_connection import ProxyConnection

class ConnectionManager:
    def __init__(self):
        self.actor_connections: Dict[str, WebSocket] = {}
        self.director_connections: Dict[str, WebSocket] = {}
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
        
        # New attributes for proxying
        self.proxy_connections: Dict[str, ProxyConnection] = {}  # session_id -> ProxyConnection
        self.proxy_sources: Dict[str, Set[int]] = {}  # session_id -> {source_indices}
        self.current_proxy_active: Dict[str, bool] = {}  # session_id -> bool

        # New attribute to store the latest intensity value for each session
        self.animation_intensity: Dict[str, float] = {}
    
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
        
    async def connect_director(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        
        # Initialize list for this session if it doesn't exist
        if session_id not in self.director_connections:
            self.director_connections[session_id] = []
            
        # Add this viewer to the session's viewers
        self.director_connections[session_id].append(websocket)
        logger.info(f"Director connected to session: {session_id}, total directors: {len(self.director_connections[session_id])}")
        
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
            
            # Clean up proxy connections when actor disconnects
            if session_id in self.proxy_connections:
                asyncio.create_task(self.proxy_connections[session_id].disconnect())
                del self.proxy_connections[session_id]
            if session_id in self.proxy_sources:
                del self.proxy_sources[session_id]
            if session_id in self.current_proxy_active:
                del self.current_proxy_active[session_id]
            
    def disconnect_director(self, session_id: str, websocket: WebSocket):
        if session_id in self.director_connections:
            try:
                self.director_connections[session_id].remove(websocket)
                logger.info(f"Viewer disconnected from session: {session_id}, remaining directors: {len(self.director_connections[session_id])}")
                
                # Remove the session entry if no more directors
                if not self.director_connections[session_id]:
                    del self.director_connections[session_id]
                    
            except ValueError:
                # WebSocket was not in the list
                pass
            
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
    
    def set_proxy_sources(self, session_id: str, proxy_sources: List[int], proxy_target: str):
        """Configure which sources should be proxied for a session"""
        self.proxy_sources[session_id] = set(proxy_sources)
        
        # Create the proxy connection if it doesn't exist
        if session_id not in self.proxy_connections:
            self.proxy_connections[session_id] = ProxyConnection(proxy_target, session_id)
        
        logger.info(f"Session {session_id} configured to proxy sources {proxy_sources} to {proxy_target}")
    
    def is_source_proxied(self, session_id: str, source_index: int) -> bool:
        """Check if the current source should be proxied"""
        return session_id in self.proxy_sources and source_index in self.proxy_sources[session_id]
        
    async def handle_proxy_switch(self, session_id: str, from_index: int, to_index: int):
        """Handle switching between local and proxied sources"""
        was_proxied = self.is_source_proxied(session_id, from_index)
        will_be_proxied = self.is_source_proxied(session_id, to_index)
        
        # No change in proxy status
        if was_proxied == will_be_proxied:
            return
            
        if will_be_proxied:
            # Switching to a proxied source
            logger.info(f"Switching to proxied source {to_index} for session {session_id}")
            proxy = self.proxy_connections[session_id]
            await proxy.connect()
            self.current_proxy_active[session_id] = True
        else:
            # Switching from a proxied source to a local one
            logger.info(f"Switching from proxied source {from_index} to local source {to_index}")
            if session_id in self.proxy_connections:
                self.proxy_connections[session_id].disconnect()
            self.current_proxy_active[session_id] = False
    
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
                    
                    # Check if we're using a proxy for the current source
                    current_source_index = processor.current_source_index
                    use_proxy = (session_id in self.proxy_sources and 
                                current_source_index in self.proxy_sources[session_id] and
                                session_id in self.proxy_connections)
                    
                    if use_proxy:
                        # PROXY MODE: Send frame to remote server and wait for result
                        proxy_start_time = time.time()
                        proxy = self.proxy_connections[session_id]
                        
                        # Make sure we're connected
                        if not proxy.connected:
                            success = await proxy.connect()
                            if not success:
                                logger.error(f"Failed to connect to proxy for source {current_source_index}; falling back to local processing")
                                use_proxy = False
                        
                        if use_proxy:  # Still using proxy after connection check
                            # Encode the frame to JPEG for sending
                            encode_start_time = time.time()
                            success, encoded_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            if not success:
                                logger.error("Failed to encode frame for proxy")
                                continue
                            encode_time = time.time() - encode_start_time
                            self._update_metric("encode_time", encode_time * 1000)
                            
                            # Send to remote server
                            send_to_proxy_start_time = time.time()
                            sent = await proxy.send_frame(encoded_img.tobytes())
                            if not sent:
                                logger.error("Failed to send frame to proxy")
                                continue
                            send_to_proxy_time = time.time() - send_to_proxy_start_time
                            
                            # Wait for processed result
                            proxy_wait_start_time = time.time()
                            binary_img = await proxy.get_processed_frame()
                            proxy_wait_time = time.time() - proxy_wait_start_time
                            
                            if binary_img is None:
                                logger.error("Failed to get processed frame from proxy")
                                continue
                            
                            # Record total proxy time
                            proxy_time = time.time() - proxy_start_time
                            self._update_metric("process_time", proxy_time * 1000)
                            
                            # Send to viewers
                            send_start_time = time.time()
                            await self._send_bytes_to_viewers(binary_img, session_id)
                            send_time = time.time() - send_start_time
                            self._update_metric("send_time", send_time * 1000)
                            
                            # Total time
                            total_time = time.time() - total_start_time
                            self._update_metric("total_time", total_time * 1000)
                            
                            # Update stats
                            self.frames_processed[session_id] += 1
                            
                    if not use_proxy:
                        # LOCAL MODE: Process the frame locally as before
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
        logger.info(f"Notifying viewers of source switch in session {session_id}: {source_index} - {source_name}")
        if session_id not in self.viewer_connections:
            logger.warning(f"No viewers connected for session {session_id}")
            return
            
        message = {
            "status": "source_switched", 
            "current_source": source_index,
            "source_name": source_name
        }
        
        disconnected_viewers = []
        for viewer_websocket in self.viewer_connections[session_id]:
            try:
                logger.info(f"Sending source switch notification to viewer in session {session_id}: {message}")
                await viewer_websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending source switch notification to viewer in session {session_id}: {e}")
                disconnected_viewers.append(viewer_websocket)
                
        # Remove any disconnected viewers
        for websocket in disconnected_viewers:
            self.viewer_connections[session_id].remove(websocket)

    # New method to send notifications to viewers when source is switched
    async def notify_actors_source_switched(self, session_id: str, source_index: int, source_name: str):
        """Notify all actors that the source image has been switched"""
        logger.info(f"Notifying actors of source switch in session {session_id}: {source_index} - {source_name}")
        if session_id not in self.actor_connections:
            return
            
        message = {
            "status": "source_switched", 
            "current_source": source_index,
            "source_name": source_name
        }
        
        disconnected_actors = []
        for actor_websocket in self.actor_connections[session_id]:
            try:
                await actor_websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending source switch notification to actor in session {session_id}: {e}")
                disconnected_actors.append(actor_websocket)
                
        # Remove any disconnected actors
        for websocket in disconnected_actors:
            self.actor_connections[session_id].remove(websocket)

    async def broadcast_message(self, session_id: str, message: dict, exclude_websocket=None):
        """Broadcast a message to all connected clients for a session"""
        try:
            message_json = json.dumps(message)
            
            # Send to all viewers
            if session_id in self.viewer_connections:
                for websocket in self.viewer_connections[session_id]:
                    if websocket != exclude_websocket:
                        try:
                            await websocket.send_text(message_json)
                        except Exception as e:
                            logger.error(f"Error sending message to viewer in session {session_id}: {e}")
            
            # Send to actor
            if session_id in self.actor_connections:
                if self.actor_connections[session_id] != exclude_websocket:
                    try:
                        await self.actor_connections[session_id].send_text(message_json)
                    except Exception as e:
                        logger.error(f"Error sending message to actor in session {session_id}: {e}")
            
            # Send to directors
            if session_id in self.director_connections:
                for websocket in self.director_connections[session_id]:
                    if websocket != exclude_websocket:
                        try:
                            await websocket.send_text(message_json)
                        except Exception as e:
                            logger.error(f"Error sending message to director in session {session_id}: {e}")
                            
        except Exception as e:
            logger.error(f"Error broadcasting message for session {session_id}: {e}")
    
    async def handle_intensity_update(self, session_id: str, intensity: float, websocket: WebSocket):
        """Handle an intensity update from any client and broadcast to all others"""
        try:
            # Store the latest intensity value
            self.animation_intensity[session_id] = float(intensity)
            
            # Create the message to broadcast
            message = {
                "status": "intensity_update",
                "value": intensity
            }
            
            logger.info(f"Broadcasting intensity update for session {session_id}: {intensity}")
            
            # Broadcast to all clients except the sender
            await self.broadcast_message(session_id, message, exclude_websocket=websocket)
            
        except Exception as e:
            logger.error(f"Error handling intensity update for session {session_id}: {e}")

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
