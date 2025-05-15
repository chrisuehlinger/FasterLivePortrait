import asyncio
import logging
from typing import Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proxy_connection")
from websockets import connect
from websockets.legacy.client import WebSocketClientProtocol

class ProxyConnection:
    """Manages a connection to a remote WebSocket server for frame processing"""
    def __init__(self, target_url: str, session_id: str) -> None:
        self.target_url: str = target_url
        self.session_id: str = session_id
        self.ws_connection: Optional[WebSocketClientProtocol] = None
        self.connected: bool = False
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1)  # Queue for processed frames from remote
        self.task: Optional[asyncio.Task] = None
        logger.info(f"Created proxy connection to {target_url} for session {session_id}")
        
    async def connect(self) -> bool:
        """Establish WebSocket connection to remote server"""
            
        try:
            # Connect to remote actor endpoint
            remote_actor_url: str = f"ws://{self.target_url}/ws/actor/{self.session_id}"
            logger.info(f"Connecting to proxy target: {remote_actor_url}")
            self.ws_connection = await connect(remote_actor_url)
            self.connected = True
            
            # Start the receive task
            self.task = asyncio.create_task(self._receive_loop())
            logger.info(f"Successfully connected to proxy target for session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to proxy target: {e}")
            return False
            
    async def disconnect(self) -> None:
        """Close the connection to the remote server"""
        logger.info(f"Disconnecting proxy for session {self.session_id}")
        if self.task:
            self.task.cancel()
            self.task = None
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
        self.connected = False
            
    async def send_frame(self, frame_data: bytes) -> bool:
        """Send a frame to the remote server for processing"""
        if not self.connected or not self.ws_connection:
            logger.error("Cannot send frame: not connected to proxy")
            return False
            
        try:
            await self.ws_connection.send(frame_data)
            return True
        except Exception as e:
            logger.error(f"Error sending frame to proxy: {e}")
            self.connected = False
            return False
            
    async def get_processed_frame(self) -> Optional[bytes]:
        """Get a processed frame from the remote server"""
        try:
            # Use the queue with a timeout
            return await asyncio.wait_for(self.queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for processed frame from proxy")
            return None
        except Exception as e:
            logger.error(f"Error getting processed frame: {e}")
            return None
            
    async def _receive_loop(self) -> None:
        """Background task to receive frames from the remote server"""
        try:
            if not self.ws_connection:
                logger.error("Cannot start receive loop: not connected to proxy")
                return
                
            # First message is the connection status
            response = await self.ws_connection.recv()
            if isinstance(response, str):
                logger.info(f"Proxy connection setup complete: {response}")
                
            # Then we enter the main receive loop
            while True:
                try:
                    message = await self.ws_connection.recv()
                    
                    # If it's binary data (a frame)
                    if isinstance(message, bytes):
                        # Clear the queue to make room for new frame
                        while not self.queue.empty():
                            try:
                                self.queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                                
                        # Add the frame to the queue
                        await self.queue.put(message)
                    # If it's a text message (status update or command)
                    elif isinstance(message, str):
                        logger.info(f"Received proxy message: {message}")
                        
                except Exception as e:
                    logger.error(f"Error in proxy receive loop: {e}")
                    break
                    
        except asyncio.CancelledError:
            logger.info(f"Proxy receive loop for {self.session_id} cancelled")
        except Exception as e:
            logger.error(f"Error in proxy receive loop: {e}")
            self.connected = False
