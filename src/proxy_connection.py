
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("proxy_connection")
from websockets.client import connect as ws_connect

class ProxyConnection:
    """Manages a connection to a remote WebSocket server for frame processing"""
    def __init__(self, target_url, session_id):
        self.target_url = target_url
        self.session_id = session_id
        self.ws_connection = None
        self.connected = False
        self.queue = asyncio.Queue(maxsize=1)  # Queue for processed frames from remote
        self.task = None
        logger.info(f"Created proxy connection to {target_url} for session {session_id}")
        
    async def connect(self):
        """Establish WebSocket connection to remote server"""
            
        try:
            # Connect to remote actor endpoint
            remote_actor_url = f"ws://{self.target_url}/ws/actor/{self.session_id}"
            logger.info(f"Connecting to proxy target: {remote_actor_url}")
            self.ws_connection = await ws_connect(remote_actor_url)
            self.connected = True
            
            # Start the receive task
            self.task = asyncio.create_task(self._receive_loop())
            logger.info(f"Successfully connected to proxy target for session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to proxy target: {e}")
            return False
            
    async def disconnect(self):
        """Close the connection to the remote server"""
        logger.info(f"Disconnecting proxy for session {self.session_id}")
        if self.task:
            self.task.cancel()
            self.task = None
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
        self.connected = False
            
    async def send_frame(self, frame_data):
        """Send a frame to the remote server for processing"""
        if not self.connected or not self.ws_connection:
            return False
        try:
            await self.ws_connection.send(frame_data)
            return True
        except Exception as e:
            logger.error(f"Error sending frame to proxy: {e}")
            self.connected = False
            return False
            
    async def get_processed_frame(self):
        """Get a processed frame from the remote server"""
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for frame from proxy for session {self.session_id}")
            return None
            
    async def _receive_loop(self):
        """Background task to receive processed frames from the remote server"""
        try:
            while self.connected:
                try:
                    message = await self.ws_connection.recv()
                    
                    # Clear queue if it has an old frame
                    while not self.queue.empty():
                        try:
                            self.queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                            
                    # Add new frame to queue
                    await self.queue.put(message)
                except Exception as e:
                    logger.error(f"Error receiving from proxy: {e}")
                    self.connected = False
                    break
        except asyncio.CancelledError:
            # Task is being cancelled
            pass
        except Exception as e:
            logger.error(f"Error in proxy receive loop: {e}")
            self.connected = False
