import sys # Added
import os # Added
import argparse # Added
import time # Add time import

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import asyncio
import base64
import io
import logging
import os
import ssl  # Added for SSL support
from typing import Dict, List, Optional
from pathlib import Path  # Added for path handling

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse  # Added RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np # Added
import cv2 # Added
import torch # Added
from omegaconf import OmegaConf # Added
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline # Added

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Faster Live Portrait Server')
parser.add_argument('--cfg', type=str, default='configs/trt_infer.yaml', help='Path to the inference configuration file.')
parser.add_argument('--animal', action='store_true', help='Use the animal model variant.')
# Add other arguments if needed, e.g., for source image
parser.add_argument('--source_image', type=str, default='assets/examples/source/s2.jpg', help='Path to the source image.')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Global Variables for Pipeline ---
live_portrait_pipeline: Optional[FasterLivePortraitPipeline] = None
source_image_pil: Optional[Image.Image] = None
source_image_np: Optional[np.ndarray] = None # Store as numpy array (RGB)
source_info: Optional[dict] = None # Store prepared source info

# --- Environment variables and configuration ---
# Get the directory where server.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
certs_dir = os.path.join(current_dir, "certs")
ssl_keyfile = os.path.join(certs_dir, "key.pem")
ssl_certfile = os.path.join(certs_dir, "cert.pem")

# Check if SSL certificates exist
has_ssl = os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile)
if has_ssl:
    logger.info(f"SSL certificates found at: {certs_dir}")
else:
    logger.warning(f"SSL certificates not found at: {certs_dir}. Running in HTTP mode.")

# --- Mount Static Directory ---
# Check if the directory exists before mounting
if os.path.exists(static_dir) and os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Mounted static directory at: {static_dir}")
else:
    logger.warning(f"Static directory not found at: {static_dir}. Static files will not be served.")
    # Create static directory if it doesn't exist
    os.makedirs(static_dir, exist_ok=True)
    logger.info(f"Created static directory at: {static_dir}")

# --- WebSocket Connection Management ---

class ConnectionManager:
    def __init__(self):
        self.actor_connections: List[WebSocket] = [] # Should ideally be only one actor
        self.viewer_connections: List[WebSocket] = []
        self.latest_actor_frame: Optional[bytes] = None
        self.latest_processed_frame: Optional[bytes] = None
        self.processing_lock = asyncio.Lock()
        self.frame_updated = asyncio.Event() # Signal when a new actor frame arrives
        self.processed_frame_updated = asyncio.Event() # Signal when a new processed frame is ready
        self.latest_processing_time_ms: Optional[float] = None
        self.latest_fps: Optional[float] = None
        self._frame_times = [] # For FPS calculation
        self._max_frame_times = 50 # Calculate FPS over the last 50 frames

    async def connect_actor(self, websocket: WebSocket):
        await websocket.accept()
        # Allow only one actor connection for simplicity, disconnect others
        while self.actor_connections:
             existing_ws = self.actor_connections.pop()
             try:
                 await existing_ws.close(code=1000, reason="New actor connected")
             except Exception:
                 pass # Ignore errors closing old sockets
        self.actor_connections.append(websocket)
        self.latest_actor_frame = None # Reset frame on new connection
        logger.info("Actor connected.")

    async def disconnect_actor(self, websocket: WebSocket):
        if websocket in self.actor_connections:
            self.actor_connections.remove(websocket)
        self.latest_actor_frame = None # Clear frame on disconnect
        logger.info("Actor disconnected.")

    async def connect_viewer(self, websocket: WebSocket):
        await websocket.accept()
        self.viewer_connections.append(websocket)
        logger.info(f"Viewer connected. Total viewers: {len(self.viewer_connections)}")
        # Send the latest processed frame immediately if available
        if self.latest_processed_frame:
             try:
                 await websocket.send_bytes(self.latest_processed_frame)
             except WebSocketDisconnect:
                 self.disconnect_viewer(websocket) # Handle immediate disconnect
             except Exception as e:
                 logger.error(f"Error sending initial frame to viewer: {e}")


    async def disconnect_viewer(self, websocket: WebSocket):
        if websocket in self.viewer_connections:
            self.viewer_connections.remove(websocket)
        logger.info(f"Viewer disconnected. Total viewers: {len(self.viewer_connections)}")

    def set_actor_frame(self, frame_bytes: bytes):
        self.latest_actor_frame = frame_bytes
        self.frame_updated.set() # Signal that a new frame is available

    def set_processed_frame(self, frame_bytes: bytes):
        self.latest_processed_frame = frame_bytes
        self.processed_frame_updated.set() # Signal that a new processed frame is ready

    def update_stats(self, processing_time_sec: float):
        """Calculates and stores processing time and FPS."""
        self.latest_processing_time_ms = processing_time_sec * 1000

        # Update FPS calculation
        now = time.time()
        self._frame_times.append(now)
        # Keep only the last N timestamps
        self._frame_times = self._frame_times[-self._max_frame_times:]

        if len(self._frame_times) > 1:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self.latest_fps = (len(self._frame_times) - 1) / elapsed
            else:
                self.latest_fps = float('inf') # Avoid division by zero
        else:
            self.latest_fps = 0.0

        # Log the stats
        logger.info(f"Frame Processed: {self.latest_processing_time_ms:.2f} ms, FPS: {self.latest_fps:.2f}")

    async def send_stats_to_actor(self):
        """Sends the latest stats to the connected actor."""
        if not self.actor_connections:
            return # No actor connected

        actor_ws = self.actor_connections[0]
        stats_payload = {
            "type": "stats",
            "processing_time_ms": self.latest_processing_time_ms,
            "fps": self.latest_fps
        }
        try:
            await actor_ws.send_json(stats_payload)
        except WebSocketDisconnect:
            logger.info("Actor disconnected while trying to send stats.")
            # Disconnect logic is handled elsewhere
        except Exception as e:
            logger.error(f"Error sending stats to actor: {e}")

    async def broadcast_processed_frame(self):
        """Sends the latest processed frame to all connected viewers."""
        self.processed_frame_updated.clear() # Reset event until next frame
        if not self.latest_processed_frame:
            return

        # Encode to base64 for easy display in HTML img tag
        try:
            encoded_frame = base64.b64encode(self.latest_processed_frame).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
            return

        # Iterate over a copy of the list to allow safe removal during iteration
        for connection in list(self.viewer_connections):
            try:
                await connection.send_text(encoded_frame)
            except WebSocketDisconnect:
                # Remove the disconnected viewer immediately
                logger.info("Viewer disconnected during broadcast.")
                # Use the existing disconnect method for proper cleanup and logging
                await self.disconnect_viewer(connection)
            except Exception as e:
                # Handle other potential send errors (e.g., connection broken unexpectedly)
                logger.error(f"Error sending frame to viewer: {e}")
                # Assume connection is broken and remove it
                await self.disconnect_viewer(connection)

        # No need for the separate disconnected_viewers list and cleanup loop


manager = ConnectionManager()

# --- Background Processing Task ---

async def process_frames():
    """Continuously process the latest actor frame using FasterLivePortrait."""
    global live_portrait_pipeline, source_image_np, source_info # Access globals

    logger.info("Starting frame processing loop...")
    if not live_portrait_pipeline or source_image_np is None or source_info is None: # Check all required components
        logger.error("Pipeline or source image not initialized. Waiting for startup.")
        await asyncio.sleep(5) # Wait a bit for startup
        if not live_portrait_pipeline or source_image_np is None or source_info is None:
             logger.error("Pipeline failed to initialize. Processing loop cannot run.")
             return # Exit the task if initialization failed

    while True:
        await manager.frame_updated.wait() # Wait until a new frame arrives
        manager.frame_updated.clear() # Reset the event

        start_time = time.time() # Start timer
        processed_successfully = False

        async with manager.processing_lock:
            if manager.latest_actor_frame:
                frame_bytes = manager.latest_actor_frame
                try:
                    # 1. Decode frame bytes to PIL Image
                    img_pil = Image.open(io.BytesIO(frame_bytes)).convert('RGB')

                    # 2. Convert PIL Image to NumPy array (BGR format for pipeline.run)
                    driving_frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                    # 3. Call FasterLivePortrait inference function
                    # Ensure pipeline is ready (redundant check, but safe)
                    if live_portrait_pipeline and source_image_np is not None and source_info is not None:
                        # logger.debug(f"Processing frame {live_portrait_pipeline.frame_id}") # Reduce log spam
                        # The run method expects BGR driving frame, RGB source image, and prepared source info
                        # It returns: dri_crop, out_crop, out_org, dri_motion_info
                        # We want out_crop (index 1) instead of out_org (index 2)
                        result = live_portrait_pipeline.run(
                            driving_frame_bgr,
                            source_image_np, # Use the stored NumPy source image (RGB)
                            source_info,     # Use the stored prepared source info
                            # first_frame is handled internally by the pipeline's frame_id counter
                        )

                        # Use out_crop (result[1]) instead of out_org (result[2])
                        if result and result[1] is not None: # Check if out_crop exists
                            processed_frame_np_rgb = result[1] # This should be RGB
                            processed_pil = Image.fromarray(processed_frame_np_rgb)
                            # logger.info(f"Frame {live_portrait_pipeline.frame_id-1} processed successfully.") # Reduce log spam

                            # 4. Encode processed PIL Image back to bytes (e.g., JPEG)
                            buffer = io.BytesIO()
                            processed_pil.save(buffer, format="JPEG")
                            processed_bytes = buffer.getvalue()

                            # 5. Update the latest processed frame
                            manager.set_processed_frame(processed_bytes)
                            processed_successfully = True # Mark as processed
                        else:
                            # Handle cases where face detection might fail on a frame
                            logger.warning(f"Frame {live_portrait_pipeline.frame_id-1 if live_portrait_pipeline.frame_id > 0 else 0} processing failed or no face detected.")
                            # Optionally: send the previous frame again or a placeholder?
                            # For now, just don't update the processed frame.

                    else:
                        logger.warning("Pipeline not ready, skipping frame processing.")

                except Exception as e:
                    logger.error(f"Error processing frame: {e}", exc_info=True) # Log traceback
            # else: # No need for this else block, wait() handles it
            #     logger.debug("No actor frame to process.")

        # Calculate and update stats outside the lock if processing occurred
        if processed_successfully:
            processing_time = time.time() - start_time
            manager.update_stats(processing_time)
            # Send stats back to actor
            await manager.send_stats_to_actor()


async def broadcast_loop():
    """Continuously broadcasts the latest processed frame to viewers."""
    logger.info("Starting broadcast loop...")
    while True:
        await manager.processed_frame_updated.wait() # Wait for a new processed frame
        await manager.broadcast_processed_frame()
        # Add a small delay to control broadcast rate if needed,
        # but waiting on the event is usually sufficient.
        # await asyncio.sleep(0.01) # Optional small delay


@app.on_event("startup")
async def startup_event():
    global live_portrait_pipeline, source_image_pil, source_image_np, source_info # Modify globals

    logger.info("Server starting up. Initializing FasterLivePortrait pipeline...")
    try:
        # --- Load Configuration using OmegaConf ---
        config_path = args.cfg # Use path from argparse
        if not os.path.exists(config_path):
             logger.error(f"Configuration file not found: {config_path}")
             return

        # Load directly using OmegaConf.load
        cfg = OmegaConf.load(config_path)
        logger.info(f"Loaded configuration from: {config_path}")

        # --- Initialize Pipeline ---
        # Ensure CUDA is available if specified in config
        # Access device_id directly from the loaded cfg
        if hasattr(cfg, 'infer_params') and hasattr(cfg.infer_params, 'device_id') and cfg.infer_params.device_id >= 0:
            if not torch.cuda.is_available():
                logger.error("CUDA specified in config but not available. Pipeline initialization failed.")
                return
            torch.cuda.set_device(cfg.infer_params.device_id)
            logger.info(f"Using CUDA device: {cfg.infer_params.device_id}")
        else:
             logger.info("Using CPU or device_id not specified/negative.")

        # Initialize pipeline with the loaded config and animal flag
        live_portrait_pipeline = FasterLivePortraitPipeline(cfg=cfg, is_animal=args.animal)

        # --- Prepare Source Image ---
        source_image_path = args.source_image # Use path from argparse
        if not os.path.exists(source_image_path):
             logger.error(f"Source image not found: {source_image_path}")
             live_portrait_pipeline = None # Prevent processing
             return

        logger.info(f"Loading and preparing source image: {source_image_path}")

        # Load image and call prepare_source
        source_image_pil = Image.open(source_image_path).convert('RGB')
        source_image_np = np.array(source_image_pil) # Keep as RGB numpy array

        # Call prepare_source - it populates pipeline.src_imgs and pipeline.src_infos
        success = live_portrait_pipeline.prepare_source(source_image_path) # Pass path directly

        if success and live_portrait_pipeline.src_infos:
            # Store the prepared info from the pipeline instance
            # Assuming single image source, take the first item
            source_info = live_portrait_pipeline.src_infos[0]
            logger.info("Source image prepared successfully.")
        else:
            logger.error(f"Failed to prepare source image: {source_image_path}")
            live_portrait_pipeline = None # Prevent processing
            return

        logger.info("FasterLivePortrait pipeline initialized successfully.")

    except Exception as e:
        logger.error(f"Error during pipeline initialization: {e}", exc_info=True)
        live_portrait_pipeline = None # Ensure pipeline is None if init fails

    # Start background tasks only if initialization seems okay (pipeline is not None)
    if live_portrait_pipeline:
        asyncio.create_task(process_frames())
        asyncio.create_task(broadcast_loop())
    else:
        logger.error("Background processing tasks not started due to initialization failure.")


# --- FastAPI Routes ---

@app.get("/", response_class=HTMLResponse)
async def get_root():
    # Simple landing page linking to actor/viewer HTML files with proper protocol handling
    protocol = "https" if has_ssl else "http"
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FasterLivePortrait WebSocket Server</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #333;
            }}
            .link-container {{
                margin-top: 20px;
            }}
            a {{
                display: inline-block;
                margin: 10px;
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 4px;
            }}
            a:hover {{
                background-color: #45a049;
            }}
        </style>
    </head>
    <body>
        <h1>FasterLivePortrait WebSocket Server</h1>
        <p>This server connects actor webcams to viewer displays through the FasterLivePortrait AI model.</p>
        <div class="link-container">
            <a href="/actor">Actor View</a>
            <a href="/viewer">Viewer View</a>
        </div>
        <p>Running with {'HTTPS' if has_ssl else 'HTTP'} on port {8443 if has_ssl else 8000}</p>
    </body>
    </html>
    """

@app.get("/actor", response_class=FileResponse)
async def get_actor():
    actor_path = os.path.join(static_dir, "actor.html")
    if os.path.exists(actor_path):
        return FileResponse(actor_path)
    else:
        return HTMLResponse("<html><body>Actor page not found. Please ensure the static/actor.html file exists.</body></html>", status_code=404)

@app.get("/viewer", response_class=FileResponse)
async def get_viewer():
    viewer_path = os.path.join(static_dir, "viewer.html")
    if os.path.exists(viewer_path):
        return FileResponse(viewer_path)
    else:
        return HTMLResponse("<html><body>Viewer page not found. Please ensure the static/viewer.html file exists.</body></html>", status_code=404)

@app.websocket("/ws/actor")
async def websocket_actor_endpoint(websocket: WebSocket):
    await manager.connect_actor(websocket)
    try:
        while True:
            # Receive frame data (expecting bytes/blob)
            data = await websocket.receive_bytes()
            manager.set_actor_frame(data) # Store the latest frame, overwriting previous
            # Actor doesn't need responses in this setup
    except WebSocketDisconnect:
        logger.info("Actor WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Actor WebSocket error: {e}")
    finally:
        await manager.disconnect_actor(websocket)


@app.websocket("/ws/viewer")
async def websocket_viewer_endpoint(websocket: WebSocket):
    await manager.connect_viewer(websocket)
    try:
        # Keep connection open, broadcasting is handled by the background task
        while True:
            # Viewers primarily receive, but we need to keep the connection alive
            # and handle potential messages or disconnects.
            # A simple way is to wait for a message that might never come,
            # or periodically ping if needed (FastAPI handles some keep-alive).
            await websocket.receive_text() # Or receive_bytes, just to detect disconnect
    except WebSocketDisconnect:
        logger.info("Viewer WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Viewer WebSocket error: {e}")
    finally:
        manager.disconnect_viewer(websocket) # Ensure cleanup

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "ssl_enabled": has_ssl}

if __name__ == "__main__":
    # Argument parsing is now done globally above

    # Port configuration - use 8443 for HTTPS, 8000 for HTTP
    port = 8443 if has_ssl else 8000
    
    # Configure SSL context if certificates exist
    if has_ssl:
        # Create SSL context
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            ssl_context.load_cert_chain(ssl_certfile, ssl_keyfile)
            logger.info(f"Running with HTTPS on port {port}")
            uvicorn.run(
                "server:app", 
                host="0.0.0.0", 
                port=port, 
                reload=True, 
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile
            )
        except Exception as e:
            logger.error(f"Error loading SSL certificates: {e}")
            logger.warning("Falling back to HTTP")
            uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
    else:
        # Run without SSL
        logger.info(f"Running with HTTP on port {port}")
        uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)

