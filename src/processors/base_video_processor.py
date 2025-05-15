import cv2
import time
import numpy as np
from typing import Optional

class BaseVideoProcessor:
    def __init__(self) -> None:
        self.frame_counter: int = 0
        self.fps_counter: int = 0
        self.fps_start_time: float = time.time()
        self.fps: int = 0
        
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a single frame using simple transformations"""
        # Calculate FPS
        self.fps_counter += 1
        current_time: float = time.time()
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
        info_text: str = f"FPS: {self.fps} | Frame: {self.frame_counter} | Size: {width}x{height}"
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
