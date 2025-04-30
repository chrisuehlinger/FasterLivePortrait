#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preview Panel Component for FLPSF Editor
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import cv2
import threading
import tempfile
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QComboBox, QSlider, QCheckBox, QFrame,
    QGridLayout, QSizePolicy, QProgressBar, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QMutex, QSize
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QIcon

# Import controller
from controllers.app_controller import AppController

# Add parent directory to path for importing FasterLivePortrait modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Conditionally import FasterLivePortrait pipeline
try:
    from src.pipeline.faster_live_portrait_pipeline import FasterLivePortraitPipeline
    from src.utils.flpsf import verify_flpsf_file, extract_src_infos, save_flpsf_file
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False
    logger = logging.getLogger('flpsf_editor.views.preview_panel')
    logger.warning("Could not import FasterLivePortrait pipeline. Limited preview functionality available.")

logger = logging.getLogger('flpsf_editor.views.preview_panel')

class AnimationThread(QThread):
    """
    Thread for generating animation frames
    """
    # Signal to send a generated frame
    frame_ready = pyqtSignal(object, int)
    # Signal for progress updates
    progress_updated = pyqtSignal(int)
    # Signal when animation generation is complete
    generation_complete = pyqtSignal()
    # Signal for errors
    error_occurred = pyqtSignal(str)
    
    def __init__(self, controller: AppController, animation_type: str, 
                 show_landmarks: bool = True, show_mesh: bool = False, 
                 num_frames: int = 30):
        """
        Initialize animation thread
        
        Args:
            controller (AppController): Application controller
            animation_type (str): Type of animation to generate
            show_landmarks (bool): Whether to show landmarks
            show_mesh (bool): Whether to show 3D mesh
            num_frames (int): Number of frames to generate
        """
        super().__init__()
        self.controller = controller
        self.animation_type = animation_type
        self.show_landmarks = show_landmarks
        self.show_mesh = show_mesh
        self.num_frames = num_frames
        self.stop_flag = False
        self.mutex = QMutex()
        
    def run(self):
        """
        Run the animation generation
        """
        try:
            # Generate animation frames
            if not HAS_PIPELINE:
                # Fall back to placeholder animations if FasterLivePortrait pipeline is not available
                self._generate_placeholder_animation()
            else:
                self._generate_pipeline_animation()
                
            # Signal completion
            if not self.stop_flag:
                self.generation_complete.emit()
                
        except Exception as e:
            logger.error(f"Error generating animation: {str(e)}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Error generating animation: {str(e)}")
            
    def stop(self):
        """
        Stop the animation generation
        """
        self.mutex.lock()
        self.stop_flag = True
        self.mutex.unlock()
        
    def _generate_placeholder_animation(self):
        """
        Generate placeholder animation frames
        """
        # Check if we have the necessary data
        if self.controller.model.image is None:
            self.error_occurred.emit("No image available for preview")
            return
            
        # Make a copy of the original landmarks
        original_landmarks = None
        if self.controller.model.landmarks is not None:
            original_landmarks = self.controller.model.landmarks.copy()
        else:
            self.error_occurred.emit("No landmarks available for preview")
            return
            
        # Generate frames based on animation type
        if self.animation_type == "Rotate Head":
            self._generate_head_rotation_animation(original_landmarks)
        elif self.animation_type == "Blink":
            self._generate_blink_animation(original_landmarks)
        elif self.animation_type == "Talk":
            self._generate_talk_animation(original_landmarks)
        elif self.animation_type == "Smile":
            self._generate_smile_animation(original_landmarks)
        else:
            self._generate_head_rotation_animation(original_landmarks)
            
    def _generate_head_rotation_animation(self, original_landmarks):
        """Generate head rotation animation frames"""
        for i in range(self.num_frames):
            # Check if stopped
            if self.stop_flag:
                return
                
            # Create a modified image with simulated head rotation
            angle = 30 * np.sin(2 * np.pi * i / self.num_frames)
            
            # Make a copy of the image to draw on
            frame = self.controller.model.image.copy()
            
            # Apply a simple transform to landmarks
            center_x = np.mean(original_landmarks[:, 0])
            center_y = np.mean(original_landmarks[:, 1])
            
            # Create a rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            
            # Apply rotation to landmarks
            simulated_landmarks = np.zeros_like(original_landmarks)
            for j, (x, y) in enumerate(original_landmarks):
                new_x = rotation_matrix[0, 0] * x + rotation_matrix[0, 1] * y + rotation_matrix[0, 2]
                new_y = rotation_matrix[1, 0] * x + rotation_matrix[1, 1] * y + rotation_matrix[1, 2]
                simulated_landmarks[j] = [new_x, new_y]
            
            # Draw simulated landmarks if enabled
            if self.show_landmarks:
                for x, y in simulated_landmarks:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                    
            # Emit frame
            self.frame_ready.emit(frame, i)
            
            # Update progress
            self.progress_updated.emit(int(100 * (i + 1) / self.num_frames))
            
    def _generate_blink_animation(self, original_landmarks):
        """Generate eye blinking animation frames"""
        # Find eye landmarks - simplified for demonstration
        eye_indices = []
        for i in range(len(original_landmarks)):
            # Simple heuristic to find eye landmarks
            if original_landmarks[i, 1] < np.mean(original_landmarks[:, 1]):  # Upper face landmarks
                eye_indices.append(i)
                
        for i in range(self.num_frames):
            # Check if stopped
            if self.stop_flag:
                return
                
            # Create a modified image with simulated blinking
            frame = self.controller.model.image.copy()
            
            # Apply a simple transform to eye landmarks
            blink_factor = np.sin(np.pi * i / self.num_frames)
            
            # Copy original landmarks
            simulated_landmarks = original_landmarks.copy()
            
            # Modify eye landmarks
            for idx in eye_indices:
                # Move eye landmarks vertically to simulate blinking
                simulated_landmarks[idx, 1] += 3 * blink_factor
                
            # Draw simulated landmarks if enabled
            if self.show_landmarks:
                for x, y in simulated_landmarks:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                    
            # Emit frame
            self.frame_ready.emit(frame, i)
            
            # Update progress
            self.progress_updated.emit(int(100 * (i + 1) / self.num_frames))
            
    def _generate_talk_animation(self, original_landmarks):
        """Generate talking animation frames"""
        # Find mouth landmarks - simplified for demonstration
        mouth_indices = []
        for i in range(len(original_landmarks)):
            # Simple heuristic to find mouth landmarks
            if original_landmarks[i, 1] > np.mean(original_landmarks[:, 1]):  # Lower face landmarks
                mouth_indices.append(i)
                
        for i in range(self.num_frames):
            # Check if stopped
            if self.stop_flag:
                return
                
            # Create a modified image with simulated talking
            frame = self.controller.model.image.copy()
            
            # Apply a simple transform to mouth landmarks
            talk_factor = np.sin(2 * np.pi * i / self.num_frames) * 0.5 + 0.5
            
            # Copy original landmarks
            simulated_landmarks = original_landmarks.copy()
            
            # Modify mouth landmarks
            for idx in mouth_indices:
                # Move mouth landmarks vertically to simulate talking
                y_offset = 5 * talk_factor
                if simulated_landmarks[idx, 1] > np.mean(original_landmarks[mouth_indices, 1]):
                    simulated_landmarks[idx, 1] += y_offset
                else:
                    simulated_landmarks[idx, 1] -= y_offset
                
            # Draw simulated landmarks if enabled
            if self.show_landmarks:
                for x, y in simulated_landmarks:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                    
            # Emit frame
            self.frame_ready.emit(frame, i)
            
            # Update progress
            self.progress_updated.emit(int(100 * (i + 1) / self.num_frames))
            
    def _generate_smile_animation(self, original_landmarks):
        """Generate smile animation frames"""
        # Find mouth landmarks - simplified for demonstration
        mouth_indices = []
        for i in range(len(original_landmarks)):
            # Simple heuristic to find mouth landmarks
            if original_landmarks[i, 1] > np.mean(original_landmarks[:, 1]):  # Lower face landmarks
                mouth_indices.append(i)
                
        for i in range(self.num_frames):
            # Check if stopped
            if self.stop_flag:
                return
                
            # Create a modified image with simulated smile
            frame = self.controller.model.image.copy()
            
            # Apply a simple transform to mouth landmarks
            smile_factor = np.sin(np.pi * i / self.num_frames)
            
            # Copy original landmarks
            simulated_landmarks = original_landmarks.copy()
            
            # Modify mouth landmarks
            for idx in mouth_indices:
                # Move mouth landmarks to simulate smile
                x_diff = simulated_landmarks[idx, 0] - np.mean(original_landmarks[mouth_indices, 0])
                y_diff = simulated_landmarks[idx, 1] - np.mean(original_landmarks[mouth_indices, 1])
                
                # Corners of mouth move up and out
                simulated_landmarks[idx, 0] += x_diff * smile_factor * 0.2
                simulated_landmarks[idx, 1] -= y_diff * smile_factor * 0.2
                
            # Draw simulated landmarks if enabled
            if self.show_landmarks:
                for x, y in simulated_landmarks:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                    
            # Emit frame
            self.frame_ready.emit(frame, i)
            
            # Update progress
            self.progress_updated.emit(int(100 * (i + 1) / self.num_frames))
            
    def _generate_pipeline_animation(self):
        """
        Generate animation frames using FasterLivePortrait pipeline
        """
        # Create temporary FLPSF file
        with tempfile.NamedTemporaryFile(suffix='.flpsf', delete=False) as temp_file:
            flpsf_path = temp_file.name
        
        try:
            # Save current FLPSF data to temporary file
            flpsf_data = {
                'image': self.controller.model.image,
                'landmarks': self.controller.model.landmarks,
                'motion_params': self.controller.model.motion_params,
                'pasteback_info': self.controller.model.pasteback_info
            }
            save_flpsf_file(flpsf_path, flpsf_data)
            
            # Get driving video based on animation type
            driving_video = self._get_driving_video_for_animation()
            if driving_video is None:
                self.error_occurred.emit("Could not find driving video for animation")
                return
                
            # Initialize pipeline
            config_path = os.path.join(parent_dir, 'configs', 'onnx_infer.yaml')
            pipeline = FasterLivePortraitPipeline(config_path)
            
            # Get source info from FLPSF file
            src_infos = extract_src_infos(flpsf_path)
            if src_infos is None:
                self.error_occurred.emit("Failed to extract source info from FLPSF")
                return
                
            # Initialize pipeline with source info
            pipeline.initialize_source(src_infos)
            
            # Process driving video frames
            cap = cv2.VideoCapture(driving_video)
            if not cap.isOpened():
                self.error_occurred.emit(f"Failed to open driving video: {driving_video}")
                return
                
            # Calculate total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_frames = 0
            
            while True:
                # Check if stopped
                if self.stop_flag:
                    break
                    
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame through pipeline
                result_frame = pipeline.process_frame(frame)
                
                # Draw landmarks if enabled
                if self.show_landmarks and self.controller.model.landmarks is not None:
                    # Since the pipeline already processes landmarks, we would need to
                    # extract the transformed landmarks from the pipeline result
                    # This is just a placeholder for now
                    landmarks = self.controller.model.landmarks
                    for x, y in landmarks:
                        cv2.circle(result_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                        
                # Emit frame
                self.frame_ready.emit(result_frame, processed_frames)
                
                # Update progress
                processed_frames += 1
                self.progress_updated.emit(int(100 * processed_frames / min(total_frames, self.num_frames)))
                
                # Limit number of frames
                if processed_frames >= self.num_frames:
                    break
                    
            # Close video
            cap.release()
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(flpsf_path)
            except:
                pass
                
    def _get_driving_video_for_animation(self):
        """
        Get appropriate driving video path for the selected animation type
        
        Returns:
            str: Path to driving video
        """
        # Check for sample driving videos in assets folder
        assets_dir = os.path.join(parent_dir, 'assets')
        
        # Map animation types to potential video files
        video_map = {
            "Rotate Head": ["head_rotation.mp4", "rotate_sample.mp4", "head_turn.mp4"],
            "Blink": ["eye_blink.mp4", "blink_sample.mp4"],
            "Talk": ["talking.mp4", "speech_sample.mp4"],
            "Smile": ["smile.mp4", "expression_sample.mp4"]
        }
        
        # Check if animation type exists in map
        if self.animation_type not in video_map:
            return None
            
        # Try to find one of the videos
        for video_name in video_map[self.animation_type]:
            video_path = os.path.join(assets_dir, video_name)
            if os.path.exists(video_path):
                return video_path
                
        # If no specific video found, try to use a default video
        default_videos = ["sample_drive.mp4", "example.mp4", "driving.mp4"]
        for video_name in default_videos:
            video_path = os.path.join(assets_dir, video_name)
            if os.path.exists(video_path):
                return video_path
                
        # If no videos found, return None
        return None


class PreviewPanel(QWidget):
    """
    Preview panel component that shows animation preview
    """
    
    def __init__(self, controller: AppController):
        """
        Initialize the preview panel
        
        Args:
            controller (AppController): Application controller
        """
        super().__init__()
        
        self.controller = controller
        self.is_playing = False
        self.animation_frame = 0
        self.animation_frames = {}  # Dictionary mapping frame index to image
        self.animation_thread = None
        
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.on_animation_timer)
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """
        Setup UI components
        """
        # Main layout
        self.layout = QVBoxLayout(self)
        
        # Preview display
        self.preview_group = QGroupBox("Preview")
        self.preview_layout = QVBoxLayout(self.preview_group)
        
        # Image display
        self.image_container = QWidget()
        self.image_container.setMinimumSize(256, 256)  # Minimum size for the preview
        self.image_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_container.setStyleSheet("background-color: #333333;")
        
        self.image_layout = QVBoxLayout(self.image_container)
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(256, 256)
        self.image_label.setScaledContents(False)
        self.image_layout.addWidget(self.image_label)
        
        self.preview_layout.addWidget(self.image_container)
        
        # Progress bar for animation generation
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.preview_layout.addWidget(self.progress_bar)
        
        # Controls
        self.controls_layout = QHBoxLayout()
        self.preview_layout.addLayout(self.controls_layout)
        
        # Animation type
        self.animation_label = QLabel("Animation:")
        self.controls_layout.addWidget(self.animation_label)
        
        self.animation_combo = QComboBox()
        self.animation_combo.addItems(["Rotate Head", "Blink", "Talk", "Smile"])
        self.animation_combo.currentIndexChanged.connect(self.on_animation_changed)
        self.controls_layout.addWidget(self.animation_combo)
        
        # Play button
        self.play_button = QPushButton("▶ Generate & Play")
        self.play_button.clicked.connect(self.on_play_clicked)
        self.controls_layout.addWidget(self.play_button)
        
        # Save preview button
        self.save_button = QPushButton("💾 Save Animation")
        self.save_button.clicked.connect(self.on_save_animation)
        self.save_button.setEnabled(False)  # Disabled until animation is generated
        self.controls_layout.addWidget(self.save_button)
        
        # Options
        self.options_layout = QHBoxLayout()
        self.preview_layout.addLayout(self.options_layout)
        
        # Display options
        self.display_group = QGroupBox("Display Options")
        self.display_layout = QVBoxLayout(self.display_group)
        
        # Show landmarks
        self.show_landmarks_check = QCheckBox("Show Landmarks")
        self.show_landmarks_check.setChecked(True)
        self.show_landmarks_check.stateChanged.connect(self.on_display_options_changed)
        self.display_layout.addWidget(self.show_landmarks_check)
        
        # Show mesh
        self.show_mesh_check = QCheckBox("Show 3D Mesh")
        self.show_mesh_check.setChecked(False)
        self.show_mesh_check.stateChanged.connect(self.on_display_options_changed)
        self.display_layout.addWidget(self.show_mesh_check)
        
        # Show mask
        self.show_mask_check = QCheckBox("Show Pasteback Mask")
        self.show_mask_check.setChecked(False)
        self.show_mask_check.stateChanged.connect(self.on_display_options_changed)
        self.display_layout.addWidget(self.show_mask_check)
        
        self.options_layout.addWidget(self.display_group)
        
        # Speed control
        self.speed_group = QGroupBox("Animation Speed")
        self.speed_layout = QVBoxLayout(self.speed_group)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(5)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        self.speed_layout.addWidget(self.speed_slider)
        
        self.options_layout.addWidget(self.speed_group)
        
        # Main layout
        self.layout.addWidget(self.preview_group)
        
        # Show message for pipeline availability
        if not HAS_PIPELINE:
            self.warning_label = QLabel(
                "FasterLivePortrait pipeline not available. Using placeholder animations."
            )
            self.warning_label.setStyleSheet("color: #FF6700; background-color: #FFF3E0; padding: 5px;")
            self.warning_label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.warning_label)
        else:
            # Add frame count setting for pipeline animations
            self.frame_group = QGroupBox("Animation Settings")
            self.frame_layout = QHBoxLayout(self.frame_group)
            
            self.frame_label = QLabel("Frames:")
            self.frame_layout.addWidget(self.frame_label)
            
            self.frame_slider = QSlider(Qt.Horizontal)
            self.frame_slider.setMinimum(10)
            self.frame_slider.setMaximum(120)
            self.frame_slider.setValue(30)
            self.frame_slider.setTickPosition(QSlider.TicksBelow)
            self.frame_slider.setTickInterval(10)
            self.frame_layout.addWidget(self.frame_slider, 1)  # Stretch factor
            
            self.frame_count_label = QLabel("30")
            self.frame_slider.valueChanged.connect(
                lambda v: self.frame_count_label.setText(str(v))
            )
            self.frame_layout.addWidget(self.frame_count_label)
            
            self.layout.addWidget(self.frame_group)
        
    def update_view(self, event_type: str = None, data: Any = None) -> None:
        """
        Update the view based on model changes
        
        Args:
            event_type (str): Type of event that occurred
            data (Any): Additional data related to the event
        """
        if event_type == 'file_loaded' or event_type == 'image_loaded':
            self.stop_animation()
            self.animation_frames = {}  # Clear cached frames
            self.display_static_preview()
            self.save_button.setEnabled(False)
            
        elif event_type == 'landmarks_updated':
            self.display_static_preview()
            self.animation_frames = {}  # Clear cached frames
            self.save_button.setEnabled(False)
            
        elif event_type == 'param_updated':
            self.display_static_preview()
            self.animation_frames = {}  # Clear cached frames
            self.save_button.setEnabled(False)
            
        elif event_type == 'reset':
            self.stop_animation()
            self.animation_frames = {}  # Clear cached frames
            self.clear_preview()
            self.save_button.setEnabled(False)
            
        elif event_type == 'undo' or event_type == 'redo':
            self.display_static_preview()
            self.animation_frames = {}  # Clear cached frames
            self.save_button.setEnabled(False)
            
    def display_static_preview(self):
        """
        Display a static preview of the current image with landmarks
        """
        if self.controller.model.image is None:
            self.clear_preview()
            return
            
        # Make a copy of the image to draw on
        image = self.controller.model.image.copy()
        
        # Draw landmarks if enabled and available
        if self.show_landmarks_check.isChecked() and self.controller.model.landmarks is not None:
            for x, y in self.controller.model.landmarks:
                cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
                
        # Draw pasteback mask if enabled and available
        if self.show_mask_check.isChecked() and 'mask_ori_float' in self.controller.model.pasteback_info:
            mask = self.controller.model.pasteback_info['mask_ori_float']
            if mask is not None:
                # Create colored overlay
                overlay = image.copy()
                mask_visual = np.zeros_like(image)
                if len(mask.shape) == 2:
                    # Convert single channel mask to RGB
                    mask_rgb = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    mask_visual = cv2.applyColorMap(mask_rgb, cv2.COLORMAP_JET)
                else:
                    mask_visual = mask.astype(np.uint8)
                    
                cv2.addWeighted(mask_visual, 0.5, image, 0.5, 0, overlay)
                image = overlay
                
        # Convert to QImage and display
        self.display_image(image)
            
    def display_image(self, image):
        """
        Display an image in the preview panel
        
        Args:
            image (numpy.ndarray): Image to display
        """
        if image is None:
            return
            
        # Convert to RGB if needed (OpenCV uses BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Convert to QImage
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Create pixmap and scale to fit the container while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        
        # Get container size
        container_width = self.image_container.width()
        container_height = self.image_container.height()
        
        # Scale pixmap to fit container
        pixmap = pixmap.scaled(
            container_width - 10, container_height - 10,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # Set pixmap to label
        self.image_label.setPixmap(pixmap)
            
    def clear_preview(self):
        """
        Clear the preview display
        """
        self.image_label.clear()
        self.stop_animation()
            
    def generate_animation(self):
        """
        Generate animation frames
        """
        # Check if we have the necessary data
        if self.controller.model.image is None:
            self.show_error_message("No image loaded")
            return False
            
        if self.controller.model.landmarks is None:
            self.show_error_message("No landmarks detected")
            return False
            
        # Stop any existing animation and thread
        self.stop_animation()
        if self.animation_thread is not None:
            self.animation_thread.stop()
            self.animation_thread.wait()
            
        # Clear existing frames
        self.animation_frames = {}
        
        # Get animation type
        animation_type = self.animation_combo.currentText()
        
        # Get frame count from slider if available
        num_frames = 30
        if hasattr(self, 'frame_slider'):
            num_frames = self.frame_slider.value()
            
        # Create and start animation thread
        self.animation_thread = AnimationThread(
            self.controller,
            animation_type,
            self.show_landmarks_check.isChecked(),
            self.show_mesh_check.isChecked(),
            num_frames
        )
        
        # Connect signals
        self.animation_thread.frame_ready.connect(self.on_frame_ready)
        self.animation_thread.progress_updated.connect(self.on_generation_progress)
        self.animation_thread.generation_complete.connect(self.on_generation_complete)
        self.animation_thread.error_occurred.connect(self.show_error_message)
        
        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Update button
        self.play_button.setText("⏹ Cancel Generation")
        self.play_button.setEnabled(True)
        
        # Start thread
        self.animation_thread.start()
        
        return True
        
    def start_animation_playback(self):
        """
        Start the animation playback
        """
        if not self.animation_frames:
            self.show_error_message("No animation frames available")
            return
            
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Reset animation frame index
        self.animation_frame = 0
        
        # Start timer
        speed = self.speed_slider.value()
        interval = int(100 / speed)  # Faster speed = smaller interval
        self.animation_timer.start(interval)
        
        # Update button
        self.play_button.setText("■ Stop")
        self.is_playing = True
        
    def stop_animation(self):
        """
        Stop the animation playback
        """
        # Stop timer
        self.animation_timer.stop()
        
        # Update button text based on whether we have frames
        if self.animation_frames:
            self.play_button.setText("▶ Play")
            self.is_playing = False
        else:
            self.play_button.setText("▶ Generate & Play")
            self.is_playing = False
            
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Display static preview
        self.display_static_preview()
        
    def on_animation_timer(self):
        """
        Handle animation timer event to display the current frame
        """
        if not self.animation_frames:
            self.stop_animation()
            return
            
        # Get sorted frame indices
        frame_indices = sorted(self.animation_frames.keys())
        if not frame_indices:
            self.stop_animation()
            return
            
        # Calculate current frame index
        current_index = self.animation_frame % len(frame_indices)
        frame_idx = frame_indices[current_index]
        
        # Display current frame
        if frame_idx in self.animation_frames:
            self.display_image(self.animation_frames[frame_idx])
            
        # Increment frame index
        self.animation_frame = (self.animation_frame + 1) % len(frame_indices)
        
    def on_play_clicked(self):
        """
        Handle play/stop button click
        """
        if self.animation_thread and self.animation_thread.isRunning():
            # Cancel generation
            self.animation_thread.stop()
            self.animation_thread.wait()
            self.play_button.setText("▶ Generate & Play")
            self.progress_bar.setVisible(False)
        elif self.is_playing:
            # Stop playback
            self.stop_animation()
        elif self.animation_frames:
            # Start playback of existing frames
            self.start_animation_playback()
        else:
            # Generate new frames
            self.generate_animation()
            
    def on_frame_ready(self, frame, index):
        """
        Handle a new frame from the animation thread
        
        Args:
            frame (numpy.ndarray): Frame image
            index (int): Frame index
        """
        # Store the frame
        self.animation_frames[index] = frame
        
        # If this is the first frame and we're not already playing, show it
        if index == 0 and not self.is_playing:
            self.display_image(frame)
            
    def on_generation_progress(self, progress):
        """
        Handle progress update from the animation thread
        
        Args:
            progress (int): Progress percentage (0-100)
        """
        self.progress_bar.setValue(progress)
        
    def on_generation_complete(self):
        """
        Handle animation generation completion
        """
        # Start playback if we have frames
        if self.animation_frames:
            self.start_animation_playback()
            self.save_button.setEnabled(True)
        else:
            self.play_button.setText("▶ Generate & Play")
            self.progress_bar.setVisible(False)
            
    def show_error_message(self, message):
        """
        Show an error message
        
        Args:
            message (str): Error message
        """
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Preview Error", message)
        
        # Reset UI
        self.play_button.setText("▶ Generate & Play")
        self.progress_bar.setVisible(False)
        
    def on_animation_changed(self, index):
        """
        Handle animation type change
        
        Args:
            index: Combo box index
        """
        # If animation was playing, stop it
        if self.is_playing:
            self.stop_animation()
            
        # Mark that we need to regenerate animation
        self.animation_frames = {}
        self.save_button.setEnabled(False)
            
    def on_display_options_changed(self):
        """
        Handle display options change
        """
        # If animation is playing, just update the display options
        # The next time animation is generated, it will use the new options
        if not self.is_playing:
            # Just update the static preview
            self.display_static_preview()
        
        # Mark that we need to regenerate animation
        self.animation_frames = {}
        self.save_button.setEnabled(False)
            
    def on_speed_changed(self, value):
        """
        Handle speed slider change
        
        Args:
            value: Slider value
        """
        # If animation is playing, update timer interval
        if self.is_playing:
            interval = int(100 / value)  # Faster speed = smaller interval
            self.animation_timer.setInterval(interval)
            
    def on_save_animation(self):
        """
        Handle save animation button click
        """
        # Check if we have frames
        if not self.animation_frames:
            self.show_error_message("No animation frames to save")
            return
            
        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Animation",
            os.path.join(os.path.expanduser("~"), "animation.mp4"),
            "Video Files (*.mp4);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            # Get sorted frame indices
            frame_indices = sorted(self.animation_frames.keys())
            if not frame_indices:
                self.show_error_message("No animation frames to save")
                return
                
            # Get first frame to determine dimensions
            first_frame = self.animation_frames[frame_indices[0]]
            height, width = first_frame.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
            
            # Add all frames to video
            for idx in frame_indices:
                # Convert RGB to BGR if needed (OpenCV uses BGR)
                frame = self.animation_frames[idx]
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                out.write(frame)
                
            # Release video writer
            out.release()
            
            # Show success message
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Animation Saved",
                f"Animation saved successfully to:\n{file_path}"
            )
            
        except Exception as e:
            self.show_error_message(f"Error saving animation: {str(e)}")
            import traceback
            traceback.print_exc()