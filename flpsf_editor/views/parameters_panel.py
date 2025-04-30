#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parameters Panel Component for FLPSF Editor
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QGroupBox, QSlider, QDoubleSpinBox, QTabWidget,
    QFormLayout, QCheckBox, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QSignalBlocker

# Import controller
from controllers.app_controller import AppController

logger = logging.getLogger('flpsf_editor.views.parameters_panel')

class ParametersPanel(QScrollArea):
    """
    Parameters panel component that displays and edits FLPSF parameters
    """
    
    def __init__(self, controller: AppController):
        """
        Initialize the parameters panel
        
        Args:
            controller (AppController): Application controller
        """
        super().__init__()
        
        self.controller = controller
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """
        Setup UI components
        """
        # Main widget and layout
        self.main_widget = QWidget()
        self.layout = QVBoxLayout(self.main_widget)
        
        # Set up scroll area
        self.setWidget(self.main_widget)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_motion_tab()
        self.create_derived_tab()
        self.create_norm_tab()
        self.create_pasteback_tab()
        
        # Add buttons
        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)
        
        self.compute_all_button = QPushButton("Compute All Parameters")
        self.compute_all_button.clicked.connect(self.on_compute_all)
        self.button_layout.addWidget(self.compute_all_button)
        
        self.reset_button = QPushButton("Reset Parameters")
        self.reset_button.clicked.connect(self.on_reset_params)
        self.button_layout.addWidget(self.reset_button)
        
    def create_motion_tab(self):
        """
        Create Motion Parameters Tab
        """
        # Create tab
        self.motion_tab = QWidget()
        self.motion_layout = QVBoxLayout(self.motion_tab)
        self.tab_widget.addTab(self.motion_tab, "Motion")
        
        # Pose Group
        self.pose_group = QGroupBox("Pose")
        self.pose_layout = QFormLayout(self.pose_group)
        self.motion_layout.addWidget(self.pose_group)
        
        # Pitch
        self.pitch_layout = QHBoxLayout()
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setMinimum(-90)
        self.pitch_slider.setMaximum(90)
        self.pitch_slider.setTickPosition(QSlider.TicksBelow)
        self.pitch_spinner = QDoubleSpinBox()
        self.pitch_spinner.setRange(-90, 90)
        self.pitch_spinner.setSingleStep(1)
        self.pitch_spinner.setDecimals(2)
        self.pitch_layout.addWidget(self.pitch_slider, 7)
        self.pitch_layout.addWidget(self.pitch_spinner, 3)
        self.pose_layout.addRow("Pitch:", self.pitch_layout)
        
        # Yaw
        self.yaw_layout = QHBoxLayout()
        self.yaw_slider = QSlider(Qt.Horizontal)
        self.yaw_slider.setMinimum(-90)
        self.yaw_slider.setMaximum(90)
        self.yaw_slider.setTickPosition(QSlider.TicksBelow)
        self.yaw_spinner = QDoubleSpinBox()
        self.yaw_spinner.setRange(-90, 90)
        self.yaw_spinner.setSingleStep(1)
        self.yaw_spinner.setDecimals(2)
        self.yaw_layout.addWidget(self.yaw_slider, 7)
        self.yaw_layout.addWidget(self.yaw_spinner, 3)
        self.pose_layout.addRow("Yaw:", self.yaw_layout)
        
        # Roll
        self.roll_layout = QHBoxLayout()
        self.roll_slider = QSlider(Qt.Horizontal)
        self.roll_slider.setMinimum(-90)
        self.roll_slider.setMaximum(90)
        self.roll_slider.setTickPosition(QSlider.TicksBelow)
        self.roll_spinner = QDoubleSpinBox()
        self.roll_spinner.setRange(-90, 90)
        self.roll_spinner.setSingleStep(1)
        self.roll_spinner.setDecimals(2)
        self.roll_layout.addWidget(self.roll_slider, 7)
        self.roll_layout.addWidget(self.roll_spinner, 3)
        self.pose_layout.addRow("Roll:", self.roll_layout)
        
        # Scale
        self.scale_layout = QHBoxLayout()
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(50)
        self.scale_slider.setMaximum(200)
        self.scale_slider.setTickPosition(QSlider.TicksBelow)
        self.scale_spinner = QDoubleSpinBox()
        self.scale_spinner.setRange(0.5, 2.0)
        self.scale_spinner.setSingleStep(0.1)
        self.scale_spinner.setDecimals(2)
        self.scale_layout.addWidget(self.scale_slider, 7)
        self.scale_layout.addWidget(self.scale_spinner, 3)
        self.pose_layout.addRow("Scale:", self.scale_layout)
        
        # Translation Group
        self.translation_group = QGroupBox("Translation")
        self.translation_layout = QFormLayout(self.translation_group)
        self.motion_layout.addWidget(self.translation_group)
        
        # Translation X
        self.tx_layout = QHBoxLayout()
        self.tx_slider = QSlider(Qt.Horizontal)
        self.tx_slider.setMinimum(-100)
        self.tx_slider.setMaximum(100)
        self.tx_slider.setTickPosition(QSlider.TicksBelow)
        self.tx_spinner = QDoubleSpinBox()
        self.tx_spinner.setRange(-10, 10)
        self.tx_spinner.setSingleStep(0.1)
        self.tx_spinner.setDecimals(2)
        self.tx_layout.addWidget(self.tx_slider, 7)
        self.tx_layout.addWidget(self.tx_spinner, 3)
        self.translation_layout.addRow("X:", self.tx_layout)
        
        # Translation Y
        self.ty_layout = QHBoxLayout()
        self.ty_slider = QSlider(Qt.Horizontal)
        self.ty_slider.setMinimum(-100)
        self.ty_slider.setMaximum(100)
        self.ty_slider.setTickPosition(QSlider.TicksBelow)
        self.ty_spinner = QDoubleSpinBox()
        self.ty_spinner.setRange(-10, 10)
        self.ty_spinner.setSingleStep(0.1)
        self.ty_spinner.setDecimals(2)
        self.ty_layout.addWidget(self.ty_slider, 7)
        self.ty_layout.addWidget(self.ty_spinner, 3)
        self.translation_layout.addRow("Y:", self.ty_layout)
        
        # Translation Z
        self.tz_layout = QHBoxLayout()
        self.tz_slider = QSlider(Qt.Horizontal)
        self.tz_slider.setMinimum(-100)
        self.tz_slider.setMaximum(100)
        self.tz_slider.setTickPosition(QSlider.TicksBelow)
        self.tz_spinner = QDoubleSpinBox()
        self.tz_spinner.setRange(-10, 10)
        self.tz_spinner.setSingleStep(0.1)
        self.tz_spinner.setDecimals(2)
        self.tz_layout.addWidget(self.tz_slider, 7)
        self.tz_layout.addWidget(self.tz_spinner, 3)
        self.translation_layout.addRow("Z:", self.tz_layout)
        
        # Expression Group
        self.expression_group = QGroupBox("Expression Control")
        self.expression_layout = QVBoxLayout(self.expression_group)
        self.motion_layout.addWidget(self.expression_group)
        
        # Expression description
        self.expression_desc = QLabel("Expression parameter editing is not available in this version.")
        self.expression_layout.addWidget(self.expression_desc)
        
        # Compute button
        self.compute_motion_button = QPushButton("Compute Motion Parameters")
        self.compute_motion_button.clicked.connect(self.on_compute_motion)
        self.motion_layout.addWidget(self.compute_motion_button)
        
        self.motion_layout.addStretch()
        
        # Connect signals
        self.connect_motion_signals()
        
    def create_derived_tab(self):
        """
        Create Derived Parameters Tab
        """
        # Create tab
        self.derived_tab = QWidget()
        self.derived_layout = QVBoxLayout(self.derived_tab)
        self.tab_widget.addTab(self.derived_tab, "Derived")
        
        # Appearance Group
        self.appearance_group = QGroupBox("Appearance Features")
        self.appearance_layout = QVBoxLayout(self.appearance_group)
        self.derived_layout.addWidget(self.appearance_group)
        
        # Appearance description
        self.appearance_desc = QLabel("Appearance feature vector (256 dimensions):")
        self.appearance_layout.addWidget(self.appearance_desc)
        
        # Feature status
        self.feature_status = QLabel("No appearance features computed")
        self.appearance_layout.addWidget(self.feature_status)
        
        # Compute button
        self.compute_appearance_button = QPushButton("Compute Appearance Features")
        self.compute_appearance_button.clicked.connect(self.on_compute_appearance)
        self.appearance_layout.addWidget(self.compute_appearance_button)
        
        # Rotation Group
        self.rotation_group = QGroupBox("Rotation Matrix")
        self.rotation_layout = QVBoxLayout(self.rotation_group)
        self.derived_layout.addWidget(self.rotation_group)
        
        # Rotation description
        self.rotation_desc = QLabel("Rotation matrix (3x3):")
        self.rotation_layout.addWidget(self.rotation_desc)
        
        # Rotation status
        self.rotation_status = QLabel("No rotation matrix computed")
        self.rotation_layout.addWidget(self.rotation_status)
        
        # Compute button
        self.compute_rotation_button = QPushButton("Compute Rotation Matrix")
        self.compute_rotation_button.clicked.connect(self.on_compute_derived)
        self.rotation_layout.addWidget(self.compute_rotation_button)
        
        self.derived_layout.addStretch()
        
    def create_norm_tab(self):
        """
        Create Normalization Parameters Tab
        """
        # Create tab
        self.norm_tab = QWidget()
        self.norm_layout = QVBoxLayout(self.norm_tab)
        self.tab_widget.addTab(self.norm_tab, "Normalization")
        
        # Lip Group
        self.lip_group = QGroupBox("Lip Normalization")
        self.lip_layout = QVBoxLayout(self.lip_group)
        self.norm_layout.addWidget(self.lip_group)
        
        # Lip zero checkbox
        self.lip_zero_check = QCheckBox("Zero lip position before animation")
        self.lip_layout.addWidget(self.lip_zero_check)
        
        # Eye Group
        self.eye_group = QGroupBox("Eye Normalization")
        self.eye_layout = QVBoxLayout(self.eye_group)
        self.norm_layout.addWidget(self.eye_group)
        
        # Eye description
        self.eye_desc = QLabel("Eye normalization parameters:")
        self.eye_layout.addWidget(self.eye_desc)
        
        # Eye status
        self.eye_status = QLabel("No eye normalization parameters")
        self.eye_layout.addWidget(self.eye_status)
        
        self.norm_layout.addStretch()
        
        # Connect signals
        self.connect_norm_signals()
        
    def create_pasteback_tab(self):
        """
        Create Pasteback Tab
        """
        # Create tab
        self.pasteback_tab = QWidget()
        self.pasteback_layout = QVBoxLayout(self.pasteback_tab)
        self.tab_widget.addTab(self.pasteback_tab, "Pasteback")
        
        # Mask Group
        self.mask_group = QGroupBox("Pasteback Mask")
        self.mask_layout = QVBoxLayout(self.mask_group)
        self.pasteback_layout.addWidget(self.mask_group)
        
        # Mask description
        self.mask_desc = QLabel("Pasteback mask status:")
        self.mask_layout.addWidget(self.mask_desc)
        
        # Mask status
        self.mask_status = QLabel("No pasteback mask generated")
        self.mask_layout.addWidget(self.mask_status)
        
        # Generate button
        self.generate_mask_button = QPushButton("Generate Pasteback Mask")
        self.generate_mask_button.clicked.connect(self.on_generate_mask)
        self.mask_layout.addWidget(self.generate_mask_button)
        
        self.pasteback_layout.addStretch()
        
    def connect_motion_signals(self):
        """
        Connect signals for motion parameters
        """
        # Pitch signals
        self.pitch_slider.valueChanged.connect(self.on_pitch_slider_changed)
        self.pitch_spinner.valueChanged.connect(self.on_pitch_spinner_changed)
        
        # Yaw signals
        self.yaw_slider.valueChanged.connect(self.on_yaw_slider_changed)
        self.yaw_spinner.valueChanged.connect(self.on_yaw_spinner_changed)
        
        # Roll signals
        self.roll_slider.valueChanged.connect(self.on_roll_slider_changed)
        self.roll_spinner.valueChanged.connect(self.on_roll_spinner_changed)
        
        # Scale signals
        self.scale_slider.valueChanged.connect(self.on_scale_slider_changed)
        self.scale_spinner.valueChanged.connect(self.on_scale_spinner_changed)
        
        # Translation signals
        self.tx_slider.valueChanged.connect(self.on_tx_slider_changed)
        self.tx_spinner.valueChanged.connect(self.on_tx_spinner_changed)
        self.ty_slider.valueChanged.connect(self.on_ty_slider_changed)
        self.ty_spinner.valueChanged.connect(self.on_ty_spinner_changed)
        self.tz_slider.valueChanged.connect(self.on_tz_slider_changed)
        self.tz_spinner.valueChanged.connect(self.on_tz_spinner_changed)
        
    def connect_norm_signals(self):
        """
        Connect signals for normalization parameters
        """
        self.lip_zero_check.stateChanged.connect(self.on_lip_zero_changed)
        
    def update_view(self, event_type: str = None, data: Any = None) -> None:
        """
        Update the view based on model changes
        
        Args:
            event_type (str): Type of event that occurred
            data (Any): Additional data related to the event
        """
        if event_type == 'file_loaded' or event_type == 'image_loaded':
            self.update_all_parameters()
            
        elif event_type == 'param_updated':
            param_type = data.get('type', '')
            param_name = data.get('param', '')
            
            if param_type == 'motion':
                self.update_motion_parameter(param_name)
            elif param_type == 'derived':
                self.update_derived_parameter(param_name)
            elif param_type == 'norm':
                self.update_norm_parameter(param_name)
            elif param_type == 'pasteback':
                self.update_pasteback_parameter(param_name)
                
        elif event_type == 'reset':
            self.update_all_parameters()
            
        elif event_type == 'undo' or event_type == 'redo':
            self.update_all_parameters()
            
    def update_all_parameters(self):
        """
        Update all parameters in the panel
        """
        # Update motion parameters
        for param in self.controller.model.motion_params:
            self.update_motion_parameter(param)
            
        # Update derived parameters
        for param in self.controller.model.derived_params:
            self.update_derived_parameter(param)
            
        # Update normalization parameters
        for param in self.controller.model.norm_params:
            self.update_norm_parameter(param)
            
        # Update pasteback parameters
        for param in self.controller.model.pasteback_info:
            self.update_pasteback_parameter(param)
            
    def update_motion_parameter(self, param_name: str):
        """
        Update a motion parameter in the UI
        
        Args:
            param_name (str): Parameter name
        """
        value = self.controller.model.motion_params[param_name]
        
        if param_name == 'pitch' and value is not None:
            # Block signals to prevent feedback loop
            with QSignalBlocker(self.pitch_slider), QSignalBlocker(self.pitch_spinner):
                # Convert from radians to degrees
                degrees = float(value) * 180.0 / np.pi
                self.pitch_slider.setValue(int(degrees))
                self.pitch_spinner.setValue(degrees)
                
        elif param_name == 'yaw' and value is not None:
            with QSignalBlocker(self.yaw_slider), QSignalBlocker(self.yaw_spinner):
                degrees = float(value) * 180.0 / np.pi
                self.yaw_slider.setValue(int(degrees))
                self.yaw_spinner.setValue(degrees)
                
        elif param_name == 'roll' and value is not None:
            with QSignalBlocker(self.roll_slider), QSignalBlocker(self.roll_spinner):
                degrees = float(value) * 180.0 / np.pi
                self.roll_slider.setValue(int(degrees))
                self.roll_spinner.setValue(degrees)
                
        elif param_name == 'scale' and value is not None:
            with QSignalBlocker(self.scale_slider), QSignalBlocker(self.scale_spinner):
                scale_val = float(value)
                self.scale_slider.setValue(int(scale_val * 100))
                self.scale_spinner.setValue(scale_val)
                
        elif param_name == 't' and value is not None:
            with QSignalBlocker(self.tx_slider), QSignalBlocker(self.tx_spinner), \
                 QSignalBlocker(self.ty_slider), QSignalBlocker(self.ty_spinner), \
                 QSignalBlocker(self.tz_slider), QSignalBlocker(self.tz_spinner):
                 
                self.tx_slider.setValue(int(value[0] * 10))
                self.tx_spinner.setValue(float(value[0]))
                self.ty_slider.setValue(int(value[1] * 10))
                self.ty_spinner.setValue(float(value[1]))
                self.tz_slider.setValue(int(value[2] * 10))
                self.tz_spinner.setValue(float(value[2]))
                
        elif param_name == 'exp':
            # Expression parameters are complex and not directly editable in basic UI
            pass
            
        elif param_name == 'kp':
            # 3D keypoints not directly editable in basic UI
            pass
            
    def update_derived_parameter(self, param_name: str):
        """
        Update a derived parameter in the UI
        
        Args:
            param_name (str): Parameter name
        """
        value = self.controller.model.derived_params[param_name]
        
        if param_name == 'f_s' and value is not None:
            self.feature_status.setText(f"Appearance features computed ({value.shape[0]} dimensions)")
            
        elif param_name == 'R_s' and value is not None:
            self.rotation_status.setText("Rotation matrix computed")
            
        elif param_name == 'x_s' or param_name == 'x_c_s':
            # Not directly displayed in basic UI
            pass
            
    def update_norm_parameter(self, param_name: str):
        """
        Update a normalization parameter in the UI
        
        Args:
            param_name (str): Parameter name
        """
        value = self.controller.model.norm_params[param_name]
        
        if param_name == 'flag_lip_zero' and value is not None:
            with QSignalBlocker(self.lip_zero_check):
                self.lip_zero_check.setChecked(bool(value))
                
        elif param_name == 'lip_delta_before_animation' and value is not None:
            pass  # Complex parameter, not directly editable
            
        elif param_name == 'eye_delta_before_animation' and value is not None:
            self.eye_status.setText("Eye normalization parameters computed")
            
    def update_pasteback_parameter(self, param_name: str):
        """
        Update a pasteback parameter in the UI
        
        Args:
            param_name (str): Parameter name
        """
        value = self.controller.model.pasteback_info[param_name]
        
        if param_name == 'mask_ori_float' and value is not None:
            self.mask_status.setText("Pasteback mask generated")
            
        elif param_name == 'M' and value is not None:
            pass  # Complex parameter, not directly editable
            
    # Slot methods for parameter changes
    
    def on_pitch_slider_changed(self, value):
        """Handle pitch slider change"""
        self.pitch_spinner.setValue(value)
        self._update_pitch_value(value)
        
    def on_pitch_spinner_changed(self, value):
        """Handle pitch spinner change"""
        self.pitch_slider.setValue(int(value))
        self._update_pitch_value(value)
        
    def _update_pitch_value(self, value):
        """Update pitch value in model"""
        radians = float(value) * np.pi / 180.0
        pitch = np.array([radians], dtype=np.float32)
        self.controller.update_motion_param('pitch', pitch)
        
    def on_yaw_slider_changed(self, value):
        """Handle yaw slider change"""
        self.yaw_spinner.setValue(value)
        self._update_yaw_value(value)
        
    def on_yaw_spinner_changed(self, value):
        """Handle yaw spinner change"""
        self.yaw_slider.setValue(int(value))
        self._update_yaw_value(value)
        
    def _update_yaw_value(self, value):
        """Update yaw value in model"""
        radians = float(value) * np.pi / 180.0
        yaw = np.array([radians], dtype=np.float32)
        self.controller.update_motion_param('yaw', yaw)
        
    def on_roll_slider_changed(self, value):
        """Handle roll slider change"""
        self.roll_spinner.setValue(value)
        self._update_roll_value(value)
        
    def on_roll_spinner_changed(self, value):
        """Handle roll spinner change"""
        self.roll_slider.setValue(int(value))
        self._update_roll_value(value)
        
    def _update_roll_value(self, value):
        """Update roll value in model"""
        radians = float(value) * np.pi / 180.0
        roll = np.array([radians], dtype=np.float32)
        self.controller.update_motion_param('roll', roll)
        
    def on_scale_slider_changed(self, value):
        """Handle scale slider change"""
        scale_val = value / 100.0
        self.scale_spinner.setValue(scale_val)
        self._update_scale_value(scale_val)
        
    def on_scale_spinner_changed(self, value):
        """Handle scale spinner change"""
        self.scale_slider.setValue(int(value * 100))
        self._update_scale_value(value)
        
    def _update_scale_value(self, value):
        """Update scale value in model"""
        scale = np.array([value], dtype=np.float32)
        self.controller.update_motion_param('scale', scale)
        
    def on_tx_slider_changed(self, value):
        """Handle tx slider change"""
        tx_val = value / 10.0
        self.tx_spinner.setValue(tx_val)
        self._update_translation_value()
        
    def on_tx_spinner_changed(self, value):
        """Handle tx spinner change"""
        self.tx_slider.setValue(int(value * 10))
        self._update_translation_value()
        
    def on_ty_slider_changed(self, value):
        """Handle ty slider change"""
        ty_val = value / 10.0
        self.ty_spinner.setValue(ty_val)
        self._update_translation_value()
        
    def on_ty_spinner_changed(self, value):
        """Handle ty spinner change"""
        self.ty_slider.setValue(int(value * 10))
        self._update_translation_value()
        
    def on_tz_slider_changed(self, value):
        """Handle tz slider change"""
        tz_val = value / 10.0
        self.tz_spinner.setValue(tz_val)
        self._update_translation_value()
        
    def on_tz_spinner_changed(self, value):
        """Handle tz spinner change"""
        self.tz_slider.setValue(int(value * 10))
        self._update_translation_value()
        
    def _update_translation_value(self):
        """Update translation value in model"""
        tx = self.tx_spinner.value()
        ty = self.ty_spinner.value()
        tz = self.tz_spinner.value()
        translation = np.array([tx, ty, tz], dtype=np.float32).reshape(1, 3)
        self.controller.update_motion_param('t', translation)
        
    def on_lip_zero_changed(self, state):
        """Handle lip zero checkbox change"""
        value = bool(state == Qt.Checked)
        self.controller.update_norm_param('flag_lip_zero', value)
        
    def on_compute_motion(self):
        """Handle compute motion button click"""
        if self.controller.compute_motion_parameters():
            self.update_all_parameters()
            
    def on_compute_appearance(self):
        """Handle compute appearance button click"""
        if self.controller.compute_appearance_features():
            self.update_derived_parameter('f_s')
            
    def on_compute_derived(self):
        """Handle compute derived button click"""
        if self.controller.compute_derived_parameters():
            self.update_derived_parameter('R_s')
            
    def on_generate_mask(self):
        """Handle generate mask button click"""
        if self.controller.generate_pasteback_mask():
            self.update_pasteback_parameter('mask_ori_float')
            
    def on_compute_all(self):
        """Handle compute all button click"""
        # Make sure we have landmarks first
        if self.controller.model.landmarks is None:
            if not self.controller.detect_landmarks():
                return
                
        # Compute motion parameters
        if self.controller.compute_motion_parameters():
            # Compute derived parameters
            self.controller.compute_derived_parameters()
            
            # Compute appearance features
            self.controller.compute_appearance_features()
            
            # Generate pasteback mask
            self.controller.generate_pasteback_mask()
            
            # Update all UI elements
            self.update_all_parameters()
            
    def on_reset_params(self):
        """Handle reset parameters button click"""
        # Reset motion parameters
        for param in self.controller.model.motion_params:
            self.controller.model.motion_params[param] = None
            
        # Reset derived parameters
        for param in self.controller.model.derived_params:
            self.controller.model.derived_params[param] = None
            
        # Reset normalization parameters
        self.controller.model.norm_params['flag_lip_zero'] = False
        self.controller.model.norm_params['lip_delta_before_animation'] = None
        self.controller.model.norm_params['eye_delta_before_animation'] = None
            
        # Reset pasteback parameters
        for param in self.controller.model.pasteback_info:
            self.controller.model.pasteback_info[param] = None
            
        # Update UI
        self.controller.model.is_modified = True
        self.update_all_parameters()