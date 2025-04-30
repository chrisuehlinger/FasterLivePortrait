#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Landmark Editor Component for FLPSF Editor
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import cv2
import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGridLayout, QListWidget, QListWidgetItem, QScrollArea,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QSpinBox, QComboBox, QInputDialog,
    QFileDialog, QMessageBox, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QBrush, QIcon

# Import controller
from controllers.app_controller import AppController

logger = logging.getLogger('flpsf_editor.views.landmark_editor')

class LandmarkEditor(QWidget):
    """
    Landmark editor component that allows editing facial landmarks
    """
    
    # Signal to notify image view about selected landmark
    landmark_selected = pyqtSignal(int, QPoint)
    
    # Predefined landmark group colors for visualization
    LANDMARK_GROUPS = {
        "face_contour": {"color": QColor(255, 0, 0), "indices": list(range(0, 17))},
        "left_eyebrow": {"color": QColor(0, 255, 0), "indices": list(range(17, 22))},
        "right_eyebrow": {"color": QColor(0, 255, 0), "indices": list(range(22, 27))},
        "nose_bridge": {"color": QColor(0, 0, 255), "indices": list(range(27, 31))},
        "nose_tip": {"color": QColor(0, 0, 255), "indices": list(range(31, 36))},
        "left_eye": {"color": QColor(255, 255, 0), "indices": list(range(36, 42))},
        "right_eye": {"color": QColor(255, 255, 0), "indices": list(range(42, 48))},
        "outer_lips": {"color": QColor(255, 0, 255), "indices": list(range(48, 60))},
        "inner_lips": {"color": QColor(255, 0, 255), "indices": list(range(60, 68))},
        # Common animal face landmarks (may vary based on model)
        "animal_eyes": {"color": QColor(255, 255, 0), "indices": list(range(0, 4))},
        "animal_nose": {"color": QColor(0, 0, 255), "indices": list(range(4, 6))},
        "animal_mouth": {"color": QColor(255, 0, 255), "indices": list(range(6, 10))},
        "animal_ears": {"color": QColor(0, 255, 0), "indices": list(range(10, 14))},
    }
    
    def __init__(self, controller: AppController):
        """
        Initialize the landmark editor
        
        Args:
            controller (AppController): Application controller
        """
        super().__init__()
        
        self.controller = controller
        self.current_landmark_index = -1
        
        # Original landmarks for reset functionality
        self.original_landmarks = None
        
        # Template management
        self.templates = {}
        self.current_template = None
        self.load_templates()
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """
        Setup UI components
        """
        # Main layout
        self.layout = QVBoxLayout(self)
        
        # Create top section with splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.layout.addWidget(self.main_splitter)
        
        # Left side - Landmark list
        self.landmark_container = QWidget()
        self.landmark_layout = QVBoxLayout(self.landmark_container)
        
        # Landmark table with grouping
        self.table_group = QGroupBox("Landmark Points")
        self.table_layout = QVBoxLayout(self.table_group)
        
        # Add group selection combo box
        self.group_selector_layout = QHBoxLayout()
        self.group_label = QLabel("Group:")
        self.group_selector_layout.addWidget(self.group_label)
        
        self.group_combo = QComboBox()
        self.group_combo.addItem("All Points")
        for group_name in self.LANDMARK_GROUPS.keys():
            self.group_combo.addItem(group_name)
        self.group_combo.currentTextChanged.connect(self.on_group_changed)
        self.group_selector_layout.addWidget(self.group_combo)
        
        self.table_layout.addLayout(self.group_selector_layout)
        
        # Landmark table
        self.landmark_table = QTableWidget(0, 4)  # 4 columns: Index, Group, X, Y
        self.landmark_table.setHorizontalHeaderLabels(["Index", "Group", "X", "Y"])
        self.landmark_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.landmark_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.landmark_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.landmark_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # X column
        self.landmark_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)  # Y column
        self.landmark_table.verticalHeader().setVisible(False)
        self.landmark_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked)
        self.landmark_table.itemSelectionChanged.connect(self.on_landmark_selection_changed)
        self.landmark_table.itemChanged.connect(self.on_landmark_item_changed)
        
        self.table_layout.addWidget(self.landmark_table)
        
        # Add landmark navigation buttons
        self.nav_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("◄ Previous")
        self.prev_button.clicked.connect(self.on_prev_landmark)
        self.nav_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("Next ►")
        self.next_button.clicked.connect(self.on_next_landmark)
        self.nav_layout.addWidget(self.next_button)
        
        self.table_layout.addLayout(self.nav_layout)
        
        self.landmark_layout.addWidget(self.table_group)
        
        # Right side - Editing tools
        self.edit_container = QWidget()
        self.edit_layout = QVBoxLayout(self.edit_container)
        
        # Edit section
        self.edit_group = QGroupBox("Edit Landmark")
        self.edit_inner_layout = QVBoxLayout(self.edit_group)
        
        # Selected landmark info
        self.selected_layout = QGridLayout()
        self.edit_inner_layout.addLayout(self.selected_layout)
        
        self.selected_label = QLabel("Selected: None")
        self.selected_layout.addWidget(self.selected_label, 0, 0, 1, 2)
        
        self.group_indicator = QLabel("Group: None")
        self.selected_layout.addWidget(self.group_indicator, 1, 0, 1, 2)
        
        self.x_label = QLabel("X:")
        self.selected_layout.addWidget(self.x_label, 2, 0)
        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, 10000)  # Large enough for typical image sizes
        self.x_spin.valueChanged.connect(self.on_x_changed)
        self.selected_layout.addWidget(self.x_spin, 2, 1)
        
        self.y_label = QLabel("Y:")
        self.selected_layout.addWidget(self.y_label, 3, 0)
        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, 10000)  # Large enough for typical image sizes
        self.y_spin.valueChanged.connect(self.on_y_changed)
        self.selected_layout.addWidget(self.y_spin, 3, 1)
        
        self.fine_tuning_layout = QGridLayout()
        self.edit_inner_layout.addLayout(self.fine_tuning_layout)
        
        self.fine_tuning_label = QLabel("Fine Tuning:")
        self.fine_tuning_layout.addWidget(self.fine_tuning_label, 0, 0, 1, 4)
        
        # Fine tuning buttons
        self.move_up_btn = QPushButton("▲")
        self.move_up_btn.setToolTip("Move landmark up by 1 pixel")
        self.move_up_btn.clicked.connect(lambda: self.move_landmark(0, -1))
        self.fine_tuning_layout.addWidget(self.move_up_btn, 1, 1)
        
        self.move_left_btn = QPushButton("◄")
        self.move_left_btn.setToolTip("Move landmark left by 1 pixel")
        self.move_left_btn.clicked.connect(lambda: self.move_landmark(-1, 0))
        self.fine_tuning_layout.addWidget(self.move_left_btn, 2, 0)
        
        self.move_right_btn = QPushButton("►")
        self.move_right_btn.setToolTip("Move landmark right by 1 pixel")
        self.move_right_btn.clicked.connect(lambda: self.move_landmark(1, 0))
        self.fine_tuning_layout.addWidget(self.move_right_btn, 2, 2)
        
        self.move_down_btn = QPushButton("▼")
        self.move_down_btn.setToolTip("Move landmark down by 1 pixel")
        self.move_down_btn.clicked.connect(lambda: self.move_landmark(0, 1))
        self.fine_tuning_layout.addWidget(self.move_down_btn, 3, 1)
        
        self.edit_inner_layout.addStretch()
        
        # Action buttons
        self.reset_button = QPushButton("Reset Selected")
        self.reset_button.clicked.connect(self.on_reset_selected)
        self.edit_inner_layout.addWidget(self.reset_button)
        
        self.reset_all_button = QPushButton("Reset All Landmarks")
        self.reset_all_button.clicked.connect(self.on_reset_all)
        self.edit_inner_layout.addWidget(self.reset_all_button)
        
        self.edit_layout.addWidget(self.edit_group)
        
        # Template management
        self.template_group = QGroupBox("Landmark Templates")
        self.template_layout = QVBoxLayout(self.template_group)
        
        self.template_actions_layout = QHBoxLayout()
        
        self.template_combo = QComboBox()
        self.update_template_combo()
        self.template_combo.currentTextChanged.connect(self.on_template_selected)
        self.template_actions_layout.addWidget(self.template_combo, 2)
        
        self.save_template_btn = QPushButton("Save")
        self.save_template_btn.clicked.connect(self.on_save_template)
        self.template_actions_layout.addWidget(self.save_template_btn)
        
        self.load_template_btn = QPushButton("Apply")
        self.load_template_btn.clicked.connect(self.on_apply_template)
        self.template_actions_layout.addWidget(self.load_template_btn)
        
        self.template_layout.addLayout(self.template_actions_layout)
        
        self.template_manage_layout = QHBoxLayout()
        
        self.new_template_btn = QPushButton("New Template")
        self.new_template_btn.clicked.connect(self.on_new_template)
        self.template_manage_layout.addWidget(self.new_template_btn)
        
        self.delete_template_btn = QPushButton("Delete Template")
        self.delete_template_btn.clicked.connect(self.on_delete_template)
        self.template_manage_layout.addWidget(self.delete_template_btn)
        
        self.template_layout.addLayout(self.template_manage_layout)
        
        self.edit_layout.addWidget(self.template_group)
        
        # Detection section
        self.detect_group = QGroupBox("Landmark Detection")
        self.detect_layout = QVBoxLayout(self.detect_group)
        
        self.detect_layout_grid = QGridLayout()
        
        self.detect_human_btn = QPushButton("Detect Human Face")
        self.detect_human_btn.clicked.connect(self.on_detect_human)
        self.detect_layout_grid.addWidget(self.detect_human_btn, 0, 0)
        
        self.detect_animal_btn = QPushButton("Detect Animal Face")
        self.detect_animal_btn.clicked.connect(self.on_detect_animal)
        self.detect_layout_grid.addWidget(self.detect_animal_btn, 0, 1)
        
        self.detect_layout.addLayout(self.detect_layout_grid)
        
        self.edit_layout.addWidget(self.detect_group)
        
        # Add components to splitter
        self.main_splitter.addWidget(self.landmark_container)
        self.main_splitter.addWidget(self.edit_container)
        
        # Set splitter sizes
        self.main_splitter.setSizes([300, 200])  # Default sizes for left/right panels
        
    def load_templates(self):
        """
        Load landmark templates from file
        """
        try:
            # Check for templates directory
            template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "templates")
            os.makedirs(template_dir, exist_ok=True)
            
            # Check for templates file
            template_file = os.path.join(template_dir, "landmark_templates.json")
            if os.path.exists(template_file):
                with open(template_file, 'r') as f:
                    self.templates = json.load(f)
            else:
                # Create default templates file
                self.templates = {
                    "human_face_106": {
                        "description": "106-point human face landmarks",
                        "type": "human",
                        "points": 106
                    },
                    "animal_face_14": {
                        "description": "14-point animal face landmarks",
                        "type": "animal",
                        "points": 14
                    }
                }
                with open(template_file, 'w') as f:
                    json.dump(self.templates, f, indent=2)
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
            self.templates = {}
            
    def save_templates(self):
        """
        Save landmark templates to file
        """
        try:
            # Check for templates directory
            template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "templates")
            os.makedirs(template_dir, exist_ok=True)
            
            # Save templates file
            template_file = os.path.join(template_dir, "landmark_templates.json")
            with open(template_file, 'w') as f:
                json.dump(self.templates, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving templates: {str(e)}")
            
    def update_template_combo(self):
        """
        Update the template combo box
        """
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        
        # Add templates
        for template_name, template_info in self.templates.items():
            self.template_combo.addItem(f"{template_name} ({template_info.get('description', '')})", template_name)
            
        # Restore current selection if applicable
        if self.current_template:
            for i in range(self.template_combo.count()):
                if self.template_combo.itemData(i) == self.current_template:
                    self.template_combo.setCurrentIndex(i)
                    break
                    
        self.template_combo.blockSignals(False)
        
    def update_view(self, event_type: str = None, data: Any = None) -> None:
        """
        Update the view based on model changes
        
        Args:
            event_type (str): Type of event that occurred
            data (Any): Additional data related to the event
        """
        if event_type == 'file_loaded' or event_type == 'image_loaded':
            self.populate_landmark_table()
            self.clear_selection()
            
            # Store original landmarks for reset functionality
            if self.controller.model.landmarks is not None:
                self.original_landmarks = self.controller.model.landmarks.copy()
            else:
                self.original_landmarks = None
            
        elif event_type == 'landmarks_updated':
            self.populate_landmark_table()
            
            # If this is the first time landmarks are set, store as original
            if self.original_landmarks is None and self.controller.model.landmarks is not None:
                self.original_landmarks = self.controller.model.landmarks.copy()
            
        elif event_type == 'reset':
            self.populate_landmark_table()
            self.clear_selection()
            self.original_landmarks = None
            
        elif event_type == 'undo' or event_type == 'redo':
            self.populate_landmark_table()
            
    def populate_landmark_table(self):
        """
        Populate the landmark table with current landmark data
        """
        # Block signals during update
        self.landmark_table.blockSignals(True)
        
        # Clear table
        self.landmark_table.setRowCount(0)
        
        # Check if landmarks exist
        if self.controller.model.landmarks is None:
            self.landmark_table.blockSignals(False)
            self.clear_selection()
            return
            
        # Get current group filter
        current_group = self.group_combo.currentText()
        
        # Add landmark data
        landmarks = self.controller.model.landmarks
        
        # If filtering by group, get indices for that group
        indices_to_show = list(range(len(landmarks)))
        if current_group != "All Points" and current_group in self.LANDMARK_GROUPS:
            indices_to_show = self.LANDMARK_GROUPS[current_group]["indices"]
            # Filter indices that actually exist in the landmarks
            indices_to_show = [i for i in indices_to_show if i < len(landmarks)]
            
        self.landmark_table.setRowCount(len(indices_to_show))
        
        for table_row, i in enumerate(indices_to_show):
            if i < len(landmarks):
                x, y = landmarks[i]
                
                # Index
                index_item = QTableWidgetItem(str(i))
                index_item.setFlags(index_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
                self.landmark_table.setItem(table_row, 0, index_item)
                
                # Group
                group_name = self.get_group_for_landmark(i)
                group_item = QTableWidgetItem(group_name)
                group_item.setFlags(group_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
                
                # Set background color based on group
                if group_name in self.LANDMARK_GROUPS:
                    color = self.LANDMARK_GROUPS[group_name]["color"]
                    group_item.setBackground(QBrush(color.lighter(150)))
                    
                self.landmark_table.setItem(table_row, 1, group_item)
                
                # X coordinate
                x_item = QTableWidgetItem(str(int(x)))
                self.landmark_table.setItem(table_row, 2, x_item)
                
                # Y coordinate
                y_item = QTableWidgetItem(str(int(y)))
                self.landmark_table.setItem(table_row, 3, y_item)
        
        # Restore signals
        self.landmark_table.blockSignals(False)
        
        # If previously had a selection, try to restore it
        if self.current_landmark_index >= 0:
            # Find the row in the filtered table that corresponds to current_landmark_index
            found = False
            for row in range(self.landmark_table.rowCount()):
                idx_item = self.landmark_table.item(row, 0)
                if idx_item and int(idx_item.text()) == self.current_landmark_index:
                    self.landmark_table.selectRow(row)
                    found = True
                    break
            
            if not found:
                self.clear_selection()
        else:
            self.clear_selection()
            
    def get_group_for_landmark(self, landmark_idx):
        """
        Get the group name for a landmark index
        
        Args:
            landmark_idx (int): Landmark index
            
        Returns:
            str: Group name
        """
        for group_name, group_info in self.LANDMARK_GROUPS.items():
            if landmark_idx in group_info["indices"]:
                return group_name
                
        return "Other"
        
    def clear_selection(self):
        """
        Clear the current landmark selection
        """
        self.current_landmark_index = -1
        self.selected_label.setText("Selected: None")
        self.group_indicator.setText("Group: None")
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.x_spin.setValue(0)
        self.y_spin.setValue(0)
        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        
    def on_landmark_selection_changed(self):
        """
        Handle selection change in landmark table
        """
        selected_items = self.landmark_table.selectedItems()
        if not selected_items:
            self.clear_selection()
            return
            
        # Get the row of the selected item
        row = selected_items[0].row()
        
        # Get the actual landmark index from the first column
        index_item = self.landmark_table.item(row, 0)
        if not index_item:
            self.clear_selection()
            return
            
        landmark_idx = int(index_item.text())
        self.current_landmark_index = landmark_idx
        
        # Update the selection info
        self.selected_label.setText(f"Selected: Point {landmark_idx}")
        
        # Update group info
        group_name = self.get_group_for_landmark(landmark_idx)
        self.group_indicator.setText(f"Group: {group_name}")
        
        # Update the spin boxes
        if self.controller.model.landmarks is not None and landmark_idx < len(self.controller.model.landmarks):
            x, y = self.controller.model.landmarks[landmark_idx]
            self.x_spin.blockSignals(True)
            self.y_spin.blockSignals(True)
            self.x_spin.setValue(int(x))
            self.y_spin.setValue(int(y))
            self.x_spin.blockSignals(False)
            self.y_spin.blockSignals(False)
            
            # Emit signal to notify image view about selection
            self.landmark_selected.emit(landmark_idx, QPoint(int(x), int(y)))
            
    def on_landmark_item_changed(self, item):
        """
        Handle change in landmark table item
        
        Args:
            item: Changed table item
        """
        # Only handle changes to X and Y columns (columns 2 and 3)
        if item.column() < 2:  # Index and Group columns are read-only
            return
            
        # Get row 
        row = item.row()
        col = item.column()
        
        # Get the actual landmark index from the first column
        index_item = self.landmark_table.item(row, 0)
        if not index_item:
            return
            
        landmark_idx = int(index_item.text())
        
        try:
            # Try to convert cell text to integer
            value = int(item.text())
            
            # Ensure value is positive
            if value < 0:
                value = 0
                item.setText(str(value))
                
            # Update landmark
            if self.controller.model.landmarks is not None and landmark_idx < len(self.controller.model.landmarks):
                landmarks = self.controller.model.landmarks.copy()
                
                if col == 2:  # X column
                    landmarks[landmark_idx, 0] = value
                    if landmark_idx == self.current_landmark_index:
                        self.x_spin.blockSignals(True)
                        self.x_spin.setValue(value)
                        self.x_spin.blockSignals(False)
                else:  # Y column
                    landmarks[landmark_idx, 1] = value
                    if landmark_idx == self.current_landmark_index:
                        self.y_spin.blockSignals(True)
                        self.y_spin.setValue(value)
                        self.y_spin.blockSignals(False)
                    
                # Update model
                self.controller.update_landmarks(landmarks)
                
                # Notify image view if this is the selected landmark
                if landmark_idx == self.current_landmark_index:
                    x, y = landmarks[landmark_idx]
                    self.landmark_selected.emit(landmark_idx, QPoint(int(x), int(y)))
                
        except ValueError:
            # If conversion fails, restore the original value
            if self.controller.model.landmarks is not None and landmark_idx < len(self.controller.model.landmarks):
                x, y = self.controller.model.landmarks[landmark_idx]
                
                if col == 2:  # X column
                    item.setText(str(int(x)))
                else:  # Y column
                    item.setText(str(int(y)))
                    
    def on_x_changed(self, value):
        """
        Handle change in X spin box
        
        Args:
            value: New X value
        """
        if self.current_landmark_index < 0:
            return
            
        # Update landmark
        if self.controller.model.landmarks is not None and self.current_landmark_index < len(self.controller.model.landmarks):
            landmarks = self.controller.model.landmarks.copy()
            landmarks[self.current_landmark_index, 0] = value
            
            # Update model
            self.controller.update_landmarks(landmarks)
            
            # Update table
            self.update_table_for_current_landmark()
            
            # Notify image view
            self.landmark_selected.emit(self.current_landmark_index, QPoint(value, int(landmarks[self.current_landmark_index, 1])))
            
    def on_y_changed(self, value):
        """
        Handle change in Y spin box
        
        Args:
            value: New Y value
        """
        if self.current_landmark_index < 0:
            return
            
        # Update landmark
        if self.controller.model.landmarks is not None and self.current_landmark_index < len(self.controller.model.landmarks):
            landmarks = self.controller.model.landmarks.copy()
            landmarks[self.current_landmark_index, 1] = value
            
            # Update model
            self.controller.update_landmarks(landmarks)
            
            # Update table
            self.update_table_for_current_landmark()
            
            # Notify image view
            self.landmark_selected.emit(self.current_landmark_index, QPoint(int(landmarks[self.current_landmark_index, 0]), value))
            
    def update_table_for_current_landmark(self):
        """
        Update the table entry for the current landmark
        """
        if self.current_landmark_index < 0 or self.controller.model.landmarks is None:
            return
            
        # Find the row in the table for current_landmark_index
        for row in range(self.landmark_table.rowCount()):
            idx_item = self.landmark_table.item(row, 0)
            if idx_item and int(idx_item.text()) == self.current_landmark_index:
                x, y = self.controller.model.landmarks[self.current_landmark_index]
                
                # Update X and Y in the table
                self.landmark_table.blockSignals(True)
                self.landmark_table.item(row, 2).setText(str(int(x)))
                self.landmark_table.item(row, 3).setText(str(int(y)))
                self.landmark_table.blockSignals(False)
                break
                
    def move_landmark(self, dx, dy):
        """
        Move the selected landmark by the given delta
        
        Args:
            dx (int): X delta
            dy (int): Y delta
        """
        if self.current_landmark_index < 0 or self.controller.model.landmarks is None:
            return
            
        # Get current values
        current_x = self.x_spin.value()
        current_y = self.y_spin.value()
        
        # Update spin boxes (which will trigger the update)
        self.x_spin.setValue(current_x + dx)
        self.y_spin.setValue(current_y + dy)
        
    def on_prev_landmark(self):
        """
        Select the previous landmark
        """
        if self.landmark_table.rowCount() == 0:
            return
            
        current_row = self.landmark_table.currentRow()
        if current_row > 0:
            self.landmark_table.selectRow(current_row - 1)
        else:
            # Wrap to the last row
            self.landmark_table.selectRow(self.landmark_table.rowCount() - 1)
            
    def on_next_landmark(self):
        """
        Select the next landmark
        """
        if self.landmark_table.rowCount() == 0:
            return
            
        current_row = self.landmark_table.currentRow()
        if current_row < self.landmark_table.rowCount() - 1:
            self.landmark_table.selectRow(current_row + 1)
        else:
            # Wrap to the first row
            self.landmark_table.selectRow(0)
            
    def on_reset_selected(self):
        """
        Reset the selected landmark to its original position
        """
        if self.current_landmark_index < 0 or self.original_landmarks is None:
            return
            
        # Check if the original landmark exists
        if self.current_landmark_index < len(self.original_landmarks):
            # Get the original coordinates
            orig_x, orig_y = self.original_landmarks[self.current_landmark_index]
            
            # Update the current landmarks
            if self.controller.model.landmarks is not None:
                landmarks = self.controller.model.landmarks.copy()
                landmarks[self.current_landmark_index] = [orig_x, orig_y]
                
                # Update model
                self.controller.update_landmarks(landmarks)
                
                # Update UI
                self.x_spin.blockSignals(True)
                self.y_spin.blockSignals(True)
                self.x_spin.setValue(int(orig_x))
                self.y_spin.setValue(int(orig_y))
                self.x_spin.blockSignals(False)
                self.y_spin.blockSignals(False)
                
                # Update table
                self.update_table_for_current_landmark()
                
                # Notify image view
                self.landmark_selected.emit(self.current_landmark_index, QPoint(int(orig_x), int(orig_y)))
                
    def on_reset_all(self):
        """
        Reset all landmarks to their original positions
        """
        if self.original_landmarks is None:
            return
            
        # Update model with original landmarks
        self.controller.update_landmarks(self.original_landmarks.copy())
        
        # Update UI
        self.populate_landmark_table()
        
    def on_detect_human(self):
        """
        Detect human face landmarks
        """
        if self.controller.detect_landmarks('insightface'):
            # Store original landmarks for reset functionality
            self.original_landmarks = self.controller.model.landmarks.copy()
            
            # Update UI
            self.populate_landmark_table()
            self.clear_selection()
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Failed to detect human face landmarks"
            )
            
    def on_detect_animal(self):
        """
        Detect animal face landmarks
        """
        if self.controller.detect_landmarks('xpose'):
            # Store original landmarks for reset functionality
            self.original_landmarks = self.controller.model.landmarks.copy()
            
            # Update UI
            self.populate_landmark_table()
            self.clear_selection()
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Failed to detect animal face landmarks"
            )
            
    def on_group_changed(self, group_name):
        """
        Handle change in landmark group selection
        
        Args:
            group_name (str): New group name
        """
        # Store current selection
        current_idx = self.current_landmark_index
        
        # Update table
        self.populate_landmark_table()
        
        # Restore selection if possible
        if current_idx >= 0:
            for row in range(self.landmark_table.rowCount()):
                idx_item = self.landmark_table.item(row, 0)
                if idx_item and int(idx_item.text()) == current_idx:
                    self.landmark_table.selectRow(row)
                    break
                    
    def on_template_selected(self, template_text):
        """
        Handle template selection
        
        Args:
            template_text (str): Selected template text
        """
        # Extract template name from display text
        template_name = self.template_combo.currentData()
        if template_name:
            self.current_template = template_name
            
    def on_save_template(self):
        """
        Save current landmarks as a template
        """
        if self.controller.model.landmarks is None:
            QMessageBox.warning(
                self,
                "Warning",
                "No landmarks to save"
            )
            return
            
        # Get template name
        name, ok = QInputDialog.getText(
            self,
            "Save Template",
            "Template Name:",
            text=self.current_template if self.current_template else ""
        )
        
        if ok and name:
            description, ok = QInputDialog.getText(
                self,
                "Save Template",
                "Description:"
            )
            
            if ok:
                # Determine if human or animal based on landmark count
                template_type = "human" if len(self.controller.model.landmarks) > 30 else "animal"
                
                # Save template
                self.templates[name] = {
                    "description": description,
                    "type": template_type,
                    "points": len(self.controller.model.landmarks),
                    "landmarks": self.controller.model.landmarks.tolist()
                }
                
                # Save to file
                self.save_templates()
                
                # Update UI
                self.current_template = name
                self.update_template_combo()
                
                QMessageBox.information(
                    self,
                    "Template Saved",
                    f"Template '{name}' saved successfully."
                )
                
    def on_apply_template(self):
        """
        Apply selected template to current image
        """
        template_name = self.template_combo.currentData()
        if not template_name or template_name not in self.templates:
            QMessageBox.warning(
                self,
                "Warning",
                "No template selected"
            )
            return
            
        # Get template
        template = self.templates[template_name]
        if "landmarks" not in template:
            QMessageBox.warning(
                self,
                "Warning",
                f"Template '{template_name}' does not contain landmark data"
            )
            return
            
        # Apply template
        landmarks = np.array(template["landmarks"], dtype=np.float32)
        self.controller.update_landmarks(landmarks)
        
        # Store as original landmarks for reset functionality
        self.original_landmarks = landmarks.copy()
        
        # Update UI
        self.populate_landmark_table()
        self.clear_selection()
        
        QMessageBox.information(
            self,
            "Template Applied",
            f"Template '{template_name}' applied successfully."
        )
        
    def on_new_template(self):
        """
        Create a new empty template
        """
        # Get template name
        name, ok = QInputDialog.getText(
            self,
            "New Template",
            "Template Name:"
        )
        
        if ok and name:
            description, ok = QInputDialog.getText(
                self,
                "New Template",
                "Description:"
            )
            
            if ok:
                # Create template with placeholder
                point_count, ok = QInputDialog.getInt(
                    self,
                    "New Template",
                    "Number of points:",
                    value=106,
                    min=1,
                    max=200
                )
                
                if ok:
                    # Determine template type based on point count
                    template_type = "human" if point_count > 30 else "animal"
                    
                    # Create empty landmarks
                    landmarks = np.zeros((point_count, 2), dtype=np.float32)
                    
                    # Save template
                    self.templates[name] = {
                        "description": description,
                        "type": template_type,
                        "points": point_count,
                        "landmarks": landmarks.tolist()
                    }
                    
                    # Save to file
                    self.save_templates()
                    
                    # Update UI
                    self.current_template = name
                    self.update_template_combo()
                    
                    QMessageBox.information(
                        self,
                        "Template Created",
                        f"Template '{name}' created successfully."
                    )
                    
    def on_delete_template(self):
        """
        Delete the selected template
        """
        template_name = self.template_combo.currentData()
        if not template_name or template_name not in self.templates:
            QMessageBox.warning(
                self,
                "Warning",
                "No template selected"
            )
            return
            
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Delete Template",
            f"Are you sure you want to delete template '{template_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Delete template
            del self.templates[template_name]
            
            # Save to file
            self.save_templates()
            
            # Update UI
            self.current_template = None
            self.update_template_combo()
            
            QMessageBox.information(
                self,
                "Template Deleted",
                f"Template '{template_name}' deleted successfully."
            )

    # Method to update landmark position from image view (when user clicks on image)
    def update_landmark_position(self, x, y):
        """
        Update the position of the currently selected landmark
        
        Args:
            x (int): New X coordinate
            y (int): New Y coordinate
        """
        if self.current_landmark_index < 0 or self.controller.model.landmarks is None:
            return
            
        # Update landmark position
        landmarks = self.controller.model.landmarks.copy()
        landmarks[self.current_landmark_index] = [x, y]
        
        # Update model
        self.controller.update_landmarks(landmarks)
        
        # Update UI
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.x_spin.setValue(int(x))
        self.y_spin.setValue(int(y))
        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        
        # Update table
        self.update_table_for_current_landmark()