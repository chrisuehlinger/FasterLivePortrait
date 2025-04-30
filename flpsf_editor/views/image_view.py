#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image View Component for FLPSF Editor
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import cv2
from PyQt5.QtWidgets import (
    QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QVBoxLayout, QLabel, QScrollArea, QSizePolicy,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsItem
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QTransform, QCursor

# Import controller
from controllers.app_controller import AppController

logger = logging.getLogger('flpsf_editor.views.image_view')

class LandmarkItem(QGraphicsEllipseItem):
    """
    Custom graphics item for landmarks with enhanced interaction
    """
    
    def __init__(self, x, y, size, index, parent=None):
        """
        Initialize the landmark item
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            size (float): Size of landmark point
            index (int): Landmark index
            parent: Parent item
        """
        super().__init__(x - size/2, y - size/2, size, size, parent)
        
        self.index = index
        self.size = size
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        
        # Store original position for movement calculations
        self.original_pos = QPointF(x, y)
        
    def hoverEnterEvent(self, event):
        """
        Handle hover enter event
        
        Args:
            event: Hover event
        """
        # Show hand cursor
        QApplication.setOverrideCursor(QCursor(Qt.PointingHandCursor))
        # Make point slightly larger
        self.prepareGeometryChange()
        self.setRect(self.rect().x() - 2, self.rect().y() - 2, 
                     self.size + 4, self.size + 4)
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        """
        Handle hover leave event
        
        Args:
            event: Hover event
        """
        # Restore cursor
        QApplication.restoreOverrideCursor()
        # Restore original size
        self.prepareGeometryChange()
        self.setRect(self.original_pos.x() - self.size/2, 
                    self.original_pos.y() - self.size/2,
                    self.size, self.size)
        super().hoverLeaveEvent(event)
        
    def itemChange(self, change, value):
        """
        Handle item changes
        
        Args:
            change: Type of change
            value: New value
            
        Returns:
            value: Possibly adjusted value
        """
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            # Get new position
            new_pos = value
            # Update original position
            self.original_pos = new_pos
            
            # Notify parent view of movement
            if hasattr(self.scene().views()[0], 'on_landmark_moved'):
                self.scene().views()[0].on_landmark_moved(self.index, new_pos.x(), new_pos.y())
                
        return super().itemChange(change, value)


class ImageView(QGraphicsView):
    """
    Image view component that displays the source image and landmarks
    """
    
    # Signal emitted when a landmark position changes
    landmark_moved = pyqtSignal(int, int, int)  # index, x, y
    # Signal emitted when a landmark is clicked
    landmark_clicked = pyqtSignal(int)  # index
    
    def __init__(self, controller: AppController):
        """
        Initialize the image view
        
        Args:
            controller (AppController): Application controller
        """
        super().__init__()
        
        self.controller = controller
        
        # Setup scene
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Setup variables
        self.image_item = None
        self.landmark_items = []
        self.connection_items = []
        self.landmark_groups = {}  # Mapping of landmark groups and colors
        self.scale_factor = 1.0
        self.selected_landmark_index = -1
        
        # Setup view options
        self.setRenderHint(self.renderHints().Antialiasing)
        self.setRenderHint(self.renderHints().SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setInteractive(True)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)
        
        # Connect signals
        
    def update_view(self, event_type: str = None, data: Any = None) -> None:
        """
        Update the view based on model changes
        
        Args:
            event_type (str): Type of event that occurred
            data (Any): Additional data related to the event
        """
        if event_type == 'file_loaded' or event_type == 'image_loaded':
            self.display_image()
            
        elif event_type == 'landmarks_updated':
            self.update_landmarks()
            
        elif event_type == 'reset':
            self.clear()
            
        elif event_type == 'undo' or event_type == 'redo':
            self.display_image()
            self.update_landmarks()
            
    def display_image(self) -> None:
        """
        Display the image from the model
        """
        # Clear scene
        self.scene.clear()
        self.image_item = None
        self.landmark_items = []
        self.connection_items = []
        
        # Check if image exists
        if self.controller.model.image is None:
            return
            
        # Convert image to QImage
        height, width, channel = self.controller.model.image.shape
        bytes_per_line = 3 * width
        q_img = QImage(self.controller.model.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to pixmap and add to scene
        pixmap = QPixmap.fromImage(q_img)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        
        # Fit to view
        self.fit_to_view()
        
        # Update landmarks
        self.update_landmarks()
        
    def update_landmarks(self) -> None:
        """
        Update landmarks on the image
        """
        # Clear existing landmarks
        for item in self.landmark_items:
            self.scene.removeItem(item)
        self.landmark_items = []
        
        for item in self.connection_items:
            self.scene.removeItem(item)
        self.connection_items = []
        
        # Check if landmarks exist
        if self.controller.model.landmarks is None or self.image_item is None:
            return
            
        # Get config values
        point_size = self.controller.model.config.get("editor.landmarks.point_size", 5)
        normal_color = self.controller.model.config.get("editor.landmarks.colors.normal", [0, 255, 0])
        selected_color = self.controller.model.config.get("editor.landmarks.colors.selected", [255, 0, 0])
        line_width = self.controller.model.config.get("editor.landmarks.line_width", 1)
        connection_color = self.controller.model.config.get("editor.landmarks.colors.connection", [0, 255, 255])
        
        # Create QColor objects
        normal_qcolor = QColor(normal_color[0], normal_color[1], normal_color[2])
        selected_qcolor = QColor(selected_color[0], selected_color[1], selected_color[2])
        connection_qcolor = QColor(connection_color[0], connection_color[1], connection_color[2])
        
        # Setup landmark groups and colors (same as LandmarkEditor)
        self.landmark_groups = {
            "face_contour": {"color": QColor(255, 0, 0), "indices": list(range(0, 17))},
            "left_eyebrow": {"color": QColor(0, 255, 0), "indices": list(range(17, 22))},
            "right_eyebrow": {"color": QColor(0, 255, 0), "indices": list(range(22, 27))},
            "nose_bridge": {"color": QColor(0, 0, 255), "indices": list(range(27, 31))},
            "nose_tip": {"color": QColor(0, 0, 255), "indices": list(range(31, 36))},
            "left_eye": {"color": QColor(255, 255, 0), "indices": list(range(36, 42))},
            "right_eye": {"color": QColor(255, 255, 0), "indices": list(range(42, 48))},
            "outer_lips": {"color": QColor(255, 0, 255), "indices": list(range(48, 60))},
            "inner_lips": {"color": QColor(255, 0, 255), "indices": list(range(60, 68))},
            "animal_eyes": {"color": QColor(255, 255, 0), "indices": list(range(0, 4))},
            "animal_nose": {"color": QColor(0, 0, 255), "indices": list(range(4, 6))},
            "animal_mouth": {"color": QColor(255, 0, 255), "indices": list(range(6, 10))},
            "animal_ears": {"color": QColor(0, 255, 0), "indices": list(range(10, 14))},
        }
        
        # Draw facial landmark connections
        connections = self._get_landmark_connections()
        landmarks = self.controller.model.landmarks
        
        # Draw connections first (behind landmarks)
        for connection in connections:
            # Only draw connections when both points exist
            if all(idx < len(landmarks) for idx in connection["points"]):
                points = [landmarks[idx] for idx in connection["points"]]
                color = connection.get("color", connection_qcolor)
                width = connection.get("width", line_width)
                
                pen = QPen(color)
                pen.setWidth(width)
                
                # For polygons (closed shapes)
                if connection.get("closed", False):
                    # Create lines for each segment
                    for i in range(len(points)):
                        line = QGraphicsLineItem(
                            points[i][0], points[i][1],
                            points[(i+1) % len(points)][0], points[(i+1) % len(points)][1]
                        )
                        line.setPen(pen)
                        self.scene.addItem(line)
                        self.connection_items.append(line)
                else:
                    # For open paths, connect sequential points
                    for i in range(len(points) - 1):
                        line = QGraphicsLineItem(
                            points[i][0], points[i][1],
                            points[i+1][0], points[i+1][1]
                        )
                        line.setPen(pen)
                        self.scene.addItem(line)
                        self.connection_items.append(line)
        
        # Draw landmarks
        for i, (x, y) in enumerate(landmarks):
            # Determine color based on group
            group_color = normal_qcolor
            for group_name, group_data in self.landmark_groups.items():
                if i in group_data["indices"]:
                    group_color = group_data["color"]
                    break
            
            # Use selected color if this is the selected landmark
            if i == self.selected_landmark_index:
                pen = QPen(selected_qcolor)
                pen.setWidth(2)
                brush = QBrush(selected_qcolor)
            else:
                pen = QPen(group_color)
                brush = QBrush(group_color)
            
            # Create landmark item
            landmark = LandmarkItem(x, y, point_size, i)
            landmark.setPen(pen)
            landmark.setBrush(brush)
            
            self.scene.addItem(landmark)
            self.landmark_items.append(landmark)
    
    def _get_landmark_connections(self) -> List[Dict]:
        """
        Get landmark connections based on the type of landmarks
        
        Returns:
            List[Dict]: List of connection specifications
        """
        # Check landmark count to determine type
        if self.controller.model.landmarks is None:
            return []
            
        num_landmarks = len(self.controller.model.landmarks)
        
        # Human face connections (different configurations based on number of landmarks)
        if num_landmarks >= 68:
            # Full 68-point facial landmarks
            connections = [
                # Face contour
                {"points": list(range(0, 17)), "color": QColor(255, 0, 0), "width": 1},
                # Left eyebrow
                {"points": list(range(17, 22)), "color": QColor(0, 255, 0), "width": 1},
                # Right eyebrow
                {"points": list(range(22, 27)), "color": QColor(0, 255, 0), "width": 1},
                # Nose bridge
                {"points": list(range(27, 31)), "color": QColor(0, 0, 255), "width": 1},
                # Nose bottom
                {"points": list(range(31, 36)), "color": QColor(0, 0, 255), "width": 1},
                # Left eye
                {"points": list(range(36, 42)) + [36], "color": QColor(255, 255, 0), "width": 1, "closed": True},
                # Right eye
                {"points": list(range(42, 48)) + [42], "color": QColor(255, 255, 0), "width": 1, "closed": True},
                # Outer mouth
                {"points": list(range(48, 60)) + [48], "color": QColor(255, 0, 255), "width": 1, "closed": True},
                # Inner mouth
                {"points": list(range(60, 68)) + [60], "color": QColor(255, 0, 255), "width": 1, "closed": True},
            ]
            return connections
        elif num_landmarks <= 20:
            # Animal face landmarks or simplified face model
            connections = [
                # Eyes
                {"points": [0, 1], "color": QColor(255, 255, 0), "width": 1},
                {"points": [2, 3], "color": QColor(255, 255, 0), "width": 1},
                # Mouth
                {"points": [6, 7, 8, 9, 6], "color": QColor(255, 0, 255), "width": 1, "closed": True},
                # Ears
                {"points": [10, 11], "color": QColor(0, 255, 0), "width": 1},
                {"points": [12, 13], "color": QColor(0, 255, 0), "width": 1},
            ]
            return connections
        
        # Default empty connections
        return []
            
    def zoom_in(self) -> None:
        """
        Zoom in the view
        """
        self.scale(1.2, 1.2)
        self.scale_factor *= 1.2
        self._update_status_zoom()
        
    def zoom_out(self) -> None:
        """
        Zoom out the view
        """
        self.scale(1/1.2, 1/1.2)
        self.scale_factor /= 1.2
        self._update_status_zoom()
        
    def fit_to_view(self) -> None:
        """
        Fit the image to the view
        """
        if self.image_item is None:
            return
            
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        self.scale_factor = 1.0
        self._update_status_zoom()
        
    def clear(self) -> None:
        """
        Clear the view
        """
        self.scene.clear()
        self.image_item = None
        self.landmark_items = []
        self.connection_items = []
        self.selected_landmark_index = -1
        
    def _update_status_zoom(self) -> None:
        """
        Update zoom status in main window
        """
        # Find parent main window
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'zoom_label'):
                parent.zoom_label.setText(f"Zoom: {self.scale_factor * 100:.0f}%")
                break
            parent = parent.parent()
            
    def on_landmark_moved(self, index, x, y):
        """
        Handle landmark movement
        
        Args:
            index (int): Landmark index
            x (int): New X coordinate
            y (int): New Y coordinate
        """
        # Update landmarks in model
        if self.controller.model.landmarks is not None and index < len(self.controller.model.landmarks):
            landmarks = self.controller.model.landmarks.copy()
            landmarks[index] = [int(x), int(y)]
            
            # Update model
            self.controller.update_landmarks(landmarks)
            
            # Emit signal for landmark movement
            self.landmark_moved.emit(index, int(x), int(y))
            
    def select_landmark(self, index, point=None):
        """
        Select a landmark
        
        Args:
            index (int): Landmark index
            point (QPoint, optional): Point location
        """
        # Store selected landmark index
        self.selected_landmark_index = index
        
        # Update landmark appearance
        self.update_landmarks()
        
        # Emit signal for landmark clicked
        self.landmark_clicked.emit(index)
        
        # Ensure the landmark is visible
        if point is not None and self.image_item is not None:
            # Create visible rect around landmark
            rect = QRectF(point.x() - 20, point.y() - 20, 40, 40)
            self.ensureVisible(rect)
            
    def wheelEvent(self, event):
        """
        Handle mouse wheel events
        
        Args:
            event: Wheel event
        """
        if event.modifiers() == Qt.ControlModifier:
            # Zoom with mouse wheel
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            # Default scroll behavior
            super().wheelEvent(event)
            
    def mousePressEvent(self, event):
        """
        Handle mouse press events
        
        Args:
            event: Mouse press event
        """
        if event.button() == Qt.LeftButton:
            # Check if a landmark was clicked
            pos = self.mapToScene(event.pos())
            item = self.scene.itemAt(pos, self.transform())
            
            if isinstance(item, LandmarkItem):
                # Landmark clicked
                self.select_landmark(item.index)
            elif isinstance(item, QGraphicsPixmapItem) and item == self.image_item:
                # Image clicked
                if event.modifiers() == Qt.ShiftModifier and self.selected_landmark_index >= 0:
                    # Shift+Click: Move the selected landmark to the clicked position
                    self.on_landmark_moved(self.selected_landmark_index, pos.x(), pos.y())
                    
                    # Update the landmark item position
                    if 0 <= self.selected_landmark_index < len(self.landmark_items):
                        landmark = self.landmark_items[self.selected_landmark_index]
                        landmark.setPos(pos.x() - landmark.rect().width()/2, 
                                       pos.y() - landmark.rect().height()/2)
                
        super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        """
        Handle mouse move events
        
        Args:
            event: Mouse move event
        """
        # Update position in status bar
        pos = self.mapToScene(event.pos())
        
        # Find parent main window
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'position_label'):
                parent.position_label.setText(f"Position: {int(pos.x())}, {int(pos.y())}")
                break
            parent = parent.parent()
                
        super().mouseMoveEvent(event)
        
    def keyPressEvent(self, event):
        """
        Handle key press events
        
        Args:
            event: Key press event
        """
        if self.selected_landmark_index >= 0 and self.controller.model.landmarks is not None:
            # Arrow keys: move selected landmark
            if event.key() == Qt.Key_Up:
                self.move_selected_landmark(0, -1)
                event.accept()
            elif event.key() == Qt.Key_Down:
                self.move_selected_landmark(0, 1)
                event.accept()
            elif event.key() == Qt.Key_Left:
                self.move_selected_landmark(-1, 0)
                event.accept()
            elif event.key() == Qt.Key_Right:
                self.move_selected_landmark(1, 0)
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)
            
    def move_selected_landmark(self, dx, dy):
        """
        Move the selected landmark by the given delta
        
        Args:
            dx (int): X delta
            dy (int): Y delta
        """
        if self.selected_landmark_index < 0 or self.controller.model.landmarks is None:
            return
            
        # Get current position
        if self.selected_landmark_index < len(self.controller.model.landmarks):
            x, y = self.controller.model.landmarks[self.selected_landmark_index]
            
            # Update position
            self.on_landmark_moved(self.selected_landmark_index, x + dx, y + dy)