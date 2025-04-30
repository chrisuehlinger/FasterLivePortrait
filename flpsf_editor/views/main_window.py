#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Window View for FLPSF Editor
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QAction, QFileDialog, QMessageBox, QDockWidget, QToolBar,
    QStatusBar, QLabel, QMenu
)
from PyQt5.QtCore import Qt, QSettings, QSize, QTimer
from PyQt5.QtGui import QIcon, QKeySequence

# Import custom widgets
from views.image_view import ImageView
from views.parameters_panel import ParametersPanel
from views.landmark_editor import LandmarkEditor
from views.preview_panel import PreviewPanel

# Import controller
from controllers.app_controller import AppController

logger = logging.getLogger('flpsf_editor.views.main_window')

class MainWindow(QMainWindow):
    """
    Main window for the FLPSF Editor application
    """
    
    def __init__(self, controller: AppController):
        """
        Initialize the main window
        
        Args:
            controller (AppController): Application controller
        """
        super().__init__()
        
        self.controller = controller
        self.settings = QSettings("FasterLivePortrait", "FLPSF Editor")
        
        # Initialize UI components
        self._init_ui()
        self._create_actions()
        self._create_menus()
        self._create_toolbars()
        self._create_status_bar()
        self._create_dock_widgets()
        self._restore_window_settings()
        
        # Initialize timers
        self._init_timers()
        
        # Set window title
        self.setWindowTitle("FLPSF Editor")
        
    def _init_ui(self) -> None:
        """
        Initialize UI components
        """
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout with splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Create image view
        self.image_view = ImageView(self.controller)
        self.main_splitter.addWidget(self.image_view)
        
        # Create parameters panel container
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.parameters_panel = ParametersPanel(self.controller)
        self.params_layout.addWidget(self.parameters_panel)
        self.main_splitter.addWidget(self.params_container)
        
        # Set splitter sizes
        self.main_splitter.setSizes([700, 300])  # Default sizes for left/right panels
        
        # Create main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.main_splitter)
        
        # Set window size
        self.resize(1200, 800)
        
        # Connect image view and landmark editor signals
        self._connect_view_signals()
        
    def _connect_view_signals(self) -> None:
        """
        Connect signals between view components
        """
        # Connect landmark editor signals to image view
        self.landmark_dock = None  # Will be created in _create_dock_widgets
        
    def _create_actions(self) -> None:
        """
        Create actions for menus and toolbars
        """
        # File actions
        self.new_action = QAction("&New", self)
        self.new_action.setShortcut(QKeySequence.New)
        self.new_action.setStatusTip("Create a new file")
        self.new_action.triggered.connect(self._on_new)
        
        self.open_action = QAction("&Open...", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.setStatusTip("Open a file")
        self.open_action.triggered.connect(self._on_open)
        
        self.save_action = QAction("&Save", self)
        self.save_action.setShortcut(QKeySequence.Save)
        self.save_action.setStatusTip("Save the current file")
        self.save_action.triggered.connect(self._on_save)
        
        self.save_as_action = QAction("Save &As...", self)
        self.save_as_action.setShortcut(QKeySequence.SaveAs)
        self.save_as_action.setStatusTip("Save the current file with a new name")
        self.save_as_action.triggered.connect(self._on_save_as)
        
        self.import_image_action = QAction("&Import Image...", self)
        self.import_image_action.setStatusTip("Import an image")
        self.import_image_action.triggered.connect(self._on_import_image)
        
        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut(QKeySequence.Quit)
        self.exit_action.setStatusTip("Exit the application")
        self.exit_action.triggered.connect(self.close)
        
        # Edit actions
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.setStatusTip("Undo the last action")
        self.undo_action.triggered.connect(self._on_undo)
        
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.setStatusTip("Redo the last undone action")
        self.redo_action.triggered.connect(self._on_redo)
        
        # View actions
        self.zoom_in_action = QAction("Zoom &In", self)
        self.zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        self.zoom_in_action.setStatusTip("Zoom in")
        self.zoom_in_action.triggered.connect(self._on_zoom_in)
        
        self.zoom_out_action = QAction("Zoom &Out", self)
        self.zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        self.zoom_out_action.setStatusTip("Zoom out")
        self.zoom_out_action.triggered.connect(self._on_zoom_out)
        
        self.fit_to_view_action = QAction("&Fit to View", self)
        self.fit_to_view_action.setShortcut("Ctrl+0")
        self.fit_to_view_action.setStatusTip("Fit the image to the view")
        self.fit_to_view_action.triggered.connect(self._on_fit_to_view)
        
        # Tools actions
        self.detect_landmarks_action = QAction("&Detect Landmarks", self)
        self.detect_landmarks_action.setStatusTip("Detect landmarks in the current image")
        self.detect_landmarks_action.triggered.connect(self._on_detect_landmarks)
        
        self.detect_animal_landmarks_action = QAction("Detect &Animal Landmarks", self)
        self.detect_animal_landmarks_action.setStatusTip("Detect animal landmarks in the current image")
        self.detect_animal_landmarks_action.triggered.connect(self._on_detect_animal_landmarks)
        
        self.compute_motion_params_action = QAction("Compute &Motion Parameters", self)
        self.compute_motion_params_action.setStatusTip("Compute motion parameters from landmarks")
        self.compute_motion_params_action.triggered.connect(self._on_compute_motion_params)
        
        self.compute_appearance_action = QAction("Compute &Appearance Features", self)
        self.compute_appearance_action.setStatusTip("Compute appearance features from the image")
        self.compute_appearance_action.triggered.connect(self._on_compute_appearance)
        
        self.generate_mask_action = QAction("Generate &Pasteback Mask", self)
        self.generate_mask_action.setStatusTip("Generate pasteback mask for the image")
        self.generate_mask_action.triggered.connect(self._on_generate_mask)
        
        self.run_preview_action = QAction("Run &Preview", self)
        self.run_preview_action.setStatusTip("Run a preview animation")
        self.run_preview_action.triggered.connect(self._on_run_preview)
        
        # New actions for FLPSF validation and auto-completion
        self.validate_flpsf_action = QAction("&Validate FLPSF Data", self)
        self.validate_flpsf_action.setShortcut("Ctrl+Alt+V")
        self.validate_flpsf_action.setStatusTip("Validate current FLPSF data for pipeline compatibility")
        self.validate_flpsf_action.triggered.connect(self._on_validate_flpsf)
        
        self.auto_complete_flpsf_action = QAction("Auto-&Complete FLPSF Data", self)
        self.auto_complete_flpsf_action.setShortcut("Ctrl+Alt+C")
        self.auto_complete_flpsf_action.setStatusTip("Automatically complete missing FLPSF data using models")
        self.auto_complete_flpsf_action.triggered.connect(self._on_auto_complete_flpsf)
        
    def _create_menus(self) -> None:
        """
        Create menus
        """
        # File menu
        self.file_menu = self.menuBar().addMenu("&File")
        self.file_menu.addAction(self.new_action)
        self.file_menu.addAction(self.open_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(self.save_as_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.import_image_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)
        
        # Edit menu
        self.edit_menu = self.menuBar().addMenu("&Edit")
        self.edit_menu.addAction(self.undo_action)
        self.edit_menu.addAction(self.redo_action)
        
        # View menu
        self.view_menu = self.menuBar().addMenu("&View")
        self.view_menu.addAction(self.zoom_in_action)
        self.view_menu.addAction(self.zoom_out_action)
        self.view_menu.addAction(self.fit_to_view_action)
        self.view_menu.addSeparator()
        
        # Tools menu
        self.tools_menu = self.menuBar().addMenu("&Tools")
        self.tools_menu.addAction(self.detect_landmarks_action)
        self.tools_menu.addAction(self.detect_animal_landmarks_action)
        self.tools_menu.addSeparator()
        self.tools_menu.addAction(self.compute_motion_params_action)
        self.tools_menu.addAction(self.compute_appearance_action)
        self.tools_menu.addAction(self.generate_mask_action)
        self.tools_menu.addSeparator()
        self.tools_menu.addAction(self.run_preview_action)
        self.tools_menu.addSeparator()
        self.tools_menu.addAction(self.validate_flpsf_action)
        self.tools_menu.addAction(self.auto_complete_flpsf_action)
        
        # Help menu
        self.help_menu = self.menuBar().addMenu("&Help")
        self.help_action = QAction("&About", self)
        self.help_action.triggered.connect(self._on_about)
        self.help_menu.addAction(self.help_action)
        
    def _create_toolbars(self) -> None:
        """
        Create toolbars
        """
        # Main toolbar
        self.main_toolbar = self.addToolBar("Main")
        self.main_toolbar.setObjectName("MainToolbar")
        self.main_toolbar.addAction(self.new_action)
        self.main_toolbar.addAction(self.open_action)
        self.main_toolbar.addAction(self.save_action)
        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self.undo_action)
        self.main_toolbar.addAction(self.redo_action)
        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self.zoom_in_action)
        self.main_toolbar.addAction(self.zoom_out_action)
        self.main_toolbar.addAction(self.fit_to_view_action)
        
        # Tools toolbar
        self.tools_toolbar = self.addToolBar("Tools")
        self.tools_toolbar.setObjectName("ToolsToolbar")
        self.tools_toolbar.addAction(self.detect_landmarks_action)
        self.tools_toolbar.addAction(self.compute_motion_params_action)
        self.tools_toolbar.addAction(self.compute_appearance_action)
        self.tools_toolbar.addAction(self.run_preview_action)
        
    def _create_status_bar(self) -> None:
        """
        Create status bar
        """
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add widgets to status bar
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Add position label
        self.position_label = QLabel("Position: ---, ---")
        self.status_bar.addPermanentWidget(self.position_label)
        
        # Add zoom label
        self.zoom_label = QLabel("Zoom: 100%")
        self.status_bar.addPermanentWidget(self.zoom_label)
        
    def _create_dock_widgets(self) -> None:
        """
        Create dock widgets
        """
        # Landmark editor dock
        self.landmark_dock = QDockWidget("Landmark Editor", self)
        self.landmark_dock.setObjectName("LandmarkEditorDock")
        self.landmark_editor = LandmarkEditor(self.controller)
        self.landmark_dock.setWidget(self.landmark_editor)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.landmark_dock)
        self.view_menu.addAction(self.landmark_dock.toggleViewAction())
        
        # Connect signals between landmark editor and image view
        self.landmark_editor.landmark_selected.connect(self.image_view.select_landmark)
        self.image_view.landmark_moved.connect(self.landmark_editor.update_landmark_position)
        self.image_view.landmark_clicked.connect(lambda idx: self.landmark_editor.landmark_table.selectRow(idx))
        
        # Preview panel dock
        self.preview_dock = QDockWidget("Preview", self)
        self.preview_dock.setObjectName("PreviewDock")
        self.preview_panel = PreviewPanel(self.controller)
        self.preview_dock.setWidget(self.preview_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.preview_dock)
        self.view_menu.addAction(self.preview_dock.toggleViewAction())
        
    def _restore_window_settings(self) -> None:
        """
        Restore window settings from previous session
        """
        if self.settings.contains("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))
        if self.settings.contains("windowState"):
            self.restoreState(self.settings.value("windowState"))
            
    def _init_timers(self) -> None:
        """
        Initialize timers
        """
        # Auto-save timer
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self._on_auto_save)
        
        # Check if auto-save is enabled in settings
        if self.controller.model.config.get("application.auto_save", False):
            interval = self.controller.model.config.get("application.auto_save_interval", 300) * 1000
            self.auto_save_timer.start(interval)
            
    def update_view(self, event_type: str = None, data: Any = None) -> None:
        """
        Update the view based on model changes
        
        Args:
            event_type (str): Type of event that occurred
            data (Any): Additional data related to the event
        """
        # Update UI based on event type
        if event_type == 'file_loaded':
            self.status_label.setText(f"Loaded: {os.path.basename(data['path'])}")
            self.image_view.update_view(event_type, data)
            self.parameters_panel.update_view(event_type, data)
            self.landmark_editor.update_view(event_type, data)
            self.preview_panel.update_view(event_type, data)
            
        elif event_type == 'file_saved':
            self.status_label.setText(f"Saved: {os.path.basename(data['path'])}")
            
        elif event_type == 'image_loaded':
            self.status_label.setText(f"Image loaded: {os.path.basename(data['path'])}")
            self.image_view.update_view(event_type, data)
            self.parameters_panel.update_view(event_type, data)
            self.landmark_editor.update_view(event_type, data)
            self.preview_panel.update_view(event_type, data)
            
        elif event_type == 'landmarks_updated':
            self.status_label.setText("Landmarks updated")
            self.image_view.update_view(event_type, data)
            self.landmark_editor.update_view(event_type, data)
            
        elif event_type == 'param_updated':
            param_type = data.get('type', '')
            param_name = data.get('param', '')
            self.status_label.setText(f"{param_type.capitalize()} parameter '{param_name}' updated")
            self.parameters_panel.update_view(event_type, data)
            
        elif event_type == 'reset':
            self.status_label.setText("Reset")
            self.image_view.update_view(event_type, data)
            self.parameters_panel.update_view(event_type, data)
            self.landmark_editor.update_view(event_type, data)
            self.preview_panel.update_view(event_type, data)
            
        elif event_type == 'undo':
            self.status_label.setText(f"Undo: {data.get('action_type', '')}")
            self.image_view.update_view(event_type, data)
            self.parameters_panel.update_view(event_type, data)
            self.landmark_editor.update_view(event_type, data)
            
        elif event_type == 'redo':
            self.status_label.setText(f"Redo: {data.get('action_type', '')}")
            self.image_view.update_view(event_type, data)
            self.parameters_panel.update_view(event_type, data)
            self.landmark_editor.update_view(event_type, data)
            
    def _on_new(self) -> None:
        """
        Handle New action
        """
        # Check if there are unsaved changes
        if self._check_unsaved_changes():
            self.controller.new_file()
            self.status_label.setText("New file")
            
    def _on_open(self) -> None:
        """
        Handle Open action
        """
        # Check if there are unsaved changes
        if self._check_unsaved_changes():
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open FLPSF File",
                self.controller.model.config.get("paths.default_save_dir", "."),
                "FLPSF Files (*.flpsf);;All Files (*)"
            )
            
            if file_path:
                if self.controller.open_file(file_path):
                    self.status_label.setText(f"Opened: {os.path.basename(file_path)}")
                else:
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Failed to open file: {file_path}"
                    )
                    
    def _on_save(self) -> None:
        """
        Handle Save action
        """
        if self.controller.model.flpsf_path:
            if self.controller.save_file():
                self.status_label.setText(f"Saved: {os.path.basename(self.controller.model.flpsf_path)}")
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Failed to save file"
                )
        else:
            self._on_save_as()
            
    def _on_save_as(self) -> None:
        """
        Handle Save As action
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save FLPSF File",
            self.controller.model.config.get("paths.default_save_dir", "."),
            "FLPSF Files (*.flpsf);;All Files (*)"
        )
        
        if file_path:
            if not file_path.endswith(".flpsf"):
                file_path += ".flpsf"
                
            if self.controller.save_file(file_path):
                self.status_label.setText(f"Saved: {os.path.basename(file_path)}")
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save file: {file_path}"
                )
                
    def _on_import_image(self) -> None:
        """
        Handle Import Image action
        """
        # Check if there are unsaved changes
        if self._check_unsaved_changes():
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Image",
                self.controller.model.config.get("paths.default_save_dir", "."),
                "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
            )
            
            if file_path:
                if self.controller.load_image(file_path):
                    self.status_label.setText(f"Imported image: {os.path.basename(file_path)}")
                else:
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Failed to import image: {file_path}"
                    )
                    
    def _on_undo(self) -> None:
        """
        Handle Undo action
        """
        if self.controller.undo():
            self.status_label.setText("Undo")
        else:
            self.status_label.setText("Nothing to undo")
            
    def _on_redo(self) -> None:
        """
        Handle Redo action
        """
        if self.controller.redo():
            self.status_label.setText("Redo")
        else:
            self.status_label.setText("Nothing to redo")
            
    def _on_zoom_in(self) -> None:
        """
        Handle Zoom In action
        """
        self.image_view.zoom_in()
        
    def _on_zoom_out(self) -> None:
        """
        Handle Zoom Out action
        """
        self.image_view.zoom_out()
        
    def _on_fit_to_view(self) -> None:
        """
        Handle Fit to View action
        """
        self.image_view.fit_to_view()
        
    def _on_detect_landmarks(self) -> None:
        """
        Handle Detect Landmarks action
        """
        if self.controller.detect_landmarks('insightface'):
            self.status_label.setText("Landmarks detected")
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Failed to detect landmarks"
            )
            
    def _on_detect_animal_landmarks(self) -> None:
        """
        Handle Detect Animal Landmarks action
        """
        if self.controller.detect_landmarks('xpose'):
            self.status_label.setText("Animal landmarks detected")
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Failed to detect animal landmarks"
            )
            
    def _on_compute_motion_params(self) -> None:
        """
        Handle Compute Motion Parameters action
        """
        if self.controller.compute_motion_parameters():
            self.status_label.setText("Motion parameters computed")
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Failed to compute motion parameters"
            )
            
    def _on_compute_appearance(self) -> None:
        """
        Handle Compute Appearance Features action
        """
        if self.controller.compute_appearance_features():
            self.status_label.setText("Appearance features computed")
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Failed to compute appearance features"
            )
            
    def _on_generate_mask(self) -> None:
        """
        Handle Generate Pasteback Mask action
        """
        if self.controller.generate_pasteback_mask():
            self.status_label.setText("Pasteback mask generated")
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Failed to generate pasteback mask"
            )
            
    def _on_run_preview(self) -> None:
        """
        Handle Run Preview action
        """
        # TODO: Implement preview functionality
        self.status_label.setText("Preview not implemented yet")
        
    def _on_validate_flpsf(self) -> None:
        """
        Handle Validate FLPSF Data action
        """
        if self.controller.validate_flpsf_data():
            self.status_label.setText("FLPSF data validated successfully")
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "FLPSF data validation failed"
            )
            
    def _on_auto_complete_flpsf(self) -> None:
        """
        Handle Auto-Complete FLPSF Data action
        """
        if self.controller.auto_complete_flpsf_data():
            self.status_label.setText("FLPSF data auto-completed successfully")
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "FLPSF data auto-completion failed"
            )
        
    def _on_about(self) -> None:
        """
        Handle About action
        """
        QMessageBox.about(
            self,
            "About FLPSF Editor",
            "<h3>FLPSF Editor</h3>"
            "<p>FasterLivePortrait Source Format Editor</p>"
            "<p>Version 1.0.0</p>"
            "<p>FLPSF Editor is a tool for creating and editing FLPSF files "
            "used by FasterLivePortrait for source image processing.</p>"
        )
        
    def _on_auto_save(self) -> None:
        """
        Handle auto-save timer
        """
        if self.controller.model.is_modified and self.controller.model.flpsf_path:
            if self.controller.save_file():
                self.status_label.setText(f"Auto-saved: {os.path.basename(self.controller.model.flpsf_path)}")
                
    def _check_unsaved_changes(self) -> bool:
        """
        Check if there are unsaved changes and ask user what to do
        
        Returns:
            bool: True if operation can proceed, False otherwise
        """
        if not self.controller.model.is_modified:
            return True
            
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "There are unsaved changes. Save before proceeding?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save
        )
        
        if reply == QMessageBox.Save:
            return self._on_save()
        elif reply == QMessageBox.Discard:
            return True
        else:  # Cancel
            return False
            
    def closeEvent(self, event) -> None:
        """
        Handle window close event
        
        Args:
            event: Close event
        """
        # Check for unsaved changes
        if not self._check_unsaved_changes():
            event.ignore()
            return
            
        # Save window settings
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        
        # Accept close event
        event.accept()