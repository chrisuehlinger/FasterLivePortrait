#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPD Editor - Graphical User Interface

This module provides a GUI for working with SPD files using Tkinter
for minimal external dependencies.
"""

import os
import sys
import logging
import threading
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable

# Add the project root directory to the Python path to make src modules accessible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger("spd_editor.gui")

# Diagnostic check for common import issue cause
# This check is added to help diagnose the "No module named src.data_process" error.
# It verifies if the __init__.py file exists, which is necessary for 'data_process' to be a package.
expected_init_py_path = os.path.join(project_root, "src", "data_process", "__init__.py")
if not os.path.exists(expected_init_py_path):
    logger.warning(
        f"DIAGNOSTIC: The file '{expected_init_py_path}' was not found. "
        f"This is a likely cause for 'No module named src.data_process' errors. "
        f"Please ensure 'src/data_process/' is a Python package by creating an empty __init__.py file in it."
    )

# Check if tkinter is available
TKINTER_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
    from tkinter.font import Font
    TKINTER_AVAILABLE = True
except ImportError:
    # Create dummy tk module for environments where tkinter is not available
    class DummyTk:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    tk = DummyTk()
    ttk = DummyTk()
    filedialog = DummyTk()
    messagebox = DummyTk()
    simpledialog = DummyTk()
    Font = lambda *args, **kwargs: None

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    cv2 = DummyTk()
    np = DummyTk()
    Image = DummyTk()
    ImageTk = DummyTk()
    logger.warning("OpenCV or PIL not available. Image processing will be disabled.")

# Import OmegaConf for YAML configuration
try:
    from omegaconf import OmegaConf
    HAVE_OMEGACONF = True
except ImportError:
    HAVE_OMEGACONF = False
    logger.warning("OmegaConf not available. Default parameters will be used for analysis modules.")

# Import SPD modules
from spd_editor.spd.reader import SPDReader, SPDError, SPDSectionError
from spd_editor.spd.writer import SPDWriter
from spd_editor.spd.format import (
    HeaderFlags, ImageSection, LandmarksSection, 
    MotionParamsSection, MaskSection
)
from spd_editor.spd.validator import (
    SPDValidator, ValidationLevel, ValidationReport
)
from spd_editor.utils.visualization import (
    draw_landmarks, LandmarkVisualizationOptions
)

# Import face detection and landmark extraction modules
try:
    from spd_editor.analysis.face_detector import FaceDetector
    from spd_editor.analysis.landmark_extractor import LandmarkExtractor
    HAVE_FACE_DETECTION = True
except ImportError:
    HAVE_FACE_DETECTION = False
    logger.warning("Face detection modules not available. Face detection will be disabled.")
    FaceDetector = lambda *args, **kwargs: None
    LandmarkExtractor = lambda *args, **kwargs: None

# Try to import matplotlib for 3D visualization
HAVE_PLT = False
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for embedding in tkinter
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False
    logger.warning("Matplotlib not available. 3D visualization will be disabled.")

# Constants
DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 800
DEFAULT_PADDING = 10
SIDEBAR_WIDTH = 300


class SPDEditorApp:
    """Main application class for the SPD Editor GUI."""
    
    def __init__(self, root: tk.Tk):
        """Initialize the application.
        
        Args:
            root: The root tkinter window
        """
        self.root = root
        self.root.title("SPD Editor")
        self.root.geometry(f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
        
        # Set up state variables
        self.current_file: Optional[str] = None
        self.spd_reader: Optional[SPDReader] = None
        self.image: Optional[np.ndarray] = None
        self.landmarks: Optional[List[List[float]]] = None
        self.mask: Optional[np.ndarray] = None
        self.parameters: Dict[str, float] = {}
        self.visualization_options = LandmarkVisualizationOptions()
        self.status_text = tk.StringVar()
        self.status_text.set("Ready")
        self.progress_var = tk.DoubleVar(value=0)
        self.show_landmarks = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=False)
        self._pan_start = None

        # Load pipeline configuration
        self.pipeline_cfg = None
        self.crop_params = {}
        self.infer_params = {}
        if HAVE_OMEGACONF:
            try:
                config_path = os.path.join(project_root, "configs", "trt_infer.yaml")
                if os.path.exists(config_path):
                    self.pipeline_cfg = OmegaConf.load(config_path)
                    self.crop_params = self.pipeline_cfg.get('crop_params', {})
                    self.infer_params = self.pipeline_cfg.get('infer_params', {})
                    logger.info(f"Loaded configuration from {config_path}")
                else:
                    logger.warning(f"Configuration file not found: {config_path}. Using defaults.")
            except Exception as e:
                logger.error(f"Error loading trt_infer.yaml: {e}. Using defaults.")
        
        # Create the UI
        self._create_menu()
        self._create_layout()
        
        # Set up event bindings
        self.root.bind("<Configure>", self._on_resize)
        
        # Set up keyboard shortcuts
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-s>", lambda e: self.save_file())
        self.root.bind("<Control-e>", lambda e: self.export_image())
        self.root.bind("<Control-n>", lambda e: self.create_new_spd())
    
    def _create_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New SPD File...", command=self.create_new_spd, accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_file_as)
        file_menu.add_separator()
        file_menu.add_command(label="Export Image...", command=self.export_image, accelerator="Ctrl+E")
        file_menu.add_command(label="Export Landmarks...", command=self.export_landmarks)
        file_menu.add_command(label="Export Mask...", command=self.export_mask)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Validate SPD File", command=self.validate_spd)
        edit_menu.add_separator()
        edit_menu.add_command(label="Preferences...", command=self.show_preferences)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_checkbutton(label="Show Landmarks", variable=self.show_landmarks, 
                                 command=self.update_display)
        view_menu.add_checkbutton(label="Show Landmark Labels", variable=self.show_labels,
                                 command=self.update_display)
        view_menu.add_separator()
        view_menu.add_command(label="3D Visualization", command=self.show_3d_visualization)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def _create_layout(self):
        """Create the main application layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=DEFAULT_PADDING, pady=DEFAULT_PADDING)
        
        # Image and parameters panels side by side with a separator
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Image viewer
        left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(left_frame, weight=3)
        
        # Image canvas with scrollbars
        image_frame = ttk.Frame(left_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for displaying the image
        self.canvas = tk.Canvas(image_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars for the canvas
        h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar.pack(fill=tk.X, side=tk.BOTTOM)
        v_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        
        # Configure canvas scrolling
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        self.canvas.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        # Add panning with middle mouse button
        self.canvas.bind("<ButtonPress-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._pan_image)
        self.canvas.bind("<ButtonRelease-2>", self._end_pan)
        
        # Right panel - Parameters and controls
        right_frame = ttk.Frame(self.paned_window, width=SIDEBAR_WIDTH)
        self.paned_window.add(right_frame, weight=1)
        
        # Create sections for the sidebar
        self._create_file_info_section(right_frame)
        self._create_visualization_section(right_frame)
        self._create_parameters_section(right_frame)
        
        # Stretch the last frame to fill remaining space
        right_frame.pack_propagate(False)
        
        # Status bar at the bottom
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(DEFAULT_PADDING, 0))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            status_frame, 
            variable=self.progress_var,
            mode='determinate'
        )
        self.progress_bar.pack(side=tk.LEFT, padx=(0, DEFAULT_PADDING), fill=tk.X, expand=True)
        
        # Status text
        status_label = ttk.Label(status_frame, textvariable=self.status_text, anchor=tk.W)
        status_label.pack(side=tk.RIGHT)
    
    def _start_pan(self, event):
        """Start panning the canvas.
        
        Args:
            event: The mouse event
        """
        self.canvas.config(cursor="fleur")  # Change cursor to indicate panning
        self._pan_start = (event.x, event.y)
    
    def _pan_image(self, event):
        """Pan the canvas.
        
        Args:
            event: The mouse event
        """
        if self._pan_start:
            dx = self._pan_start[0] - event.x
            dy = self._pan_start[1] - event.y
            self.canvas.xview_scroll(int(dx), "units")
            self.canvas.yview_scroll(int(dy), "units")
            self._pan_start = (event.x, event.y)
    
    def _end_pan(self, event):
        """End panning the canvas.
        
        Args:
            event: The mouse event
        """
        self.canvas.config(cursor="")  # Reset cursor
        self._pan_start = None
    
    def _create_file_info_section(self, parent: ttk.Frame):
        """Create the file information section.
        
        Args:
            parent: Parent frame
        """
        frame = ttk.LabelFrame(parent, text="File Information")
        frame.pack(fill=tk.X, pady=(0, DEFAULT_PADDING), padx=DEFAULT_PADDING)
        
        # File path
        ttk.Label(frame, text="File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.file_path_var = tk.StringVar(value="No file loaded")
        ttk.Label(frame, textvariable=self.file_path_var, wraplength=SIDEBAR_WIDTH-40).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Image dimensions
        ttk.Label(frame, text="Dimensions:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.dimensions_var = tk.StringVar(value="N/A")
        ttk.Label(frame, textvariable=self.dimensions_var).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Landmark count
        ttk.Label(frame, text="Landmarks:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.landmark_count_var = tk.StringVar(value="N/A")
        ttk.Label(frame, textvariable=self.landmark_count_var).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Landmark type
        ttk.Label(frame, text="Type:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.landmark_type_var = tk.StringVar(value="N/A")
        ttk.Label(frame, textvariable=self.landmark_type_var).grid(
            row=3, column=1, sticky=tk.W, padx=5, pady=2)
    
    def _create_visualization_section(self, parent: ttk.Frame):
        """Create the visualization options section.
        
        Args:
            parent: Parent frame
        """
        frame = ttk.LabelFrame(parent, text="Visualization Options")
        frame.pack(fill=tk.X, pady=(0, DEFAULT_PADDING), padx=DEFAULT_PADDING)
        
        # Landmark visibility
        ttk.Checkbutton(
            frame, 
            text="Show Landmarks", 
            variable=self.show_landmarks,
            command=self.update_display
        ).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Landmark label visibility
        ttk.Checkbutton(
            frame, 
            text="Show Landmark Labels", 
            variable=self.show_labels,
            command=self.update_display
        ).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Point size
        ttk.Label(frame, text="Point Size:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        
        point_size_frame = ttk.Frame(frame)
        point_size_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.point_size_var = tk.IntVar(value=self.visualization_options.point_size)
        ttk.Spinbox(
            point_size_frame,
            from_=1, to=10, width=5, 
            textvariable=self.point_size_var,
            command=lambda: self._update_visualization_option('point_size', self.point_size_var.get())
        ).pack(side=tk.LEFT)
        
        # Point color
        ttk.Label(frame, text="Point Color:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Button(
            frame, 
            text="Change...", 
            command=self._choose_point_color
        ).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Connection visibility
        self.show_connections = tk.BooleanVar(value=self.visualization_options.draw_connections)
        ttk.Checkbutton(
            frame, 
            text="Show Connections", 
            variable=self.show_connections,
            command=lambda: self._update_visualization_option('draw_connections', self.show_connections.get())
        ).grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Connection thickness
        ttk.Label(frame, text="Connection Thickness:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        
        conn_thickness_frame = ttk.Frame(frame)
        conn_thickness_frame.grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.conn_thickness_var = tk.IntVar(value=self.visualization_options.connection_thickness)
        ttk.Spinbox(
            conn_thickness_frame,
            from_=1, to=5, width=5, 
            textvariable=self.conn_thickness_var,
            command=lambda: self._update_visualization_option('connection_thickness', self.conn_thickness_var.get())
        ).pack(side=tk.LEFT)
    
    def _create_parameters_section(self, parent: ttk.Frame):
        """Create the parameters section for editing values.
        
        Args:
            parent: Parent frame
        """
        self.params_frame = ttk.LabelFrame(parent, text="Parameters")
        self.params_frame.pack(fill=tk.BOTH, expand=True, padx=DEFAULT_PADDING)
        
        # Placeholder text when no parameters are available
        self.no_params_label = ttk.Label(
            self.params_frame, 
            text="No parameters available.\nParameters will appear when a file with motion parameters is loaded.",
            justify=tk.CENTER
        )
        self.no_params_label.pack(expand=True)
        
        # We'll create parameter sliders dynamically when loading a file with parameters
    
    def _update_visualization_option(self, option_name: str, value: Any):
        """Update a visualization option and refresh the display.
        
        Args:
            option_name: Name of the option to update
            value: New value for the option
        """
        setattr(self.visualization_options, option_name, value)
        self.update_display()
    
    def _choose_point_color(self):
        """Open a color chooser dialog for the landmark points."""
        try:
            # Convert current BGR color to RGB for the color chooser
            current_color = self.visualization_options.point_color
            current_rgb = (current_color[2], current_color[1], current_color[0])
            
            # Open the color chooser dialog
            from tkinter import colorchooser
            rgb_color = colorchooser.askcolor(color=f"#{current_rgb[0]:02x}{current_rgb[1]:02x}{current_rgb[2]:02x}")
            
            if rgb_color[0]:  # If a color was selected
                # Convert RGB to BGR for OpenCV
                r, g, b = [int(c) for c in rgb_color[0]]
                self.visualization_options.point_color = (b, g, r)
                self.update_display()
                
        except Exception as e:
            logger.error(f"Error choosing color: {e}")
            messagebox.showerror("Color Selection Error", str(e))
    
    def _on_resize(self, event: tk.Event):
        """Handle window resize event.
        
        Args:
            event: The resize event
        """
        # Only handle events from the root window
        if event.widget == self.root and self.image is not None:
            self.update_display()
    
    def update_display(self):
        """Update the displayed image with landmarks and other visualizations."""
        if self.image is None:
            return
        
        # Make a copy of the image for visualization
        display_img = self.image.copy()
        
        # Draw landmarks if requested
        if self.show_landmarks.get() and self.landmarks is not None:
            # Update visualization options based on current settings
            self.visualization_options.show_labels = self.show_labels.get()
            
            # Draw the landmarks
            display_img = draw_landmarks(display_img, self.landmarks, self.visualization_options)
        
        # Convert the image for display
        self._display_image(display_img)
    
    def _display_image(self, image: np.ndarray):
        """Display an image on the canvas.
        
        Args:
            image: The image to display (in BGR format)
        """
        if image is None:
            return
            
        # Convert BGR to RGB for PIL
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Create a PhotoImage object
        self.photo_image = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _create_parameter_sliders(self):
        """Create sliders for motion parameters."""
        # Clear existing widgets
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        if not self.parameters:
            self.no_params_label = ttk.Label(
                self.params_frame, 
                text="No parameters available.\nParameters will appear when a file with motion parameters is loaded.",
                justify=tk.CENTER
            )
            self.no_params_label.pack(expand=True)
            return
            
        # Create a canvas with scrollbar for parameters
        param_canvas = tk.Canvas(self.params_frame)
        scrollbar = ttk.Scrollbar(self.params_frame, orient="vertical", command=param_canvas.yview)
        scrollable_frame = ttk.Frame(param_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: param_canvas.configure(scrollregion=param_canvas.bbox("all"))
        )
        
        param_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        param_canvas.configure(yscrollcommand=scrollbar.set)
        
        param_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create a slider and label for each parameter
        for i, (param_name, param_value) in enumerate(self.parameters.items()):
            param_frame = ttk.Frame(scrollable_frame)
            param_frame.pack(fill=tk.X, pady=2)
            
            # Parameter label
            ttk.Label(param_frame, text=f"{param_name}:").pack(anchor=tk.W)
            
            # Create a frame for the slider and value
            slider_frame = ttk.Frame(param_frame)
            slider_frame.pack(fill=tk.X)
            
            # Slider variable
            var = tk.DoubleVar(value=param_value)
            
            # Create the slider
            slider = ttk.Scale(
                slider_frame, 
                from_=0.0, 
                to=1.0, 
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda v, name=param_name: self._on_parameter_change(name, float(v))
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Value label
            value_label = ttk.Label(slider_frame, width=5)
            value_label.pack(side=tk.RIGHT)
            
            # Update value label
            def update_label(val, label=value_label):
                label.configure(text=f"{float(val):.2f}")
            
            # Initialize label text
            update_label(param_value)
            
            # Configure slider to update label
            slider.configure(command=lambda v, name=param_name, update_fn=update_label: 
                            self._on_parameter_change(name, float(v), update_fn))
                
    def _on_parameter_change(self, param_name: str, value: float, update_label_fn: Optional[Callable]=None):
        """Handle parameter slider change events.
        
        Args:
            param_name: The name of the parameter that changed
            value: The new parameter value
            update_label_fn: Function to update the display label
        """
        # Update the parameter value
        self.parameters[param_name] = value
        
        # Update the label if provided
        if update_label_fn:
            update_label_fn(value)
    
    def open_file(self):
        """Open an SPD file dialog and load the selected file."""
        file_path = filedialog.askopenfilename(
            title="Open SPD File",
            filetypes=[("SPD Files", "*.spd"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.load_file(file_path)
            
        except Exception as e:
            logger.error(f"Error opening file: {e}")
            messagebox.showerror("Error", f"Failed to open file: {e}")
    
    def load_file(self, file_path: str):
        """Load an SPD file.
        
        Args:
            file_path: Path to the SPD file to load
        """
        try:
            self.status_text.set(f"Loading {file_path}...")
            self.progress_var.set(10)
            
            # Close any currently open file
            if self.spd_reader:
                try:
                    self.spd_reader.close()
                except Exception as e:
                    logger.warning(f"Error closing previous SPDReader: {e}")
                
            self.spd_reader = SPDReader(file_path)
            self.current_file = file_path
            
            self.file_path_var.set(os.path.basename(file_path))
            
            self.image = None
            self.landmarks = None
            self.mask = None
            self.parameters = {}
            self._original_image_data: Optional[np.ndarray] = None
            self._original_landmarks_data: Optional[List[List[float]]] = None
            self._original_mask_data: Optional[np.ndarray] = None
            self._original_motion_params: Optional[Dict[str, Any]] = None

            self.progress_var.set(30)
            
            # Load image if available
            try:
                if self.spd_reader.has_section("image"): # Check using reader's method
                    image_section = self.spd_reader.image
                    if image_section and image_section.data:
                        self.image = np.frombuffer(
                            image_section.data, 
                            dtype=np.uint8
                        ).reshape(
                            image_section.height, 
                            image_section.width, 
                            image_section.channels
                        )
                        # Ensure image is BGR if it's 3 channels, common for OpenCV
                        if self.image.ndim == 3 and self.image.shape[2] == 3 and image_section.format.upper() == "RGB":
                            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                        
                        self._original_image_data = self.image.copy()
                        self.dimensions_var.set(f"{image_section.width} × {image_section.height}")
                    else:
                        logger.info("Image section data is empty in SPD file.")
                else:
                    logger.info("Image section not present in SPD file.")
            except SPDSectionError:
                logger.info("Image section not found or error during loading.") # More generic catch
            except Exception as e:
                logger.error(f"Error loading image section: {e}", exc_info=True)
                messagebox.showwarning("Warning", f"Error loading image section: {e}")
            
            self.progress_var.set(50)
            
            # Load landmarks if available
            try:
                if self.spd_reader.has_section("landmarks"):
                    landmarks_section = self.spd_reader.landmarks
                    if landmarks_section and landmarks_section.points:
                        self.landmarks = landmarks_section.points
                        self._original_landmarks_data = [list(p) for p in self.landmarks]
                        self.landmark_count_var.set(f"{landmarks_section.count}")
                        self.landmark_type_var.set(landmarks_section.landmark_type)
                    else:
                        logger.info("Landmarks section data is empty in SPD file.")
                else:
                    logger.info("Landmarks section not present in SPD file.")
            except SPDSectionError:
                logger.info("Landmarks section not found or error during loading.")
            except Exception as e:
                logger.error(f"Error loading landmarks section: {e}", exc_info=True)
                messagebox.showwarning("Warning", f"Error loading landmarks section: {e}")
            
            self.progress_var.set(70)
            
            # Load motion parameters if available
            try:
                if self.spd_reader.has_section("motion_params"):
                    motion_params_section = self.spd_reader.motion_params
                    if motion_params_section and motion_params_section.param_names:
                        self.parameters = {
                            name: value
                            for name, value in zip(motion_params_section.param_names, motion_params_section.values)
                        }
                        self._original_motion_params = {
                            'param_names': list(motion_params_section.param_names),
                            'values': list(motion_params_section.values)
                        }
                    else:
                        logger.info("Motion parameters section data is empty in SPD file.")
                else:
                    logger.info("Motion parameters section not present in SPD file.")
            except SPDSectionError:
                logger.info("Motion parameters section not found or error during loading.")
            except Exception as e:
                logger.error(f"Error loading motion parameters section: {e}", exc_info=True)
                messagebox.showwarning("Warning", f"Error loading motion parameters section: {e}")
            
            # Load mask if available
            try:
                if self.spd_reader.has_section("mask"):
                    mask_section_obj = self.spd_reader.mask # Renamed to avoid conflict
                    if mask_section_obj and mask_section_obj.data:
                        # Assuming mask is single channel uint8 based on reader.py and format.py
                        # If MaskSection.format gives more info (e.g. "float32"), adjust dtype.
                        # For now, stick to uint8 as per previous logic.
                        self.mask = np.frombuffer(
                            mask_section_obj.data, 
                            dtype=np.uint8 
                        ).reshape(
                            mask_section_obj.height, 
                            mask_section_obj.width
                            # If mask_section_obj.format implied channels, add it here.
                        )
                        self._original_mask_data = self.mask.copy()
                    else:
                        logger.info("Mask section data is empty in SPD file.")
                else:
                    logger.info("Mask section not present in SPD file.")
            except SPDSectionError:
                logger.info("Mask section not found or error during loading.")
            except Exception as e:
                logger.error(f"Error loading mask section: {e}", exc_info=True)
                messagebox.showwarning("Warning", f"Error loading mask section: {e}")
            
            self.progress_var.set(90)
            
            self._create_parameter_sliders()
            self.update_display()
            
            self.progress_var.set(100)
            self.status_text.set(f"Loaded {os.path.basename(file_path)}")
            
        except SPDError as e:
            logger.error(f"SPD Error loading file {file_path}: {e}", exc_info=True)
            messagebox.showerror("SPD Error", f"Failed to load SPD file: {e}")
            self.status_text.set("Error loading SPD file")
            self.progress_var.set(0)
        except Exception as e:
            logger.error(f"Generic error loading file {file_path}: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load file: {e}")
            self.status_text.set("Error loading file")
            self.progress_var.set(0)

    def _prepare_image_section(self) -> Optional[np.ndarray]:
        if self.image is not None:
            return self.image
        return None

    def _prepare_landmarks_section(self) -> Optional[np.ndarray]:
        if self.landmarks is not None and len(self.landmarks) > 0:
            try:
                landmarks_array = np.array(self.landmarks, dtype=np.float32)
                if landmarks_array.ndim == 2:
                    return landmarks_array
                else:
                    logger.warning(f"Landmarks array is not 2D: shape {landmarks_array.shape}")
                    return None
            except Exception as e:
                logger.error(f"Error converting landmarks to NumPy array: {e}")
                return None
        return None

    def _prepare_mask_section(self) -> Optional[np.ndarray]:
        if self.mask is not None:
            if self.mask.ndim == 2:
                return self.mask
            elif self.mask.ndim == 3 and self.mask.shape[2] == 1:
                return self.mask.squeeze(axis=-1)
            else:
                logger.warning(f"Mask has unexpected shape for saving: {self.mask.shape}")
                return None
        return None

    def _prepare_motion_params_section(self) -> Optional[Dict[str, Any]]:
        if self.parameters:
            # SPDWriter now expects {'names': List[str], 'values': List[float]}
            param_names = list(self.parameters.keys())
            param_values = [float(self.parameters[name]) for name in param_names]
            
            if not param_names: # If parameters dict was empty
                return None

            return {
                "names": param_names,
                "values": param_values
            }
        return None

    def _perform_save_to_path(self, filepath: str) -> bool:
        self.status_text.set(f"Saving to {os.path.basename(filepath)}...")
        self.progress_var.set(0)
        
        spd_data_dict: Dict[str, Any] = {}

        try:
            prepared_image = self._prepare_image_section()
            if prepared_image is not None:
                spd_data_dict['original_image'] = prepared_image
                # Determine image format (assuming BGR if 3 channels, else grayscale or other)
                if prepared_image.ndim == 3 and prepared_image.shape[2] == 3:
                    spd_data_dict['original_image_format'] = "BGR" 
                elif prepared_image.ndim == 2 or (prepared_image.ndim == 3 and prepared_image.shape[2] == 1):
                    spd_data_dict['original_image_format'] = "GRAYSCALE"
                else:
                    spd_data_dict['original_image_format'] = "UNKNOWN" # Or handle more formats
            self.progress_var.set(10)

            prepared_landmarks = self._prepare_landmarks_section()
            if prepared_landmarks is not None:
                spd_data_dict['landmarks_original'] = prepared_landmarks
                spd_data_dict['landmarks_original_type'] = self.landmark_type_var.get() if self.landmark_type_var.get() else "unknown"
            self.progress_var.set(20)

            prepared_mask = self._prepare_mask_section()
            if prepared_mask is not None:
                spd_data_dict['mask_data'] = prepared_mask
                spd_data_dict['mask_data_format'] = "alpha" # Assuming mask is alpha
            self.progress_var.set(30)

            prepared_motion_params = self._prepare_motion_params_section()
            if prepared_motion_params is not None:
                spd_data_dict['motion_params'] = prepared_motion_params
            self.progress_var.set(40)

            if not spd_data_dict:
                messagebox.showerror("Save Error", "No data available to save.")
                self.status_text.set("Save cancelled - no data.")
                self.progress_var.set(0)
                return False
            
            self.progress_var.set(50)
            writer = SPDWriter(filepath=filepath, data=spd_data_dict)
            self.progress_var.set(60)
            writer.write()
            self.progress_var.set(90)
            
            self.progress_var.set(100)
            self.status_text.set(f"Successfully saved to {os.path.basename(filepath)}")
            logger.info(f"SPD file saved to {filepath}")
            
            if 'original_image' in spd_data_dict and self.image is not None: self._original_image_data = self.image.copy()
            if 'landmarks_original' in spd_data_dict and self.landmarks is not None: self._original_landmarks_data = [list(p) for p in self.landmarks]
            if 'mask_data' in spd_data_dict and self.mask is not None: self._original_mask_data = self.mask.copy()
            if 'motion_params' in spd_data_dict and self.parameters:
                 self._original_motion_params = {
                    'param_names': list(self.parameters.keys()),
                    'values': [float(self.parameters[name]) for name in self.parameters.keys()]
                }
            return True
            
        except Exception as e:
            logger.error(f"Error saving SPD file to {filepath}: {e}", exc_info=True)
            messagebox.showerror("Save Error", f"Failed to save file: {e}")
            self.status_text.set(f"Error saving file: {e}")
            self.progress_var.set(0)
            return False

    def save_file(self, event=None):
        """Save the current SPD file."""
        if not self.current_file:
            self.save_file_as()
            return
            
        logger.info(f"Saving current file: {self.current_file}")
        self._perform_save_to_path(self.current_file)
    
    def save_file_as(self):
        """Save the current SPD file with a new name."""
        default_filename = os.path.basename(self.current_file) if self.current_file else "untitled.spd"
        initial_dir = os.path.dirname(self.current_file) if self.current_file else "."

        file_path = filedialog.asksaveasfilename(
            title="Save SPD File As",
            initialfile=default_filename,
            initialdir=initial_dir,
            filetypes=[("SPD Files", "*.spd")],
            defaultextension=".spd"
        )
        
        if not file_path:
            self.status_text.set("Save As cancelled.")
            return
            
        logger.info(f"Saving file as: {file_path}")
        if self._perform_save_to_path(file_path):
            self.current_file = file_path
            self.file_path_var.set(os.path.basename(file_path))
            if self.spd_reader:
                 try:
                    self.spd_reader.close()
                 except Exception as e:
                    logger.warning(f"Could not close previous SPDReader: {e}")
    
    def create_new_spd(self, event=None):
        """Create a new SPD file from a source image."""
        image_path = filedialog.askopenfilename(
            title="Select Source Image",
            initialdir="/root/FasterLivePortrait/spd_editor/data",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*")
            ]
        )
        
        if not image_path:
            return
            
        output_path = filedialog.asksaveasfilename(
            title="Save SPD File",
            filetypes=[("SPD Files", "*.spd")],
            defaultextension=".spd"
        )
        
        if not output_path:
            return
            
        try:
            self.status_text.set("Creating SPD file...")
            self.progress_var.set(10)
            
            def create_spd_task():
                try:
                    img_cv2 = cv2.imread(image_path)
                    if img_cv2 is None:
                        raise ValueError(f"Failed to read source image: {image_path}")
                    
                    spd_data_for_writer: Dict[str, Any] = {}
                    spd_data_for_writer['original_image'] = img_cv2
                    # Add image format string
                    if img_cv2.ndim == 3 and img_cv2.shape[2] == 3:
                        spd_data_for_writer['original_image_format'] = "BGR"
                    elif img_cv2.ndim == 2 or (img_cv2.ndim == 3 and img_cv2.shape[2] == 1):
                         spd_data_for_writer['original_image_format'] = "GRAYSCALE"
                    else:
                        spd_data_for_writer['original_image_format'] = "UNKNOWN"
                    
                    self.root.after(0, lambda: self.status_text.set("Image data prepared..."))
                    self.root.after(0, lambda: self.progress_var.set(30))
                        
                    landmarks_added_to_dict = False
                    if HAVE_FACE_DETECTION and HAVE_CV2:
                        try:
                            self.root.after(0, lambda: self.status_text.set("Detecting face..."))
                            
                            current_predict_type = self.infer_params.get('predict_type', 'ort')
                            logger.info(f"Using predict_type: {current_predict_type} for FaceDetector")

                            face_detector = FaceDetector(predict_type=current_predict_type)
                            largest_face = face_detector.detect_largest_face(img_cv2)
                            
                            if largest_face and largest_face.get('landmark') is not None:
                                self.root.after(0, lambda: self.status_text.set("Landmarks obtained from FaceDetector."))
                                self.root.after(0, lambda: self.progress_var.set(60))
                                
                                landmarks_np_from_detector = largest_face['landmark']
                                    
                                if landmarks_np_from_detector is not None and landmarks_np_from_detector.size > 0:
                                    spd_data_for_writer['landmarks_original'] = landmarks_np_from_detector.astype(np.float32)
                                    # Add landmark type string
                                    spd_data_for_writer['landmarks_original_type'] = "face_detector_mediapipe" # Example, adjust if detector gives type
                                    self.root.after(0, lambda: self.status_text.set("Landmarks data prepared..."))
                                    self.root.after(0, lambda: self.progress_var.set(70))
                                    landmarks_added_to_dict = True
                                else:
                                    logger.info("FaceDetector returned landmarks, but they are empty or None.")
                                    self.root.after(0, lambda: self.status_text.set("Empty landmarks from FaceDetector."))
                            else:
                                logger.info("No face detected or landmarks not found by FaceDetector.")
                                self.root.after(0, lambda: self.status_text.set("No landmarks found by FaceDetector."))
                        except Exception as e:
                            logger.error(f"Error during face detection or landmark processing: {e}", exc_info=True)
                            self.root.after(0, lambda: self.status_text.set(f"Error in face/landmark processing: {e}"))
                    
                    if not landmarks_added_to_dict:
                        self.root.after(0, lambda: self.status_text.set("No landmarks detected or face detection unavailable"))
                    
                    writer_instance = SPDWriter(filepath=output_path, data=spd_data_for_writer)
                    writer_instance.write()
                    
                    self.root.after(0, lambda: self.status_text.set("SPD file created"))
                    self.root.after(0, lambda: self.progress_var.set(100))
                    
                    self.root.after(0, lambda: self.load_file(output_path))
                    
                except Exception as e:
                    logger.error(f"Error creating SPD file: {e}", exc_info=True)
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to create SPD file: {e}"))
                    self.root.after(0, lambda: self.status_text.set("Error creating SPD file"))
                    self.root.after(0, lambda: self.progress_var.set(0))
            
            thread = threading.Thread(target=create_spd_task)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            logger.error(f"Error starting SPD creation: {e}")
            messagebox.showerror("Error", f"Failed to start SPD creation: {e}")
            self.status_text.set("Error creating SPD file")
            self.progress_var.set(0)
    
    def export_image(self, event=None):
        """Export the image from the SPD file."""
        if not self.image is not None:
            messagebox.showerror("Error", "No image to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Image",
            filetypes=[
                ("JPEG Images", "*.jpg"),
                ("PNG Images", "*.png"),
                ("BMP Images", "*.bmp"),
                ("All Files", "*.*")
            ],
            defaultextension=".jpg"
        )
        
        if not file_path:
            return
            
        try:
            include_landmarks = messagebox.askyesno(
                "Include Landmarks", 
                "Include landmarks in the exported image?"
            )
            
            if include_landmarks and self.landmarks is not None:
                export_img = draw_landmarks(self.image, self.landmarks, self.visualization_options)
            else:
                export_img = self.image
            
            cv2.imwrite(file_path, export_img)
            self.status_text.set(f"Image exported to {os.path.basename(file_path)}")
            
        except Exception as e:
            logger.error(f"Error exporting image: {e}")
            messagebox.showerror("Error", f"Failed to export image: {e}")
    
    def export_landmarks(self):
        """Export landmarks from the SPD file."""
        if self.landmarks is None:
            messagebox.showerror("Error", "No landmarks to export")
            return
            
        format_options = ["CSV", "JSON"]
        format_var = tk.StringVar(value=format_options[0])
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Format")
        dialog.geometry("300x120")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select export format:").pack(pady=10)
        
        format_frame = ttk.Frame(dialog)
        format_frame.pack(pady=5)
        
        for option in format_options:
            ttk.Radiobutton(
                format_frame, 
                text=option,
                variable=format_var,
                value=option
            ).pack(side=tk.LEFT, padx=10)
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10, fill=tk.X)
        
        def on_cancel():
            dialog.destroy()
            
        def on_ok():
            export_format = format_var.get().lower()
            dialog.destroy()
            
            file_ext = ".csv" if export_format == "csv" else ".json"
            file_path = filedialog.asksaveasfilename(
                title=f"Export Landmarks as {export_format.upper()}",
                filetypes=[(f"{export_format.upper()} Files", f"*{file_ext}")],
                defaultextension=file_ext
            )
            
            if not file_path:
                return
                
            try:
                if export_format == "csv":
                    import csv
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        for point in self.landmarks:
                            writer.writerow(point)
                else:
                    import json
                    with open(file_path, 'w') as f:
                        json.dump({
                            "landmark_type": self.landmark_type_var.get(),
                            "dimensions": len(self.landmarks[0]) if self.landmarks else 0,
                            "count": len(self.landmarks),
                            "landmarks": self.landmarks
                        }, f, indent=2)
                
                self.status_text.set(f"Landmarks exported to {os.path.basename(file_path)}")
                
            except Exception as e:
                logger.error(f"Error exporting landmarks: {e}")
                messagebox.showerror("Error", f"Failed to export landmarks: {e}")
        
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=5)
        
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
    
    def export_mask(self):
        """Export mask from the SPD file."""
        if self.mask is None:
            messagebox.showerror("Error", "No mask to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Mask",
            filetypes=[
                ("PNG Images", "*.png"),
                ("JPEG Images", "*.jpg"),
                ("BMP Images", "*.bmp"),
                ("All Files", "*.*")
            ],
            defaultextension=".png"
        )
        
        if not file_path:
            return
            
        try:
            cv2.imwrite(file_path, self.mask)
            self.status_text.set(f"Mask exported to {os.path.basename(file_path)}")
            
        except Exception as e:
            logger.error(f"Error exporting mask: {e}")
            messagebox.showerror("Error", f"Failed to export mask: {e}")
    
    def validate_spd(self):
        """Validate the current SPD file."""
        if not self.spd_reader:
            messagebox.showerror("Error", "No file loaded")
            return
            
        try:
            validator = SPDValidator(self.current_file)
            report = validator.validate(level=ValidationLevel.STANDARD)
            
            dialog = tk.Toplevel(self.root)
            dialog.title("SPD Validation Results")
            dialog.geometry("600x400")
            dialog.transient(self.root)
            dialog.grab_set()
            
            status_frame = ttk.Frame(dialog)
            status_frame.pack(fill=tk.X, padx=10, pady=10)
            
            status = "VALID" if report.is_valid else "INVALID"
            status_color = "green" if report.is_valid else "red"
            status_label = ttk.Label(
                status_frame, 
                text=f"Status: {status}",
                foreground=status_color,
                font=("TkDefaultFont", 12, "bold")
            )
            status_label.pack(side=tk.LEFT)
            
            info_frame = ttk.Frame(dialog)
            info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            ttk.Label(info_frame, text=f"Validation Level: {report.validation_level.name}").pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"Sections Checked: {', '.join(report.sections_checked)}").pack(anchor=tk.W)
            
            if report.sections_invalid:
                ttk.Label(info_frame, 
                         text=f"Invalid Sections: {', '.join(report.sections_invalid)}", 
                         foreground="red").pack(anchor=tk.W)
            
            ttk.Label(info_frame, 
                     text=f"Issues: {report.get_error_count()} errors, " +
                           f"{report.get_warning_count()} warnings, " +
                           f"{report.get_info_count()} info").pack(anchor=tk.W)
            
            if report.issues:
                issues_frame = ttk.LabelFrame(dialog, text="Issues")
                issues_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
                
                issues_text = tk.Text(issues_frame, wrap=tk.WORD, height=10)
                issues_scroll = ttk.Scrollbar(issues_frame, orient=tk.VERTICAL, command=issues_text.yview)
                issues_text.configure(yscrollcommand=issues_scroll.set)
                
                issues_scroll.pack(side=tk.RIGHT, fill=tk.Y)
                issues_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                
                issues_text.tag_configure("ERROR", foreground="red")
                issues_text.tag_configure("WARNING", foreground="orange")
                issues_text.tag_configure("INFO", foreground="blue")
                
                for i, issue in enumerate(report.issues, 1):
                    severity = issue.severity.name
                    issues_text.insert(tk.END, f"{i}. [{severity}] {issue.section}: {issue.message}\n", severity)
                    if issue.suggestion:
                        issues_text.insert(tk.END, f"   Suggestion: {issue.suggestion}\n")
                    issues_text.insert(tk.END, "\n")
                
                issues_text.configure(state=tk.DISABLED)
            
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
            
            dialog.update_idletasks()
            x = self.root.winfo_rootx() + (self.root.winfo_width() - dialog.winfo_width()) // 2
            y = self.root.winfo_rooty() + (self.root.winfo_height() - dialog.winfo_height()) // 2
            dialog.geometry(f"+{x}+{y}")
            
        except Exception as e:
            logger.error(f"Error validating SPD file: {e}")
            messagebox.showerror("Error", f"Failed to validate SPD file: {e}")
    
    def show_preferences(self):
        """Show preferences dialog."""
        messagebox.showinfo("Not Implemented", "Preferences dialog will be implemented in a future update")
    
    def show_3d_visualization(self):
        """Show 3D visualization of landmarks."""
        if not HAVE_PLT:
            messagebox.showerror("Error", "Matplotlib is required for 3D visualization")
            return
            
        if not self.landmarks or not self.landmarks[0] or len(self.landmarks[0]) < 3:
            messagebox.showinfo("3D Visualization", 
                              "3D visualization requires 3D landmarks (current landmarks are 2D).")
            return
            
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            viz_window = tk.Toplevel(self.root)
            viz_window.title("3D Landmark Visualization")
            viz_window.geometry("800x600")
            
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            x = [p[0] for p in self.landmarks]
            y = [p[1] for p in self.landmarks]
            z = [p[2] for p in self.landmarks]
            
            ax.scatter(x, y, z, c='g', marker='o', s=20)
            
            landmark_type = None
            num_landmarks = len(self.landmarks)
            if num_landmarks == 68:
                landmark_type = 'dlib68'
            
            if landmark_type in self.visualization_options.connections:
                connections = self.visualization_options.connections[landmark_type]
                for start_idx, end_idx in connections:
                    if start_idx < len(self.landmarks) and end_idx < len(self.landmarks):
                        ax.plot([x[start_idx], x[end_idx]],
                               [y[start_idx], y[end_idx]],
                               [z[start_idx], z[end_idx]], 'b-', linewidth=0.5)
            
            ax.set_title("3D Face Mesh")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_box_aspect([1, 1, 1])
            
            toolbar_frame = ttk.Frame(viz_window)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            slider_frame = ttk.Frame(toolbar_frame)
            slider_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
            
            ttk.Label(slider_frame, text="Elevation:").grid(row=0, column=0, padx=5, sticky=tk.W)
            elev_var = tk.IntVar(value=30)
            elev_slider = ttk.Scale(
                slider_frame,
                from_=0,
                to=180,
                orient=tk.HORIZONTAL,
                variable=elev_var
            )
            elev_slider.grid(row=0, column=1, padx=5, sticky=tk.EW)
            
            ttk.Label(slider_frame, text="Azimuth:").grid(row=1, column=0, padx=5, sticky=tk.W)
            azim_var = tk.IntVar(value=45)
            azim_slider = ttk.Scale(
                slider_frame,
                from_=0,
                to=360,
                orient=tk.HORIZONTAL,
                variable=azim_var
            )
            azim_slider.grid(row=1, column=1, padx=5, sticky=tk.EW)
            
            def update_view(*args):
                ax.view_init(elev=elev_var.get(), azim=azim_var.get())
                canvas.draw()
            
            elev_slider.configure(command=lambda *args: update_view())
            azim_slider.configure(command=lambda *args: update_view())
            
            canvas = FigureCanvasTkAgg(fig, master=viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            update_view()
            
        except Exception as e:
            logger.error(f"Error creating 3D visualization: {e}")
            messagebox.showerror("Error", f"Failed to create 3D visualization: {e}")
    
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About SPD Editor",
            "SPD Editor v0.1.0\n\n"
            "A tool for creating, editing, and visualizing SPD (Source Portrait Descriptor) files "
            "for the FasterLivePortrait system.\n\n"
            "© 2025 FasterLivePortrait Team"
        )
    
    def show_documentation(self):
        """Show documentation dialog."""
        doc_window = tk.Toplevel(self.root)
        doc_window.title("SPD Editor Documentation")
        doc_window.geometry("800x600")
        
        notebook = ttk.Notebook(doc_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Overview")
        
        overview_text = tk.Text(overview_frame, wrap=tk.WORD, padx=10, pady=10)
        overview_scroll = ttk.Scrollbar(overview_frame, orient=tk.VERTICAL, command=overview_text.yview)
        overview_text.configure(yscrollcommand=overview_scroll.set)
        
        overview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        overview_text.pack(fill=tk.BOTH, expand=True)
        
        overview_text.insert(tk.END, """\
# SPD Editor

## Overview
The Source Portrait Descriptor (SPD) Editor is a tool for the FasterLivePortrait system that helps create, manage, and manipulate SPD files.

## What is an SPD file?
An SPD (Source Portrait Descriptor) is a binary file format that stores pre-processed facial data to eliminate the need for face detection during animation. This speeds up the animation process significantly by having all the necessary facial landmark information readily available.

## Features
- Create SPD files from images or videos
- Inspect and validate existing SPD files
- Modify SPD file parameters
- Extract facial features from SPD files
- Visualize SPD data
""")
        
        overview_text.configure(state=tk.DISABLED)
        
        usage_frame = ttk.Frame(notebook)
        notebook.add(usage_frame, text="Usage")
        
        usage_text = tk.Text(usage_frame, wrap=tk.WORD, padx=10, pady=10)
        usage_scroll = ttk.Scrollbar(usage_frame, orient=tk.VERTICAL, command=usage_text.yview)
        usage_text.configure(yscrollcommand=usage_scroll.set)
        
        usage_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        usage_text.pack(fill=tk.BOTH, expand=True)
        
        usage_text.insert(tk.END, """\
# Usage Guide

## Opening SPD Files
1. Click on File > Open or press Ctrl+O
2. Select an SPD file to open

## Creating New SPD Files
1. Click on File > New SPD File or press Ctrl+N
2. Select a source image
3. Choose the output file location

## Visualizing Landmarks
1. Open an SPD file with landmarks
2. Use the checkboxes in the Visualization Options panel to show/hide landmarks and labels
3. Adjust point size and connection thickness as needed

## Exporting Data
1. Open an SPD file
2. Use the File > Export menu to export images, landmarks, or masks

## Validating SPD Files
1. Open an SPD file
2. Click on Edit > Validate SPD File
3. Review the validation report
""")
        
        usage_text.configure(state=tk.DISABLED)
        
        shortcuts_frame = ttk.Frame(notebook)
        notebook.add(shortcuts_frame, text="Shortcuts")
        
        shortcuts_text = tk.Text(shortcuts_frame, wrap=tk.WORD, padx=10, pady=10)
        shortcuts_scroll = ttk.Scrollbar(shortcuts_frame, orient=tk.VERTICAL, command=shortcuts_text.yview)
        shortcuts_text.configure(yscrollcommand=shortcuts_scroll.set)
        
        shortcuts_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        shortcuts_text.pack(fill=tk.BOTH, expand=True)
        
        shortcuts_text.insert(tk.END, """\
# Keyboard Shortcuts

- Ctrl+N: Create a new SPD file
- Ctrl+O: Open an existing SPD file
- Ctrl+S: Save the current SPD file
- Ctrl+E: Export the image
""")
        
        shortcuts_text.configure(state=tk.DISABLED)
        
        close_button = ttk.Button(doc_window, text="Close", command=doc_window.destroy)
        close_button.pack(pady=10)
        
        doc_window.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - doc_window.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - doc_window.winfo_height()) // 2
        doc_window.geometry(f"+{x}+{y}")


def main():
    """Main entry point for the GUI."""
    if not TKINTER_AVAILABLE:
        logger.error("Tkinter is not available. The GUI cannot be started.")
        sys.exit(1)
    root = tk.Tk()
    app = SPDEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()