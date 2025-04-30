#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FLPSF Editor - FasterLivePortrait Source Format Editor
Main application entry point
"""

import sys
import os
import argparse
import logging
from PyQt5.QtWidgets import QApplication

# Add parent directory to path to allow importing FasterLivePortrait modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from views.main_window import MainWindow
from controllers.app_controller import AppController
from models.app_model import AppModel
from utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('flpsf_editor')


def parse_args():
    parser = argparse.ArgumentParser(description='FLPSF Editor - FasterLivePortrait Source Format Editor')
    parser.add_argument('--file', type=str, help='Path to FLPSF file to open')
    parser.add_argument('--image', type=str, help='Path to source image')
    parser.add_argument('--config', type=str, default='config/editor_config.yaml',
                        help='Path to editor configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = Config(config_path)
    
    # Initialize application
    app = QApplication(sys.argv)
    app.setApplicationName("FLPSF Editor")
    app.setOrganizationName("FasterLivePortrait")
    
    # Create MVC components
    model = AppModel(config)
    controller = AppController(model)
    view = MainWindow(controller)
    
    # Connect model to view
    model.register_view(view)
    
    # Load file if specified
    if args.file and os.path.exists(args.file):
        controller.open_file(args.file)
    elif args.image and os.path.exists(args.image):
        controller.load_image(args.image)
    
    # Show main window
    view.show()
    
    # Start event loop
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())