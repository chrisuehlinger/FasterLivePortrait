#!/usr/bin/env python3
"""
Main entry point for SPD Editor.
"""
import argparse
import sys
from typing import List, Optional

from spd_editor import __version__
from spd_editor.cli import main as cli_main


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for SPD Editor.
    
    Args:
        args: Command line arguments (defaults to sys.argv if None)
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    if args is None:
        args = sys.argv[1:]
    
    # If no arguments are provided, show version and help message
    if not args:
        print(f"SPD Editor v{__version__} - Source Portrait Descriptor Editor for FasterLivePortrait")
        print("Use --help to see available commands")
        print("Use 'gui' to launch the graphical interface")
        return 0
        
    # If --version is the only argument, show version
    if len(args) == 1 and args[0] in ["--version", "-v"]:
        print(f"SPD Editor v{__version__}")
        return 0
        
    # Check if GUI mode is requested
    if args[0] == "gui":
        try:
            from spd_editor.gui import main as gui_main
            return gui_main()
        except ImportError as e:
            print(f"Error: Could not launch GUI. {e}")
            print("Make sure you have the required dependencies installed:")
            print("- tkinter (usually comes with Python)")
            print("- Pillow (pip install pillow)")
            print("- numpy (pip install numpy)")
            print("- opencv-python (pip install opencv-python)")
            print("- matplotlib (optional, for 3D visualization)")
            return 1
        
    # Route all other commands to CLI
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())