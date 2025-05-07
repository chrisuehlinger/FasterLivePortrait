#!/usr/bin/env python3
"""
Main entry point for SPD Editor.
"""
import argparse
import sys
from typing import List, Optional
import numpy as np
import cv2 # For creating dummy image data
from spd_editor.spd import SPDWriter, DEFAULT_NUM_LANDMARKS, RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT

from spd_editor import __version__
from spd_editor.cli import main as cli_main


def create_dummy_spd_data():
    """Creates a dictionary with dummy data for testing SPDWriter."""
    data = {}

    # Dummy Images (RGB, HWC, uint8)
    data['original_image'] = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    data['cropped_image'] = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    data['resized_image_256'] = np.random.randint(0, 256, (RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, 3), dtype=np.uint8)

    # Dummy Landmarks (Nx2, float32)
    num_lmk = DEFAULT_NUM_LANDMARKS
    data['landmarks_original'] = np.random.rand(num_lmk, 2).astype(np.float32) * np.array([640, 480])
    data['landmarks_cropped'] = np.random.rand(num_lmk, 2).astype(np.float32) * np.array([300, 300])
    data['landmarks_resized'] = np.random.rand(num_lmk, 2).astype(np.float32) * np.array([RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT])
    
    # Dummy Motion Parameters
    data['motion_params'] = {
        'pitch': 0.1, 'yaw': -0.05, 'roll': 0.02,
        'translation': np.array([1.0, 2.0, -50.0], dtype=np.float32),
        'expression': np.random.rand(50).astype(np.float32), # 50 expression parameters
        'scale': 1.05,
        'keypoints_3d': np.random.rand(num_lmk, 3).astype(np.float32) # 3D keypoints
    }

    # Dummy Appearance Features
    data['appearance_features'] = np.random.rand(256).astype(np.float32) # 256-dim feature vector

    # Dummy Transformation Matrices (3x3, float32)
    data['transform_matrices'] = {
        'M_c2o': np.eye(3, dtype=np.float32) * 1.1,
        'M_o2c': np.linalg.inv(np.eye(3, dtype=np.float32) * 1.1).astype(np.float32)
    }
    
    # Dummy Mask Data (HxW, float32 alpha)
    data['mask_data'] = np.random.rand(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH).astype(np.float32)

    # Dummy Additional Flags
    data['additional_flags'] = {
        'lips_normalized': True,
        'eye_ratio': 0.8,
        'lip_ratio': 0.5,
        'lip_delta_data': np.random.rand(10).astype(np.float32) # 10 lip delta values
    }
    return data


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
    print("SPD Editor Main - Example Usage")

    # This is where your CLI or GUI would gather data.
    # For now, we use dummy data.
    spd_content = create_dummy_spd_data()
    
    output_filepath = "dummy_test.spd"

    # --- How the GUI/CLI would use SPDWriter ---
    # 1. Collect all necessary data into a dictionary (like spd_content).
    #    This data would come from image loading, face analysis, user edits, etc.
    #
    # 2. Instantiate SPDWriter with the target filepath and the data.
    #    writer = SPDWriter(filepath="output.spd", data=collected_data_dict)
    #
    # 3. Call the write method.
    #    writer.write()
    #
    # Example:
    print(f"Attempting to write dummy SPD data to {output_filepath}...")
    try:
        writer = SPDWriter(filepath=output_filepath, data=spd_content)
        writer.write()
        print(f"Successfully wrote SPD file: {output_filepath}")
        print("You would now need an SPDReader to verify its contents.")
    except Exception as e:
        print(f"Error writing SPD file: {e}")
        import traceback
        traceback.print_exc()

    # --- Example of how the "Create SPD from image" CLI command might work ---
    # (Conceptual - requires face_detector.py, landmark_extractor.py etc.)
    #
    # if command == "create":
    #   source_image_path = args.source_image
    #   output_spd_path = args.output_spd
    #
    #   # 1. Load source image
    #   img_original_bgr = cv2.imread(source_image_path)
    #   img_original_rgb = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)
    #
    #   # 2. Perform face analysis (using components from spd_editor/analysis/)
    #   #    face_data = FaceDetector().detect(img_original_rgb)
    #   #    landmarks_original = LandmarkExtractor().extract(img_original_rgb, face_data['bbox'])
    #   #    # ... crop image, get resized image, get landmarks in those spaces ...
    #   #    # ... extract motion params, appearance features, mask, etc. ...
    #
    #   # 3. Populate the data dictionary
    #   #    current_spd_data = {
    #   #        'original_image': img_original_rgb,
    #   #        'resized_image_256': processed_img_256_rgb,
    #   #        'landmarks_original': landmarks_original_np,
    #   #        'landmarks_resized': landmarks_resized_np,
    #   #        # ... other fields ...
    #   #    }
    #
    #   # 4. Write SPD file
    #   #    writer = SPDWriter(output_spd_path, current_spd_data)
    #   #    writer.write()
    sys.exit(main())