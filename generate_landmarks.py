# -*- coding: utf-8 -*-
"""
Generate and save custom landmarks from an image.
This script can be used to extract facial landmarks from images and save them for use with
FasterLivePortrait when automated landmark detection is not working properly.
"""

import os
import argparse
import numpy as np
import torch
import cv2
import pickle
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

def display_landmarks(img, landmarks, save_path=None):
    """Display image with landmarks for verification"""
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, c='r', marker='.')
    
    # Label landmark indices
    for i, (x, y) in enumerate(landmarks):
        plt.text(x, y, str(i), fontsize=8, color='white', 
                 bbox=dict(facecolor='red', alpha=0.5))
    
    plt.title("Facial Landmarks")
    if save_path:
        plt.savefig(save_path)
        print(f"Landmark visualization saved to {save_path}")
    plt.tight_layout()
    plt.show()

def extract_save_landmarks(args):
    """Extract landmarks and save to file"""
    # Load configuration
    infer_cfg = OmegaConf.load(args.cfg)
    
    # Initialize pipeline (without full initialization for speed)
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    
    # Read image
    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Extract landmarks
    if args.animal:
        print("Extracting animal landmarks...")
        from PIL import Image
        img_rgb_pil = Image.fromarray(img_rgb)
        with torch.no_grad():
            landmarks = pipe.model_dict["xpose"].run(
                img_rgb_pil,
                'face',
                'animal_face',
                0,
                0
            )
    else:
        print("Extracting human landmarks...")
        faces = pipe.model_dict["face_analysis"].predict(img_bgr)
        
        if len(faces) == 0:
            print("No faces detected in the image.")
            if args.force:
                print("Creating blank landmark template with 106 points (insightface format)")
                # Create a blank template with 106 landmarks positioned in face-like arrangement
                # This is just a placeholder for manual editing
                landmarks = np.zeros((106, 2), dtype=np.float32)
                h, w = img_bgr.shape[:2]
                # Position in center with default face proportions
                center_x, center_y = w // 2, h // 2
                for i in range(106):
                    landmarks[i] = [center_x + (i % 20), center_y + (i // 10)]
            else:
                return
        else:
            # Get the largest face by default
            face_idx = args.face_index if args.face_index is not None else 0
            if face_idx >= len(faces):
                print(f"Warning: Requested face index {face_idx} but only {len(faces)} faces found.")
                face_idx = 0
                
            landmarks = faces[face_idx]
            landmarks = pipe.model_dict["landmark"].predict(img_rgb, landmarks)
    
    if landmarks is None:
        print("Failed to extract landmarks.")
        return
    
    print(f"Extracted landmarks with shape: {landmarks.shape}")
    
    # Save landmarks
    if args.output.endswith('.npy'):
        np.save(args.output, landmarks)
    else:
        with open(args.output, 'wb') as f:
            pickle.dump(landmarks, f)
    
    print(f"Landmarks saved to {args.output}")
    
    # Visualize landmarks if requested
    if args.visualize:
        vis_path = args.output.rsplit('.', 1)[0] + '_visualization.png'
        display_landmarks(img_bgr, landmarks, vis_path)

def extract_with_manual_editing(args):
    """Extract landmarks and allow manual editing"""
    # This is a placeholder for future implementation of interactive landmark editing
    # For now we'll just call the regular extraction function
    print("Manual editing mode not fully implemented yet. Extracting landmarks normally.")
    extract_save_landmarks(args)
    print("\nTo manually edit landmarks:")
    print("1. Load the saved .npy file with np.load()")
    print("2. Modify the landmark coordinates as needed")
    print("3. Save with np.save() or pickle.dump()")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and save facial landmarks from an image')
    parser.add_argument('--image', required=True, type=str, help='Path to source image')
    parser.add_argument('--output', required=True, type=str, help='Path to save landmarks (.npy or .pkl)')
    parser.add_argument('--cfg', type=str, default="configs/onnx_infer.yaml", help='Inference config')
    parser.add_argument('--animal', action='store_true', help='Use animal model')
    parser.add_argument('--face-index', type=int, help='Index of face to use (default: 0, largest face)')
    parser.add_argument('--visualize', action='store_true', help='Visualize landmarks on image')
    parser.add_argument('--manual', action='store_true', help='Enable manual editing of landmarks')
    parser.add_argument('--force', action='store_true', help='Create template even if no face is detected')
    
    args = parser.parse_args()
    
    if args.manual:
        extract_with_manual_editing(args)
    else:
        extract_save_landmarks(args)