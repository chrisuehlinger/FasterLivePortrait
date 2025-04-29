# -*- coding: utf-8 -*-
# @Project : FasterLivePortrait
# @FileName: custom_landmark_demo.py

"""
Demo script for using FasterLivePortrait with custom face landmarks
when automatic face detection fails.

Usage:
    python custom_landmark_demo.py --src_image path/to/source_image.jpg --dri_video path/to/driving_video.mp4

Optional arguments:
    --cfg: Path to config file (default: configs/onnx_infer.yaml)
    --realtime: Enable real-time processing
    --paste_back: Enable paste-back feature
    --landmarks_path: Path to pre-saved landmarks file (default: None, will create example landmarks)
    --no_crop: Disable source image cropping
"""

import os
import sys
import argparse
import numpy as np
import cv2
import pickle
import time
import datetime
import logging
from omegaconf import OmegaConf

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import video_has_audio, resize_to_limit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("custom_landmark_demo")

def load_or_create_landmarks(image_path, landmarks_path=None):
    """
    Either load landmarks from a file or create example landmarks for demonstration.
    
    Args:
        image_path: Path to the source image
        landmarks_path: Path to a landmarks file (.npy or .pkl) or None
        
    Returns:
        numpy.ndarray: Face landmarks array with shape (N, 2)
    """
    if landmarks_path and os.path.exists(landmarks_path):
        # Load landmarks from file
        try:
            if landmarks_path.endswith('.npy'):
                landmarks = np.load(landmarks_path)
                logger.info(f"Loaded landmarks from {landmarks_path} with shape {landmarks.shape}")
                
                if len(landmarks.shape) != 2 or landmarks.shape[1] != 2:
                    logger.warning(f"⚠️ Invalid landmarks format! Expected shape (N,2), got {landmarks.shape}")
                    if len(landmarks.shape) > 1 and landmarks.shape[1] > 2:
                        # Try to fix: use just x,y coordinates if there are more
                        landmarks = landmarks[:, :2]
                        logger.info(f"✓ Fixed landmarks: using first two columns as x,y coordinates. New shape: {landmarks.shape}")
            elif landmarks_path.endswith('.pkl'):
                with open(landmarks_path, 'rb') as f:
                    landmarks_data = pickle.load(f)
                
                # Handle different pickle formats
                if isinstance(landmarks_data, dict) and 'lmk_crop_256x256' in landmarks_data:
                    landmarks = landmarks_data['lmk_crop_256x256']
                    logger.info(f"Found lmk_crop_256x256 in pickle data with shape {landmarks.shape}")
                elif isinstance(landmarks_data, dict) and 'lmk_crop' in landmarks_data:
                    landmarks = landmarks_data['lmk_crop'] 
                    logger.info(f"Found lmk_crop in pickle data with shape {landmarks.shape}")
                elif isinstance(landmarks_data, np.ndarray):
                    landmarks = landmarks_data
                    logger.info(f"Loaded numpy array from pickle with shape {landmarks.shape}")
                else:
                    logger.warning("Cannot determine landmarks format in pickle file")
                    raise ValueError(f"Unsupported landmarks format in {landmarks_path}")
            
            # Debug: print some landmark points to verify format
            logger.info(f"Sample landmark points (first 3): {landmarks[:3]}")
            
            return landmarks
        except Exception as e:
            logger.error(f"Error loading landmarks from {landmarks_path}: {str(e)}")
            logger.info("Falling back to generating example landmarks...")
    
    # Create example landmarks
    # We'll load the image and create a basic facial landmark approximation
    # based on face proportions (not accurate but demonstrates the API)
    logger.info("Creating example landmarks (for demonstration purposes only)")
    
    # Load image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = img.shape[:2]
    img = resize_to_limit(img, 512, 1)
    h, w = img.shape[:2]
    
    # Create a simple face landmark approximation
    # These are very basic landmarks that may not work well but demonstrate the API
    face_center_x = w // 2
    face_center_y = h // 2
    face_width = w // 3
    face_height = h // 2
    
    # Create 106 landmarks (same number as used by insightface)
    # This is a very basic approximation of facial landmarks
    landmarks = []
    
    # Create face outline (about 33 points)
    for i in range(17):
        angle = np.pi - (np.pi * i / 16)
        x = int(face_center_x - np.cos(angle) * face_width / 2)
        y = int(face_center_y - np.sin(angle) * face_height / 2)
        landmarks.append([x, y])
    
    for i in range(16):
        angle = 0 + (np.pi * i / 16)
        x = int(face_center_x + np.cos(angle) * face_width / 2)
        y = int(face_center_y - np.sin(angle) * face_height / 2)
        landmarks.append([x, y])
    
    # Create eyebrows (about 8 points)
    left_eyebrow_y = face_center_y - face_height // 4
    for i in range(5):
        x = int(face_center_x - face_width // 4 + (i * face_width // 10))
        y = int(left_eyebrow_y - abs((i - 2) * 3))
        landmarks.append([x, y])
    
    right_eyebrow_y = face_center_y - face_height // 4
    for i in range(5):
        x = int(face_center_x + (i * face_width // 10))
        y = int(right_eyebrow_y - abs((i - 2) * 3))
        landmarks.append([x, y])
    
    # Create eyes (about 12 points)
    left_eye_y = face_center_y - face_height // 8
    left_eye_x = face_center_x - face_width // 4
    eye_width = face_width // 6
    eye_height = face_height // 10
    for i in range(6):
        angle = 2 * np.pi * i / 6
        x = int(left_eye_x + np.cos(angle) * eye_width)
        y = int(left_eye_y + np.sin(angle) * eye_height)
        landmarks.append([x, y])
    
    right_eye_y = face_center_y - face_height // 8
    right_eye_x = face_center_x + face_width // 4
    for i in range(6):
        angle = 2 * np.pi * i / 6
        x = int(right_eye_x + np.cos(angle) * eye_width)
        y = int(right_eye_y + np.sin(angle) * eye_height)
        landmarks.append([x, y])
    
    # Create nose (about 9 points)
    nose_y = face_center_y
    nose_x = face_center_x
    for i in range(9):
        x = int(nose_x + ((i % 3) - 1) * face_width // 8)
        y = int(nose_y - face_height // 8 + (i // 3) * face_height // 10)
        landmarks.append([x, y])
    
    # Create mouth (about 20 points)
    mouth_y = face_center_y + face_height // 4
    mouth_width = face_width // 3
    mouth_height = face_height // 8
    
    # Outer lip
    for i in range(12):
        angle = 2 * np.pi * i / 12
        x = int(face_center_x + np.cos(angle) * mouth_width)
        y = int(mouth_y + np.sin(angle) * mouth_height)
        landmarks.append([x, y])
        
    # Inner lip (slightly smaller)
    for i in range(8):
        angle = 2 * np.pi * i / 8
        x = int(face_center_x + np.cos(angle) * mouth_width * 0.7)
        y = int(mouth_y + np.sin(angle) * mouth_height * 0.7)
        landmarks.append([x, y])
    
    # Ensure we have 106 points (the expected number) by adding any needed extra points
    while len(landmarks) < 106:
        # Add some random points inside the face area
        x = np.random.randint(face_center_x - face_width//2, face_center_x + face_width//2)
        y = np.random.randint(face_center_y - face_height//2, face_center_y + face_height//2)
        landmarks.append([x, y])
    
    # Limit to exactly 106 points if we have more
    landmarks = landmarks[:106]
    
    # Convert to numpy array
    landmarks = np.array(landmarks, dtype=np.float32)
    
    # Save the example landmarks for future use
    output_dir = "custom_landmarks"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"example_landmarks_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npy")
    np.save(output_path, landmarks)
    logger.info(f"Saved example landmarks to {output_path}")
    
    return landmarks

def visualize_landmarks(image_path, landmarks):
    """
    Visualize landmarks on the image
    
    Args:
        image_path: Path to the image
        landmarks: Numpy array of landmarks with shape (N, 2)
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not load image: {image_path}")
        return
    
    # Resize image if too large
    img = resize_to_limit(img, 1024, 1)
    
    # Draw landmarks
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
        if i % 10 == 0:  # Display some landmark indices
            cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    # Display image
    cv2.imshow("Image with landmarks", img)
    logger.info("Displaying landmarks on image. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_with_video(args, custom_landmarks):
    """
    Run FasterLivePortrait with custom landmarks
    
    Args:
        args: Command line arguments
        custom_landmarks: Custom face landmarks
    """
    logger.info(f"Running with video: {args.dri_video}")
    logger.info(f"Original custom landmarks shape: {custom_landmarks.shape}")
    logger.info(f"First 3 points of custom landmarks: {custom_landmarks[:3]}")
    
    # Load configuration
    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = args.paste_back
    
    # Set flag_do_crop based on the no_crop argument (inverted logic)
    flag_do_crop = not args.no_crop
    logger.info(f"Source image cropping: {'disabled' if args.no_crop else 'enabled'}")
    
    # Initialize pipeline
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=False)
    
    # Prepare source with custom landmarks
    logger.info("Preparing source with custom landmarks")
    success = pipe.prepare_source_with_custom_landmarks(args.src_image, custom_landmarks, 
                                                       realtime=args.realtime, 
                                                       flag_do_crop=flag_do_crop)
    
    if not success:
        logger.error("Failed to prepare source with custom landmarks")
        return
    
    logger.info("Successfully prepared source with custom landmarks")
    
    # Check if landmarks were properly stored
    if len(pipe.src_infos) > 0 and len(pipe.src_infos[0]) > 0 and len(pipe.src_infos[0][0]) > 1:
        stored_landmarks = pipe.src_infos[0][0][1]
        logger.info(f"Pipeline stored landmarks shape: {stored_landmarks.shape}")
        logger.info(f"First 3 points of stored landmarks: {stored_landmarks[:3]}")
        
        # Verify if landmarks match
        if np.array_equal(stored_landmarks[:3], custom_landmarks[:3]):
            logger.info("✅ Landmarks match: Pipeline is using the provided landmarks correctly")
        else:
            logger.info("❌ Landmarks don't match: Pipeline might be modifying or not using the provided landmarks")
    else:
        logger.warning("Cannot verify landmarks: src_infos structure is unexpected")
    
    # Set up video capture
    if not args.dri_video or not os.path.exists(args.dri_video):
        # Read frames from camera if no driving video input
        vcap = cv2.VideoCapture(0)
        if not vcap.isOpened():
            logger.error("No camera found! Exiting.")
            return
    else:
        vcap = cv2.VideoCapture(args.dri_video)
    
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    h, w = pipe.src_imgs[0].shape[:2]
    
    # Create output directory
    save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}_custom_landmarks"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the landmarks that were actually used
    if len(pipe.src_infos) > 0 and len(pipe.src_infos[0]) > 0 and len(pipe.src_infos[0][0]) > 1:
        used_landmarks_path = os.path.join(save_dir, "used_landmarks.npy")
        np.save(used_landmarks_path, pipe.src_infos[0][0][1])
        logger.info(f"Saved landmarks actually used by pipeline to {used_landmarks_path}")
    
    # Set up video writer if not in realtime mode
    if not args.realtime:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vsave_crop_path = os.path.join(save_dir, f"custom_landmarks_result_crop.mp4")
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512 * 2, 512))
        vsave_org_path = os.path.join(save_dir, f"custom_landmarks_result_org.mp4")
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))
    
    # Process video
    frame_ind = 0
    infer_times = []
    
    logger.info("Processing frames...")
    while vcap.isOpened():
        start_time = time.time()
        
        # Read frame
        ret, frame = vcap.read()
        if not ret:
            break
        
        # Process frame
        first_frame = frame_ind == 0
        dri_crop, out_crop, out_org, dri_motion_info = pipe.run(
            frame, pipe.src_imgs[0], pipe.src_infos[0], first_frame=first_frame
        )
        
        frame_ind += 1
        process_time = time.time() - start_time
        infer_times.append(process_time)
        
        if out_crop is None:
            logger.warning(f"No face detected in driving frame: {frame_ind}")
            continue
        
        # Prepare output
        dri_crop = cv2.resize(dri_crop, (512, 512))
        out_crop = np.concatenate([dri_crop, out_crop], axis=1)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        
        # Display or save output
        if not args.realtime:
            vout_crop.write(out_crop)
            if out_org is not None:
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                vout_org.write(out_org)
        else:
            if infer_cfg.infer_params.flag_pasteback and out_org is not None:
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                cv2.imshow('Custom Landmarks Animation', out_org)
            else:
                cv2.imshow('Custom Landmarks Animation', out_crop)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Display FPS every 10 frames
        if frame_ind % 10 == 0:
            fps = 1.0 / (sum(infer_times[-10:]) / 10)
            logger.info(f"Frame: {frame_ind}, FPS: {fps:.2f}")
    
    # Clean up
    vcap.release()
    if not args.realtime:
        vout_crop.release()
        if vout_org is not None:
            vout_org.release()
        logger.info(f"Results saved to {save_dir}")
    else:
        cv2.destroyAllWindows()
    
    # Display performance statistics
    logger.info(f"Processed {frame_ind} frames")
    logger.info(f"Inference time: {np.mean(infer_times) * 1000:.2f} ms/frame (mean), "
                f"{np.median(infer_times) * 1000:.2f} ms/frame (median)")

def main():
    parser = argparse.ArgumentParser(description='FasterLivePortrait with Custom Landmarks Demo')
    parser.add_argument('--src_image', required=False, type=str, default="assets/examples/source/s12.jpg",
                        help='Source image path')
    parser.add_argument('--dri_video', required=False, type=str, default="assets/examples/driving/d14.mp4",
                        help='Driving video path')
    parser.add_argument('--cfg', required=False, type=str, default="configs/onnx_infer.yaml",
                        help='Inference config path')
    parser.add_argument('--realtime', action='store_true', help='Enable realtime inference')
    parser.add_argument('--paste_back', action='store_true', default=False,
                        help='Paste back to original image')
    parser.add_argument('--landmarks_path', type=str, default=None,
                        help='Path to custom landmarks file (.npy or .pkl)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Visualize landmarks before processing')
    parser.add_argument('--no_crop', action='store_true', default=False,
                        help='Disable source image cropping')
    args = parser.parse_args()
    
    # Check if source image exists
    if not os.path.exists(args.src_image):
        logger.error(f"Source image not found: {args.src_image}")
        return
    
    # Check if config file exists
    if not os.path.exists(args.cfg):
        logger.error(f"Config file not found: {args.cfg}")
        return
    
    # Load or create landmarks
    custom_landmarks = load_or_create_landmarks(args.src_image, args.landmarks_path)
    
    # Optionally visualize landmarks
    if args.visualize:
        visualize_landmarks(args.src_image, custom_landmarks)
    
    # Process with custom landmarks
    run_with_video(args, custom_landmarks)

if __name__ == '__main__':
    main()