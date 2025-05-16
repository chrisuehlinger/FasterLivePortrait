#!/usr/bin/env python3
"""
Utility script to display information about an FSP (FasterLivePortrait Source) file.
"""
import argparse
import pickle
import os
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Display information about an FSP file")
    parser.add_argument("file", help="Path to the FSP file")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed information")
    args = parser.parse_args()
    
    try:
        with open(args.file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading FSP file: {e}")
        return 1
    
    # Display basic info
    print(f"File: {args.file}")
    print(f"Version: {data.get('version', 'Not specified')}")
    print(f"Source is video: {data.get('is_source_video', False)}")
    print(f"Original source path: {data.get('source_path', 'Not specified')}")
    
    # Images info
    if 'src_imgs' in data:
        imgs = data['src_imgs']
        print(f"Number of source images: {len(imgs)}")
        for i, img in enumerate(imgs):
            print(f"  Image {i}: shape={img.shape}, dtype={img.dtype}, range=({img.min()}, {img.max()})")
    else:
        print("No source images found")
    
    # Face info
    if 'src_infos' in data:
        infos = data['src_infos']
        print(f"Number of frames with info: {len(infos)}")
        for i, frame_info in enumerate(infos):
            print(f"  Frame {i}: {len(frame_info)} faces")
            
            if args.detailed:
                for j, face_info in enumerate(frame_info):
                    print(f"    Face {j}:")
                    # Display common fields
                    if len(face_info) >= 10:
                        print(f"      Motion info: {type(face_info[0])}")
                        print(f"      Landmarks: shape={face_info[1].shape if isinstance(face_info[1], np.ndarray) else 'N/A'}")
                        print(f"      Rotation matrix: shape={face_info[2].shape if isinstance(face_info[2], np.ndarray) else 'N/A'}")
                        print(f"      Feature vector: shape={face_info[3].shape if isinstance(face_info[3], np.ndarray) else 'N/A'}")
                        print(f"      Transformed keypoints: shape={face_info[4].shape if isinstance(face_info[4], np.ndarray) else 'N/A'}")
                        print(f"      Canonical keypoints: shape={face_info[5].shape if isinstance(face_info[5], np.ndarray) else 'N/A'}")
                        print(f"      Lip delta before animation: {type(face_info[6])}")
                        print(f"      Flag lip zero: {face_info[7]}")
                        print(f"      Mask: {type(face_info[8])}")
                        print(f"      Transform: {type(face_info[9])}")
    else:
        print("No source info found")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
