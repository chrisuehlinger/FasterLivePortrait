#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
CLI for copying landmarks from one FSP file to another
"""
import argparse
import pickle
import copy
from typing import Dict, Any


def main():
    parser = argparse.ArgumentParser(description="Copy landmarks from one FSP file to another")
    parser.add_argument('--source', required=True, help='Path to source FSP file (landmarks will be copied from this)')
    parser.add_argument('--target', required=True, help='Path to target FSP file (landmarks will be copied to this)')
    parser.add_argument('--output', required=True, help='Path to save the resulting FSP file')
    args = parser.parse_args()

    # Load source FSP file (contains landmarks to be copied)
    try:
        with open(args.source, 'rb') as f:
            source_data = pickle.load(f)
        print(f"Loaded source FSP from {args.source}")
    except Exception as e:
        print(f"Failed to load source FSP file: {e}")
        return 1

    # Load target FSP file (will receive the landmarks)
    try:
        with open(args.target, 'rb') as f:
            target_data = pickle.load(f)
        print(f"Loaded target FSP from {args.target}")
    except Exception as e:
        print(f"Failed to load target FSP file: {e}")
        return 1

    # Verify FSP file version compatibility
    if source_data.get('version', 0) != target_data.get('version', 0):
        print(f"Warning: FSP file versions don't match. Source: {source_data.get('version')}, Target: {target_data.get('version')}")
    
    # Create a copy of the target data to modify
    result_data = copy.deepcopy(target_data)
    
    # Copy landmarks (src_infos) from source to target
    if 'src_infos' not in source_data:
        print("Error: Source FSP file doesn't contain landmarks (src_infos)")
        return 1
    
    # Check that the number of frames matches
    if len(source_data['src_infos']) != len(result_data['src_infos']):
        print(f"Warning: Number of frames doesn't match. Source: {len(source_data['src_infos'])}, Target: {len(result_data['src_infos'])}")
        print("Continuing anyway - this might cause issues if the landmarks don't align properly")
    
    # Copy the landmarks
    print(str(result_data['src_infos'][0][0][4].shape))
    result_data['src_infos'][0][0][1] = copy.deepcopy(source_data['src_infos'][0][0][1])
    result_data['src_infos'][0][0][4] = copy.deepcopy(source_data['src_infos'][0][0][4])
    result_data['src_infos'][0][0][5] = copy.deepcopy(source_data['src_infos'][0][0][5])
    print(f"Copied landmarks from {args.source} to target data")

    # Save the resulting data
    try:
        with open(args.output, 'wb') as f:
            pickle.dump(result_data, f)
        print(f"Saved FSP file with copied landmarks to {args.output}")
    except Exception as e:
        print(f"Failed to save output FSP file: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
