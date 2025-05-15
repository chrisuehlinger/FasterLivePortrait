#!/usr/bin/env python
"""
SPD Editor - Command-Line Interface

This module provides a command-line interface for working with SPD files.
It allows users to create, inspect, validate, extract, convert, and visualize 
SPD (Source Portrait Descriptor) files.
"""

import os
import sys
import argparse
import csv
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, cast, TypeVar
from typing_extensions import TypedDict, Protocol

from spd_editor.cli_types import (
    ExitCode, CommandOptions, CreateOptions, InspectOptions,
    ExtractOptions, ConvertOptions, ValidateOptions, VisualizeOptions
)

# Import SPD modules
from spd_editor.spd.format import (
    HeaderFlags, ImageSection, LandmarksSection, 
    MotionParamsSection, AppearanceSection, 
    TransformationSection, MaskSection, 
    AdditionalFlagsSection, flags_to_dict
)
from spd_editor.spd.reader import SPDReader, SPDError, SPDFormatError, SPDSectionError
from spd_editor.spd.writer import SPDWriter
from spd_editor.spd.validator import (
    SPDValidator, ValidationLevel, ValidationSeverity, ValidationReport
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger("spd_editor.cli")

# Try to import additional visualization libraries
try:
    import numpy as np
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False


class ColorFormatter:
    """Utility for formatting terminal output with colors."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    
    @staticmethod
    def red(text: str) -> str:
        return f"{ColorFormatter.RED}{text}{ColorFormatter.RESET}"
    
    @staticmethod
    def green(text: str) -> str:
        return f"{ColorFormatter.GREEN}{text}{ColorFormatter.RESET}"
    
    @staticmethod
    def yellow(text: str) -> str:
        return f"{ColorFormatter.YELLOW}{text}{ColorFormatter.RESET}"
    
    @staticmethod
    def blue(text: str) -> str:
        return f"{ColorFormatter.BLUE}{text}{ColorFormatter.RESET}"
    
    @staticmethod
    def magenta(text: str) -> str:
        return f"{ColorFormatter.MAGENTA}{text}{ColorFormatter.RESET}"
    
    @staticmethod
    def cyan(text: str) -> str:
        return f"{ColorFormatter.CYAN}{text}{ColorFormatter.RESET}"
    
    @staticmethod
    def bold(text: str) -> str:
        return f"{ColorFormatter.BOLD}{text}{ColorFormatter.RESET}"
    
    @staticmethod
    def error(text: str) -> str:
        return f"{ColorFormatter.RED}{ColorFormatter.BOLD}ERROR: {text}{ColorFormatter.RESET}"
    
    @staticmethod
    def warning(text: str) -> str:
        return f"{ColorFormatter.YELLOW}WARNING: {text}{ColorFormatter.RESET}"
    
    @staticmethod
    def success(text: str) -> str:
        return f"{ColorFormatter.GREEN}{ColorFormatter.BOLD}SUCCESS: {text}{ColorFormatter.RESET}"
    
    @staticmethod
    def info(text: str) -> str:
        return f"{ColorFormatter.BLUE}INFO: {text}{ColorFormatter.RESET}"


def create_cmd(args: argparse.Namespace) -> int:
    """
    Create an SPD file from a source image.
    
    Args:
        args: Command-line arguments
            - source: Path to the source image
            - output: Path to the output SPD file
            - landmarks: Path to landmarks file (optional)
            - landmark_type: Type of landmarks (e.g., "mediapipe", "dlib68")
            - mask: Path to mask file (optional)
            - include_image: Whether to include the source image in the SPD file
    
    Returns:
        0 for success, non-zero for failure
    """
    start_time = time.time()
    
    try:
        if not HAVE_CV2:
            logger.error("OpenCV is required for creating SPD files.")
            logger.error("Install it with: pip install opencv-python")
            return 1
        
        # Read source image
        if not os.path.exists(args.source):
            logger.error(f"Source image not found: {args.source}")
            return 1
        
        logger.info(f"Reading source image: {args.source}")
        img = cv2.imread(args.source)
        if img is None:
            logger.error(f"Failed to read source image: {args.source}")
            return 1
        
        # Create SPD file
        with SPDWriter(args.output) as writer:
            # Add image section if requested
            if args.include_image:
                logger.info("Adding image section")
                height, width = img.shape[:2]
                channels = 3 if len(img.shape) == 3 else 1
                
                # Convert image to raw bytes
                img_bytes = img.tobytes()
                
                # Create and write image section
                img_section = ImageSection(
                    width=width,
                    height=height,
                    channels=channels,
                    format="BGR" if channels == 3 else "GRAY",
                    data=img_bytes
                )
                writer.write_image_section(img_section)
            
            # Add landmarks section if provided
            if args.landmarks:
                if not os.path.exists(args.landmarks):
                    logger.error(f"Landmarks file not found: {args.landmarks}")
                    return 1
                
                logger.info(f"Reading landmarks from: {args.landmarks}")
                landmarks_type = args.landmark_type or "unknown"
                landmarks_points = []
                
                # Read landmarks from file based on extension
                landmarks_path = Path(args.landmarks)
                if landmarks_path.suffix.lower() in ['.csv', '.txt']:
                    # Assume CSV format: x,y,z or x,y per line
                    with open(landmarks_path, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if not row or row[0].startswith('#'):
                                continue
                            try:
                                point = [float(val) for val in row]
                                landmarks_points.append(point)
                            except ValueError:
                                logger.error(f"Invalid landmark format in row: {row}")
                                return 1
                elif landmarks_path.suffix.lower() in ['.json', '.jsonl']:
                    # Assume JSON format
                    with open(landmarks_path, 'r') as f:
                        try:
                            data = json.load(f)
                            # Try to determine the structure
                            if isinstance(data, list):
                                if all(isinstance(item, list) for item in data):
                                    landmarks_points = data  # [[x,y,z], [x,y,z], ...]
                                else:
                                    # Try to find landmarks in the JSON structure
                                    logger.error("Unrecognized JSON landmarks format")
                                    return 1
                            elif isinstance(data, dict) and 'landmarks' in data:
                                landmarks_points = data['landmarks']
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON in landmarks file: {args.landmarks}")
                            return 1
                else:
                    logger.error(f"Unsupported landmarks file format: {landmarks_path.suffix}")
                    return 1
                
                # Determine dimensions from the first point
                if landmarks_points:
                    dimensions = len(landmarks_points[0])
                    
                    # Create and write landmarks section
                    landmarks_section = LandmarksSection(
                        count=len(landmarks_points),
                        dimensions=dimensions,
                        landmark_type=landmarks_type,
                        points=landmarks_points
                    )
                    
                    writer.write_landmarks_section(landmarks_section)
                    logger.info(f"Added {len(landmarks_points)} {dimensions}D landmarks of type '{landmarks_type}'")
                else:
                    logger.warning("No landmarks found in the file")
            
            # Add mask section if provided
            if args.mask:
                if not os.path.exists(args.mask):
                    logger.error(f"Mask file not found: {args.mask}")
                    return 1
                
                logger.info(f"Reading mask from: {args.mask}")
                mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    logger.error(f"Failed to read mask image: {args.mask}")
                    return 1
                
                # Create and write mask section
                mask_section = MaskSection(
                    width=mask.shape[1],
                    height=mask.shape[0],
                    format="grayscale",
                    data=mask.tobytes()
                )
                writer.write_mask_section(mask_section)
                logger.info(f"Added mask of size {mask.shape[1]}x{mask.shape[0]}")
            
            # Finalize SPD file
            writer.finalize()
        
        elapsed_time = time.time() - start_time
        logger.info(ColorFormatter.success(f"SPD file created: {args.output} ({elapsed_time:.2f} seconds)"))
        return 0
    
    except (SPDError, Exception) as e:
        logger.error(f"Failed to create SPD file: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


def info_cmd(args: argparse.Namespace) -> int:
    """
    Display information about an SPD file.
    
    Args:
        args: Command-line arguments
            - file: Path to the SPD file
            - json: Output in JSON format
            - verbose: Show detailed information
    
    Returns:
        0 for success, non-zero for failure
    """
    try:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return 1
        
        reader = SPDReader(args.file)
        
        # Get header information
        header = reader.header
        flags = flags_to_dict(header.flags)
        available_sections = reader.get_available_sections()
        
        # Prepare information dictionary
        info = {
            "file": args.file,
            "version": header.version,
            "timestamp": header.timestamp,
            "timestamp_readable": time.ctime(header.timestamp),
            "flags": {k: v for k, v in flags.items() if v},
            "sections": available_sections
        }
        
        # Add detailed section information if verbose
        if args.verbose:
            if "image" in available_sections:
                try:
                    image = reader.image
                    info["image"] = {
                        "width": image.width,
                        "height": image.height,
                        "channels": image.channels,
                        "format": image.format,
                        "data_size": len(image.data)
                    }
                except SPDSectionError as e:
                    info["image"] = {"error": str(e)}
            
            if "landmarks" in available_sections:
                try:
                    landmarks = reader.landmarks
                    info["landmarks"] = {
                        "count": landmarks.count,
                        "dimensions": landmarks.dimensions,
                        "type": landmarks.landmark_type
                    }
                    if args.very_verbose:
                        info["landmarks"]["points"] = landmarks.points
                except SPDSectionError as e:
                    info["landmarks"] = {"error": str(e)}
            
            if "motion_params" in available_sections:
                try:
                    motion_params = reader.motion_params
                    info["motion_params"] = {
                        "num_params": motion_params.num_params,
                        "param_names": motion_params.param_names
                    }
                    if args.very_verbose:
                        info["motion_params"]["values"] = motion_params.values
                except SPDSectionError as e:
                    info["motion_params"] = {"error": str(e)}
            
            if "appearance" in available_sections:
                try:
                    appearance = reader.appearance
                    info["appearance"] = {
                        "feature_dim": appearance.feature_dim,
                        "feature_type": appearance.feature_type
                    }
                    if args.very_verbose:
                        info["appearance"]["features"] = appearance.features
                except SPDSectionError as e:
                    info["appearance"] = {"error": str(e)}
            
            if "transforms" in available_sections:
                try:
                    transforms = reader.transforms
                    info["transforms"] = {
                        "num_transforms": transforms.num_transforms,
                        "transform_types": transforms.transform_types
                    }
                    if args.very_verbose:
                        info["transforms"]["matrices"] = transforms.matrices
                except SPDSectionError as e:
                    info["transforms"] = {"error": str(e)}
            
            if "mask" in available_sections:
                try:
                    mask = reader.mask
                    info["mask"] = {
                        "width": mask.width,
                        "height": mask.height,
                        "format": mask.format,
                        "data_size": len(mask.data)
                    }
                except SPDSectionError as e:
                    info["mask"] = {"error": str(e)}
            
            if "additional_flags" in available_sections:
                try:
                    additional_flags = reader.additional_flags
                    info["additional_flags"] = {
                        "num_flags": additional_flags.num_flags,
                        "flags": additional_flags.flags
                    }
                except SPDSectionError as e:
                    info["additional_flags"] = {"error": str(e)}
        
        # Output information
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(ColorFormatter.bold(f"SPD File: {args.file}"))
            print(f"Version: {info['version']}")
            print(f"Created: {info['timestamp_readable']}")
            print(f"Flags: {', '.join(k for k, v in flags.items() if v)}")
            print(f"Sections: {', '.join(available_sections)}")
            
            if args.verbose:
                print("\nDetailed Section Information:")
                
                if "image" in info:
                    print(ColorFormatter.cyan("\nImage Section:"))
                    img_info = info["image"]
                    if "error" in img_info:
                        print(f"  Error: {img_info['error']}")
                    else:
                        print(f"  Dimensions: {img_info['width']}x{img_info['height']}")
                        print(f"  Channels: {img_info['channels']}")
                        print(f"  Format: {img_info['format']}")
                        print(f"  Data size: {img_info['data_size']} bytes")
                
                if "landmarks" in info:
                    print(ColorFormatter.cyan("\nLandmarks Section:"))
                    lm_info = info["landmarks"]
                    if "error" in lm_info:
                        print(f"  Error: {lm_info['error']}")
                    else:
                        print(f"  Count: {lm_info['count']} points")
                        print(f"  Dimensions: {lm_info['dimensions']}D")
                        print(f"  Type: {lm_info['type']}")
                        if args.very_verbose and "points" in lm_info:
                            print("  Points:")
                            for i, point in enumerate(lm_info['points'][:10]):
                                print(f"    {i}: {point}")
                            if len(lm_info['points']) > 10:
                                print(f"    ... ({len(lm_info['points']) - 10} more points)")
                
                if "motion_params" in info:
                    print(ColorFormatter.cyan("\nMotion Parameters Section:"))
                    mp_info = info["motion_params"]
                    if "error" in mp_info:
                        print(f"  Error: {mp_info['error']}")
                    else:
                        print(f"  Number of parameters: {mp_info['num_params']}")
                        print(f"  Parameter names: {', '.join(mp_info['param_names'])}")
                        if args.very_verbose and "values" in mp_info:
                            print("  Parameter values:")
                            for name, value in zip(mp_info['param_names'], mp_info['values']):
                                print(f"    {name}: {value}")
                
                if "appearance" in info:
                    print(ColorFormatter.cyan("\nAppearance Section:"))
                    app_info = info["appearance"]
                    if "error" in app_info:
                        print(f"  Error: {app_info['error']}")
                    else:
                        print(f"  Feature dimension: {app_info['feature_dim']}")
                        print(f"  Feature type: {app_info['feature_type']}")
                
                if "transforms" in info:
                    print(ColorFormatter.cyan("\nTransformations Section:"))
                    tx_info = info["transforms"]
                    if "error" in tx_info:
                        print(f"  Error: {tx_info['error']}")
                    else:
                        print(f"  Number of transforms: {tx_info['num_transforms']}")
                        print(f"  Transform types: {', '.join(tx_info['transform_types'])}")
                
                if "mask" in info:
                    print(ColorFormatter.cyan("\nMask Section:"))
                    mask_info = info["mask"]
                    if "error" in mask_info:
                        print(f"  Error: {mask_info['error']}")
                    else:
                        print(f"  Dimensions: {mask_info['width']}x{mask_info['height']}")
                        print(f"  Format: {mask_info['format']}")
                        print(f"  Data size: {mask_info['data_size']} bytes")
                
                if "additional_flags" in info:
                    print(ColorFormatter.cyan("\nAdditional Flags Section:"))
                    flags_info = info["additional_flags"]
                    if "error" in flags_info:
                        print(f"  Error: {flags_info['error']}")
                    else:
                        print(f"  Number of flags: {flags_info['num_flags']}")
                        if "flags" in flags_info and flags_info["flags"]:
                            print("  Flags:")
                            for key, value in flags_info["flags"].items():
                                print(f"    {key}: {value}")
        
        reader.close()
        return 0
    
    except (SPDError, Exception) as e:
        logger.error(f"Failed to get SPD file information: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


def validate_cmd(args: argparse.Namespace) -> int:
    """
    Validate an SPD file and report any issues.
    
    Args:
        args: Command-line arguments
            - file: Path to the SPD file
            - level: Validation level (basic, standard, strict)
            - json: Output in JSON format
    
    Returns:
        0 for success, non-zero for failure
    """
    try:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return 1
        
        # Determine validation level
        level_map = {
            "basic": ValidationLevel.BASIC,
            "standard": ValidationLevel.STANDARD,
            "strict": ValidationLevel.STRICT
        }
        validation_level = level_map.get(args.level.lower(), ValidationLevel.STANDARD)
        
        # Validate the file
        validator = SPDValidator(args.file, validation_level)
        report = validator.validate()
        
        # Output validation report
        if args.json:
            # Convert report to JSON serializable format
            json_report = {
                "file": args.file,
                "is_valid": report.is_valid,
                "validation_level": report.validation_level.name,
                "error_count": report.get_error_count(),
                "warning_count": report.get_warning_count(),
                "info_count": report.get_info_count(),
                "sections_checked": report.sections_checked,
                "sections_valid": report.sections_valid,
                "sections_invalid": report.sections_invalid,
                "elapsed_time_ms": report.elapsed_time_ms,
                "issues": []
            }
            
            for issue in report.issues:
                json_report["issues"].append({
                    "section": issue.section,
                    "severity": issue.severity.name,
                    "message": issue.message,
                    "suggestion": issue.suggestion
                })
            
            print(json.dumps(json_report, indent=2))
        else:
            # Print formatted report
            status = ColorFormatter.green("VALID") if report.is_valid else ColorFormatter.red("INVALID")
            
            print(ColorFormatter.bold(f"SPD Validation Report: {args.file}"))
            print(f"Status: {status}")
            print(f"Validation Level: {report.validation_level.name}")
            print(f"Time: {report.elapsed_time_ms:.2f} ms")
            print(f"Sections Checked: {', '.join(report.sections_checked)}")
            
            if report.sections_invalid:
                print(f"Invalid Sections: {ColorFormatter.red(', '.join(report.sections_invalid))}")
            
            print(f"Errors: {report.get_error_count()}, "
                  f"Warnings: {report.get_warning_count()}, "
                  f"Info: {report.get_info_count()}")
            
            if report.issues:
                print("\nIssues:")
                for i, issue in enumerate(report.issues, 1):
                    if issue.severity == ValidationSeverity.ERROR:
                        severity = ColorFormatter.red(issue.severity.name)
                    elif issue.severity == ValidationSeverity.WARNING:
                        severity = ColorFormatter.yellow(issue.severity.name)
                    else:
                        severity = ColorFormatter.blue(issue.severity.name)
                    
                    print(f"{i}. [{severity}] {issue.section}: {issue.message}")
                    if issue.suggestion:
                        print(f"   Suggestion: {issue.suggestion}")
            else:
                print("\nNo issues found.")
        
        # Return success if validation passed, otherwise failure
        return 0 if report.is_valid else 1
    
    except (SPDError, Exception) as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


def extract_cmd(args: argparse.Namespace) -> int:
    """
    Extract data from an SPD file (image, landmarks, etc.).
    
    Args:
        args: Command-line arguments
            - file: Path to the SPD file
            - image: Extract the image to this path
            - landmarks: Extract landmarks to this path
            - format: Format for extracted data (auto, csv, json)
            - mask: Extract the mask to this path
    
    Returns:
        0 for success, non-zero for failure
    """
    try:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return 1
        
        reader = SPDReader(args.file)
        available_sections = reader.get_available_sections()
        output_count = 0
        
        # Extract image
        if args.image:
            if "image" in available_sections:
                if not HAVE_CV2:
                    logger.error("OpenCV is required for extracting images.")
                    logger.error("Install it with: pip install opencv-python")
                    return 1
                
                try:
                    logger.info(f"Extracting image to: {args.image}")
                    
                    # Get image section
                    image_section = reader.image
                    
                    # Convert to numpy array
                    img = np.frombuffer(
                        image_section.data, 
                        dtype=np.uint8
                    ).reshape(
                        image_section.height, 
                        image_section.width, 
                        image_section.channels
                    )
                    
                    # Save image
                    cv2.imwrite(args.image, img)
                    output_count += 1
                    logger.info(ColorFormatter.success(f"Image extracted to {args.image}"))
                except SPDSectionError as e:
                    logger.error(f"Failed to extract image: {e}")
            else:
                logger.error("The SPD file does not contain an image section")
        
        # Extract landmarks
        if args.landmarks:
            if "landmarks" in available_sections:
                try:
                    logger.info(f"Extracting landmarks to: {args.landmarks}")
                    
                    # Get landmarks section
                    landmarks_section = reader.landmarks
                    
                    # Determine output format
                    landmarks_path = Path(args.landmarks)
                    format_arg = args.format.lower() if args.format else "auto"
                    
                    if format_arg == "auto":
                        # Try to determine format from file extension
                        ext = landmarks_path.suffix.lower()
                        if ext in ['.csv', '.txt']:
                            format_type = "csv"
                        elif ext in ['.json', '.jsonl']:
                            format_type = "json"
                        else:
                            format_type = "json"  # Default
                    else:
                        format_type = format_arg
                    
                    # Save landmarks in the requested format
                    if format_type == "csv":
                        with open(landmarks_path, 'w', newline='') as f:
                            writer = csv.writer(f)
                            for point in landmarks_section.points:
                                writer.writerow(point)
                    elif format_type == "json":
                        with open(landmarks_path, 'w') as f:
                            json_data = {
                                "landmark_type": landmarks_section.landmark_type,
                                "dimensions": landmarks_section.dimensions,
                                "count": landmarks_section.count,
                                "landmarks": landmarks_section.points
                            }
                            json.dump(json_data, f, indent=2)
                    else:
                        logger.error(f"Unsupported landmarks output format: {format_type}")
                        return 1
                    
                    output_count += 1
                    logger.info(ColorFormatter.success(
                        f"Landmarks extracted to {args.landmarks} in {format_type} format"
                    ))
                except SPDSectionError as e:
                    logger.error(f"Failed to extract landmarks: {e}")
            else:
                logger.error("The SPD file does not contain a landmarks section")
        
        # Extract mask
        if args.mask:
            if "mask" in available_sections:
                if not HAVE_CV2:
                    logger.error("OpenCV is required for extracting masks.")
                    logger.error("Install it with: pip install opencv-python")
                    return 1
                
                try:
                    logger.info(f"Extracting mask to: {args.mask}")
                    
                    # Get mask section
                    mask_section = reader.mask
                    
                    # Convert to numpy array
                    mask = np.frombuffer(
                        mask_section.data, 
                        dtype=np.uint8
                    ).reshape(
                        mask_section.height, 
                        mask_section.width
                    )
                    
                    # Save mask
                    cv2.imwrite(args.mask, mask)
                    output_count += 1
                    logger.info(ColorFormatter.success(f"Mask extracted to {args.mask}"))
                except SPDSectionError as e:
                    logger.error(f"Failed to extract mask: {e}")
            else:
                logger.error("The SPD file does not contain a mask section")
        
        reader.close()
        
        if output_count == 0:
            logger.warning("No data was extracted. Specify at least one output type.")
            return 1
        
        return 0
    
    except (SPDError, Exception) as e:
        logger.error(f"Failed to extract data from SPD file: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


def convert_cmd(args: argparse.Namespace) -> int:
    """
    Convert between different formats (e.g., extract landmarks to CSV).
    
    Args:
        args: Command-line arguments
            - file: Path to the SPD file
            - output: Path to the output file
            - type: Type of data to convert (landmarks, transforms, motion_params)
            - format: Output format (csv, json)
    
    Returns:
        0 for success, non-zero for failure
    """
    try:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return 1
        
        reader = SPDReader(args.file)
        available_sections = reader.get_available_sections()
        data_type = args.type.lower()
        output_format = args.format.lower()
        
        # Check if the requested section is available
        if data_type not in available_sections:
            logger.error(f"The SPD file does not contain a {data_type} section")
            return 1
        
        # Convert the requested data
        logger.info(f"Converting {data_type} data to {output_format} format")
        output_path = Path(args.output)
        
        if data_type == "landmarks":
            # Get landmarks data
            landmarks = reader.landmarks
            
            # Save in requested format
            if output_format == "csv":
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    for point in landmarks.points:
                        writer.writerow(point)
            elif output_format == "json":
                with open(output_path, 'w') as f:
                    json_data = {
                        "landmark_type": landmarks.landmark_type,
                        "dimensions": landmarks.dimensions,
                        "count": landmarks.count,
                        "landmarks": landmarks.points
                    }
                    json.dump(json_data, f, indent=2)
            else:
                logger.error(f"Unsupported output format for landmarks: {output_format}")
                return 1
        
        elif data_type == "transforms":
            # Get transforms data
            transforms = reader.transforms
            
            # Save in requested format
            if output_format == "csv":
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write header
                    writer.writerow(["transform_type", "matrix_elements"])
                    # Write data
                    for transform_type, matrix in zip(transforms.transform_types, transforms.matrices):
                        writer.writerow([transform_type] + matrix)
            elif output_format == "json":
                with open(output_path, 'w') as f:
                    json_data = {
                        "num_transforms": transforms.num_transforms,
                        "transforms": [
                            {
                                "type": t_type,
                                "matrix": matrix
                            }
                            for t_type, matrix in zip(transforms.transform_types, transforms.matrices)
                        ]
                    }
                    json.dump(json_data, f, indent=2)
            else:
                logger.error(f"Unsupported output format for transforms: {output_format}")
                return 1
        
        elif data_type == "motion_params":
            # Get motion parameters data
            motion_params = reader.motion_params
            
            # Save in requested format
            if output_format == "csv":
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write header
                    writer.writerow(["param_name", "value"])
                    # Write data
                    for name, value in zip(motion_params.param_names, motion_params.values):
                        writer.writerow([name, value])
            elif output_format == "json":
                with open(output_path, 'w') as f:
                    json_data = {
                        "num_params": motion_params.num_params,
                        "parameters": {
                            name: value
                            for name, value in zip(motion_params.param_names, motion_params.values)
                        }
                    }
                    json.dump(json_data, f, indent=2)
            else:
                logger.error(f"Unsupported output format for motion parameters: {output_format}")
                return 1
        
        else:
            logger.error(f"Conversion not supported for section type: {data_type}")
            return 1
        
        logger.info(ColorFormatter.success(f"Data converted and saved to {args.output}"))
        reader.close()
        return 0
    
    except (SPDError, Exception) as e:
        logger.error(f"Failed to convert SPD data: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


def visualize_cmd(args: argparse.Namespace) -> int:
    """
    Visualize SPD data (landmarks, transforms, etc.).
    
    Args:
        args: Command-line arguments
            - file: Path to the SPD file
            - output: Path to save the visualization (optional)
            - show: Display visualization in a window
            - type: Type of visualization (landmarks, mask, combined)
    
    Returns:
        0 for success, non-zero for failure
    """
    try:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return 1
        
        if not HAVE_CV2 or not HAVE_PLT:
            logger.error("OpenCV and Matplotlib are required for visualization.")
            logger.error("Install them with: pip install opencv-python matplotlib")
            return 1
        
        reader = SPDReader(args.file)
        available_sections = reader.get_available_sections()
        viz_type = args.type.lower()
        
        # Check visualization type
        if viz_type == "landmarks" and "landmarks" not in available_sections:
            logger.error("The SPD file does not contain landmarks")
            return 1
        elif viz_type == "mask" and "mask" not in available_sections:
            logger.error("The SPD file does not contain a mask")
            return 1
        elif viz_type == "combined" and ("landmarks" not in available_sections or "image" not in available_sections):
            logger.error("Combined visualization requires both image and landmarks sections")
            return 1
        
        # Create visualization
        fig = plt.figure(figsize=(12, 8))
        
        if viz_type == "landmarks":
            # Visualize landmarks
            landmarks = reader.landmarks
            points = np.array(landmarks.points)
            
            # Plot landmarks
            plt.scatter(points[:, 0], points[:, 1], c='r', marker='o')
            plt.title(f"Landmarks Visualization ({landmarks.landmark_type})")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
            
            # Invert y-axis for image-like coordinates
            ax = plt.gca()
            ax.invert_yaxis()
        
        elif viz_type == "mask":
            # Visualize mask
            mask = reader.mask
            mask_array = np.frombuffer(
                mask.data, 
                dtype=np.uint8
            ).reshape(
                mask.height, 
                mask.width
            )
            
            # Plot mask
            plt.imshow(mask_array, cmap='gray')
            plt.title("Mask Visualization")
            plt.axis('off')
        
        elif viz_type == "combined":
            # Combined visualization of image and landmarks
            image = reader.image
            landmarks = reader.landmarks
            
            # Convert image data to numpy array
            img = np.frombuffer(
                image.data, 
                dtype=np.uint8
            ).reshape(
                image.height, 
                image.width, 
                image.channels
            )
            
            # Convert BGR to RGB if needed
            if image.format == "BGR":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get landmarks
            points = np.array(landmarks.points)
            
            # Plot image
            plt.imshow(img)
            
            # Plot landmarks on top of image
            plt.scatter(points[:, 0], points[:, 1], c='r', marker='o', s=10)
            
            # Add landmark indices if requested
            if args.label_landmarks:
                for i, (x, y) in enumerate(points):
                    plt.annotate(str(i), (x, y), fontsize=8, color='white', 
                                backgroundcolor='black', alpha=0.7)
            
            plt.title("Combined Visualization")
            plt.axis('off')
        
        else:
            logger.error(f"Unsupported visualization type: {viz_type}")
            return 1
        
        # Save or show visualization
        if args.output:
            plt.tight_layout()
            plt.savefig(args.output, dpi=300)
            logger.info(ColorFormatter.success(f"Visualization saved to {args.output}"))
        
        if args.show:
            plt.tight_layout()
            plt.show()
        
        if not args.output and not args.show:
            logger.warning("No output specified. Use --output or --show to see the visualization.")
            return 1
        
        reader.close()
        return 0
    
    except (SPDError, Exception) as e:
        logger.error(f"Failed to visualize SPD data: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="SPD Editor - Command Line Interface for working with SPD files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create an SPD file from an image with landmarks
  spd_editor create --source image.jpg --output face.spd --landmarks face_landmarks.csv
        
  # Display information about an SPD file
  spd_editor info face.spd
        
  # Extract image from an SPD file
  spd_editor extract --file face.spd --image extracted_face.jpg
        
  # Validate an SPD file
  spd_editor validate face.spd
        
  # Convert landmarks to CSV format
  spd_editor convert --file face.spd --type landmarks --output landmarks.csv --format csv
        
  # Visualize landmarks
  spd_editor visualize --file face.spd --type landmarks --show
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create command
    create_parser = subparsers.add_parser(
        "create",
        help="Create an SPD file from a source image"
    )
    create_parser.add_argument(
        "--source", "-s",
        required=True,
        help="Path to the source image"
    )
    create_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to the output SPD file"
    )
    create_parser.add_argument(
        "--landmarks", "-l",
        help="Path to landmarks file (CSV or JSON)"
    )
    create_parser.add_argument(
        "--landmark-type",
        help="Type of landmarks (e.g., mediapipe, dlib68)"
    )
    create_parser.add_argument(
        "--mask", "-m",
        help="Path to mask file"
    )
    create_parser.add_argument(
        "--include-image",
        action="store_true",
        help="Include the source image in the SPD file"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Display information about an SPD file"
    )
    info_parser.add_argument(
        "file",
        help="Path to the SPD file"
    )
    info_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    info_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information"
    )
    info_parser.add_argument(
        "--very-verbose", "-vv",
        action="store_true",
        help="Show all data (may be large)"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate an SPD file and report any issues"
    )
    validate_parser.add_argument(
        "file",
        help="Path to the SPD file"
    )
    validate_parser.add_argument(
        "--level", "-l",
        choices=["basic", "standard", "strict"],
        default="standard",
        help="Validation level (default: standard)"
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract data from an SPD file (image, landmarks, etc.)"
    )
    extract_parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the SPD file"
    )
    extract_parser.add_argument(
        "--image", "-i",
        help="Path to save the extracted image"
    )
    extract_parser.add_argument(
        "--landmarks", "-l",
        help="Path to save the extracted landmarks"
    )
    extract_parser.add_argument(
        "--mask", "-m",
        help="Path to save the extracted mask"
    )
    extract_parser.add_argument(
        "--format",
        choices=["auto", "csv", "json"],
        default="auto",
        help="Format for extracted data (default: auto)"
    )
    
    # Convert command
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert between different formats (e.g., extract landmarks to CSV)"
    )
    convert_parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the SPD file"
    )
    convert_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to the output file"
    )
    convert_parser.add_argument(
        "--type", "-t",
        required=True,
        choices=["landmarks", "transforms", "motion_params"],
        help="Type of data to convert"
    )
    convert_parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)"
    )
    
    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Create visualizations of SPD data"
    )
    visualize_parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the SPD file"
    )
    visualize_parser.add_argument(
        "--output", "-o",
        help="Path to save the visualization"
    )
    visualize_parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display visualization in a window"
    )
    visualize_parser.add_argument(
        "--type", "-t",
        choices=["landmarks", "mask", "combined"],
        default="combined",
        help="Type of visualization (default: combined)"
    )
    visualize_parser.add_argument(
        "--label-landmarks",
        action="store_true",
        help="Label landmarks with their indices (for combined visualization)"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        # Call the appropriate function based on the command
        if args.command == "create":
            return create_cmd(args)
        elif args.command == "info":
            return info_cmd(args)
        elif args.command == "validate":
            return validate_cmd(args)
        elif args.command == "extract":
            return extract_cmd(args)
        elif args.command == "convert":
            return convert_cmd(args)
        elif args.command == "visualize":
            return visualize_cmd(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())