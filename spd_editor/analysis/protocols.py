"""
Protocol definitions for SPD Editor analysis components.

This module provides Protocol classes to define interfaces for face analysis components.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, BinaryIO, Callable
from typing_extensions import Protocol, runtime_checkable
import numpy as np

from spd_editor.spd.types import LandmarkPoint, FacialLandmarks, AppearanceFeatures


@runtime_checkable
class FaceDetectorProtocol(Protocol):
    """Protocol defining the interface for face detection implementations."""
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected faces, each as a dictionary with bounding box and confidence
        """
        ...


@runtime_checkable
class LandmarkExtractorProtocol(Protocol):
    """Protocol defining the interface for facial landmark extraction implementations."""
    
    def extract_landmarks(self, image: np.ndarray, face_location: Optional[Dict[str, Any]] = None) -> FacialLandmarks:
        """
        Extract facial landmarks from an image.
        
        Args:
            image: Input image as numpy array
            face_location: Optional face location dict (if None, will detect face first)
            
        Returns:
            Dictionary with facial landmarks information
        """
        ...
    
    def get_landmark_type(self) -> str:
        """
        Get the type of landmarks that this extractor provides.
        
        Returns:
            String identifier of the landmark type (e.g., "mediapipe", "dlib68")
        """
        ...


@runtime_checkable
class FeatureExtractorProtocol(Protocol):
    """Protocol defining the interface for facial feature extraction implementations."""
    
    def extract_features(self, image: np.ndarray, face_location: Optional[Dict[str, Any]] = None) -> AppearanceFeatures:
        """
        Extract facial appearance features from an image.
        
        Args:
            image: Input image as numpy array
            face_location: Optional face location dict (if None, will detect face first)
            
        Returns:
            Dictionary with facial appearance features
        """
        ...
    
    def get_feature_type(self) -> str:
        """
        Get the type of features that this extractor provides.
        
        Returns:
            String identifier of the feature type
        """
        ...


@runtime_checkable
class SPDProcessorProtocol(Protocol):
    """Protocol defining the interface for SPD processors."""
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image and prepare SPD data.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary with all extracted SPD data
        """
        ...
    
    def save_spd(self, output_path: str, data: Dict[str, Any]) -> None:
        """
        Save SPD data to a file.
        
        Args:
            output_path: Path where the SPD file will be saved
            data: Dictionary with SPD data to save
        """
        ...
