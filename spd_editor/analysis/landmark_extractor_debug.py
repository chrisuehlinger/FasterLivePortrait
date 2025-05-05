import logging
import numpy as np
from typing import Dict, List, Optional, Union

# Wrap the existing LandmarkExtractor to provide debug information
class DebugLandmarkExtractor:
    def __init__(self, original_extractor):
        self.extractor = original_extractor
        self.logger = logging.getLogger("landmark_extractor_debug")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
    
    def extract_landmarks(self, image, face_box):
        """Debug wrapper for landmark extraction"""
        self.logger.debug(f"Extracting landmarks from face box: {face_box}")
        
        # Get the underlying model 
        model = self.extractor.model if hasattr(self.extractor, 'model') else None
        
        if model and hasattr(model, 'predictor'):
            # Print model predictor info
            self.logger.debug(f"Model predictor type: {model.predict_type}")
            self.logger.debug(f"Model input shape: {image.shape}")
            
            # Add a hook to intercept predictions
            original_predict = model.predict
            
            def debug_predict(*args, **kwargs):
                self.logger.debug(f"Predict called with args: {[arg.shape if hasattr(arg, 'shape') else type(arg) for arg in args]}")
                outputs = original_predict(*args, **kwargs)
                
                # Debug output
                if isinstance(outputs, list):
                    self.logger.debug(f"Model output is a list of length {len(outputs)}")
                    for i, out in enumerate(outputs):
                        self.logger.debug(f"Output[{i}] shape: {out.shape if hasattr(out, 'shape') else 'N/A'}")
                elif hasattr(outputs, 'shape'):
                    self.logger.debug(f"Model output shape: {outputs.shape}")
                else:
                    self.logger.debug(f"Model output type: {type(outputs)}")
                
                return outputs
            
            # Replace with debug version temporarily
            model.predict = debug_predict
            
            try:
                # Call the original method
                result = self.extractor.extract_landmarks(image, face_box)
                return result
            finally:
                # Restore original method
                model.predict = original_predict
        else:
            # Fall back to original if we can't intercept
            return self.extractor.extract_landmarks(image, face_box)
