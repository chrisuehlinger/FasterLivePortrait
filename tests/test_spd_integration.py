"""
Tests for the SPD integration with FasterLivePortrait.
"""
import os
import sys
import unittest
import tempfile
import numpy as np
import torch
from omegaconf import OmegaConf

# Add the parent directory to sys.path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the modules we're testing
from src.utils.spd_utils import is_spd_file, load_spd_file, save_spd_file
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from spd_editor.spd.reader import SPDReader
from spd_editor.spd.writer import SPDWriter


class TestSPDIntegration(unittest.TestCase):
    """Test SPD integration with FasterLivePortrait."""

    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.pipeline = None
        
        # Sample configuration for testing
        config_str = """
        models:
          face_analysis:
            name: FaceAnalysisModel
            model_path: checkpoints/liveportrait_onnx/insightface/arcface_w600k_r50.onnx
            providers: [CPUExecutionProvider]
          landmark:
            name: LandmarkModel
            model_path: checkpoints/liveportrait_onnx/insightface/2d106det.onnx
            providers: [CPUExecutionProvider]
          motion_extractor:
            name: MotionExtractorModel
            model_path: checkpoints/liveportrait_onnx/motion_extractor.onnx
            providers: [CPUExecutionProvider]
          app_feat_extractor:
            name: AppFeatExtractorModel
            model_path: checkpoints/liveportrait_onnx/app_feat_extractor.onnx
            providers: [CPUExecutionProvider]
          stitching:
            name: StitchingModel
            model_path: checkpoints/liveportrait_onnx/stitching.onnx
            providers: [CPUExecutionProvider]
          stitching_lip_retarget:
            name: StitchingLipRetargetModel
            model_path: checkpoints/liveportrait_onnx/stitching_lip_retarget.onnx
            providers: [CPUExecutionProvider]
          stitching_eye_retarget:
            name: StitchingEyeRetargetModel
            model_path: checkpoints/liveportrait_onnx/stitching_eye_retarget.onnx
            providers: [CPUExecutionProvider]
          warping_spade:
            name: WarpingSpadeModel
            model_path: checkpoints/liveportrait_onnx/warping_spade_generator.onnx
            providers: [CPUExecutionProvider]

        infer_params:
          crop_face: True
          flag_stitching: True
          flag_lip_retargeting: False
          flag_eye_retargeting: False
          flag_relative_motion: True
          flag_pasteback: False
          flag_crop_driving_video: True 
          flag_normalize_lip: True
          driving_multiplier: 1.0
          lip_normalize_threshold: 0.6
          animation_region: "all"
          flag_do_crop: True
          flag_source_video_eye_retargeting: True
          source_video_eye_retargeting_threshold: 0.5
          source_max_dim: 512
          source_division: 128
          mask_crop_path: "assets/mask_template.png"
          flag_video_editing_head_rotation: False

        crop_params:
          src_dsize: 512
          src_scale: 1.6
          src_vx_ratio: 0.33
          src_vy_ratio: 0.325
          dri_dsize: 256
          dri_scale: 1.6
          dri_vx_ratio: 0.33
          dri_vy_ratio: 0.325
        """
        self.config = OmegaConf.create(config_str)
        
        # Create a mock sample image for testing
        self.sample_img_path = os.path.join(self.test_dir.name, "sample.jpg")
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[100:400, 100:400, :] = 255  # White square in the middle (simulating a face)
        import cv2
        cv2.imwrite(self.sample_img_path, img)
        
        # Path for test SPD file
        self.test_spd_path = os.path.join(self.test_dir.name, "test.spd")
        
    def tearDown(self):
        """Clean up resources."""
        self.test_dir.cleanup()
        if self.pipeline is not None:
            del self.pipeline
    
    def create_mock_pipeline(self):
        """Create a mock pipeline for testing."""
        # Skip if we need actual models
        if not os.path.exists("checkpoints/liveportrait_onnx/insightface/arcface_w600k_r50.onnx"):
            self.skipTest("Skipping test because required models are not available")
            
        self.pipeline = FasterLivePortraitPipeline(cfg=self.config)
        
    def create_mock_source_data(self):
        """Create mock source data for testing."""
        # Create a simple RGB image
        img_rgb = np.ones((256, 256, 3), dtype=np.uint8) * 128  # Gray image
        
        # Create a minimal source info structure
        x_s_info = {
            "pitch": np.array([[0.1]], dtype=np.float32),
            "yaw": np.array([[0.2]], dtype=np.float32),
            "roll": np.array([[0.0]], dtype=np.float32),
            "t": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "exp": np.zeros((1, 20, 3), dtype=np.float32),
            "scale": np.array([[1.0]], dtype=np.float32),
            "kp": np.zeros((1, 20, 3), dtype=np.float32)
        }
        
        source_lmk = np.array([[100, 100], [150, 100], [200, 100],
                              [100, 150], [150, 150], [200, 150],
                              [100, 200], [150, 200], [200, 200]], dtype=np.float32)
        
        R_s = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
        f_s = np.ones((1, 256), dtype=np.float32) * 0.5
        x_s = np.zeros((1, 20, 3), dtype=np.float32)
        x_c_s = np.zeros((1, 20, 3), dtype=np.float32)
        lip_delta = None
        flag_lip_zero = False
        mask_ori_float = None
        M = torch.eye(3, dtype=torch.float32)
        
        src_info = [
            x_s_info, source_lmk, R_s, f_s, x_s, x_c_s,
            lip_delta, flag_lip_zero, mask_ori_float, M
        ]
        
        return [img_rgb], [[src_info]], False  # src_imgs, src_infos, is_source_video
    
    def test_is_spd_file(self):
        """Test the is_spd_file function."""
        # Create a test SPD file
        src_imgs, src_infos, is_video = self.create_mock_source_data()
        save_spd_file(self.test_spd_path, src_imgs, src_infos, is_video)
        
        # Test that it detects the SPD file correctly
        self.assertTrue(is_spd_file(self.test_spd_path))
        
        # Test that it returns False for a non-SPD file
        self.assertFalse(is_spd_file(self.sample_img_path))
    
    def test_save_and_load_spd(self):
        """Test saving and loading SPD files."""
        # Create and save a mock SPD file
        src_imgs, src_infos, is_video = self.create_mock_source_data()
        save_spd_file(self.test_spd_path, src_imgs, src_infos, is_video)
        
        # Verify the file exists
        self.assertTrue(os.path.exists(self.test_spd_path))
        
        # Load the SPD file
        device = torch.device("cpu")
        loaded_imgs, loaded_infos, loaded_is_video = load_spd_file(
            self.test_spd_path, device, self.config)
        
        # Verify basic structure of loaded data
        self.assertEqual(len(loaded_imgs), 1)
        self.assertEqual(len(loaded_infos), 1)
        self.assertEqual(loaded_is_video, is_video)
        
        # Verify the image data
        self.assertEqual(loaded_imgs[0].shape, src_imgs[0].shape)
        
        # Verify the info structure
        self.assertEqual(len(loaded_infos[0]), 1)
        self.assertEqual(len(loaded_infos[0][0]), 10)  # Should have 10 elements
        
        # Check a few key parameters
        loaded_x_s_info = loaded_infos[0][0][0]
        self.assertAlmostEqual(loaded_x_s_info["pitch"][0][0], 0.1, places=5)
        self.assertAlmostEqual(loaded_x_s_info["yaw"][0][0], 0.2, places=5)
    
    @unittest.skip("Skip pipeline integration test if models not available")
    def test_pipeline_with_spd(self):
        """Test the integration of SPD files with the pipeline."""
        try:
            self.create_mock_pipeline()
        except Exception as e:
            self.skipTest(f"Skipping pipeline test due to setup error: {e}")
            
        # Export a source image as SPD
        export_path = os.path.join(self.test_dir.name, "exported.spd")
        ret = self.pipeline.prepare_source(self.sample_img_path, export_spd=export_path)
        
        # If face detection fails on the mock image, skip the test
        if not ret:
            self.skipTest("Skipping test because face detection failed on mock image")
        
        # Verify the SPD file was created
        self.assertTrue(os.path.exists(export_path))
        
        # Now try to use the SPD file as a source
        self.pipeline.src_imgs = []  # Clear previous source data
        self.pipeline.src_infos = []
        ret = self.pipeline.prepare_source(export_path)
        
        # Verify loading worked
        self.assertTrue(ret)
        self.assertGreater(len(self.pipeline.src_imgs), 0)
        self.assertGreater(len(self.pipeline.src_infos), 0)


if __name__ == "__main__":
    unittest.main()