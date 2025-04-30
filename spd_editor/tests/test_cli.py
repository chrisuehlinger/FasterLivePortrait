"""
Tests for SPD Editor Command-Line Interface.
"""
import os
import sys
import json
import pytest
import tempfile
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import cv2

from spd_editor.cli import (
    create_cmd, info_cmd, validate_cmd, extract_cmd, 
    convert_cmd, visualize_cmd, main
)
from spd_editor.spd.format import (
    ImageSection, LandmarksSection, MaskSection
)
from spd_editor.spd.writer import SPDWriter


# Helper functions for tests
def create_test_image(path: str, width: int = 100, height: int = 100) -> None:
    """Create a simple test image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[30:70, 30:70] = [255, 128, 64]  # Face-like region
    cv2.imwrite(path, img)


def create_test_landmarks(path: str, format_type: str = "csv") -> None:
    """Create test landmarks file in specified format."""
    # Define some test landmarks (5 points)
    landmarks = [
        [30, 40],
        [70, 40],
        [50, 50],
        [40, 70],
        [60, 70]
    ]
    
    if format_type == "csv":
        with open(path, 'w', newline='') as f:
            for point in landmarks:
                f.write(f"{point[0]},{point[1]}\n")
    else:  # json
        with open(path, 'w') as f:
            json_data = {
                "landmark_type": "test",
                "dimensions": 2,
                "count": len(landmarks),
                "landmarks": landmarks
            }
            json.dump(json_data, f)


def create_test_mask(path: str, width: int = 100, height: int = 100) -> None:
    """Create a simple test mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # Mask region
    cv2.imwrite(path, mask)


def create_test_spd(path: str, include_image: bool = True, include_landmarks: bool = True,
                   include_mask: bool = False) -> None:
    """Create a test SPD file with optional sections."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data files
        temp_img_path = os.path.join(temp_dir, "test_image.jpg")
        temp_landmarks_path = os.path.join(temp_dir, "test_landmarks.csv")
        temp_mask_path = os.path.join(temp_dir, "test_mask.jpg")
        
        create_test_image(temp_img_path)
        create_test_landmarks(temp_landmarks_path)
        if include_mask:
            create_test_mask(temp_mask_path)
        
        # Create SPD file
        with SPDWriter(path) as writer:
            if include_image:
                # Read image
                img = cv2.imread(temp_img_path)
                height, width = img.shape[:2]
                channels = 3
                
                # Create and write image section
                img_section = ImageSection(
                    width=width,
                    height=height,
                    channels=channels,
                    format="BGR",
                    data=img.tobytes()
                )
                writer.write_image_section(img_section)
            
            if include_landmarks:
                # Create landmarks
                landmarks_points = []
                with open(temp_landmarks_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            x, y = line.strip().split(',')
                            landmarks_points.append([float(x), float(y)])
                
                # Create and write landmarks section
                landmarks_section = LandmarksSection(
                    count=len(landmarks_points),
                    dimensions=2,
                    landmark_type="test",
                    points=landmarks_points
                )
                writer.write_landmarks_section(landmarks_section)
            
            if include_mask:
                # Read mask
                mask = cv2.imread(temp_mask_path, cv2.IMREAD_GRAYSCALE)
                height, width = mask.shape
                
                # Create and write mask section
                mask_section = MaskSection(
                    width=width,
                    height=height,
                    format="grayscale",
                    data=mask.tobytes()
                )
                writer.write_mask_section(mask_section)


# Test fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def test_image(temp_dir):
    """Create a test image file."""
    img_path = os.path.join(temp_dir, "test_image.jpg")
    create_test_image(img_path)
    return img_path


@pytest.fixture
def test_landmarks_csv(temp_dir):
    """Create a test landmarks CSV file."""
    landmarks_path = os.path.join(temp_dir, "test_landmarks.csv")
    create_test_landmarks(landmarks_path, "csv")
    return landmarks_path


@pytest.fixture
def test_landmarks_json(temp_dir):
    """Create a test landmarks JSON file."""
    landmarks_path = os.path.join(temp_dir, "test_landmarks.json")
    create_test_landmarks(landmarks_path, "json")
    return landmarks_path


@pytest.fixture
def test_mask(temp_dir):
    """Create a test mask file."""
    mask_path = os.path.join(temp_dir, "test_mask.jpg")
    create_test_mask(mask_path)
    return mask_path


@pytest.fixture
def test_spd_basic(temp_dir):
    """Create a basic test SPD file with image and landmarks."""
    spd_path = os.path.join(temp_dir, "test_basic.spd")
    create_test_spd(spd_path, include_image=True, include_landmarks=True)
    return spd_path


@pytest.fixture
def test_spd_full(temp_dir):
    """Create a full test SPD file with image, landmarks, and mask."""
    spd_path = os.path.join(temp_dir, "test_full.spd")
    create_test_spd(spd_path, include_image=True, include_landmarks=True, include_mask=True)
    return spd_path


# Tests for CLI commands
class TestCreateCommand:
    """Tests for the 'create' command."""
    
    def test_create_basic(self, temp_dir, test_image, test_landmarks_csv):
        """Test creating a basic SPD file with image and landmarks."""
        output_path = os.path.join(temp_dir, "output.spd")
        
        # Create mock args
        args = mock.MagicMock(
            source=test_image,
            output=output_path,
            landmarks=test_landmarks_csv,
            landmark_type="test",
            mask=None,
            include_image=True
        )
        
        # Run command
        result = create_cmd(args)
        
        # Verify
        assert result == 0
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    
    def test_create_with_mask(self, temp_dir, test_image, test_landmarks_csv, test_mask):
        """Test creating an SPD file with image, landmarks, and mask."""
        output_path = os.path.join(temp_dir, "output_with_mask.spd")
        
        # Create mock args
        args = mock.MagicMock(
            source=test_image,
            output=output_path,
            landmarks=test_landmarks_csv,
            landmark_type="test",
            mask=test_mask,
            include_image=True
        )
        
        # Run command
        result = create_cmd(args)
        
        # Verify
        assert result == 0
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    
    def test_create_with_json_landmarks(self, temp_dir, test_image, test_landmarks_json):
        """Test creating an SPD file with JSON landmarks format."""
        output_path = os.path.join(temp_dir, "output_json_landmarks.spd")
        
        # Create mock args
        args = mock.MagicMock(
            source=test_image,
            output=output_path,
            landmarks=test_landmarks_json,
            landmark_type="test",
            mask=None,
            include_image=True
        )
        
        # Run command
        result = create_cmd(args)
        
        # Verify
        assert result == 0
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    
    def test_create_no_image(self, temp_dir, test_image, test_landmarks_csv):
        """Test creating an SPD file without including the image."""
        output_path = os.path.join(temp_dir, "output_no_image.spd")
        
        # Create mock args
        args = mock.MagicMock(
            source=test_image,
            output=output_path,
            landmarks=test_landmarks_csv,
            landmark_type="test",
            mask=None,
            include_image=False
        )
        
        # Run command
        result = create_cmd(args)
        
        # Verify
        assert result == 0
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    
    def test_create_invalid_source(self, temp_dir):
        """Test creating an SPD file with an invalid source image."""
        output_path = os.path.join(temp_dir, "output_invalid.spd")
        invalid_source = os.path.join(temp_dir, "nonexistent_image.jpg")
        
        # Create mock args
        args = mock.MagicMock(
            source=invalid_source,
            output=output_path,
            landmarks=None,
            landmark_type=None,
            mask=None,
            include_image=True
        )
        
        # Run command
        result = create_cmd(args)
        
        # Verify
        assert result != 0
        assert not os.path.exists(output_path)


class TestInfoCommand:
    """Tests for the 'info' command."""
    
    def test_info_basic(self, test_spd_basic):
        """Test getting basic info from an SPD file."""
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_basic,
            json=False,
            verbose=False,
            very_verbose=False
        )
        
        # Run command with captured output
        with mock.patch('sys.stdout', new_callable=sys.StringIO) as mock_stdout:
            result = info_cmd(args)
            output = mock_stdout.getvalue()
        
        # Verify
        assert result == 0
        assert "SPD File:" in output
        assert "Version:" in output
        assert "Created:" in output
        assert "Sections:" in output
    
    def test_info_verbose(self, test_spd_basic):
        """Test getting verbose info from an SPD file."""
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_basic,
            json=False,
            verbose=True,
            very_verbose=False
        )
        
        # Run command with captured output
        with mock.patch('sys.stdout', new_callable=sys.StringIO) as mock_stdout:
            result = info_cmd(args)
            output = mock_stdout.getvalue()
        
        # Verify
        assert result == 0
        assert "Detailed Section Information:" in output
        assert "Image Section:" in output
        assert "Landmarks Section:" in output
    
    def test_info_json_format(self, test_spd_basic):
        """Test getting info in JSON format."""
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_basic,
            json=True,
            verbose=False,
            very_verbose=False
        )
        
        # Run command with captured output
        with mock.patch('sys.stdout', new_callable=sys.StringIO) as mock_stdout:
            result = info_cmd(args)
            output = mock_stdout.getvalue()
        
        # Verify
        assert result == 0
        
        # Parse JSON output
        try:
            info_json = json.loads(output)
            assert "file" in info_json
            assert "version" in info_json
            assert "timestamp" in info_json
            assert "sections" in info_json
        except json.JSONDecodeError:
            assert False, "Output is not valid JSON"
    
    def test_info_invalid_file(self, temp_dir):
        """Test getting info from a nonexistent SPD file."""
        invalid_file = os.path.join(temp_dir, "nonexistent.spd")
        
        # Create mock args
        args = mock.MagicMock(
            file=invalid_file,
            json=False,
            verbose=False,
            very_verbose=False
        )
        
        # Run command
        result = info_cmd(args)
        
        # Verify
        assert result != 0


class TestValidateCommand:
    """Tests for the 'validate' command."""
    
    def test_validate_basic(self, test_spd_basic):
        """Test validating a basic SPD file."""
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_basic,
            level="standard",
            json=False
        )
        
        # Run command with captured output
        with mock.patch('sys.stdout', new_callable=sys.StringIO) as mock_stdout:
            result = validate_cmd(args)
            output = mock_stdout.getvalue()
        
        # Verify
        assert result == 0
        assert "SPD Validation Report:" in output
        assert "Status: VALID" in output or "Status: \x1b[32mVALID\x1b[0m" in output
    
    def test_validate_json_format(self, test_spd_basic):
        """Test validating an SPD file with JSON output."""
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_basic,
            level="standard",
            json=True
        )
        
        # Run command with captured output
        with mock.patch('sys.stdout', new_callable=sys.StringIO) as mock_stdout:
            result = validate_cmd(args)
            output = mock_stdout.getvalue()
        
        # Verify
        assert result == 0
        
        # Parse JSON output
        try:
            validate_json = json.loads(output)
            assert "file" in validate_json
            assert "is_valid" in validate_json
            assert validate_json["is_valid"] is True
        except json.JSONDecodeError:
            assert False, "Output is not valid JSON"
    
    def test_validate_strict_level(self, test_spd_basic):
        """Test validating an SPD file with strict validation level."""
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_basic,
            level="strict",
            json=False
        )
        
        # Run command
        result = validate_cmd(args)
        
        # Verify - even with strict validation, our test file should pass
        assert result == 0
    
    def test_validate_invalid_file(self, temp_dir):
        """Test validating a nonexistent SPD file."""
        invalid_file = os.path.join(temp_dir, "nonexistent.spd")
        
        # Create mock args
        args = mock.MagicMock(
            file=invalid_file,
            level="standard",
            json=False
        )
        
        # Run command
        result = validate_cmd(args)
        
        # Verify
        assert result != 0


class TestExtractCommand:
    """Tests for the 'extract' command."""
    
    def test_extract_image(self, temp_dir, test_spd_full):
        """Test extracting an image from an SPD file."""
        output_image = os.path.join(temp_dir, "extracted_image.jpg")
        
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_full,
            image=output_image,
            landmarks=None,
            mask=None,
            format="auto"
        )
        
        # Run command
        result = extract_cmd(args)
        
        # Verify
        assert result == 0
        assert os.path.exists(output_image)
        assert os.path.getsize(output_image) > 0
    
    def test_extract_landmarks_csv(self, temp_dir, test_spd_full):
        """Test extracting landmarks to CSV format."""
        output_landmarks = os.path.join(temp_dir, "extracted_landmarks.csv")
        
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_full,
            image=None,
            landmarks=output_landmarks,
            mask=None,
            format="csv"
        )
        
        # Run command
        result = extract_cmd(args)
        
        # Verify
        assert result == 0
        assert os.path.exists(output_landmarks)
        assert os.path.getsize(output_landmarks) > 0
        
        # Check file content
        with open(output_landmarks, 'r') as f:
            content = f.read()
            # Should have at least one line with comma-separated coordinates
            assert ',' in content
    
    def test_extract_landmarks_json(self, temp_dir, test_spd_full):
        """Test extracting landmarks to JSON format."""
        output_landmarks = os.path.join(temp_dir, "extracted_landmarks.json")
        
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_full,
            image=None,
            landmarks=output_landmarks,
            mask=None,
            format="json"
        )
        
        # Run command
        result = extract_cmd(args)
        
        # Verify
        assert result == 0
        assert os.path.exists(output_landmarks)
        assert os.path.getsize(output_landmarks) > 0
        
        # Check file content
        with open(output_landmarks, 'r') as f:
            try:
                data = json.load(f)
                assert "landmarks" in data
                assert isinstance(data["landmarks"], list)
            except json.JSONDecodeError:
                assert False, "Output is not valid JSON"
    
    def test_extract_mask(self, temp_dir, test_spd_full):
        """Test extracting a mask from an SPD file."""
        output_mask = os.path.join(temp_dir, "extracted_mask.jpg")
        
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_full,
            image=None,
            landmarks=None,
            mask=output_mask,
            format="auto"
        )
        
        # Run command
        result = extract_cmd(args)
        
        # Verify
        assert result == 0
        assert os.path.exists(output_mask)
        assert os.path.getsize(output_mask) > 0
    
    def test_extract_multiple(self, temp_dir, test_spd_full):
        """Test extracting multiple sections from an SPD file."""
        output_image = os.path.join(temp_dir, "extracted_image.jpg")
        output_landmarks = os.path.join(temp_dir, "extracted_landmarks.csv")
        output_mask = os.path.join(temp_dir, "extracted_mask.jpg")
        
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_full,
            image=output_image,
            landmarks=output_landmarks,
            mask=output_mask,
            format="auto"
        )
        
        # Run command
        result = extract_cmd(args)
        
        # Verify
        assert result == 0
        assert os.path.exists(output_image)
        assert os.path.exists(output_landmarks)
        assert os.path.exists(output_mask)
    
    def test_extract_missing_section(self, temp_dir, test_spd_basic):
        """Test extracting a section that doesn't exist in the SPD file."""
        output_mask = os.path.join(temp_dir, "missing_mask.jpg")
        
        # Create mock args (test_spd_basic doesn't have a mask section)
        args = mock.MagicMock(
            file=test_spd_basic,
            image=None,
            landmarks=None,
            mask=output_mask,
            format="auto"
        )
        
        # Run command
        result = extract_cmd(args)
        
        # Verify
        assert result != 0
        assert not os.path.exists(output_mask)
    
    def test_extract_no_outputs(self, test_spd_basic):
        """Test extract command with no output paths specified."""
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_basic,
            image=None,
            landmarks=None,
            mask=None,
            format="auto"
        )
        
        # Run command
        result = extract_cmd(args)
        
        # Verify
        assert result != 0  # Should fail because no outputs were specified


class TestConvertCommand:
    """Tests for the 'convert' command."""
    
    def test_convert_landmarks_to_csv(self, temp_dir, test_spd_basic):
        """Test converting landmarks to CSV format."""
        output_file = os.path.join(temp_dir, "converted_landmarks.csv")
        
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_basic,
            output=output_file,
            type="landmarks",
            format="csv"
        )
        
        # Run command
        result = convert_cmd(args)
        
        # Verify
        assert result == 0
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0
        
        # Check file content
        with open(output_file, 'r') as f:
            content = f.read()
            # Should have at least one line with comma-separated coordinates
            assert ',' in content
    
    def test_convert_landmarks_to_json(self, temp_dir, test_spd_basic):
        """Test converting landmarks to JSON format."""
        output_file = os.path.join(temp_dir, "converted_landmarks.json")
        
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_basic,
            output=output_file,
            type="landmarks",
            format="json"
        )
        
        # Run command
        result = convert_cmd(args)
        
        # Verify
        assert result == 0
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0
        
        # Check file content
        with open(output_file, 'r') as f:
            try:
                data = json.load(f)
                assert "landmarks" in data
                assert isinstance(data["landmarks"], list)
            except json.JSONDecodeError:
                assert False, "Output is not valid JSON"
    
    def test_convert_missing_section(self, temp_dir, test_spd_basic):
        """Test converting a section that doesn't exist."""
        output_file = os.path.join(temp_dir, "nonexistent.csv")
        
        # Create mock args (test_spd_basic doesn't have motion_params)
        args = mock.MagicMock(
            file=test_spd_basic,
            output=output_file,
            type="motion_params",
            format="csv"
        )
        
        # Run command
        result = convert_cmd(args)
        
        # Verify
        assert result != 0
        assert not os.path.exists(output_file)
    
    def test_convert_invalid_format(self, temp_dir, test_spd_basic):
        """Test converting to an invalid format."""
        output_file = os.path.join(temp_dir, "invalid_format.xyz")
        
        # Create mock args with invalid format
        args = mock.MagicMock(
            file=test_spd_basic,
            output=output_file,
            type="landmarks",
            format="invalid"
        )
        
        # Run command
        result = convert_cmd(args)
        
        # Verify
        assert result != 0
        assert not os.path.exists(output_file)


class TestVisualizeCommand:
    """Tests for the 'visualize' command."""
    
    def test_visualize_landmarks(self, temp_dir, test_spd_basic):
        """Test visualizing landmarks."""
        output_file = os.path.join(temp_dir, "landmarks_viz.png")
        
        # Skip test if matplotlib not available
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")
        
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_basic,
            output=output_file,
            show=False,
            type="landmarks",
            label_landmarks=False
        )
        
        # Run command
        result = visualize_cmd(args)
        
        # Verify
        assert result == 0
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0
    
    def test_visualize_combined(self, temp_dir, test_spd_basic):
        """Test combined visualization of image and landmarks."""
        output_file = os.path.join(temp_dir, "combined_viz.png")
        
        # Skip test if matplotlib or opencv not available
        try:
            import matplotlib.pyplot as plt
            import cv2
        except ImportError:
            pytest.skip("matplotlib or opencv not available")
        
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_basic,
            output=output_file,
            show=False,
            type="combined",
            label_landmarks=True
        )
        
        # Run command
        result = visualize_cmd(args)
        
        # Verify
        assert result == 0
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0
    
    def test_visualize_invalid_type(self, temp_dir, test_spd_basic):
        """Test visualizing with an invalid type."""
        output_file = os.path.join(temp_dir, "invalid_viz.png")
        
        # Create mock args with invalid visualization type
        args = mock.MagicMock(
            file=test_spd_basic,
            output=output_file,
            show=False,
            type="invalid",
            label_landmarks=False
        )
        
        # Run command
        result = visualize_cmd(args)
        
        # Verify
        assert result != 0
        assert not os.path.exists(output_file)
    
    def test_visualize_no_output(self, test_spd_basic):
        """Test visualizing without specifying an output or show flag."""
        # Create mock args
        args = mock.MagicMock(
            file=test_spd_basic,
            output=None,
            show=False,
            type="landmarks",
            label_landmarks=False
        )
        
        # Run command
        result = visualize_cmd(args)
        
        # Verify
        assert result != 0  # Should fail because no output or show was specified


class TestMainFunction:
    """Tests for the main CLI entry point."""
    
    def test_main_no_args(self):
        """Test main function with no arguments."""
        # Mock sys.argv
        with mock.patch('sys.argv', ['spd_editor']):
            # Run main and capture the output
            with mock.patch('sys.stdout', new_callable=sys.StringIO) as mock_stdout:
                result = main()
                
            # Verify
            assert result == 1  # Return code should be failure (no command)
            assert "usage: " in mock_stdout.getvalue()  # Help message
    
    def test_main_help(self):
        """Test main function with --help argument."""
        # Mock sys.argv
        with mock.patch('sys.argv', ['spd_editor', '--help']):
            # Run main and capture the output
            with mock.patch('sys.stdout', new_callable=sys.StringIO) as mock_stdout:
                try:
                    main()
                    assert False, "Expected SystemExit"
                except SystemExit as e:
                    # --help exits with code 0
                    assert e.code == 0
            
            # Verify help message
            assert "usage: " in mock_stdout.getvalue()
            assert "Commands:" in mock_stdout.getvalue()
    
    def test_main_invalid_command(self):
        """Test main function with an invalid command."""
        # Mock sys.argv
        with mock.patch('sys.argv', ['spd_editor', 'nonexistent']):
            # Run main and capture the output
            with mock.patch('sys.stderr', new_callable=sys.StringIO) as mock_stderr:
                result = main()
                
            # Verify
            assert result == 1  # Return code should be failure
    
    def test_main_with_valid_command(self, temp_dir, test_spd_basic):
        """Test main function with a valid command."""
        # Create output path for info command
        output_file = os.path.join(temp_dir, "info_output.txt")
        
        # Use info command with JSON output to a file
        cmd = f"python -m spd_editor info {test_spd_basic} --json > {output_file}"
        
        # Run command as a subprocess
        try:
            subprocess.run(cmd, shell=True, check=True)
            
            # Verify
            assert os.path.exists(output_file)
            assert os.path.getsize(output_file) > 0
            
            # Check if output is valid JSON
            with open(output_file, 'r') as f:
                try:
                    data = json.load(f)
                    assert "file" in data
                    assert "version" in data
                except json.JSONDecodeError:
                    assert False, "Output is not valid JSON"
                    
        except subprocess.CalledProcessError:
            assert False, "Command failed"