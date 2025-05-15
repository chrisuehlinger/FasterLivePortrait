# Source Data File Format Specification

Version: 1.0

This document specifies the file format used to serialize and store the data gathered by the `prepare_source` method of `FasterLivePortraitPipeline`. The format supports both human and animal modes.

## File Type
- Extension: `.fsp` (FasterLivePortrait Source) or `.pkl`
- Serialization: Python `pickle` of a top-level `dict`.

## Top-Level Structure
```yaml
{
  version: int,                # Format version (1)
  is_source_video: bool,       # True if source_path is a video
  source_path: str,            # Original file path
  src_imgs: List[np.ndarray],  # List of source frames (RGB uint8 arrays)
  src_infos: List[List[FaceInfo]]  # Per-frame list of FaceInfo entries
}
```

## FaceInfo Structure
Each frame's `src_infos` entry is a list of `FaceInfo` objects, one per detected face. A `FaceInfo` is serialized as a `dict` with the following fields:

```yaml
{
  x_s_info: {
    pitch: np.ndarray,    # Float array shape (1,)  
    yaw: np.ndarray,      # Float array shape (1,)  
    roll: np.ndarray,     # Float array shape (1,)  
    t: np.ndarray,        # Translation array shape (1, 3)  
    exp: np.ndarray,      # Expression parameters shape (1, num_kp, 3)  
    scale: np.ndarray,    # Scale array shape (1,)  
    kp: np.ndarray        # Canonical keypoints shape (1, num_kp, 3)
  },
  source_lmk: np.ndarray,          # Landmarks array shape (num_landmarks, 2)
  R_s: np.ndarray,                 # Rotation matrix shape (3, 3)
  f_s: np.ndarray,                 # Appearance feature vector or map (model-dependent)
  x_s: np.ndarray,                 # Transformed keypoints shape (1, num_kp, 3)
  x_c_s: np.ndarray,               # Canonical keypoints shape (1, num_kp, 3)
  lip_delta_before_animation: np.ndarray or None,  # Lip retargeting delta (1, num_kp, 3)
  flag_lip_zero: bool,             # Whether lip retargeting is zeroed
  mask_ori_float: np.ndarray or None,  # Pasteback mask (H, W) float tensor
  M: np.ndarray                    # Crop-to-original transform matrix shape (3, 3)
}
```

### Notes
- For **animal mode**, `lip_delta_before_animation` is always `None` and `flag_lip_zero` is `False`.
- If `flag_pasteback` is disabled in the pipeline configuration, `mask_ori_float` is `None`.
- Frame count:
  - For a still image: `len(src_imgs) == 1`.
  - For a video: `len(src_imgs) == number_of_frames_extracted`.

## Version History
- **1.0**: Initial specification supporting human and animal modes.
