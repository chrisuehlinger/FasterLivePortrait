# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: landmark_model.py
import pdb

from .base_model import BaseModel
import cv2
import numpy as np
from src.utils.crop import crop_image, _transform_pts
import torch
from torch.cuda import nvtx
from .predictor import numpy_to_torch_dtype_dict
from typing import Optional


class LandmarkModel(BaseModel):
    """
    landmark Model
    """
    MODEL_EXPECTED_INPUT_DSIZE = 192  # Define the fixed input size for the ONNX model

    def __init__(self, crop_params: Optional[dict] = None, **kwargs):
        super(LandmarkModel, self).__init__(**kwargs)
        self.crop_config_params = crop_params if crop_params is not None else {}
        # self.dsize is the fixed input size the ONNX model expects.
        self.dsize = LandmarkModel.MODEL_EXPECTED_INPUT_DSIZE

    def input_process(self, *data):
        if len(data) > 1:
            img_rgb, lmk = data
        else:
            img_rgb = data[0]
            lmk = None
        if lmk is not None:
            # Use crop parameters from config for scale and ratios
            scale_val = self.crop_config_params.get('src_scale', 1.5)
            vy_ratio_val = self.crop_config_params.get('src_vy_ratio', -0.1)
            vx_ratio_val = self.crop_config_params.get('src_vx_ratio', 0)

            crop_dct = crop_image(
                img_rgb, 
                lmk, 
                dsize=self.dsize,  # This will correctly be 192
                scale=scale_val, 
                vy_ratio=vy_ratio_val,
                vx_ratio=vx_ratio_val
            )
            img_crop_rgb = crop_dct['img_crop']
        else:
            # NOTE: force resize to self.dsize (192x192), NOT RECOMMEND if face detected!
            img_crop_rgb = cv2.resize(img_rgb, (self.dsize, self.dsize))
            # Scale for M_c2o should relate original image to self.dsize (192)
            scale_fallback = max(img_rgb.shape[:2]) / self.dsize
            crop_dct = {
                'M_c2o': np.array([
                    [scale_fallback, 0., 0.],
                    [0., scale_fallback, 0.],
                    [0., 0., 1.],
                ], dtype=np.float32),
                # Add 'img_crop' for consistency if needed by other parts, though not used in current output_process
                'img_crop': img_crop_rgb 
            }

        inp = (img_crop_rgb.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]  # HxWx3 (BGR) -> 1x3xHxW (RGB!)
        return inp, crop_dct

    def output_process(self, *data):
        out_pts, crop_dct = data
        # The ONNX model is exported with a single output named "output".
        # Thus, out_pts will be a list containing one element: the landmark tensor.
        # Accessing out_pts[0] is correct.
        
        # Assuming the ONNX model outputs landmarks normalized in the range [-1, 1]
        # for both x and y, relative to the center of the cropped image.
        lmk_normalized_minus1_to_1 = out_pts[0].reshape(-1, 2)
        
        # Convert from [-1, 1] range to [0, 1] range
        # A point at -1 (left/top) becomes 0.
        # A point at 0 (center) becomes 0.5.
        # A point at 1 (right/bottom) becomes 1.
        lmk_normalized_0_to_1 = (lmk_normalized_minus1_to_1 + 1.0) / 2.0
        
        # Scale to crop image pixel coordinates [0, dsize]
        # This ensures landmarks are correctly positioned within the dsize x dsize crop,
        # with (0,0) at the top-left of the crop.
        lmk_in_crop_coords = lmk_normalized_0_to_1 * self.dsize
        
        # Transform landmarks from cropped image coordinates back to original image coordinates
        lmk = _transform_pts(lmk_in_crop_coords, M=crop_dct['M_c2o'])
        return lmk

    def predict_trt(self, *data):
        nvtx.range_push("forward")
        feed_dict = {}
        for i, inp in enumerate(self.predictor.inputs):
            if isinstance(data[i], torch.Tensor):
                feed_dict[inp['name']] = data[i]
            else:
                feed_dict[inp['name']] = torch.from_numpy(data[i]).to(device=self.device,
                                                                      dtype=numpy_to_torch_dtype_dict[inp['dtype']])
        preds_dict = self.predictor.predict(feed_dict, self.cudaStream)
        outs = []
        for i, out in enumerate(self.predictor.outputs):
            outs.append(preds_dict[out["name"]].cpu().numpy())
        nvtx.range_pop()
        return outs

    def predict(self, *data):
        input, crop_dct = self.input_process(*data)
        if self.predict_type == "trt":
            preds = self.predict_trt(input)
        else:
            preds = self.predictor.predict(input)
        outputs = self.output_process(preds, crop_dct)
        return outputs
