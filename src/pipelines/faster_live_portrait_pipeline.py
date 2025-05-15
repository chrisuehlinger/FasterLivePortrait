# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo0611@gmail.com
# @Project : FasterLivePortrait
# @FileName: faster_live_portrait_pipeline.py

import copy
import os.path
import pdb
import time
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Deque, Callable, TypeVar, cast, Protocol, Literal
from typing_extensions import TypedDict
from omegaconf import DictConfig
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
import torch

from .. import models
from ..utils.crop import crop_image, parse_bbox_from_landmark, crop_image_by_bbox, paste_back, paste_back_pytorch
from ..utils.utils import resize_to_limit, prepare_paste_back, get_rotation_matrix, calc_lip_close_ratio, \
    calc_eye_close_ratio, transform_keypoint, concat_feat
from src.utils import utils


# Define protocol for models
class BaseModel(Protocol):
    def forward(self, *args, **kwargs) -> Any:
        ...
        
# Define TypedDict for model dictionary
class ModelDict(TypedDict, total=False):
    """Dictionary of models used in the pipeline"""
    net_recon: BaseModel
    net_distort: BaseModel
    net_disentangle: BaseModel
    xpose: BaseModel
    face_analysis: BaseModel
    landmark: BaseModel
    motion_extractor: BaseModel
    warping_spade: BaseModel
    stitching: BaseModel
    stitching_eye_retarget: BaseModel
    stitching_lip_retarget: BaseModel

# Define TypedDict for motion information
class MotionInfo(TypedDict, total=False):
    """Motion information extracted from frames"""
    pitch: np.ndarray
    yaw: np.ndarray
    roll: np.ndarray
    t: np.ndarray
    exp: np.ndarray
    scale: np.ndarray
    kp: np.ndarray
    R: Optional[np.ndarray]  # Rotation matrix
    R_d: Optional[np.ndarray]  # Alternative name for rotation matrix

# Define TypedDict for source information tuple components
class SourceInfoData(TypedDict, total=False):
    """Information extracted from a source image/frame"""
    x_s_info: MotionInfo
    source_lmk: np.ndarray
    R_s: np.ndarray  # Rotation matrix for source
    f_s: np.ndarray  # Features for source
    x_s: np.ndarray  # Keypoints for source
    x_c_s: np.ndarray  # Canonical keypoints for source
    lip_delta_before_animation: Optional[np.ndarray]
    flag_lip_zero: bool
    mask_ori_float: Optional[torch.Tensor]
    M: torch.Tensor  # Transformation matrix

# Define TypedDict for driving motion information from pickle
class DrivingMotionInfo(TypedDict, total=False):
    """Motion information for driving animation"""
    motion_lst: List[np.ndarray]
    c_eyes_lst: Optional[List[np.ndarray]]
    c_d_eyes_lst: Optional[List[np.ndarray]]
    c_lip_lst: Optional[List[np.ndarray]]
    c_d_lip_lst: Optional[List[np.ndarray]]
    output_fps: int
    face_analysis: BaseModel
    landmark: BaseModel
    motion_extractor: BaseModel
    warping_spade: BaseModel
    stitching: BaseModel
    stitching_eye_retarget: BaseModel
    stitching_lip_retarget: BaseModel

# Define TypedDict for motion information
class MotionInfo(TypedDict, total=False):
    """Motion information extracted from frames"""
    pitch: np.ndarray
    yaw: np.ndarray
    roll: np.ndarray
    t: np.ndarray
    exp: np.ndarray
    scale: np.ndarray
    kp: np.ndarray
    R: Optional[np.ndarray]  # Rotation matrix
    R_d: Optional[np.ndarray]  # Alternative name for rotation matrix

# Define TypedDict for source information tuple components
class SourceInfoData(TypedDict, total=False):
    """Information extracted from a source image/frame"""
    x_s_info: MotionInfo
    source_lmk: np.ndarray
    R_s: np.ndarray  # Rotation matrix for source
    f_s: np.ndarray  # Features for source
    x_s: np.ndarray  # Keypoints for source
    x_c_s: np.ndarray  # Canonical keypoints for source
    lip_delta_before_animation: Optional[np.ndarray]
    flag_lip_zero: bool
    mask_ori_float: Optional[torch.Tensor]
    M: torch.Tensor  # Transformation matrix

# Define TypedDict for driving motion information
class DrivingMotionInfo(TypedDict, total=False):
    """Motion information for driving animation"""
    motion_lst: List[np.ndarray]
    c_eyes_lst: Optional[List[np.ndarray]]
    c_d_eyes_lst: Optional[List[np.ndarray]]
    c_lip_lst: Optional[List[np.ndarray]]
    c_d_lip_lst: Optional[List[np.ndarray]]
    output_fps: int

class FasterLivePortraitPipeline:
    def __init__(self, cfg: DictConfig, **kwargs: Any) -> None:
        self.cfg = cfg
        self.init(**kwargs)
        # Initialize pause_animation flag as False by default
        self.pause_animation: bool = False

    def init(self, **kwargs: Any) -> None:
        self.init_vars(**kwargs)
        self.init_models(**kwargs)

    def update_cfg(self, args_user: Dict[str, Any]) -> bool:
        update_ret: bool = False
        for key in args_user:
            if key in self.cfg.infer_params:
                if self.cfg.infer_params[key] != args_user[key]:
                    update_ret = True
                print("update infer cfg {} from {} to {}".format(key, self.cfg.infer_params[key], args_user[key]))
                self.cfg.infer_params[key] = args_user[key]
            elif key in self.cfg.crop_params:
                if self.cfg.crop_params[key] != args_user[key]:
                    update_ret = True
                print("update crop cfg {} from {} to {}".format(key, self.cfg.crop_params[key], args_user[key]))
                self.cfg.crop_params[key] = args_user[key]
            else:
                if key in self.cfg.infer_params and self.cfg.infer_params[key] != args_user[key]:
                    update_ret = True
                print("add {}:{} to infer cfg".format(key, args_user[key]))
                self.cfg.infer_params[key] = args_user[key]
        return update_ret

    def clean_models(self, **kwargs: Any) -> None:
        """
        clean model
        :param kwargs:
        :return:
        """
        if hasattr(self, 'model_dict'):
            for key in list(self.model_dict.keys()):
                del self.model_dict[key]
        self.model_dict: ModelDict = {}

    def init_models(self, **kwargs: Any) -> None:
        if not kwargs.get("is_animal", False):
            print("load Human Model >>>")
            self.is_animal: bool = False
            self.model_dict: ModelDict = {}
            for model_name in self.cfg.models:
                print(f"loading model: {model_name}")
                print(self.cfg.models[model_name])
                self.model_dict[model_name] = getattr(models, self.cfg.models[model_name]["name"])(
                    **self.cfg.models[model_name])
        else:
            print("load Animal Model >>>")
            self.is_animal: bool = True
            self.model_dict: ModelDict = {}
            from src.utils.animal_landmark_runner import XPoseRunner
            from src.utils.utils import make_abs_path
            checkpoint_dir: Optional[str] = None
            for model_name in self.cfg.animal_models:
                print(f"loading model: {model_name}")
                print(self.cfg.animal_models[model_name])
                if checkpoint_dir is None and isinstance(self.cfg.animal_models[model_name].model_path, str):
                    checkpoint_dir = os.path.dirname(self.cfg.animal_models[model_name].model_path)
                self.model_dict[model_name] = getattr(models, self.cfg.animal_models[model_name]["name"])(
                    **self.cfg.animal_models[model_name])

            xpose_config_file_path: str = make_abs_path("models/XPose/config_model/UniPose_SwinT.py")
            xpose_ckpt_path: str = os.path.join(checkpoint_dir, "xpose.pth")
            xpose_embedding_cache_path: str = os.path.join(checkpoint_dir, 'clip_embedding')
            self.model_dict["xpose"] = XPoseRunner(model_config_path=xpose_config_file_path,
                                                   model_checkpoint_path=xpose_ckpt_path,
                                                   embeddings_cache_path=xpose_embedding_cache_path,
                                                   flag_use_half_precision=True)

    def init_vars(self, **kwargs: Any) -> None:
        """Initialize pipeline variables"""
        self.mask_crop: np.ndarray = cv2.imread(self.cfg.infer_params.mask_crop_path, cv2.IMREAD_COLOR)
        self.frame_id: int = 0
        self.src_lmk_pre: Optional[np.ndarray] = None
        self.R_d_0: Optional[np.ndarray] = None
        self.x_d_0_info: Optional[Dict[str, Any]] = None
        self.R_d_smooth = utils.OneEuroFilter(4, 0.3)
        self.exp_smooth = utils.OneEuroFilter(4, 0.3)

        # Source information
        self.source_path: Optional[str] = None
        self.src_infos: List[Dict[str, Any]] = []
        self.src_imgs: List[np.ndarray] = []
        self.is_source_video: bool = False
        self.device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def calc_combined_eye_ratio(self, c_d_eyes_i: List[List[float]], source_lmk: np.ndarray) -> np.ndarray:
        """
        Calculate combined eye ratio tensor for retargeting
        
        Parameters:
        -----------
        c_d_eyes_i: List[List[float]]
            Target eye ratio values
        source_lmk: np.ndarray
            Source landmarks
            
        Returns:
        --------
        np.ndarray:
            Combined eye ratio tensor with shape (1, 2)
        """
        c_s_eyes = calc_eye_close_ratio(source_lmk[None])
        c_d_eyes_i = np.array(c_d_eyes_i).reshape(1, 1)
        # [c_s,eyes, c_d,eyes,i]
        combined_eye_ratio_tensor = np.concatenate([c_s_eyes, c_d_eyes_i], axis=1)
        return combined_eye_ratio_tensor

    def calc_combined_lip_ratio(self, c_d_lip_i: Union[List[float], float], source_lmk: np.ndarray) -> np.ndarray:
        """
        Calculate combined lip ratio tensor for retargeting
        
        Parameters:
        -----------
        c_d_lip_i: Union[List[float], float]
            Target lip ratio values
        source_lmk: np.ndarray
            Source landmarks
            
        Returns:
        --------
        np.ndarray:
            Combined lip ratio tensor with shape (1, 2)
        """
        c_s_lip = calc_lip_close_ratio(source_lmk[None])
        c_d_lip_i = np.array(c_d_lip_i).reshape(1, 1)  # 1x1
        # [c_s,lip, c_d,lip,i]
        combined_lip_ratio_tensor = np.concatenate([c_s_lip, c_d_lip_i], axis=1)  # 1x2
        return combined_lip_ratio_tensor

    def prepare_source(self, source_path: str, **kwargs: Any) -> bool:
        print(f"process source:{source_path} >>>>>>>>")
        try:
            # Check if the source is an SPD file
            from src.utils.spd_utils import is_spd_file, load_spd_file
            
            if is_spd_file(source_path):
                # Load SPD file directly, skipping face detection and analysis
                print(f"Loading SPD file: {source_path}")
                self.src_imgs, self.src_infos, self.is_source_video = load_spd_file(source_path, self.device, self.cfg)
                self.source_path = source_path
                print(f"Successfully loaded SPD file: {source_path}")
                return len(self.src_infos) > 0

            # Regular image or video processing path
            if utils.is_video(source_path):
                self.is_source_video = True
            else:
                self.is_source_video = False

            if self.is_source_video:
                src_imgs_bgr = []
                src_vcap = cv2.VideoCapture(source_path)
                while True:
                    ret, frame = src_vcap.read()
                    if not ret:
                        break
                    src_imgs_bgr.append(frame)
                src_vcap.release()
            else:
                img_bgr = cv2.imread(source_path, cv2.IMREAD_COLOR)
                src_imgs_bgr = [img_bgr]

            self.src_imgs = []
            self.src_infos = []
            self.source_path = source_path

            for ii, img_bgr in tqdm(enumerate(src_imgs_bgr), total=len(src_imgs_bgr)):
                img_bgr = resize_to_limit(img_bgr, self.cfg.infer_params.source_max_dim,
                                          self.cfg.infer_params.source_division)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                src_faces = []
                if self.is_animal:
                    with torch.no_grad():
                        img_rgb_pil = Image.fromarray(img_rgb)
                        lmk = self.model_dict["xpose"].run(
                            img_rgb_pil,
                            'face',
                            'animal_face',
                            0,
                            0
                        )
                    if lmk is None:
                        continue
                    self.src_imgs.append(img_rgb)
                    src_faces.append(lmk)
                else:
                    src_faces = self.model_dict["face_analysis"].predict(img_bgr)
                    if len(src_faces) == 0:
                        print("No face detected in the this image.")
                        continue
                    self.src_imgs.append(img_rgb)
                    # 如果是实时，只关注最大的那张脸
                    if kwargs.get("realtime", False):
                        src_faces = src_faces[:1]

                crop_infos = []
                for i in range(len(src_faces)):
                    # NOTE: temporarily only pick the first face, to support multiple face in the future
                    lmk = src_faces[i]
                    # crop the face
                    ret_dct = crop_image(
                        img_rgb,  # ndarray
                        lmk,  # 106x2 or Nx2
                        dsize=self.cfg.crop_params.src_dsize,
                        scale=self.cfg.crop_params.src_scale,
                        vx_ratio=self.cfg.crop_params.src_vx_ratio,
                        vy_ratio=self.cfg.crop_params.src_vy_ratio,
                    )
                    if self.is_animal:
                        ret_dct["lmk_crop"] = lmk
                    else:
                        lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                        ret_dct["lmk_crop"] = lmk
                        ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / self.cfg.crop_params.src_dsize

                    # update a 256x256 version for network input
                    ret_dct["img_crop_256x256"] = cv2.resize(
                        ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA
                    )
                    crop_infos.append(ret_dct)

                src_infos = [[] for _ in range(len(crop_infos))]
                for i, crop_info in enumerate(crop_infos):
                    source_lmk = crop_info['lmk_crop']
                    img_crop, img_crop_256x256 = crop_info['img_crop'], crop_info['img_crop_256x256']
                    pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(
                        img_crop_256x256)
                    x_s_info = {
                        "pitch": pitch,
                        "yaw": yaw,
                        "roll": roll,
                        "t": t,
                        "exp": exp,
                        "scale": scale,
                        "kp": kp
                    }
                    src_infos[i].append(copy.deepcopy(x_s_info))
                    x_c_s = kp
                    R_s = get_rotation_matrix(pitch, yaw, roll)
                    f_s = self.model_dict["app_feat_extractor"].predict(img_crop_256x256)
                    x_s = transform_keypoint(pitch, yaw, roll, t, exp, scale, kp)
                    src_infos[i].extend([source_lmk.copy(), R_s.copy(), f_s.copy(), x_s.copy(), x_c_s.copy()])
                    if not self.is_animal:
                        flag_lip_zero = self.cfg.infer_params.flag_normalize_lip  # not overwrite
                        if flag_lip_zero:
                            # let lip-open scalar to be 0 at first
                            # 似乎要调参？
                            c_d_lip_before_animation = [0.05]
                            combined_lip_ratio_tensor_before_animation = self.calc_combined_lip_ratio(
                                c_d_lip_before_animation, source_lmk.copy())
                            if combined_lip_ratio_tensor_before_animation[0][
                                0] < self.cfg.infer_params.lip_normalize_threshold:
                                flag_lip_zero = False
                                src_infos[i].append(None)
                                src_infos[i].append(flag_lip_zero)
                            else:
                                lip_delta_before_animation = self.model_dict['stitching_lip_retarget'].predict(
                                    concat_feat(x_s, combined_lip_ratio_tensor_before_animation))
                                src_infos[i].append(lip_delta_before_animation.copy())
                                src_infos[i].append(flag_lip_zero)
                        else:
                            src_infos[i].append(None)
                            src_infos[i].append(flag_lip_zero)
                    else:
                        src_infos[i].append(None)
                        src_infos[i].append(False)

                    ######## prepare for pasteback ########
                    if self.cfg.infer_params.flag_pasteback and self.cfg.infer_params.flag_do_crop and self.cfg.infer_params.flag_stitching:
                        mask_ori_float = prepare_paste_back(self.mask_crop, crop_info['M_c2o'],
                                                            dsize=(img_rgb.shape[1], img_rgb.shape[0]))
                        mask_ori_float = torch.from_numpy(mask_ori_float).to(self.device)
                        src_infos[i].append(mask_ori_float)
                    else:
                        src_infos[i].append(None)
                    M = torch.from_numpy(crop_info['M_c2o']).to(self.device)
                    src_infos[i].append(M)
                self.src_infos.append(src_infos[:])
                
            # Export to SPD if requested
            if kwargs.get("export_spd", None):
                from src.utils.spd_utils import save_spd_file
                spd_path = kwargs.get("export_spd")
                print(f"Exporting to SPD file: {spd_path}")
                spd_path = save_spd_file(spd_path, self.src_imgs, self.src_infos, self.is_source_video)
                print(f"Successfully exported to SPD file: {spd_path}")
                
            print(f"finish process source:{source_path} >>>>>>>>")
            return len(self.src_infos) > 0
        except Exception as e:
            traceback.print_exc()
            return False

    def retarget_eye(self, kp_source: np.ndarray, eye_close_ratio: np.ndarray) -> np.ndarray:
        """
        Retarget eye expressions
        
        Parameters:
        -----------
        kp_source: np.ndarray
            Source keypoints with shape (batch_size, num_keypoints, 3)
        eye_close_ratio: np.ndarray
            Eye close ratio tensor with shape (batch_size, 2) containing source and target ratios
            
        Returns:
        --------
        np.ndarray:
            Delta values for eye expressions
        """
        feat_eye = concat_feat(kp_source, eye_close_ratio)
        delta = self.model_dict['stitching_eye_retarget'].predict(feat_eye)
        return delta

    def retarget_lip(self, kp_source: np.ndarray, lip_close_ratio: np.ndarray) -> np.ndarray:
        """
        Retarget lip expressions
        
        Parameters:
        -----------
        kp_source: np.ndarray
            Source keypoints with shape (batch_size, num_keypoints, 3)
        lip_close_ratio: np.ndarray
            Lip close ratio tensor with shape (batch_size, 2) containing source and target ratios
            
        Returns:
        --------
        np.ndarray:
            Delta values for lip expressions
        """
        feat_lip = concat_feat(kp_source, lip_close_ratio)
        delta = self.model_dict['stitching_lip_retarget'].predict(feat_lip)
        return delta

    def stitching(self, kp_source: np.ndarray, kp_driving: np.ndarray) -> np.ndarray:
        """ 
        Conduct the stitching between source and driving keypoints
        
        Parameters:
        -----------
        kp_source: np.ndarray
            Source keypoints with shape (batch_size, num_keypoints, 3)
        kp_driving: np.ndarray
            Driving keypoints with shape (batch_size, num_keypoints, 3)
            
        Returns:
        --------
        np.ndarray:
            New driving keypoints after stitching
        """
        bs, num_kp = kp_source.shape[:2]

        kp_driving_new = kp_driving.copy()

        delta = self.model_dict['stitching'].predict(concat_feat(kp_source, kp_driving_new))

        delta_exp = delta[..., :3 * num_kp].reshape(bs, num_kp, 3)  # 1x20x3
        delta_tx_ty = delta[..., 3 * num_kp:3 * num_kp + 2].reshape(bs, 1, 2)  # 1x1x2

        kp_driving_new += delta_exp
        kp_driving_new[..., :2] += delta_tx_ty

        return kp_driving_new

    def _run(self, 
             src_info: List[List], 
             x_d_i_info: MotionInfo, 
             x_d_0_info: MotionInfo, 
             R_d_i: np.ndarray, 
             R_d_0: np.ndarray, 
             realtime: bool, 
             input_eye_ratio: np.ndarray, 
             input_lip_ratio: np.ndarray,
             I_p_pstbk: torch.Tensor, 
             **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main pipeline processing function
        
        Parameters:
        -----------
        src_info: List[List]
            List of source information for each keypoint
        x_d_i_info: MotionInfo
            Motion information for current driving frame
        x_d_0_info: MotionInfo
            Motion information for reference driving frame
        R_d_i: np.ndarray
            Rotation matrix for current driving frame
        R_d_0: np.ndarray
            Rotation matrix for reference driving frame
        realtime: bool
            Whether processing in realtime mode
        input_eye_ratio: np.ndarray
            Eye ratio for current driving frame
        input_lip_ratio: np.ndarray
            Lip ratio for current driving frame
        I_p_pstbk: torch.Tensor
            Pasteback image
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]:
            Output crop and pasteback images
        """
        out_crop, out_org = None, None
        eye_delta_before_animation = None
        for j in range(len(src_info)):
            if self.is_source_video:
                x_s_info, source_lmk, R_s, f_s, x_s, x_c_s, lip_delta_before_animation, flag_lip_zero, mask_ori_float, M = \
                    src_info[j]
                # let lip-open scalar to be 0 at first if the input is a video and flag_relative_motion
                if not (self.cfg.infer_params.flag_normalize_lip and self.cfg.infer_params.flag_relative_motion):
                    lip_delta_before_animation = None
                # let eye-open scalar to be the same as the first frame if the latter is eye-open state
                if self.cfg.infer_params.flag_source_video_eye_retargeting and source_lmk is not None:
                    combined_eye_ratio_tensor_frame_zero = utils.calc_eye_close_ratio(src_info[0][1])
                    c_d_eye_before_animation_frame_zero = [
                        [combined_eye_ratio_tensor_frame_zero[0][:2].mean()]]
                    if c_d_eye_before_animation_frame_zero[0][
                        0] < self.cfg.infer_params.source_video_eye_retargeting_threshold:
                        c_d_eye_before_animation_frame_zero = [[0.39]]
                    combined_eye_ratio_tensor_before_animation = self.calc_combined_eye_ratio(
                        c_d_eye_before_animation_frame_zero, source_lmk)
                    eye_delta_before_animation = self.retarget_eye(x_s, combined_eye_ratio_tensor_before_animation)

                if not realtime and self.cfg.infer_params.flag_pasteback and self.cfg.infer_params.flag_do_crop and \
                        self.cfg.infer_params.flag_stitching:
                    mask_ori_float = prepare_paste_back(self.mask_crop, M.cpu().numpy(),
                                                        dsize=(self.src_imgs[0].shape[1], self.src_imgs[0].shape[0]))
                    mask_ori_float = torch.from_numpy(mask_ori_float).to(self.device)
            else:
                x_s_info, source_lmk, R_s, f_s, x_s, x_c_s, lip_delta_before_animation, flag_lip_zero, mask_ori_float, M = \
                    src_info[j]
            if self.cfg.infer_params.flag_relative_motion:
                if self.cfg.infer_params.animation_region in ["all", "pose"]:
                    if self.is_source_video:
                        R_new = self.R_d_smooth.process(R_d_i)
                    else:
                        R_new = (R_d_i @ np.transpose(R_d_0, (0, 2, 1))) @ R_s
                else:
                    R_new = R_s

                delta_new = x_s_info['exp'].copy()
                x_d_exp_smooth = x_d_i_info['exp'].copy()
                if self.is_source_video:
                    x_d_exp_smooth = self.exp_smooth.process(x_d_exp_smooth)
                if self.cfg.infer_params.animation_region in ["all", "exp"]:
                    if self.is_source_video:
                        for idx in [1, 2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
                            delta_new[:, idx, :] = x_d_exp_smooth[:, idx, :]
                        delta_new[:, 3:5, 1] = x_d_exp_smooth[:, 3:5, 1]
                        delta_new[:, 5, 2] = x_d_exp_smooth[:, 5, 2]
                        delta_new[:, 8, 2] = x_d_exp_smooth[:, 8, 2]
                        delta_new[:, 9, 1:] = x_d_exp_smooth[:, 9, 1:]
                    else:
                        delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                elif self.cfg.infer_params.animation_region in ["lip"]:
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        if self.is_source_video:
                            delta_new[:, lip_idx, :] = x_d_exp_smooth[:, lip_idx, :]
                        else:
                            delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:,
                                                       lip_idx, :]
                elif self.cfg.infer_params.animation_region in ["eyes"]:
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        if self.is_source_video:
                            delta_new[:, eyes_idx, :] = x_d_exp_smooth[:, eyes_idx, :]
                        else:
                            delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:,
                                                        eyes_idx, :]
                if self.cfg.infer_params.animation_region in ["all"]:
                    scale_new = x_s_info['scale'] if self.is_source_video else x_s_info['scale'] * (
                            x_d_i_info['scale'] / x_d_0_info['scale'])
                else:
                    scale_new = x_s_info['scale']
                if self.cfg.infer_params.animation_region in ["all"]:
                    t_new = x_s_info['t'] if self.is_source_video else x_s_info['t'] + (
                            x_d_i_info['t'] - x_d_0_info['t'])
                else:
                    t_new = x_s_info['t']
            else:
                if self.cfg.infer_params.animation_region in ["all", "pose"]:
                    if self.is_source_video:
                        R_new = self.R_d_smooth.process(R_d_i)
                    else:
                        R_new = R_d_i
                else:
                    R_new = R_s

                delta_new = x_s_info['exp'].copy()
                x_d_exp_smooth = x_d_i_info['exp'].copy()
                if self.is_source_video:
                    x_d_exp_smooth = self.exp_smooth.process(x_d_exp_smooth)
                if self.cfg.infer_params.animation_region in ["all", "exp"]:
                    for idx in [1, 2,  11, 13, 15, 16, 18]:
                        delta_new[:, idx, :] = x_d_exp_smooth[:, idx, :] if self.is_source_video else (5 if self.is_animal else 2)*x_d_i_info['exp'][
                                                                                                      :, idx, :]
                    for idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, idx, :] = x_d_exp_smooth[:, idx, :] if self.is_source_video else (10 if self.is_animal else 3)*x_d_i_info['exp'][
                                                                                                      :, idx, :]
                    delta_new[:, 3:5, 1] = x_d_exp_smooth[:, 3:5, 1] if self.is_source_video else x_d_i_info['exp'][:,
                                                                                                  3:5, 1]
                    delta_new[:, 5, 2] = x_d_exp_smooth[:, 5, 2] if self.is_source_video else x_d_i_info['exp'][:,
                                                                                              5, 2]
                    delta_new[:, 8, 2] = x_d_exp_smooth[:, 8, 2] if self.is_source_video else x_d_i_info['exp'][:,
                                                                                              8, 2]
                    delta_new[:, 9, 1:] = x_d_exp_smooth[:, 9, 1:] if self.is_source_video else x_d_i_info['exp'][:,
                                                                                                9, 1:]
                elif self.cfg.infer_params.animation_region in ["lip"]:
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_exp_smooth[:, lip_idx, :] if self.is_source_video else \
                            x_d_i_info['exp'][:, lip_idx, :]
                elif self.cfg.infer_params.animation_region in ["eyes"]:
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_exp_smooth[:, eyes_idx, :] if self.is_source_video else \
                            x_d_i_info['exp'][:, eyes_idx, :]
                scale_new = x_s_info['scale'].copy()
                if self.cfg.infer_params.animation_region in ["all", "pose"]:
                    t_new = x_d_i_info['t'].copy()
                else:
                    t_new = x_s_info['t'].copy()

            t_new[..., 2] = 0  # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
            if not self.is_animal:
                # Algorithm 1:
                if not self.cfg.infer_params.flag_stitching and not self.cfg.infer_params.flag_eye_retargeting and not self.cfg.infer_params.flag_lip_retargeting:
                    # without stitching or retargeting
                    if flag_lip_zero and lip_delta_before_animation is not None:
                        x_d_i_new += lip_delta_before_animation.reshape(-1, x_s.shape[1], 3)
                    if self.cfg.infer_params.flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                        x_d_i_new += eye_delta_before_animation
                elif self.cfg.infer_params.flag_stitching and not self.cfg.infer_params.flag_eye_retargeting and not self.cfg.infer_params.flag_lip_retargeting:
                    # with stitching and without retargeting
                    if flag_lip_zero and lip_delta_before_animation is not None:
                        x_d_i_new = self.stitching(x_s, x_d_i_new) + lip_delta_before_animation.reshape(
                            -1, x_s.shape[1], 3)
                    else:
                        x_d_i_new = self.stitching(x_s, x_d_i_new)
                    if self.cfg.infer_params.flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                        x_d_i_new += eye_delta_before_animation
                else:
                    eyes_delta, lip_delta = None, None
                    if self.cfg.infer_params.flag_eye_retargeting:
                        c_d_eyes_i = input_eye_ratio
                        combined_eye_ratio_tensor = self.calc_combined_eye_ratio(c_d_eyes_i,
                                                                                 source_lmk)
                        # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                        eyes_delta = self.retarget_eye(x_s, combined_eye_ratio_tensor)
                    if self.cfg.infer_params.flag_lip_retargeting:
                        c_d_lip_i = input_lip_ratio
                        combined_lip_ratio_tensor = self.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                        # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                        lip_delta = self.retarget_lip(x_s, combined_lip_ratio_tensor)

                    if self.cfg.infer_params.flag_relative_motion:  # use x_s
                        x_d_i_new = x_s + \
                                    (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                                    (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)
                    else:  # use x_d,i
                        x_d_i_new = x_d_i_new + \
                                    (eyes_delta.reshape(-1, x_s.shape[1], 3) if eyes_delta is not None else 0) + \
                                    (lip_delta.reshape(-1, x_s.shape[1], 3) if lip_delta is not None else 0)

                    if self.cfg.infer_params.flag_stitching:
                        x_d_i_new = self.stitching(x_s, x_d_i_new)
            else:
                if self.cfg.infer_params.flag_stitching:
                    x_d_i_new = self.stitching(x_s, x_d_i_new)

            x_d_i_new = x_s + (x_d_i_new - x_s) * self.cfg.infer_params.driving_multiplier
            out_crop = self.model_dict["warping_spade"].predict(f_s, x_s, x_d_i_new)
            if not realtime and self.cfg.infer_params.flag_pasteback and self.cfg.infer_params.flag_do_crop and self.cfg.infer_params.flag_stitching:
                # TODO: pasteback is slow, considering optimize it using multi-threading or GPU
                # I_p_pstbk = paste_back(out_crop, crop_info['M_c2o'], I_p_pstbk, mask_ori_float)
                I_p_pstbk = paste_back_pytorch(out_crop, M, I_p_pstbk, mask_ori_float)
        return out_crop.to(dtype=torch.uint8).cpu().numpy(), I_p_pstbk.to(dtype=torch.uint8).cpu().numpy()

    def run(self, 
            image: np.ndarray, 
            img_src: np.ndarray, 
            src_info: List[List], 
            **kwargs: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple]]:
        """
        Process a driving frame
        
        Parameters:
        -----------
        image: np.ndarray
            Driving frame 
        img_src: np.ndarray
            Source image
        src_info: List[List]
            Source information for keypoint manipulation
        
        Returns:
        --------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple]]:
            Tuple containing (driving crop, output crop, output full, motion info)
        """
        # Check if animation is paused - return source image if paused
        if self.pause_animation and img_src is not None:
            # If paused, return the source image without modification
            # We'll still need to format the return values to match the expected structure

            # Get original dimensions for proper return formatting
            h, w = img_src.shape[:2]
            if self.cfg.infer_params.flag_do_crop:
                # Return placeholder for driving crop since we're paused
                dri_crop = np.zeros((512, 512, 3), dtype=np.uint8)
                # Return unmodified source cropped to 512x512
                src_crop = cv2.resize(img_src, (512, 512))
                # Return unmodified source image
                return dri_crop, src_crop, img_src, (None, None, None)
            else:
                # Similar return structure but with full-sized image
                return None, img_src, img_src, (None, None, None)

        # If not paused, continue with regular processing
        img_bgr = image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        I_p_pstbk = torch.from_numpy(img_src).to(self.device).float()
        realtime: bool = kwargs.get("realtime", False)
        if self.cfg.infer_params.flag_crop_driving_video:
            # if self.src_lmk_pre is None:
            src_face = self.model_dict["face_analysis"].predict(img_bgr)
            if len(src_face) == 0:
                return None, None, None, None
            lmk = src_face[0]
            lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
            self.src_lmk_pre = lmk.copy()
            # else:
            #     lmk = self.model_dict["landmark"].predict(img_rgb, self.src_lmk_pre)
            #     self.src_lmk_pre = lmk.copy()

            ret_bbox = parse_bbox_from_landmark(
                lmk,
                scale=self.cfg.crop_params.dri_scale,
                vx_ratio_crop_video=self.cfg.crop_params.dri_vx_ratio,
                vy_ratio=self.cfg.crop_params.dri_vy_ratio,
            )["bbox"]
            global_bbox = [
                ret_bbox[0, 0],
                ret_bbox[0, 1],
                ret_bbox[2, 0],
                ret_bbox[2, 1],
            ]
            ret_dct = crop_image_by_bbox(
                img_rgb,
                global_bbox,
                lmk=lmk,
                dsize=kwargs.get("dsize", 512),
                flag_rot=False,
                borderValue=(0, 0, 0),
            )
            lmk_crop = ret_dct["lmk_crop"]
            img_crop = ret_dct["img_crop"]
            img_crop = cv2.resize(img_crop, (256, 256))
        else:
            if self.src_lmk_pre is None:
                src_face = self.model_dict["face_analysis"].predict(img_bgr)
                if len(src_face) == 0:
                    return None, None, None, None
                lmk = src_face[0]
                lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                self.src_lmk_pre = lmk.copy()
            else:
                lmk = self.model_dict["landmark"].predict(img_rgb, self.src_lmk_pre)
                self.src_lmk_pre = lmk.copy()
            lmk_crop = lmk.copy()
            img_crop = cv2.resize(img_rgb, (256, 256))

        input_eye_ratio = calc_eye_close_ratio(lmk_crop[None])
        input_lip_ratio = calc_lip_close_ratio(lmk_crop[None])
        pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(img_crop)
        x_d_i_info = {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "t": t,
            "exp": exp,
            "scale": scale,
            "kp": kp
        }
        R_d_i = get_rotation_matrix(pitch, yaw, roll)
        x_d_i_info["R"] = R_d_i
        x_d_i_info_copy = copy.deepcopy(x_d_i_info)
        for key in x_d_i_info_copy:
            x_d_i_info_copy[key] = x_d_i_info_copy[key].astype(np.float32)
        dri_motion_info = [x_d_i_info_copy, copy.deepcopy(input_eye_ratio.astype(np.float32)),
                           copy.deepcopy(input_lip_ratio.astype(np.float32))]
        if kwargs.get("first_frame", False) or self.R_d_0 is None:
            self.frame_id = 0
            self.R_d_0 = R_d_i.copy()
            self.x_d_0_info = copy.deepcopy(x_d_i_info)
            # realtime smooth
            self.R_d_smooth = utils.OneEuroFilter(4, 0.3)
            self.exp_smooth = utils.OneEuroFilter(4, 0.3)
        R_d_0 = self.R_d_0.copy()
        x_d_0_info = copy.deepcopy(self.x_d_0_info)
        out_crop, I_p_pstbk = self._run(src_info, x_d_i_info, x_d_0_info, R_d_i, R_d_0, realtime, input_eye_ratio,
                                        input_lip_ratio,
                                        I_p_pstbk, **kwargs)
        return img_crop, out_crop, I_p_pstbk, dri_motion_info

    def run_with_pkl(self, 
                 dri_motion_info: List[Any], 
                 img_src: np.ndarray, 
                 src_info: List[List], 
                 **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process frames using pre-computed motion info from a pickle file
        
        Parameters:
        -----------
        dri_motion_info: List[Any]
            List containing motion information from pickle file
            [0] - x_d_i_info: MotionInfo - Motion information
            [1] - input_eye_ratio: np.ndarray - Eye ratio
            [2] - input_lip_ratio: np.ndarray - Lip ratio
        img_src: np.ndarray
            Source image
        src_info: List[List]
            Source information for keypoint manipulation
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]:
            Output crop and pasteback images
        """
        # Check if animation is paused - return source image if paused
        if self.pause_animation and img_src is not None:
            # If paused, return the source image without modification
            h, w = img_src.shape[:2]

            # Return unmodified source cropped to 512x512 for crop output
            src_crop = cv2.resize(img_src, (512, 512))

            # Return unmodified source image for both outputs
            return src_crop, img_src

        # If not paused, continue with regular processing
        I_p_pstbk = torch.from_numpy(img_src).to(self.device).float()
        realtime: bool = kwargs.get("realtime", False)

        input_eye_ratio = dri_motion_info[1]
        input_lip_ratio = dri_motion_info[2]
        x_d_i_info = dri_motion_info[0]
        R_d_i = x_d_i_info["R"] if "R" in x_d_i_info else x_d_i_info["R_d"]

        if kwargs.get("first_frame", False) or self.R_d_0 is None:
            self.frame_id = 0
            self.R_d_0 = R_d_i.copy()
            self.x_d_0_info = copy.deepcopy(x_d_i_info)
            # realtime smooth
            self.R_d_smooth = utils.OneEuroFilter(4, 0.3)
            self.exp_smooth = utils.OneEuroFilter(4, 0.3)
        R_d_0 = self.R_d_0.copy()
        x_d_0_info = copy.deepcopy(self.x_d_0_info)
        out_crop, I_p_pstbk = self._run(src_info, x_d_i_info, x_d_0_info, R_d_i, R_d_0, realtime, input_eye_ratio,
                                        input_lip_ratio, I_p_pstbk, **kwargs)
        return out_crop, I_p_pstbk

    def __del__(self):
        self.clean_models()
