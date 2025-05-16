# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: run.py

"""
# video
 python run.py \
 --src_image assets/examples/driving/d13.mp4 \
 --dri_video assets/examples/driving/d11.mp4 \
 --cfg configs/trt_infer.yaml \
 --paste_back \
 --animal
# pkl
 python run.py \
 --src_image assets/examples/source/s12.jpg \
 --dri_video ./results/2024-09-13-081710/d0.mp4.pkl \
 --cfg configs/trt_infer.yaml \
 --paste_back \
 --animal
"""
import os
import argparse
import pdb
import subprocess
import ffmpeg
import cv2
import time
import numpy as np
import os
import datetime
import platform
import pickle  # added for preprocessed source data
import collections
import logging
import threading
import torch
from typing import Dict, List, Any, Deque, Optional, Tuple, Union
from omegaconf import OmegaConf
from tqdm import tqdm
from colorama import Fore, Back, Style
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import video_has_audio
from src.utils.crop import prepare_paste_back

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_py")

if platform.system().lower() == 'windows':
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"

# Performance metrics tracking
class PerformanceTracker:
    def __init__(self) -> None:
        self.metrics: Dict[str, Deque[float]] = {
            "decode_time": collections.deque(maxlen=30),
            "process_time": collections.deque(maxlen=30),
            "encode_time": collections.deque(maxlen=30),
            "display_time": collections.deque(maxlen=30),
            "total_time": collections.deque(maxlen=30),
            "frame_sizes": collections.deque(maxlen=30),
        }
        self.frames_received: int = 0
        self.frames_processed: int = 0
        self.last_metrics_log: float = time.time()
        self.metrics_log_interval: float = 5.0  # Log metrics every 5 seconds

    def update_metric(self, metric_name: str, value: float) -> None:
        """Update a performance metric"""
        self.metrics[metric_name].append(value)
        
    def calculate_avg_metrics(self) -> Dict[str, float]:
        """Calculate average metrics for logging"""
        avg_metrics: Dict[str, float] = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
            else:
                avg_metrics[key] = 0
        return avg_metrics
    
    def log_metrics(self) -> None:
        """Log performance metrics if interval has passed"""
        current_time: float = time.time()
        if current_time - self.last_metrics_log >= self.metrics_log_interval:
            self.last_metrics_log = current_time
            avg_metrics: Dict[str, float] = self.calculate_avg_metrics()
            
            # Calculate average FPS based on processing time
            avg_fps: float = 1000 / avg_metrics["total_time"] if avg_metrics["total_time"] > 0 else 0
            
            # Log detailed performance info
            logger.info(f"=== Performance metrics ===")
            logger.info(f"  Decode:    {avg_metrics['decode_time']:.2f}ms ({avg_metrics['decode_time']/avg_metrics['total_time']*100:.1f}%)")
            logger.info(f"  Processing: {avg_metrics['process_time']:.2f}ms ({avg_metrics['process_time']/avg_metrics['total_time']*100:.1f}%)")
            logger.info(f"  Encode:    {avg_metrics['encode_time']:.2f}ms ({avg_metrics['encode_time']/avg_metrics['total_time']*100:.1f}%)")
            logger.info(f"  Display:   {avg_metrics['display_time']:.2f}ms ({avg_metrics['display_time']/avg_metrics['total_time']*100:.1f}%)")
            logger.info(f"  Total Time: {avg_metrics['total_time']:.2f}ms (Theoretical max FPS: {avg_fps:.1f})")
            
            # Log frame counts
            logger.info(f"  Frames: Received {self.frames_received}, Processed {self.frames_processed}")
            
            # Add frame size info
            if self.metrics["frame_sizes"]:
                avg_size = sum(self.metrics["frame_sizes"]) / len(self.metrics["frame_sizes"])
                logger.info(f"  Avg Frame Size: {avg_size/1024:.2f} KB")
            
            return True
        return False

# Initialize performance tracker
performance_tracker = PerformanceTracker()

# Class to manage multiple source images with automatic cycling
class MultiSourceManager:
    def __init__(self, source_images, pipeline, switch_interval=5.0, is_animal=False):
        """
        Initialize the multi-source manager
        
        Args:
            source_images (list): List of paths to source images
            pipeline (FasterLivePortraitPipeline): Reference to the animation pipeline
            switch_interval (float): Time interval in seconds between source switches
            is_animal (bool): Whether to use animal model
        """
        self.source_images = source_images
        self.pipeline = pipeline
        self.switch_interval = switch_interval
        self.is_animal = is_animal
        self.current_index = 0
        self.source_count = len(source_images)
        self.last_switch_time = time.time()
        self.active = False
        self.lock = threading.RLock()  # Thread-safe lock for pipeline access
        
        # Pre-load all sources
        self.source_data = []
        for src_path in source_images:
            logger.info(f"Loading source image: {src_path}")
            try:
                # Clone the pipeline for each source to avoid conflicts
                self.pipeline.prepare_source(src_path, realtime=True)
                self.source_data.append({
                    "path": src_path,
                    "image": self.pipeline.src_imgs[0],
                    "info": self.pipeline.src_infos[0]
                })
                logger.info(f"Successfully loaded source: {src_path}")
            except Exception as e:
                logger.error(f"Error loading source {src_path}: {e}")
        
        # Make sure we have at least one valid source
        if not self.source_data:
            raise ValueError("No valid source images could be loaded")
    
    def start_auto_switching(self):
        """Start the automatic source switching"""
        self.active = True
        self.last_switch_time = time.time()
        
        # Start the auto-switch thread
        self.switch_thread = threading.Thread(target=self._auto_switch_thread, daemon=True)
        self.switch_thread.start()
    
    def stop_auto_switching(self):
        """Stop the automatic source switching"""
        self.active = False
        if hasattr(self, 'switch_thread'):
            self.switch_thread.join(timeout=1.0)
    
    def _auto_switch_thread(self):
        """Background thread to automatically switch sources"""
        while self.active:
            current_time = time.time()
            if current_time - self.last_switch_time >= self.switch_interval:
                self.switch_to_next()
                self.last_switch_time = current_time
            time.sleep(0.1)  # Sleep to avoid high CPU usage
    
    def switch_to_next(self):
        """Switch to the next source in the list"""
        with self.lock:
            self.current_index = (self.current_index + 1) % self.source_count
            logger.info(f"Switching to source {self.current_index + 1}/{self.source_count}: {self.source_data[self.current_index]['path']}")
    
    def get_current_source(self):
        """Get the currently active source data"""
        with self.lock:
            return self.source_data[self.current_index]
    
    def reset_pipeline_state(self):
        """Reset the pipeline state for a clean transition"""
        with self.lock:
            self.pipeline.frame_id = 0
            self.pipeline.R_d_0 = None
            self.pipeline.x_d_0_info = None
            self.pipeline.src_lmk_pre = None

def run_with_video(args):
    # Initialize multi_source_mode to avoid UnboundLocalError when src_data is used
    multi_source_mode = False
    print(Fore.RED+'Render,  Q > exit,  S > Stitching,  Z > RelativeMotion,  X > AnimationRegion,  C > CropDrivingVideo, KL > AdjustSourceScale, NM > AdjustDriverScale,  Space > Webcamassource,  R > SwitchRealtimeWebcamUpdate'+Style.RESET_ALL)
    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = args.paste_back

    # Initialize pipeline
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    # Load preprocessed source if provided
    if args.src_data:
        # Disable pasteback for preprocessed source data
        pipe.cfg.infer_params.flag_pasteback = False
        # Load serialized source data
        with open(args.src_data, 'rb') as f:
            data = pickle.load(f)

        print(str(data['src_infos'][0][0][5]))
        # exit(1)
        # data['src_infos'][0][0][1] = []
        # data['src_infos'][0][0][3] = np.zeros(data['src_infos'][0][0][3].shape[:], dtype=np.float32)
        # data['src_infos'][0][0][5] = np.zeros(data['src_infos'][0][0][5].shape[:], dtype=np.float32)

        pipe.is_source_video = data['is_source_video']
        pipe.src_imgs = data['src_imgs']
        pipe.src_infos = data['src_infos']
        if not pipe.src_infos:
            print(f"No source info in {args.src_data}! exit!")
            exit(1)
        # Use first frame and face info
        src_img = pipe.src_imgs[0]
        src_info = pipe.src_infos[0]
    else:
        # Setup multiple sources if provided
        source_images = [args.src_image]  # Start with the primary source image
        if args.src_image_2:
            source_images.append(args.src_image_2)
        if args.src_image_3:
            source_images.append(args.src_image_3)
        if args.src_image_4:
            source_images.append(args.src_image_4)
        if args.src_image_5:
            source_images.append(args.src_image_5)
        if args.src_image_6:
            source_images.append(args.src_image_6)
        if args.src_image_7:
            source_images.append(args.src_image_7)
        if args.src_image_8:
            source_images.append(args.src_image_8)
        
        # Check if multi-source mode is enabled
        multi_source_mode = len(source_images) > 1 and args.auto_switch
        
        if multi_source_mode:
            # Initialize the multi-source manager with all source images
            source_manager = MultiSourceManager(
                source_images=source_images, 
                pipeline=pipe, 
                switch_interval=args.switch_interval,
                is_animal=args.animal
            )
            
            # Start automatic switching if requested
            if args.auto_switch:
                source_manager.start_auto_switching()
                
            # Use the first source to start
            current_source = source_manager.get_current_source()
            src_img = current_source["image"]
            src_info = current_source["info"]
            last_source_index = source_manager.current_index
        elif not args.src_data:
            # Standard single-source mode
            ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
            if not ret:
                print(f"No face in {args.src_image}! exit!")
                exit(1)
            src_img = pipe.src_imgs[0]
            src_info = pipe.src_infos[0]
    
    if not args.dri_video or not os.path.exists(args.dri_video):
        # read frame from camera if no driving video input
        vcap = cv2.VideoCapture(0)
        if not vcap.isOpened():
            print("no camera found! exit!")
            exit(1)
    else:
        vcap = cv2.VideoCapture(args.dri_video)
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    h, w = src_img.shape[:2]
    save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # render output video
    if not args.realtime:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vsave_crop_path = os.path.join(save_dir,
                                       f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-crop.mp4")
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512 * 2, 512))
        vsave_org_path = os.path.join(save_dir,
                                      f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-org.mp4")
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

    infer_times = []
    motion_lst = []
    c_eyes_lst = []
    c_lip_lst = []

    frame_ind = 0
    while vcap.isOpened():
        # Start timing total frame processing
        total_start_time = time.time()
        
        # Timing for decoding
        decode_start_time = time.time()
        ret, frame = vcap.read()
        decode_time = time.time() - decode_start_time
        performance_tracker.update_metric("decode_time", decode_time * 1000)  # Convert to ms
        
        if not ret:
            break
        
        # Track frame info
        performance_tracker.frames_received += 1
        if frame is not None:
            frame_size = frame.nbytes
            performance_tracker.update_metric("frame_sizes", frame_size)
        
        # Check if we need to switch sources in multi-source mode
        if multi_source_mode:
            current_source = source_manager.get_current_source()
            if source_manager.current_index != last_source_index:
                # Reset pipeline state when switching sources
                source_manager.reset_pipeline_state()
                # Force treating this as a first frame
                frame_ind = 0
                last_source_index = source_manager.current_index
            
            src_img = current_source["image"]
            src_info = current_source["info"]
        
        # Timing for model processing
        process_start_time = time.time()
        first_frame = frame_ind == 0
        dri_crop, out_crop, out_org, dri_motion_info = pipe.run(frame, src_img, src_info,
                                                                first_frame=first_frame)
        process_time = time.time() - process_start_time
        performance_tracker.update_metric("process_time", process_time * 1000)  # Convert to ms
        
        frame_ind += 1
        if out_crop is None:
            print(f"no face in driving frame:{frame_ind}")
            continue

        motion_lst.append(dri_motion_info[0])
        c_eyes_lst.append(dri_motion_info[1])
        c_lip_lst.append(dri_motion_info[2])

        infer_times.append(process_time)
        performance_tracker.frames_processed += 1
        
        # Timing for encoding/post-processing
        encode_start_time = time.time()
        dri_crop = cv2.resize(dri_crop, (512, 512))
        out_crop = np.concatenate([dri_crop, out_crop], axis=1)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        encode_time = time.time() - encode_start_time
        performance_tracker.update_metric("encode_time", encode_time * 1000)  # Convert to ms
        
        # Timing for display or writing
        display_start_time = time.time()
        if not args.realtime:
            vout_crop.write(out_crop)
            out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
            vout_org.write(out_org)
        else:
            if infer_cfg.infer_params.flag_pasteback:
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                
                # Add current source info in multi-source mode
                if multi_source_mode:
                    source_info_text = f"Source: {source_manager.current_index + 1}/{source_manager.source_count}"
                    cv2.putText(out_org, source_info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    # Show source file name
                    source_name = os.path.basename(source_manager.get_current_source()["path"])
                    cv2.putText(out_org, source_name, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow('Render', out_org)
            else:
                # Show source info in multi-source mode
                if multi_source_mode:
                    source_info_text = f"Source: {source_manager.current_index + 1}/{source_manager.source_count}"
                    cv2.putText(out_crop, source_info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    # Show source file name
                    source_name = os.path.basename(source_manager.get_current_source()["path"])
                    cv2.putText(out_crop, source_name, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # image show in realtime mode
                cv2.imshow('Render', out_crop)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # Manual source switching with number keys in multi-source mode
            elif multi_source_mode and key >= ord('1') and key <= ord(str(min(source_manager.source_count, 8))):
                source_idx = key - ord('1')
                if source_idx < source_manager.source_count:
                    source_manager.current_index = source_idx
                    source_manager.last_switch_time = time.time()  # Reset timer
                    logger.info(f"Manually switched to source {source_idx + 1}")
        
        display_time = time.time() - display_start_time
        performance_tracker.update_metric("display_time", display_time * 1000)  # Convert to ms
        
        # Total processing time
        total_time = time.time() - total_start_time
        performance_tracker.update_metric("total_time", total_time * 1000)  # Convert to ms
        
        # Log metrics periodically
        performance_tracker.log_metrics()
        
    vcap.release()
    
    # Stop source switching thread if active
    if multi_source_mode and args.auto_switch:
        source_manager.stop_auto_switching()
    
    if not args.realtime:
        vout_crop.release()
        vout_org.release()
        if video_has_audio(args.dri_video):
            vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_crop_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_crop_path_new, "-y", "-shortest"])
            vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_org_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_org_path_new, "-y", "-shortest"])

            print(vsave_crop_path_new)
            print(vsave_org_path_new)
        else:
            print(vsave_crop_path)
            print(vsave_org_path)
    else:
        cv2.destroyAllWindows()

    # Calculate final stats
    avg_metrics = performance_tracker.calculate_avg_metrics()
    logger.info("=== Final Performance Summary ===")
    logger.info(f"  Processing: {avg_metrics['process_time']:.2f}ms")
    logger.info(f"  Total: {avg_metrics['total_time']:.2f}ms")
    logger.info(f"  Theoretical max FPS: {1000/avg_metrics['total_time']:.2f}")
    logger.info(f"  Frames processed: {performance_tracker.frames_processed}")
    
    print(
        "inference median time: {} ms/frame, mean time: {} ms/frame".format(np.median(infer_times) * 1000,
                                                                            np.mean(infer_times) * 1000))
    # save driving motion to pkl
    template_dct = {
        'n_frames': len(motion_lst),
        'output_fps': fps,
        'motion': motion_lst,
        'c_eyes_lst': c_eyes_lst,
        'c_lip_lst': c_lip_lst,
    }
    template_pkl_path = os.path.join(save_dir,
                                     f"{os.path.basename(args.dri_video)}.pkl")
    with open(template_pkl_path, "wb") as fw:
        pickle.dump(template_dct, fw)
    print(f"save driving motion pkl file at : {template_pkl_path}")


def run_with_pkl(args):
    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = args.paste_back

    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
    if not ret:
        print(f"no face in {args.src_image}! exit!")
        return
    with open(args.dri_video, "rb") as fin:
        dri_motion_infos = pickle.load(fin)

    fps = int(dri_motion_infos["output_fps"])
    h, w = pipe.src_imgs[0].shape[:2]
    save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # render output video
    if not args.realtime:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vsave_crop_path = os.path.join(save_dir,
                                       f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-crop.mp4")
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512, 512))
        vsave_org_path = os.path.join(save_dir,
                                      f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-org.mp4")
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

    infer_times = []
    motion_lst = dri_motion_infos["motion"]
    c_eyes_lst = dri_motion_infos["c_eyes_lst"] if "c_eyes_lst" in dri_motion_infos else dri_motion_infos[
        "c_d_eyes_lst"]
    c_lip_lst = dri_motion_infos["c_lip_lst"] if "c_lip_lst" in dri_motion_infos else dri_motion_infos["c_d_lip_lst"]

    frame_num = len(motion_lst)
    for frame_ind in tqdm(range(frame_num)):
        # Start timing total frame processing
        total_start_time = time.time()
        performance_tracker.frames_received += 1
        
        # Timing for model processing (includes everything for pkl mode)
        process_start_time = time.time()
        first_frame = frame_ind == 0
        dri_motion_info_ = [motion_lst[frame_ind], c_eyes_lst[frame_ind], c_lip_lst[frame_ind]]
        out_crop, out_org = pipe.run_with_pkl(dri_motion_info_, pipe.src_imgs[0], pipe.src_infos[0],
                                              first_frame=first_frame)
        process_time = time.time() - process_start_time
        performance_tracker.update_metric("process_time", process_time * 1000)  # Convert to ms
        
        if out_crop is None:
            print(f"no face in driving frame:{frame_ind}")
            continue
            
        performance_tracker.frames_processed += 1
        infer_times.append(process_time)
        
        # Timing for encoding/post-processing
        encode_start_time = time.time()
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        encode_time = time.time() - encode_start_time
        performance_tracker.update_metric("encode_time", encode_time * 1000)  # Convert to ms
        
        # Timing for display or writing
        display_start_time = time.time()
        if not args.realtime:
            vout_crop.write(out_crop)
            out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
            vout_org.write(out_org)
        else:
            if infer_cfg.infer_params.flag_pasteback:
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                cv2.imshow('Render', out_org)
            else:
                # image show in realtime mode
                cv2.imshow('Render', out_crop)
            
            # Check for key press
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            # Key for Interesting Params    
            if k == ord('s'):
                infer_cfg.infer_params.flag_stitching = not infer_cfg.infer_params.flag_stitching
                print('flag_stitching:'+str(infer_cfg.infer_params.flag_stitching))
            if k == ord('z'):
                infer_cfg.infer_params.flag_relative_motion = not infer_cfg.infer_params.flag_relative_motion
                print('flag_relative_motion:'+str(infer_cfg.infer_params.flag_relative_motion))                
            if k == ord('x'):
                if infer_cfg.infer_params.animation_region == "all": infer_cfg.infer_params.animation_region = "exp", print('animation_region = "exp"')
                else:infer_cfg.infer_params.animation_region = "all", print('animation_region = "all"')
            if k == ord('c'):
                infer_cfg.infer_params.flag_crop_driving_video = not infer_cfg.infer_params.flag_crop_driving_video
                print('flag_crop_driving_video:'+str(infer_cfg.infer_params.flag_crop_driving_video))  
            if k == ord('v'):
                infer_cfg.infer_params.flag_pasteback = not infer_cfg.infer_params.flag_pasteback
                print('flag_pasteback:'+str(infer_cfg.infer_params.flag_pasteback)) 
                
            if k == ord('a'):
                infer_cfg.infer_params.flag_normalize_lip = not infer_cfg.infer_params.flag_normalize_lip
                print('flag_normalize_lip:'+str(infer_cfg.infer_params.flag_normalize_lip))  
            if k == ord('d'):
                infer_cfg.infer_params.flag_source_video_eye_retargeting = not infer_cfg.infer_params.flag_source_video_eye_retargeting
                print('flag_source_video_eye_retargeting:'+str(infer_cfg.infer_params.flag_source_video_eye_retargeting))  
            if k == ord('f'):
                infer_cfg.infer_params.flag_video_editing_head_rotation = not infer_cfg.infer_params.flag_video_editing_head_rotation
                print('flag_video_editing_head_rotation:'+str(infer_cfg.infer_params.flag_video_editing_head_rotation))                 
            if k == ord('g'):
                infer_cfg.infer_params.flag_eye_retargeting = not infer_cfg.infer_params.flag_eye_retargeting
                print('flag_eye_retargeting:'+str(infer_cfg.infer_params.flag_eye_retargeting)) 
                
            if k == ord('k'):
                infer_cfg.crop_params.src_scale -= 0.1
                ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
                print('src_scale:'+str(infer_cfg.crop_params.src_scale))                
            if k == ord('l'):
                infer_cfg.crop_params.src_scale += 0.1
                ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
                print('src_scale:'+str(infer_cfg.crop_params.src_scale))  
            if k == ord('n'):
                infer_cfg.crop_params.dri_scale -= 0.1
                print('dri_scale:'+str(infer_cfg.crop_params.dri_scale))                
            if k == ord('m'):
                infer_cfg.crop_params.dri_scale += 0.1
                print('dri_scale:'+str(infer_cfg.crop_params.dri_scale))

        display_time = time.time() - display_start_time
        performance_tracker.update_metric("display_time", display_time * 1000)  # Convert to ms
        
        # Total processing time
        total_time = time.time() - total_start_time
        performance_tracker.update_metric("total_time", total_time * 1000)  # Convert to ms
        
        # Log metrics periodically
        performance_tracker.log_metrics()

    if not args.realtime:
        vout_crop.release()
        vout_org.release()
        if video_has_audio(args.dri_video):
            vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_crop_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_crop_path_new, "-y", "-shortest"])
            vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_org_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_org_path_new, "-y", "-shortest"])

            print(vsave_crop_path_new)
            print(vsave_org_path_new)
        else:
            print(vsave_crop_path)
            print(vsave_org_path)
    else:
        cv2.destroyAllWindows()

    # Calculate final stats
    avg_metrics = performance_tracker.calculate_avg_metrics()
    logger.info("=== Final Performance Summary ===")
    logger.info(f"  Processing: {avg_metrics['process_time']:.2f}ms")
    logger.info(f"  Total: {avg_metrics['total_time']:.2f}ms")
    logger.info(f"  Theoretical max FPS: {1000/avg_metrics['total_time']:.2f}")
    logger.info(f"  Frames processed: {performance_tracker.frames_processed}")
    
    print(
        "inference median time: {} ms/frame, mean time: {} ms/frame".format(np.median(infer_times) * 1000,
                                                                            np.mean(infer_times) * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster Live Portrait Pipeline')
    parser.add_argument('--src_image', required=False, type=str, default="assets/examples/source/s12.jpg",
                        help='primary source image')
    parser.add_argument('--src_data', type=str, default=None,
                        help='path to preprocessed source data file')  # added
    parser.add_argument('--src_image_2', type=str, default="",
                        help='second source image for auto-switching')
    parser.add_argument('--src_image_3', type=str, default="",
                        help='third source image for auto-switching')
    parser.add_argument('--src_image_4', type=str, default="",
                        help='fourth source image for auto-switching')
    parser.add_argument('--src_image_5', type=str, default="",
                        help='fifth source image for auto-switching')
    parser.add_argument('--src_image_6', type=str, default="",
                        help='sixth source image for auto-switching')
    parser.add_argument('--src_image_7', type=str, default="",
                        help='seventh source image for auto-switching')
    parser.add_argument('--src_image_8', type=str, default="",
                        help='eighth source image for auto-switching')
    parser.add_argument('--auto_switch', action='store_true',
                        help='automatically switch between source images')
    parser.add_argument('--switch_interval', type=float, default=5.0,
                        help='time interval in seconds between source switches')
    parser.add_argument('--dri_video', required=False, type=str, default="assets/examples/driving/d14.mp4",
                        help='driving video')
    parser.add_argument('--cfg', required=False, type=str, default="configs/onnx_infer.yaml", help='inference config')
    parser.add_argument('--realtime', action='store_true', help='realtime inference')
    parser.add_argument('--animal', action='store_true', help='use animal model')
    parser.add_argument('--paste_back', action='store_true', default=False, help='paste back to origin image')
    args, unknown = parser.parse_known_args()

    if args.dri_video.endswith(".pkl"):
        run_with_pkl(args)
    else:
        run_with_video(args)
