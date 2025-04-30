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
import pickle
import collections
import logging
from omegaconf import OmegaConf
from tqdm import tqdm
from colorama import Fore, Back, Style
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import video_has_audio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_py")

if platform.system().lower() == 'windows':
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"

# Performance metrics tracking
class PerformanceTracker:
    def __init__(self):
        self.metrics = {
            "decode_time": collections.deque(maxlen=30),
            "process_time": collections.deque(maxlen=30),
            "encode_time": collections.deque(maxlen=30),
            "display_time": collections.deque(maxlen=30),
            "total_time": collections.deque(maxlen=30),
            "frame_sizes": collections.deque(maxlen=30),
        }
        self.frames_received = 0
        self.frames_processed = 0
        self.last_metrics_log = time.time()
        self.metrics_log_interval = 5.0  # Log metrics every 5 seconds

    def update_metric(self, metric_name, value):
        """Update a performance metric"""
        self.metrics[metric_name].append(value)
        
    def calculate_avg_metrics(self):
        """Calculate average metrics for logging"""
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
            else:
                avg_metrics[key] = 0
        return avg_metrics
    
    def log_metrics(self):
        """Log performance metrics if interval has passed"""
        current_time = time.time()
        if current_time - self.last_metrics_log >= self.metrics_log_interval:
            self.last_metrics_log = current_time
            avg_metrics = self.calculate_avg_metrics()
            
            # Calculate average FPS based on processing time
            avg_fps = 1000 / avg_metrics["total_time"] if avg_metrics["total_time"] > 0 else 0
            
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

def run_with_video(args):
    print(Fore.RED+'Render,  Q > exit,  S > Stitching,  Z > RelativeMotion,  X > AnimationRegion,  C > CropDrivingVideo, KL > AdjustSourceScale, NM > AdjustDriverScale,  Space > Webcamassource,  R > SwitchRealtimeWebcamUpdate'+Style.RESET_ALL)
    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = args.paste_back

    # Load custom landmarks if provided
    custom_landmarks = None
    if args.custom_landmarks:
        if not os.path.exists(args.custom_landmarks):
            print(f"Custom landmarks file {args.custom_landmarks} doesn't exist! Exiting.")
            exit(1)
        print(f"Loading custom landmarks from {args.custom_landmarks}")
        if args.custom_landmarks.endswith('.npy'):
            custom_landmarks = np.load(args.custom_landmarks)
        elif args.custom_landmarks.endswith('.pkl'):
            with open(args.custom_landmarks, 'rb') as f:
                custom_landmarks = pickle.load(f)
        else:
            print(f"Unsupported landmarks file format. Please use .npy or .pkl. Exiting.")
            exit(1)
        print(f"Custom landmarks loaded with shape: {custom_landmarks.shape if hasattr(custom_landmarks, 'shape') else 'unknown'}")

    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    ret = pipe.prepare_source(args.src_image, custom_landmarks=custom_landmarks, realtime=args.realtime, no_crop=args.no_crop)
    if not ret:
        print(f"no face in {args.src_image}! exit!")
        exit(1)
    if not args.dri_video or not os.path.exists(args.dri_video):
        # read frame from camera if no driving video input
        vcap = cv2.VideoCapture(0)
        if not vcap.isOpened():
            print("no camera found! exit!")
            exit(1)
    else:
        vcap = cv2.VideoCapture(args.dri_video)
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    h, w = pipe.src_imgs[0].shape[:2]
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
        
        # Timing for model processing
        process_start_time = time.time()
        first_frame = frame_ind == 0
        dri_crop, out_crop, out_org, dri_motion_info = pipe.run(frame, pipe.src_imgs[0], pipe.src_infos[0],
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
                cv2.imshow('Render', out_org)
            else:
                # image show in realtime mode
                cv2.imshow('Render', out_crop)
            # Check for key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        display_time = time.time() - display_start_time
        performance_tracker.update_metric("display_time", display_time * 1000)  # Convert to ms
        
        # Total processing time
        total_time = time.time() - total_start_time
        performance_tracker.update_metric("total_time", total_time * 1000)  # Convert to ms
        
        # Log metrics periodically
        performance_tracker.log_metrics()
        
    vcap.release()
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

    # Load custom landmarks if provided
    custom_landmarks = None
    if args.custom_landmarks:
        if not os.path.exists(args.custom_landmarks):
            print(f"Custom landmarks file {args.custom_landmarks} doesn't exist! Exiting.")
            exit(1)
        print(f"Loading custom landmarks from {args.custom_landmarks}")
        if args.custom_landmarks.endswith('.npy'):
            custom_landmarks = np.load(args.custom_landmarks)
        elif args.custom_landmarks.endswith('.pkl'):
            with open(args.custom_landmarks, 'rb') as f:
                custom_landmarks = pickle.load(f)
        else:
            print(f"Unsupported landmarks file format. Please use .npy or .pkl. Exiting.")
            exit(1)
        print(f"Custom landmarks loaded with shape: {custom_landmarks.shape if hasattr(custom_landmarks, 'shape') else 'unknown'}")

    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    ret = pipe.prepare_source(args.src_image, custom_landmarks=custom_landmarks, realtime=args.realtime, no_crop=args.no_crop)
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
                cv2.imshow('Render,  Q > exit,  S > Stitching,  Z > RelativeMotion,  X > AnimationRegion,  C > CropDrivingVideo, KL > AdjustSourceScale, NM > AdjustDriverScale,  Space > Webcamassource,  R > SwitchRealtimeWebcamUpdate',out_org)
            else:
                # image show in realtime mode
                cv2.imshow('Render,  Q > exit,  S > Stitching,  Z > RelativeMotion,  X > AnimationRegion,  C > CropDrivingVideo, KL > AdjustSourceScale, NM > AdjustDriverScale,  Space > Webcamassource,  R > SwitchRealtimeWebcamUpdate', out_crop)
            
            # Handle keyboard inputs
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
                        help='source image')
    parser.add_argument('--dri_video', required=False, type=str, default="assets/examples/driving/d14.mp4",
                        help='driving video')
    parser.add_argument('--cfg', required=False, type=str, default="configs/onnx_infer.yaml", help='inference config')
    parser.add_argument('--realtime', action='store_true', help='realtime inference')
    parser.add_argument('--animal', action='store_true', help='use animal model')
    parser.add_argument('--paste_back', action='store_true', default=False, help='paste back to origin image')
    parser.add_argument('--custom_landmarks', type=str, default=None, 
                        help='Path to a .npy or .pkl file containing custom landmarks in insightface format')
    parser.add_argument('--no-crop', action='store_true', help='Prevent cropping of the source image')
    args, unknown = parser.parse_known_args()

    if args.dri_video.endswith(".pkl"):
        run_with_pkl(args)
    else:
        run_with_video(args)
