#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
CLI for preprocessing a source image/video using FasterLivePortraitPipeline.prepare_source
"""
import argparse
import pickle
from omegaconf import OmegaConf, DictConfig
from typing import cast
import torch
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

def main():
    parser = argparse.ArgumentParser(description="Preprocess a source image/video and save source data file")
    parser.add_argument('--src', required=True, help='Path to source image or video')
    parser.add_argument('--output', required=True, help='Path to save preprocessed source (e.g. .fsp)')
    parser.add_argument('--cfg', required=True, help='Inference config YAML file')
    parser.add_argument('--animal', action='store_true', help='Use animal model')
    parser.add_argument('--realtime', action='store_true', help='Enable realtime mode during source processing')
    args = parser.parse_args()

    # Load config
    cfg_raw = OmegaConf.load(args.cfg)
    cfg: DictConfig = cast(DictConfig, cfg_raw)
    # Ensure pasteback disabled for preprocessing
    cfg.infer_params.flag_pasteback = True

    # Instantiate pipeline
    pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=args.animal)
    # Run prepare_source
    ok = pipe.prepare_source(args.src, realtime=args.realtime)
    if not ok:
        print(f"Failed to prepare source for {args.src}")
        return 1

    # Build export data
    data = {
        'version': 1,
        'is_source_video': pipe.is_source_video,
        'source_path': args.src,
        'src_imgs': pipe.src_imgs,
        'src_infos': pipe.src_infos,
    }

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved preprocessed source to {args.output}")
    return 0


if __name__ == '__main__':
    exit(main())
