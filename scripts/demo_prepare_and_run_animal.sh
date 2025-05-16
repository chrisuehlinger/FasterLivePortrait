#!/usr/bin/env bash

# Demo script: preprocess an animal source image and run the pipeline with preprocessed data
set -e

# Configuration
CONFIG="configs/trt_infer.yaml"
SRC_IMAGE="assets/examples/source/chac-bolay.png"
DRI_VIDEO="assets/examples/driving/d10.mp4"
PREPROCESSED="assets/examples/source/chac-bolay_edited.fsp"

# Preprocess the animal source image
# python3 scripts/prepare_source.py --src "$SRC_IMAGE" --output "$PREPROCESSED" --cfg "$CONFIG" --animal

# Run the pipeline using the preprocessed animal source data
# python3 run.py --src_data "$PREPROCESSED" --dri_video "$DRI_VIDEO" --cfg "$CONFIG" --paste_back
python3 run.py --src_data "$PREPROCESSED" --dri_video "assets/examples/driving/d0.mp4" --cfg "$CONFIG" --paste_back
python3 run.py --src_data "$PREPROCESSED" --dri_video "assets/examples/driving/d3.mp4" --cfg "$CONFIG" --paste_back
python3 run.py --src_data "$PREPROCESSED" --dri_video "assets/examples/driving/d6.mp4" --cfg "$CONFIG" --paste_back
python3 run.py --src_data "$PREPROCESSED" --dri_video "assets/examples/driving/d9.mp4" --cfg "$CONFIG" --paste_back
python3 run.py --src_data "$PREPROCESSED" --dri_video "assets/examples/driving/d11.mp4" --cfg "$CONFIG" --paste_back
python3 run.py --src_data "$PREPROCESSED" --dri_video "assets/examples/driving/d12.mp4" --cfg "$CONFIG" --paste_back
python3 run.py --src_data "$PREPROCESSED" --dri_video "assets/examples/driving/d13.mp4" --cfg "$CONFIG" --paste_back
python3 run.py --src_data "$PREPROCESSED" --dri_video "assets/examples/driving/d14.mp4" --cfg "$CONFIG" --paste_back
python3 run.py --src_data "$PREPROCESSED" --dri_video "assets/examples/driving/d18.mp4" --cfg "$CONFIG" --paste_back
python3 run.py --src_data "$PREPROCESSED" --dri_video "assets/examples/driving/d19.mp4" --cfg "$CONFIG" --paste_back
