#!/usr/bin/env bash

# Demo script: preprocess a source image and run the pipeline with preprocessed data
set -e

# Configuration
CONFIG="configs/trt_infer.yaml"
SRC_IMAGE="assets/examples/source/ahau-kin.png"
DRI_VIDEO="assets/examples/driving/d14.mp4"
PREPROCESSED="assets/examples/source/ahau-kin.fsp"

# Preprocess the source image
python3 scripts/prepare_source.py --src "$SRC_IMAGE" --output "$PREPROCESSED" --cfg "$CONFIG"

# Run the pipeline using the preprocessed source data
python3 run.py --src_data "$PREPROCESSED" --dri_video "$DRI_VIDEO" --cfg "$CONFIG" --paste_back
