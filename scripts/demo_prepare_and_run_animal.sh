#!/usr/bin/env bash

# Demo script: preprocess an animal source image and run the pipeline with preprocessed data
set -e

# Configuration
CONFIG="configs/onnx_infer.yaml"
SRC_IMAGE="assets/examples/source/chac-bolay-head.png"
DRI_VIDEO="assets/examples/driving/d10-short.mp4"
PREPROCESSED="assets/examples/source/chac-bolay_with_ahau_landmarks.fsp"

# Preprocess the animal source image
# python3 scripts/prepare_source.py --src "$SRC_IMAGE" --output "$PREPROCESSED" --cfg "$CONFIG" --animal

# Run the pipeline using the preprocessed animal source data
python3 run.py --src_data "$PREPROCESSED" --dri_video "$DRI_VIDEO" --cfg "$CONFIG" --paste_back
