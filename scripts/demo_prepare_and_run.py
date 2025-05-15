#!/usr/bin/env python3
"""
Demo script that preprocesses a source image and runs run.py with the preprocessed data.
"""
import subprocess
import sys

def main():
    config = "configs/onnx_infer.yaml"
    src_image = "assets/examples/source/s12.jpg"
    dri_video = "assets/examples/driving/d14.mp4"
    preprocessed = "assets/examples/source/s12.fsp"

    # Preprocess source
    ret = subprocess.call([
        sys.executable, "scripts/prepare_source.py",
        "--src", src_image,
        "--output", preprocessed,
        "--cfg", config
    ])
    if ret != 0:
        print("Source preprocessing failed")
        sys.exit(1)

    # Run the pipeline with preprocessed source
    cmd = [
        sys.executable, "run.py",
        "--src_data", preprocessed,
        "--dri_video", dri_video,
        "--cfg", config,
        "--paste_back"
    ]
    subprocess.call(cmd)

if __name__ == '__main__':
    main()
