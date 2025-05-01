#!/bin/bash -ex

# huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints

export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH
# sh scripts/all_onnx2trt.sh
# sh scripts/all_onnx2trt_animal.sh

# python run.py \
#  --src_image assets/examples/source/s2.jpg \
#  --dri_video assets/examples/driving/d14.mp4 \
#  --cfg configs/trt_infer.yaml

# python run.py \
#  --src_image assets/examples/source/s2.jpg \
#  --dri_video 0 \
#  --realtime \
#  --cfg configs/trt_infer.yaml

# sh scripts/all_onnx2trt.sh
# sh scripts/all_onnx2trt_animal.sh

# python run.py \
#  --src_image assets/examples/source/s10.jpg \
#  --dri_video assets/examples/driving/d14.mp4 \
#  --cfg configs/trt_infer.yaml

python webrtc_server.py \
  --port 8080 \
  --host 0.0.0.0 \
  --config-path configs/trt_infer.yaml \
  --source-image assets/examples/source/s2.jpg \
  --source-image-2 assets/examples/source/s12.jpg \
  --source-image-3 assets/examples/source/s10.jpg \
  --ssl-cert certs/cert.pem \
  --ssl-key certs/key.pem
