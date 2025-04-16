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
  --port 9000 \
  --source-image assets/examples/source/s2.jpg \
  --config-path configs/onnx_infer.yaml \
  --ssl-cert certs/cert.pem \
  --ssl-key certs/key.pem
