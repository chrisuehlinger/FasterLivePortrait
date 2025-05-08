#!/bin/bash -ex

# huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints

export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH
# sh scripts/all_onnx2trt.sh
# sh scripts/all_onnx2trt_animal.sh

pushd /root/FasterLivePortrait/src/models/XPose/models/UniPose/ops
python setup.py build install
popd

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
#  --src_image assets/examples/source/chac-bolay-head.png \
#  --dri_video assets/examples/driving/d10.mp4 \
#  --cfg configs/trt_infer.yaml \
#  --animal

python websocket_server.py \
  --port 8080 \
  --host 0.0.0.0 \
  --config-path configs/trt_infer.yaml \
  --source-image assets/examples/source/ahau-kin.png \
  --source-image-2 assets/examples/source/ix-chel.png \
  --source-image-3 assets/examples/source/chac-bolay-head.png \
  --source-image-4 assets/examples/source/ahau-kin.png \
  --is-animal \
  --ssl-cert certs/cert.pem \
  --ssl-key certs/key.pem
