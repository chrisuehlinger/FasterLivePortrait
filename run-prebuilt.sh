#!/bin/bash -ex

docker run -it --rm \
  --gpus=all \
  -v "$(pwd):/root/FasterLivePortrait" \
  -p 9870:9870 \
  shaoguo/faster_liveportrait:v3 \
  /bin/bash