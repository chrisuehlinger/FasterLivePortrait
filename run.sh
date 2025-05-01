#!/bin/bash -ex

docker build . \
  -t uehreka/fasterliveportrait:latest

docker system prune -f

xhost +local:root
docker run -it --gpus=all \
  --rm \
  --privileged \
  --net=host \
  -e DISPLAY=${DISPLAY} \
  -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)/checkpoints:/root/FasterLivePortrait/checkpoints" \
  -v "$(pwd)/results:/root/FasterLivePortrait/results" \
  -v "$(pwd)/certs:/root/FasterLivePortrait/certs" \
  -v "$(pwd)/frontend:/root/FasterLivePortrait/frontend" \
  uehreka/fasterliveportrait:latest

sudo chown -R $USER:$USER checkpoints results