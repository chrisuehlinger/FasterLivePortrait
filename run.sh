#!/bin/bash -ex

docker build . \
  -t uehreka/fasterliveportrait:latest

# docker system prune -f

# Define environment variables for X11
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

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
  -v "$(pwd)/spd_editor/data:/root/FasterLivePortrait/spd_editor/data" \
  -v "$(pwd)/tools:/root/FasterLivePortrait/tools" \
  uehreka/fasterliveportrait:latest

sudo chown -R $USER:$USER checkpoints results