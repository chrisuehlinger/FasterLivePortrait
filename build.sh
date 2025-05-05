#!/bin/bash -ex

docker buildx build \
  --push \
  -t uehreka/fasterliveportrait:latest \
  .
