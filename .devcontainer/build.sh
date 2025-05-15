#!/bin/bash -ex

docker build \
  -t uehreka/fasterliveportrait:dev \
  .

docker push uehreka/fasterliveportrait:dev