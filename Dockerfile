FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 \
        CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST="8.6"
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update && apt-get install -y --no-install-recommends \
        make \
        wget \
        tar \
        build-essential \
        libgl1-mesa-dev \
        curl \
        unzip \
        git \
        python3-dev \
        python3-pip \
        libsm6 \
        libice6 \
        libglib2.0-0 \
        ffmpeg \
        software-properties-common \
    && apt clean && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# ARG TENSORRT_VERSION
# ENV TENSORRT_VERSION=${TENSORRT_VERSION}
# RUN test -n "$TENSORRT_VERSION" || (echo "No tensorrt version specified, please use --build-arg TENSORRT_VERSION=x.y to specify a version." && exit 1)

# # Install TensorRT + dependencies
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
# RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
# RUN apt-get update
# RUN TENSORRT_MAJOR_VERSION=`echo ${TENSORRT_VERSION} | cut -d '.' -f 1` && \
#     apt-get install --no-install-recommends -y \
#                        libnvinfer${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}.* \
#                        libnvinfer-plugin${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}.* \
#                        libnvinfer-dev=${TENSORRT_VERSION}.* \
#                        libnvinfer-headers-dev=${TENSORRT_VERSION}.* \
#                        libnvinfer-headers-plugin-dev=${TENSORRT_VERSION}.* \
#                        libnvinfer-plugin-dev=${TENSORRT_VERSION}.* \
#                        libnvonnxparsers${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}.* \
#                        libnvonnxparsers-dev=${TENSORRT_VERSION}.* \
#     && apt clean && rm -rf /var/lib/apt/lists/*




RUN echo "export PATH=/usr/local/cuda-11.8:/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc \
    && echo "export CUDA_HOME=/usr/local/cuda-11.8" >> /etc/bash.bashrc \
    && echo "export CUDA_ROOT=/usr/local/cuda" >> /etc/bash.bashrc 
    # && echo "export C_INCLUDE_PATH=/usr/local/tensorrt/include:${C_INCLUDE_PATH}" >> /etc/bash.bashrc

WORKDIR /root
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.sh \
    && chmod +x cmake-3.31.6-linux-x86_64.sh \
    && ./cmake-3.31.6-linux-x86_64.sh --skip-license --prefix=/usr/local
# RUN pip install psutil numpy
RUN git clone https://github.com/microsoft/onnxruntime \
    && cd onnxruntime \
    && git checkout liqun/ImageDecoder-cuda \
    && pip install -r requirements-dev.txt \
    && ./build.sh --parallel \
        --build_shared_lib --use_cuda \
        --cuda_version 11.8 \
        --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda/ \
        --config Release --build_wheel --skip_tests \
        --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="60;70;75;80;86" \
        --cmake_extra_defines CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        --disable_contrib_ops \
        --allow_running_as_root \
    && pip install build/Linux/Release/dist/onnxruntime_gpu-1.17.0-cp310-cp310-linux_x86_64.whl


# COPY downloads/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz /tmp/TensorRT.tar
# RUN tar -xf /tmp/TensorRT.tar -C /usr/local/ \
#     && mv /usr/local/TensorRT-8.6.1.6 /usr/local/tensorrt \
#     && python --version \
#     && pip3 install /usr/local/tensorrt/python/tensorrt-*-cp310-*.whl \
#     && rm -rf /tmp/TensorRT.tar \
#     && echo 'export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH' >> /etc/bash.bashrc
# RUN git clone https://github.com/SeanWangJS/grid-sample3d-trt-plugin
# COPY grid-sample3d-trt-plugin /root/grid-sample3d-trt-plugin
# RUN cd grid-sample3d-trt-plugin \
#     && mkdir build \
#     && cd build \
#     && cmake .. \
#     && make -j$(nproc) \
#     && cp libgrid_sample3d_plugin.so /opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so

RUN mkdir /root/FasterLivePortrait
COPY ./requirements.txt /root/FasterLivePortrait/requirements.txt
WORKDIR /root/FasterLivePortrait

RUN pip install -r requirements.txt

COPY downloads/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz /tmp/TensorRT.tar
RUN tar -xf /tmp/TensorRT.tar -C /usr/local/ \
    && mv /usr/local/TensorRT-8.6.1.6 /usr/local/tensorrt \
    && python --version \
    && pip3 install /usr/local/tensorrt/python/tensorrt-*-cp310-*.whl \
    && rm -rf /tmp/TensorRT.tar \
    && echo 'export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH' >> /etc/bash.bashrc

COPY grid-sample3d-trt-plugin /opt/grid-sample3d-trt-plugin
# RUN cd ../grid-sample3d-trt-plugin \
#     && export C_INCLUDE_PATH=/usr/local/tensorrt/include:${C_INCLUDE_PATH} \
#     && export CPP_INCLUDE_PATH=/usr/local/tensorrt/include:${CPP_INCLUDE_PATH} \
#     && export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:${LD_LIBRARY_PATH} \
#     && mkdir build \
#     && cd build \
#     && cmake .. -DTensorRT_ROOT=/usr/local/tensorrt \
#     && make -j$(nproc) \
#     && cp libgrid_sample3d_plugin.so /opt/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so

COPY ./configs /root/FasterLivePortrait/configs
COPY ./scripts /root/FasterLivePortrait/scripts
COPY ./src /root/FasterLivePortrait/src
COPY ./assets /root/FasterLivePortrait/assets
COPY ./entrypoint.sh /root/FasterLivePortrait/entrypoint.sh
COPY ./webrtc_server.py /root/FasterLivePortrait/webrtc_server.py
COPY ./run.py /root/FasterLivePortrait/run.py

ENTRYPOINT [ "./entrypoint.sh" ]
# CMD /bin/bash