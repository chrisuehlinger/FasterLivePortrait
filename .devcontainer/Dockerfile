FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 \
        CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST="8.6"
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install base dependencies
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
        # X11 dependencies
        x11-apps \
        xterm \
        xauth \
        # Docker-in-Docker
        apt-transport-https \
        ca-certificates \
        gnupg-agent \
        sudo \
    && apt clean && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install Docker
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - \
    && add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    && apt-get update \
    && apt-get install -y docker-ce docker-ce-cli containerd.io \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Configure CUDA environment variables
RUN echo "export PATH=/usr/local/cuda-11.8:/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc \
    && echo "export CUDA_HOME=/usr/local/cuda-11.8" >> /etc/bash.bashrc \
    && echo "export CUDA_ROOT=/usr/local/cuda" >> /etc/bash.bashrc 

# Install CMake
WORKDIR /root
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.sh \
    && chmod +x cmake-3.31.6-linux-x86_64.sh \
    && ./cmake-3.31.6-linux-x86_64.sh --skip-license --prefix=/usr/local

# Build and install onnxruntime
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

# Create directory structure
RUN mkdir -p /workspace
RUN mkdir -p /opt/grid-sample3d-trt-plugin/build
RUN mkdir -p /usr/local/tensorrt

# Setup workspace directory
WORKDIR /workspace

# Allow X11 connections
RUN echo "allowed_users=anybody" >> /etc/X11/Xwrapper.config

# Create a non-root user for development (optional, as we're using root in devcontainer.json)
RUN groupadd -g 1000 devuser && \
    useradd -u 1000 -g 1000 -m devuser && \
    usermod -aG sudo devuser && \
    echo "devuser ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/devuser

# Setup entry point
CMD ["/bin/bash"]