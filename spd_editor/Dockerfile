FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 \
    CUDA_HOME=/usr/local/cuda-11.8
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    build-essential \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libice6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Set up CUDA environment
RUN echo "export PATH=/usr/local/cuda-11.8:/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc \
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc \
    && echo "export CUDA_HOME=/usr/local/cuda-11.8" >> /etc/bash.bashrc \
    && echo "export CUDA_ROOT=/usr/local/cuda" >> /etc/bash.bashrc

WORKDIR /app

# Install Python dependencies with GPU support
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy SPD Editor files
COPY . /app/spd_editor/

# Install SPD Editor in development mode
WORKDIR /app/spd_editor
RUN pip install -e .

# Set up the entry point
WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command: run GUI (overridable by user)
CMD ["python", "-m", "spd_editor.gui"]