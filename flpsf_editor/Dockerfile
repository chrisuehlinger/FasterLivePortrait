FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-pyqt5 \
    python3-pyqt5.qtsvg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg \
    git \
    wget \
    curl \
    # Add additional X11 dependencies
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    xauth \
    x11-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app/flpsf_editor

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir onnx onnxruntime

# Copy FLPSF editor files
COPY . .

# Create directory to mount host data
RUN mkdir -p /data

# Set environment variables for Qt
ENV QT_X11_NO_MITSHM=1
ENV QT_GRAPHICSSYSTEM="native"

# Important: Setting the XDG_RUNTIME_DIR avoids some Qt/XCB issues
ENV XDG_RUNTIME_DIR=/tmp
ENV QT_DEBUG_PLUGINS=1

# Default to XCB platform
ENV QT_QPA_PLATFORM=xcb

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]