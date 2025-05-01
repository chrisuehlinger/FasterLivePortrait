FROM uehreka/flp-base:latest

# Install Docker
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - \
    && add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    && apt-get update \
    && apt-get install -y docker-ce docker-ce-cli containerd.io \
    && apt clean && rm -rf /var/lib/apt/lists/*

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