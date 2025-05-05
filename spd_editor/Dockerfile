FROM uehreka/flp-base:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-tk \
    libgl1-mesa-glx \
    python3-pil.imagetk \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with GPU support
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

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