FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Upgrade PyTorch to fix transformers compatibility
RUN pip install --no-cache-dir --upgrade torch torchvision

# Clone and install GroundingDINO
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
WORKDIR /workspace/GroundingDINO

RUN pip install --no-cache-dir -e .

# Download model weights
RUN mkdir -p weights && \
    cd weights && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Install additional dependencies and pin numpy to <2.0 for PyTorch 2.1.0 compatibility
RUN pip install --no-cache-dir "numpy<2.0" redis opencv-python-headless

# Copy service code
WORKDIR /workspace
COPY service.py /workspace/service.py

# Run the service
CMD ["python", "/workspace/service.py"]
