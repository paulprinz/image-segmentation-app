FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone and install GroundingDINO
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
WORKDIR /workspace/GroundingDINO
RUN pip install --no-cache-dir -e .

# Download model weights
RUN mkdir -p weights && \
    cd weights && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Install additional dependencies
RUN pip install --no-cache-dir redis

# Copy service code
WORKDIR /workspace
COPY service.py /workspace/service.py

# Run the service
CMD ["python", "/workspace/service.py"]
