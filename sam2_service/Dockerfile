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

# Clone and install SAM2
RUN git clone https://github.com/facebookresearch/sam2.git
WORKDIR /workspace/sam2
RUN pip install --no-cache-dir -e .

# Download model checkpoint
RUN mkdir -p checkpoints && \
    cd checkpoints && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Install additional dependencies
RUN pip install --no-cache-dir redis matplotlib

# Copy service code
WORKDIR /workspace
COPY service.py /workspace/service.py

# Run the service
CMD ["python", "/workspace/service.py"]
