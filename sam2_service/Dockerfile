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

# Patch SAM2's pyproject.toml to accept PyTorch 2.1.0 from base image
# This avoids downloading 3+ GB of torch>=2.5.1 dependencies
RUN sed -i 's/torch>=2.5.1/torch>=2.1.0,<2.5.0/' pyproject.toml && \
    sed -i 's/torchvision>=0.20.1/torchvision>=0.16.0,<0.20.0/' pyproject.toml

# Install SAM2's dependencies that are compatible with torch 2.1.0
RUN pip install --no-cache-dir \
    "tqdm>=4.66.1" \
    "hydra-core>=1.3.2" \
    "iopath>=0.1.10" && \
    pip install --no-cache-dir --no-deps -e .

# Download model checkpoint
RUN mkdir -p checkpoints && \
    cd checkpoints && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Install additional dependencies
# Pin matplotlib and opencv-python-headless to versions compatible with numpy<2.0
RUN pip install --no-cache-dir redis "matplotlib<3.9.0" "opencv-python-headless<4.10.0" && \
    pip install --no-cache-dir --force-reinstall "numpy<2.0"

# Install Weights & Biases for monitoring
RUN pip install --no-cache-dir wandb weave

# Copy service code
WORKDIR /workspace
COPY service.py /workspace/service.py

# Run the service
CMD ["python", "/workspace/service.py"]
