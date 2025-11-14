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

# Clone and install GroundingDINO first
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
WORKDIR /workspace/GroundingDINO

# Patch ms_deform_attn.py to use PyTorch fallback when C++ extensions are not available
RUN python3 << 'EOF'
import re

file_path = 'groundingdino/models/GroundingDINO/ms_deform_attn.py'
with open(file_path, 'r') as f:
    content = f.read()

# Add _C_available flag after the import attempt
content = content.replace(
    'try:\n    from groundingdino import _C\nexcept:\n    warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")',
    'try:\n    from groundingdino import _C\n    _C_available = True\nexcept:\n    _C_available = False\n    warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")'
)

# Use PyTorch fallback when _C is not available
content = content.replace(
    'if torch.cuda.is_available() and value.is_cuda:',
    'if _C_available and torch.cuda.is_available() and value.is_cuda:'
)

with open(file_path, 'w') as f:
    f.write(content)

print("Patched ms_deform_attn.py successfully")
EOF

# Install GroundingDINO with specific compatible versions
# Pin numpy to <2.0 for PyTorch 2.1.0 compatibility
# Pin transformers to version compatible with PyTorch 2.1.0
# Install dependencies first, then install GroundingDINO with --no-deps to prevent upgrades
RUN pip install --no-cache-dir "numpy<2.0" && \
    pip install --no-cache-dir "transformers>=4.36.0,<4.40.0" && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir --force-reinstall "numpy<2.0"

# Download model weights
RUN mkdir -p weights && \
    cd weights && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Install additional dependencies
# Pin opencv-python-headless to version compatible with numpy<2.0
RUN pip install --no-cache-dir redis "opencv-python-headless<4.10.0" && \
    pip install --no-cache-dir --force-reinstall "numpy<2.0"

# Copy service code
WORKDIR /workspace
COPY service.py /workspace/service.py

# Run the service
CMD ["python", "/workspace/service.py"]
