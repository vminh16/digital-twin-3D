# Use official PyTorch image with CUDA development tools (CUDA 12.1, Ubuntu 22.04)
# L4 GPU has SM 8.9 architecture which is fully supported by CUDA 12.1+
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
# SM 8.9 is for Ada Lovelace (e.g., L4, RTX 4090)
# SM 8.0 is for Ampere (e.g., A100), SM 7.5 for Turing (e.g., T4)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

# Install system dependencies (including OpenGL dependencies for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpng-dev \
    libjpeg-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /workspace/digital-twin-3D

# Install python dependencies
# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and other directories
COPY configs/ ./configs/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Set PYTHONPATH to include src directory
ENV PYTHONPATH=/workspace/digital-twin-3D/src

# Run smoke test by default when running the container
CMD ["python", "tests/smoke_test.py"]
