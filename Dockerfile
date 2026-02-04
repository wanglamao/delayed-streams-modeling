FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Replace apt source with Tsinghua mirror (Ubuntu 22.04 - jammy)
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy moshi and moshi-finetune from build context
# These will be copied from the parent directory (delayed-streams-modeling)
COPY moshi/moshi /workspace/moshi
COPY moshi-finetune /workspace/moshi-finetune

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install moshi first
WORKDIR /workspace/moshi
RUN pip install -e . --no-cache-dir

# Install moshi-finetune
WORKDIR /workspace/moshi-finetune
RUN pip install -e . --no-cache-dir

# Create directories
RUN mkdir -p /workspace/data /workspace/models /workspace/runs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
ENV HF_HOME=/workspace/.hf_home
ENV HF_HUB_CACHE=/workspace/.hf_home/hub
ENV TRANSFORMERS_CACHE=/workspace/.hf_home/transformers
ENV TORCH_HOME=/workspace/.hf_home/torch

# Create HF cache directories
RUN mkdir -p $HF_HUB_CACHE $TRANSFORMERS_CACHE $TORCH_HOME

# Set default command
CMD ["bash"]
