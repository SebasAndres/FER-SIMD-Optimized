FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    pkg-config \
    libopencv-dev \
    nasm \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    unzip \
    gdb \
    libgtk-3-0 \
    libx11-6 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Setup Kaggle credentials directory
RUN mkdir -p /root/.kaggle /root/.config/kaggle

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen

# Copy the rest of the project
COPY . .

# Set default command
CMD ["bash"]
