FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    libgl1 \
    libegl1 \
    libgles2 \
    libglvnd0 \
    libglx0 \
    libopengl0 \
    libglfw3 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxcursor1 \
    libxi6 \
    libxinerama1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Codex CLI
RUN curl -fsSL https://chatgpt.com/codex/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

# Headless MuJoCo rendering
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# NVIDIA GPU capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# More reliable with bind mounts / Docker volumes
ENV UV_LINK_MODE=copy

WORKDIR /workspace

RUN git config --global --add safe.directory /workspace

ENTRYPOINT ["/bin/bash"]