FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: build tools for acados, and runtime libs for rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    mesa-utils \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv globally.
RUN python -m pip install --no-cache-dir uv

# Build acados from source with qpoases enabled.
RUN git clone --recurse-submodules https://github.com/acados/acados.git /opt/acados \
    && mkdir -p /opt/acados/build \
    && cd /opt/acados/build \
    && cmake -DACADOS_WITH_QPOASES=ON .. \
    && make install -j"$(nproc)"

# Set acados environment variables.
ENV ACADOS_SOURCE_DIR=/opt/acados
ENV LD_LIBRARY_PATH=/opt/acados/lib:${LD_LIBRARY_PATH}

# Copy only metadata for better layer caching.
COPY pyproject.toml uv.lock* README.md /app/

# Create project environment.
RUN uv sync --all-groups

# Install acados Python interface in the project environment.
RUN . /app/.venv/bin/activate && pip install -e /opt/acados/interfaces/acados_template

# Copy the full repository (for image-only runs; compose mounts override this).
COPY . /app

CMD ["bash"]
