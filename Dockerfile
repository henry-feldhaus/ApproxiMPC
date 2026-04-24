FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive

# Build tools for acados + Qt6/X11 runtime for GUI rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    dbus \
    git \
    libdbus-1-3 \
    libgl1-mesa-glx \
    libx11-xcb1 \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    qt6-base-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build acados from source with qpoases enabled.
RUN git clone --recurse-submodules https://github.com/acados/acados.git /opt/acados \
    && mkdir -p /opt/acados/build \
    && cd /opt/acados/build \
    && cmake -DACADOS_WITH_QPOASES=ON .. \
    && make install -j"$(nproc)"

ENV ACADOS_SOURCE_DIR=/opt/acados
ENV LD_LIBRARY_PATH=/opt/acados/lib:${LD_LIBRARY_PATH}

# Copy package metadata and the installable package source first so editable
# install works even before the full repository is copied.
COPY pyproject.toml uv.lock* README.md /app/
COPY gymkhana /app/gymkhana

# Install baseline runtime dependencies from pyproject.
# Add optional render dependencies for GUI.
# Add LSTM inference/runtime dependencies used by src/lstm_eval.py.
# Install -e . first (which pulls core deps), then pin numpy last to ensure it stays <1.25.
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir pyqtgraph PyQt6 \
    && pip install --no-cache-dir "onnxruntime>=1.12,<1.24" "onnx>=1.20.1,<2.0.0" scikit-learn joblib \
    && pip install --no-cache-dir --force-reinstall "numpy<1.25"

# Gymnasium imports Atari wrappers by default, which pulls cv2 at import time.
# Gym-Khana does not use Atari wrappers, so remove that one import to avoid cv2 dnn typing issues
# seen on the NVIDIA base image's OpenCV build.
RUN python - <<'PY'
from pathlib import Path

p = Path('/usr/local/lib/python3.10/dist-packages/gymnasium/wrappers/__init__.py')
text = p.read_text()
line = 'from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing\n'
if line in text:
    p.write_text(text.replace(line, ''))

# Some OpenCV wheels on this base image expose cv2.dnn without DictValue.
# Patch cv2 typing hint import so cv2 can still import successfully.
cv2_typing = Path('/usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py')
if cv2_typing.exists():
    t = cv2_typing.read_text()
    old = 'LayerId = cv2.dnn.DictValue\n'
    new = "LayerId = getattr(cv2.dnn, 'DictValue', int)\n"
    if old in t:
        cv2_typing.write_text(t.replace(old, new))
PY

# Install acados Python interface.
RUN pip install --no-cache-dir -e /opt/acados/interfaces/acados_template

# Install acados tera renderer at build time so runtime does not prompt for interactive download.
RUN python - <<'PY'
import os
import platform
import stat
import urllib.request
from pathlib import Path

version = "v0.2.0"
arch_map = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
}
machine = platform.machine().lower()
arch = arch_map.get(machine)
if arch is None:
    raise RuntimeError(f"Unsupported architecture for t_renderer: {machine}")

out_dir = Path("/opt/acados/bin")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "t_renderer"
url = f"https://github.com/acados/tera_renderer/releases/download/{version}/t_renderer-{version}-linux-{arch}"

urllib.request.urlretrieve(url, out_path)
mode = out_path.stat().st_mode
out_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
print(f"Installed t_renderer from: {url}")
PY

# Copy the full repository.
COPY . /app

CMD ["bash"]
