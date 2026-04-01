FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates zip ffmpeg \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 1) Pin NumPy (قبل أي installs)
RUN python -m pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir "numpy<2"

# 2) Clone gaussian-splatting + submodules
RUN git clone --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git /workspace/gaussian-splatting
WORKDIR /workspace/gaussian-splatting
RUN git submodule update --init --recursive

# 3) Install python deps WITHOUT pulling numpy>=2
# - تجنب opencv-python لأنه بيطلب numpy>=2 في اللوج عندك
# - هنستخدم opencv-python-headless (غالبًا أقل مشاكل) + نثبت numpy<2 بعدها
RUN pip install --no-cache-dir \
    tqdm pillow imageio scipy matplotlib \
    opencv-python-headless

# Re-assert numpy pin (لأن أي package ممكن يغيره)
RUN pip install --no-cache-dir --force-reinstall "numpy<2"

# 4) Build CUDA extensions with NO build isolation so torch is visible
RUN pip install --no-cache-dir --no-build-isolation -e submodules/diff-gaussian-rasterization
RUN pip install --no-cache-dir --no-build-isolation -e submodules/simple-knn

# 5) Serverless deps (وتتضمن numpy<2 كمان)
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Final sanity check
RUN python -c "import torch; print('TORCH_OK', torch.__version__)" \
 && python -c "import numpy as np; print('NUMPY_OK', np.__version__)" \
 && python -c "import diff_gaussian_rasterization; print('DGR_OK')" \
 && python -c "import simple_knn; print('SKNN_OK')"

COPY handler.py /app/handler.py
CMD ["python", "/app/handler.py"]

