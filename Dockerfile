FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates zip ffmpeg \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Pin numpy to <2 to avoid ABI issues
RUN python -m pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir "numpy<2"

# Clone gaussian-splatting + submodules
RUN git clone --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git /workspace/gaussian-splatting
WORKDIR /workspace/gaussian-splatting
RUN git submodule update --init --recursive

# Install common deps
RUN pip install --no-cache-dir \
    tqdm pillow opencv-python imageio scipy matplotlib

# Build/install CUDA extensions that provide diff_gaussian_rasterization + simple_knn
RUN pip install --no-cache-dir -e submodules/diff-gaussian-rasterization
RUN pip install --no-cache-dir -e submodules/simple-knn

# Serverless deps
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Re-assert numpy pin after everything
RUN pip install --no-cache-dir --force-reinstall "numpy<2" \
 && python -c "import numpy as np; print('NUMPY', np.__version__)" \
 && python -c "import diff_gaussian_rasterization; print('diff_gaussian_rasterization OK')"

COPY handler.py /app/handler.py
CMD ["python", "/app/handler.py"]
