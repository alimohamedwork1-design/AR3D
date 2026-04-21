FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
# Keep this conservative; you can override in RunPod env if needed.
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

RUN apt-get update && apt-get install -y \
  colmap \
  ffmpeg \
  git \
  build-essential \
  cmake \
  ninja-build \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
  runpod \
  requests

RUN git clone --depth 1 --recursive https://github.com/graphdeco-inria/gaussian-splatting /workspace/gaussian-splatting

# Core Python deps + gaussian-splatting CUDA extensions
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" && \
  nvcc --version

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
  pip install --no-cache-dir pybind11

RUN pip install --no-cache-dir plyfile tqdm "numpy<2" scipy
RUN pip install --no-cache-dir opencv-python joblib

# Build CUDA extensions (separate layers so build logs are clear)
RUN pip install --no-cache-dir -v --no-build-isolation /workspace/gaussian-splatting/submodules/diff-gaussian-rasterization
RUN pip install --no-cache-dir -v --no-build-isolation /workspace/gaussian-splatting/submodules/simple-knn
RUN pip install --no-cache-dir -v --no-build-isolation /workspace/gaussian-splatting/submodules/fused-ssim

COPY handler.py /workspace/handler.py

CMD ["python", "-u", "/workspace/handler.py"]

