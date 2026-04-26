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
  libgl1 \
  libglib2.0-0 \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
  runpod \
  requests

RUN git clone --depth 1 --recursive https://github.com/graphdeco-inria/gaussian-splatting /workspace/gaussian-splatting

# Core Python deps + gaussian-splatting CUDA extensions
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" && \
  nvcc --version

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
  pip uninstall -y numpy || true && \
  pip install --no-cache-dir --force-reinstall "numpy==1.26.4" && \
  pip install --no-cache-dir "pybind11>=2.12" && \
  pip install --no-cache-dir plyfile tqdm scipy pygltflib
RUN pip install --no-cache-dir opencv-python joblib && \
  pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# Optional mesh export (Poisson reconstruction) for "complete" GLB meshes.
# Heavy dependency; kept last so cache churn is limited.
# NOTE: open3d wheels are not available on all architectures (e.g., linux/arm64). Don't fail the image build.
ARG TARGETARCH
RUN set -eux; \
  echo "TARGETARCH=${TARGETARCH:-unknown}"; \
  python -V; \
  python -m pip --version; \
  if [ "${TARGETARCH:-}" = "amd64" ]; then \
    python -m pip install --no-cache-dir --only-binary=:all: "open3d==0.18.0" "trimesh>=4.4.0" || ( \
      echo "WARN: open3d install failed; continuing without mesh export"; \
      python -m pip install --no-cache-dir "trimesh>=4.4.0" \
    ); \
  else \
    echo "WARN: skipping open3d install on TARGETARCH=${TARGETARCH:-unknown}"; \
    python -m pip install --no-cache-dir "trimesh>=4.4.0"; \
  fi

# Build CUDA extensions (separate layers so build logs are clear)
RUN pip install --no-cache-dir -v --no-build-isolation /workspace/gaussian-splatting/submodules/diff-gaussian-rasterization
RUN pip install --no-cache-dir -v --no-build-isolation /workspace/gaussian-splatting/submodules/simple-knn
RUN pip install --no-cache-dir -v --no-build-isolation /workspace/gaussian-splatting/submodules/fused-ssim

COPY handler.py /workspace/handler.py

CMD ["python", "-u", "/workspace/handler.py"]

