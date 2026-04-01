FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates zip ffmpeg \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Clone gaussian-splatting + submodules
RUN git clone --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git /workspace/gaussian-splatting
WORKDIR /workspace/gaussian-splatting
RUN git submodule update --init --recursive

# Install python deps (بدون numpy هنا)
RUN pip install --no-cache-dir \
    tqdm pillow opencv-python imageio scipy matplotlib

# Install serverless deps (يتضمن numpy<2)
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Force pin numpy<2 كآخر خطوة (حتى لو أي dependency حاول يرفعه)
RUN python -m pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --force-reinstall "numpy<2" \
 && python -c "import numpy as np; print('NUMPY_VERSION', np.__version__)"

COPY handler.py /app/handler.py
CMD ["python", "/app/handler.py"]
