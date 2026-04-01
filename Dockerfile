FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

# System build tools + utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates zip ffmpeg \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Clone gaussian-splatting
RUN git clone --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git /workspace/gaussian-splatting

# Pull submodules (important)
WORKDIR /workspace/gaussian-splatting
RUN git submodule update --init --recursive

# Minimal python deps (we will extend after first runtime error if needed)
RUN pip install --no-cache-dir \
    numpy tqdm pillow opencv-python imageio scipy matplotlib

# Serverless handler deps
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
