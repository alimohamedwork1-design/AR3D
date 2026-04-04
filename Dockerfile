# استخدم base image جاهزة من RunPod بدل ما تبني من الصفر
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

# Install system deps
RUN apt-get update && apt-get install -y \
    colmap \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    runpod \
    requests \
    Pillow \
    numpy \
    boto3

# Install gaussian-splatting
RUN git clone --depth 1 \
    https://github.com/graphdeco-inria/gaussian-splatting \
    /workspace/gaussian-splatting

RUN pip install --no-cache-dir \
    plyfile \
    tqdm \
    scipy

# Copy handler
COPY handler.py /workspace/handler.py

CMD ["python", "-u", "/workspace/handler.py"]
