FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

# أدوات بناء + utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates zip ffmpeg \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# مهم جدًا: ثبّت numpy<2 قبل أي packages قد تعمل compilation
RUN python -m pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir "numpy<2"

# Clone gaussian-splatting + submodules
RUN git clone --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git /workspace/gaussian-splatting
WORKDIR /workspace/gaussian-splatting
RUN git submodule update --init --recursive

# Python deps أساسية (نزودها لاحقًا لو ظهر نقص في logs)
RUN pip install --no-cache-dir \
    tqdm pillow opencv-python imageio scipy matplotlib

# Serverless handler deps
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
