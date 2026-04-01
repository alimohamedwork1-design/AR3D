FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates zip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git /workspace/gaussian-splatting

WORKDIR /workspace/gaussian-splatting
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py
CMD ["python", "/app/handler.py"]
