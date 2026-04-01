FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

# System deps (قد تحتاج colmap/ffmpeg حسب مشروعك)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl zip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 1) Clone gaussian splatting source
# مهم: لازم تحط رابط repo الصحيح هنا
# مؤقتًا حاطط placeholder — ابعتهولي أو استبدله أنت
ARG GS_REPO=https://github.com/ashawkey/gaussian-splatting.git
RUN git clone --depth 1 ${GS_REPO} /workspace/gs

# 2) Install python deps
WORKDIR /workspace/gs
RUN pip install --no-cache-dir -r requirements.txt || true
# بعض مشاريع gaussian-splatting تحتاج install إضافي:
# RUN pip install --no-cache-dir -e .

# 3) Install serverless handler deps
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
