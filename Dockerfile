FROM ashawkey/gaussian-splatting-webui:latest

WORKDIR /app

# تثبيت بايثون deps للـ handler
RUN pip install --no-cache-dir -r /dev/null || true
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# نسخ handler
COPY handler.py /app/handler.py

# تشغيل الـ serverless worker
CMD ["python", "/app/handler.py"]
