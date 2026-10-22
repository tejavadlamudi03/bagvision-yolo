# BagVision YOLO - for training and for checking saved photos (no camera window in Docker).
#   docker build -t bagvision-yolo .
#   docker run --rm -v "$PWD/models:/app/models" -v "$PWD/photos:/photos" bagvision-yolo check /photos
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU version of PyTorch (smaller); build with --build-arg TORCH=cu124 for an NVIDIA GPU
ARG TORCH=cpu
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/${TORCH} \
    && pip install --no-cache-dir -r requirements.txt

COPY bag_inspect.py .env ./
ENV YOLO_OFFLINE=1 PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/Ultralytics
VOLUME ["/app/models", "/app/bag_data", "/app/inspections"]
ENTRYPOINT ["python", "bag_inspect.py"]
CMD ["--help"]
