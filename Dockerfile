FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

COPY requirements.txt .

RUN pip install --no-cache-dir flask pillow numpy gunicorn
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir tensorflow==2.16.1

COPY . .

RUN git lfs pull || true

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "--workers", "1", "app:app"]