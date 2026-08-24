FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
# Install CPU-only torch from dedicated index (avoids 2GB+ CUDA bundle)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchaudio
# Install everything else from PyPI (torch/torchaudio already satisfied, skipped)
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY app/ ./app/
COPY weights/ ./weights/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ADFD_LOG_LEVEL=INFO \
    ADFD_API_PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
