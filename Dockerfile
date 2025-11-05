# syntax=docker/dockerfile:1.7

############################
# Stage 1: deps (Python venv)
############################
FROM python:3.11-slim AS deps
WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- системные пакеты ----
# JDK для Pyserini/Lucene, tesseract + рус/анг языки, GL/GLib для opencv,
# curl/ca-certificates для скачивания моделей EasyOCR, bash для скриптов
RUN apt-get update && apt-get install -y --no-install-recommends \
      openjdk-21-jre-headless \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus \
      libgl1 libglib2.0-0 \
      curl ca-certificates \
      bash \
    && rm -rf /var/lib/apt/lists/*

# Виртуальное окружение
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ------------ PyTorch вариант ------------
# Выбор варианта на этапе сборки:
#   --build-arg TORCH_VARIANT=cpu    (по умолчанию)
#   --build-arg TORCH_VARIANT=cu121  (если нужен CUDA 12.1)
ARG TORCH_VARIANT=cpu
ARG TORCH_VERSION=2.3.*

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$TORCH_VARIANT" = "cu121" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 "torch==${TORCH_VERSION}"; \
    else \
      pip install --no-cache-dir "torch==${TORCH_VERSION}"; \
    fi

# Остальные Python-зависимости
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# ---- подготовка EasyOCR моделей (чтобы не качались в рантайме) ----
ENV EASY_OCR_HOME=/root/.EasyOCR
RUN mkdir -p /root/.EasyOCR && \
    python - <<'PY'
import os
os.environ['EASY_OCR_HOME'] = os.path.expanduser('~/.EasyOCR')
import easyocr
# без GPU, просто загрузка моделей на слой образа
reader = easyocr.Reader(['ru','en'], gpu=False, download_enabled=True)
print("EasyOCR models ready at:", os.environ['EASY_OCR_HOME'])
PY


############################
# Stage 2: runtime
############################
FROM python:3.11-slim AS runtime
WORKDIR /app

# те же системные пакеты нужны в рантайме
RUN apt-get update && apt-get install -y --no-install-recommends \
      openjdk-21-jre-headless \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus \
      libgl1 libglib2.0-0 \
      curl ca-certificates \
      bash \
    && rm -rf /var/lib/apt/lists/*

# JAVA_HOME для Pyserini, Tesseract/ocr env
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:/opt/venv/bin:${PATH}" \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EASY_OCR_HOME=/root/.EasyOCR

# Переносим venv и предзагруженные модели EasyOCR
COPY --from=deps /opt/venv /opt/venv
COPY --from=deps /root/.EasyOCR /root/.EasyOCR

# Копируем код после зависимостей (лучший build cache)
COPY . .

# Стартовый скрипт
RUN chmod +x /app/start.sh

# --------- defaults (перекрывай через compose) ----------
ENV QDRANT_URL=http://qdrant:6333 \
    QDRANT__PREFER_GRPC=false \
    EMB_BACKEND=hf \
    HF_MODEL=BAAI/bge-m3 \
    RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
    MODEL_ID=llama3.1:8b \
    LLM_BASE_URL=http://host.docker.internal:11434 \
    APP_ENV=dev \
    APP_PORT=8000 \
    APP_HOST=0.0.0.0

EXPOSE 8000

# Если compose не переопределит, запустим так
CMD ["bash", "/app/start.sh"]
