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

# Системные пакеты для Pyserini (JDK), OCR (tesseract + языки), рендера (libgl1),
# OpenCV (libglib2.0-0), диагностики (curl), и bash для скриптов
RUN apt-get update && apt-get install -y --no-install-recommends \
      openjdk-21-jdk-headless \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus \
      libgl1 libglib2.0-0 \
      curl ca-certificates \
      bash \
    && rm -rf /var/lib/apt/lists/*

# Виртуалка для зависимостей
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ===== PyTorch =====
# Если нужен CUDA-рантайм — укажи канал:
#   --build-arg TORCH_CHANNEL=https://download.pytorch.org/whl/cu124
# или актуальный канал cu12X (см. документацию PyTorch).
# На CPU: оставь пустым (по умолчанию).
ARG TORCH_CHANNEL=""
ARG TORCH_VERSION="2.3.*"

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -n "$TORCH_CHANNEL" ]; then \
      pip install --no-cache-dir --index-url "$TORCH_CHANNEL" "torch==${TORCH_VERSION}"; \
    else \
      pip install --no-cache-dir "torch==${TORCH_VERSION}"; \
    fi

# Остальные Python-зависимости
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt


############################
# Stage 2: runtime
############################
FROM python:3.11-slim AS runtime
WORKDIR /app

# На runtime тоже нужны JDK, tesseract, libgl1/libglib2.0-0, curl и bash
RUN apt-get update && apt-get install -y --no-install-recommends \
      openjdk-21-jdk-headless \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus \
      libgl1 libglib2.0-0 \
      curl ca-certificates \
      bash \
    && rm -rf /var/lib/apt/lists/*

# JAVA_HOME для pyserini/pyjnius
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Переносим venv со всеми установленными пакетами
COPY --from=deps /opt/venv /opt/venv

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

CMD ["bash", "/app/start.sh"]
