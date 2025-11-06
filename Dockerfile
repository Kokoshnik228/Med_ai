# syntax=docker/dockerfile:1.7

############################
# Stage 0: base (OS deps)
############################
FROM python:3.11-slim AS base
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Кэшируем ТОЛЬКО /var/cache/apt (sharing=locked). /var/lib/apt НЕ кэшируем.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      openjdk-21-jdk-headless \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus \
      libgl1 libglib2.0-0 \
      curl ca-certificates \
      bash && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

############################
# Stage 1: deps (Python venv)
############################
FROM base AS deps

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PIP_CACHE_DIR=/root/.cache/pip

# 1) Копируем lock и вырезаем тяжёлые пакеты (torch/xformers/triton и nvidia-* cuda wheels)
COPY requirements.txt /tmp/requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    awk 'BEGIN{skip=0} \
         /^[[:space:]]*(torch|torchvision|torchaudio)==/ {next} \
         /^[[:space:]]*nvidia-.*-cu1[23]/ {next} \
         /^[[:space:]]*triton==/ {next} \
         {print}' /tmp/requirements.lock \
    | sed -E '/--hash=sha256:[0-9a-f]{64}/d' \
    | sed -E 's/[[:space:]]*\\$//g' \
    > /tmp/requirements.clean && \
    pip install --no-cache-dir --no-deps -r /tmp/requirements.clean

# 2) Ставим «остальные» пакеты из очищенного lock-файла БЕЗ разрешения зависимостей
#    (в lock уже зафиксированы все транзитивные зависимости, кроме удалённых нами torch/nvidia-*)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir --no-deps -r /tmp/requirements.clean

# 3) Ставим Torch с CUDA отдельно (по умолчанию cu124 — оптимально для RTX 50xx/5090)
ARG TORCH_CHANNEL="https://download.pytorch.org/whl/cu128"
ARG TORCH_VERSION="2.9.*"
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir --index-url "$TORCH_CHANNEL" "torch==${TORCH_VERSION}"
# Sanity-check (не валим билд, просто печатаем)
RUN python - <<'PY' || true
import torch
print("Torch:", torch.__version__, "| CUDA avail:", torch.cuda.is_available(), "| CUDA:", torch.version.cuda)
PY

############################
# Stage 2: runtime
############################
FROM base AS runtime
WORKDIR /app

COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY . .

RUN chmod +x /app/start.sh

ENV QDRANT_URL=http://qdrant:6333 \
    QDRANT__PREFER_GRPC=false \
    EMB_BACKEND=hf \
    HF_MODEL=BAAI/bge-m3 \
    RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
    MODEL_ID=llama3.1:8b \
    LLM_BASE_URL=http://host.docker.internal:11434 \
    APP_ENV=dev \
    APP_PORT=8000 \
    APP_HOST=0.0.0.0 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

EXPOSE 8000
CMD ["bash", "/app/start.sh"]
