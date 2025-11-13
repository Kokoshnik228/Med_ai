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
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      curl ca-certificates \
      bash && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
# Чуть ограничим JVM под Pyserini
ENV JAVA_TOOL_OPTIONS="-Xms512m -Xmx2g -XX:+UseG1GC -XX:MaxGCPauseMillis=100"

############################
# Stage 1: deps (Python venv)
############################
FROM base AS deps

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PIP_CACHE_DIR=/root/.cache/pip

# 1) Копируем requirements и вырезаем тяжёлые пакеты (torch/triton/nvidia-* cuda wheels)
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python - <<'PY'
import re, pathlib
src = pathlib.Path("/tmp/requirements.txt").read_text(encoding="utf-8").splitlines()
out=[]
skip = re.compile(r'^\s*(torch|torchvision|torchaudio|triton|nvidia-.*-cu1[23])\b', re.I)
for ln in src:
    if ln.strip().startswith("--hash="):  # чистим lock-хэши, чтобы awk/sed не нужны были
        continue
    if skip.match(ln):
        continue
    out.append(ln)
pathlib.Path("/tmp/requirements.clean").write_text("\n".join(out)+"\n", encoding="utf-8")
print("Generated /tmp/requirements.clean:", len(out), "lines")
PY

# 2) Обновляем pip и ставим все зависимости (кроме torch) с разрешением транзитивных deps
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -U pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.clean

# 3) Ставим Torch c нужным CUDA-каналом отдельным слоем (лучше кешируется)
ARG TORCH_CHANNEL="https://download.pytorch.org/whl/cu128"
ARG TORCH_VERSION="2.9.*"
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir --index-url "${TORCH_CHANNEL}" "torch==${TORCH_VERSION}" && \
    python - <<'PY' || true
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

# Базовые ENV (остальные — через .env.dev/.env.prod и docker-compose)
ENV QDRANT_URL=http://qdrant:6333 \
    QDRANT__PREFER_GRPC=false \
    EMB_BACKEND=hf \
    HF_MODEL=BAAI/bge-m3 \
    RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
    MODEL_ID=deepseek-r1:32b \
    LLM_BASE_URL=http://ollama:11434 \
    APP_ENV=dev \
    APP_PORT=8000 \
    APP_HOST=0.0.0.0 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8000
CMD ["bash", "/app/start.sh"]
