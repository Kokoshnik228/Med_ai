#!/usr/bin/env bash
set -euo pipefail

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
APP_ENV="${APP_ENV:-dev}"
QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
LLM_BASE_URL="${LLM_BASE_URL:-http://host.docker.internal:11434}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

cd /app
export PATH="/opt/venv/bin:${PATH}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"
export PATH="${JAVA_HOME}/bin:${PATH}"

export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/.cache/huggingface}"
export EASYOCR_DIR="${EASYOCR_DIR:-/root/.EasyOCR}"
mkdir -p "${TRANSFORMERS_CACHE}" "${EASYOCR_DIR}/model"

# Предпочесть GPU для эмбеддингов, если доступен
if [[ -z "${HF_DEVICE:-}" ]]; then
  python - >/dev/null 2>&1 <<'PY'
import sys
try:
    import torch
    sys.exit(0 if torch.cuda.is_available() else 1)
except Exception:
    sys.exit(1)
PY
  rc=$?
  if [[ $rc -eq 0 ]]; then export HF_DEVICE="cuda"; else export HF_DEVICE="auto"; fi
fi

echo "== med_ai start =="
echo " APP_ENV = ${APP_ENV}"
echo " APP_HOST= ${APP_HOST}"
echo " APP_PORT= ${APP_PORT}"
echo " QDRANT  = ${QDRANT_URL}"
echo " LLM_URL = ${LLM_BASE_URL}"
echo " HF_DEVICE = ${HF_DEVICE}"
echo " WEB_CONCURRENCY = ${WEB_CONCURRENCY}"

if command -v curl >/dev/null 2>&1; then
  curl -fsS "${QDRANT_URL%/}/readyz" >/dev/null 2>&1 && echo "Qdrant ready." || \
    echo "WARN: Qdrant not ready yet (${QDRANT_URL%/}/readyz). Продолжаем."
  curl -fsS "${LLM_BASE_URL%/}/api/tags" >/dev/null 2>&1 && echo "Ollama reachable." || \
    echo "WARN: Ollama not reachable at ${LLM_BASE_URL}."
fi

UVICORN_FLAGS=(
  --host "${APP_HOST}"
  --port "${APP_PORT}"
  --proxy-headers
  --forwarded-allow-ips=*
  --log-level "${LOG_LEVEL}"
)
if [[ "${WEB_CONCURRENCY}" != "1" ]]; then
  UVICORN_FLAGS+=(--workers "${WEB_CONCURRENCY}")
fi

exec python -m uvicorn api_app:app "${UVICORN_FLAGS[@]}"
