#!/usr/bin/env bash
set -euo pipefail

# -------- базовые переменные (с дефолтами) --------
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
APP_ENV="${APP_ENV:-dev}"
QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
LLM_BASE_URL="${LLM_BASE_URL:-http://ollama:11434}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

cd /app

# -------- пути и кэши --------
export PATH="/opt/venv/bin:${PATH}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"
export PATH="${JAVA_HOME}/bin:${PATH}"

export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/.cache/huggingface}"
export EASYOCR_DIR="${EASYOCR_DIR:-/root/.EasyOCR}"
mkdir -p "${TRANSFORMERS_CACHE}" "${EASYOCR_DIR}/model"

# -------- авто-детект GPU для эмбеддингов --------
if [[ -z "${HF_DEVICE:-}" ]]; then
  if python - <<'PY' >/dev/null 2>&1
import sys
try:
    import torch
    sys.exit(0 if torch.cuda.is_available() else 1)
except Exception:
    sys.exit(1)
PY
  then export HF_DEVICE="cuda"; else export HF_DEVICE="auto"; fi
fi

echo "== med_ai start =="
echo " APP_ENV  = ${APP_ENV}"
echo " APP_HOST = ${APP_HOST}"
echo " APP_PORT = ${APP_PORT}"
echo " QDRANT   = ${QDRANT_URL}"
echo " LLM_URL  = ${LLM_BASE_URL}"
echo " HF_DEVICE= ${HF_DEVICE}"
echo " WEB_CONCURRENCY = ${WEB_CONCURRENCY}"
echo " LOG_LEVEL       = ${LOG_LEVEL}"

# -------- подсказки по доступности сервисов (мягкие) --------
if command -v curl >/dev/null 2>&1; then
  curl -fsS "${QDRANT_URL%/}/readyz" >/dev/null 2>&1 \
    && echo "Qdrant ready." \
    || echo "WARN: Qdrant not ready yet (${QDRANT_URL%/}/readyz)"
  curl -fsS "${LLM_BASE_URL%/}/api/tags" >/dev/null 2>&1 \
    && echo "Ollama reachable." \
    || echo "WARN: Ollama not reachable at ${LLM_BASE_URL}"
fi

# -------- uvicorn флаги --------
UVICORN_FLAGS=(
  --host "${APP_HOST}"
  --port "${APP_PORT}"
  --proxy-headers
  --forwarded-allow-ips=*
  --log-level "${LOG_LEVEL}"
)
# ВНИМАНИЕ: больше 1 воркера заставит дублировать загрузку моделей/весов.
# Держите 1, пока не будете уверены, что это нужно.
if [[ "${WEB_CONCURRENCY}" != "1" ]]; then
  UVICORN_FLAGS+=(--workers "${WEB_CONCURRENCY}")
fi

# -------- запуск API --------
exec python -m uvicorn api_app:app "${UVICORN_FLAGS[@]}"
