#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 📦 Env (из docker-compose)
# ============================================================
APP_ENV="${APP_ENV:-dev}"             # dev | prod
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
APP_WORKERS="${APP_WORKERS:-1}"       # для prod можно 2–4
APP_MODULE="${APP_MODULE:-api_app:app}"

QDRANT_URL="${QDRANT_URL:-}"          # напр.: http://qdrant:6333
LLM_BASE_URL="${LLM_BASE_URL:-}"      # напр.: http://host.docker.internal:11434

# Embeddings/LLM (полезно для логов)
EMB_BACKEND="${EMB_BACKEND:-hf}"
HF_MODEL="${HF_MODEL:-BAAI/bge-m3}"
HF_DEVICE="${HF_DEVICE:-auto}"
MODEL_ID="${MODEL_ID:-llama3.1:8b}"

# Отключаем gRPC во всех подпроцессах
export QDRANT__PREFER_GRPC="${QDRANT__PREFER_GRPC:-false}"

# Опционально: жёстко ждать готовность сервисов
WAIT_FOR_QDRANT="${WAIT_FOR_QDRANT:-false}"   # true/false
WAIT_FOR_OLLAMA="${WAIT_FOR_OLLAMA:-false}"   # true/false
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"            # сек на ожидание

# ============================================================
# ⚙️  Helpers
# ============================================================
try_curl() {
  # try_curl <url> <timeout_sec> <desc>
  local url="${1:-}"
  local t="${2:-2}"
  local desc="${3:-service}"
  if [[ -z "$url" ]]; then
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time "$t" "$url" >/dev/null 2>&1; then
      echo "✅ $desc доступен: $url"
      return 0
    else
      echo "⚠️  $desc пока недоступен: $url"
      return 1
    fi
  fi
  return 0
}

wait_http() {
  # wait_http <url> <timeout_sec> <desc>
  local url="${1:-}"
  local timeout="${2:-60}"
  local desc="${3:-service}"
  if [[ -z "$url" ]]; then
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "ℹ нет curl, пропускаю ожидание $desc"
    return 0
  fi
  echo "⏳ Жду $desc up to ${timeout}s: $url"
  local end=$((SECONDS + timeout))
  while (( SECONDS < end )); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      echo "✅ $desc готов: $url"
      return 0
    fi
    sleep 1
  done
  echo "⚠️  Не дождался $desc за ${timeout}s (продолжаю запуск)"
  return 1
}

# ============================================================
# 🚀 Инфо о запуске
# ============================================================
echo "▶️  Запуск приложения"
echo "    APP_ENV       = $APP_ENV"
echo "    APP_MODULE    = $APP_MODULE"
echo "    HOST:PORT     = $APP_HOST:$APP_PORT"
echo "    WORKERS       = $APP_WORKERS"
echo "    QDRANT_URL    = ${QDRANT_URL:-<empty>}"
echo "    LLM_BASE_URL  = ${LLM_BASE_URL:-<empty>}"
echo "    EMB_BACKEND   = $EMB_BACKEND"
echo "    HF_MODEL      = $HF_MODEL"
echo "    HF_DEVICE     = $HF_DEVICE"
echo "    MODEL_ID      = $MODEL_ID"
echo "    QDRANT__PREFER_GRPC = $QDRANT__PREFER_GRPC"
echo "============================================================"

# ============================================================
# 🔍 Проверка APP_MODULE (фикс: читаем из env в Python)
# ============================================================
python - <<'PY'
import importlib, os, sys
raw = os.environ.get("APP_MODULE", "api_app:app")
mod, _, attr = raw.partition(":")
if not mod or not attr:
    print(f"[FATAL] Некорректный APP_MODULE='{raw}' (ожидалось 'package.module:app')", file=sys.stderr)
    sys.exit(2)
try:
    m = importlib.import_module(mod)
except Exception as e:
    print(f"[FATAL] Не удалось импортировать модуль '{mod}': {e}", file=sys.stderr)
    sys.exit(3)
if not hasattr(m, attr):
    print(f"[FATAL] В модуле '{mod}' нет объекта '{attr}'", file=sys.stderr)
    sys.exit(4)
print(f"[OK] APP_MODULE проверен: {raw}")
PY

# ============================================================
# 🌐 Проверка внешних сервисов
# ============================================================
QCOL="${QDRANT_URL:+${QDRANT_URL%/}/collections}"
OTAGS="${LLM_BASE_URL:+${LLM_BASE_URL%/}/api/tags}"

# Мягкие проверки (логируем, но не падаем)
try_curl "$QCOL" 2 "Qdrant"
try_curl "$OTAGS" 2 "Ollama"

# Жёсткое ожидание по флагам
if [[ "$WAIT_FOR_QDRANT" == "true" && -n "$QCOL" ]]; then
  wait_http "$QCOL" "$WAIT_TIMEOUT" "Qdrant"
fi
if [[ "$WAIT_FOR_OLLAMA" == "true" && -n "$OTAGS" ]]; then
  wait_http "$OTAGS" "$WAIT_TIMEOUT" "Ollama"
fi
echo "============================================================"

# ============================================================
# ⚡ Запуск Uvicorn
# ============================================================
if [[ "$APP_ENV" == "dev" ]]; then
  echo "🔧 DEV-режим (autoreload включён)"
  exec python -m uvicorn "$APP_MODULE" \
    --host "$APP_HOST" \
    --port "$APP_PORT" \
    --reload \
    --proxy-headers \
    --log-level info
else
  echo "🚀 PROD-режим (workers: $APP_WORKERS)"
  exec python -m uvicorn "$APP_MODULE" \
    --host "$APP_HOST" \
    --port "$APP_PORT" \
    --workers "$APP_WORKERS" \
    --proxy-headers \
    --log-level info
fi
