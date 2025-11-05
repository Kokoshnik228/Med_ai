#!/usr/bin/env bash
set -euo pipefail

# === базовые переменные (приходят из compose/env) ===
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
APP_ENV="${APP_ENV:-dev}"
QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"

echo "== med_ai start =="
echo " APP_ENV = ${APP_ENV}"
echo " APP_HOST= ${APP_HOST}"
echo " APP_PORT= ${APP_PORT}"
echo " QDRANT  = ${QDRANT_URL}"

cd /app

# (опционально) проверим, что Qdrant доступен — не фейлим, просто предупреждаем
if command -v curl >/dev/null 2>&1; then
  if curl -fsS "${QDRANT_URL%/}/readyz" >/dev/null 2>&1; then
    echo "Qdrant ready."
  else
    echo "WARN: Qdrant not ready yet (${QDRANT_URL}/readyz). Продолжаем, API сам переждёт."
  fi
fi

# Убедимся, что зависимости в PATH (venv переносится в Dockerfile)
export PATH="/opt/venv/bin:${PATH}"

# Запускаем API
exec uvicorn api_app:app --host "${APP_HOST}" --port "${APP_PORT}" --proxy-headers
