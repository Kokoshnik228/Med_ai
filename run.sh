#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
ACTION="${2:-up}"
ARG3="${3:-}"     # опция: для set-emb / set-gpu

usage() {
  cat <<'EOF'
⚙️  Использование: ./run.sh [dev|prod] [up|down|rebuild|restart|logs|logs-app|ps|sh|env|set-emb|set-gpu|health|pull|build|down-v] [опции]

Примеры:
  ./run.sh dev                  # Запуск dev (БЕЗ сборки)
  ./run.sh prod                 # Запуск prod (со сборкой)
  ./run.sh dev down             # Остановить dev
  ./run.sh prod rebuild         # Пересобрать и перезапустить prod
  ./run.sh dev logs             # Логи всех сервисов (Ctrl+C для выхода)
  ./run.sh dev logs-app         # Логи только сервиса app
  ./run.sh prod ps              # Состояние контейнеров prod
  ./run.sh dev sh               # Shell в контейнер app (dev)

  # Эмбеддинги (HF)
  ./run.sh dev env              # Показать текущие переменные для эмбеддинга
  ./run.sh dev set-emb hf       # Записать в .env.dev: EMB_BACKEND=hf, HF_MODEL=...

  # GPU-профиль (compose profile "gpu")
  ./run.sh dev set-gpu on       # Включить COMPOSE_PROFILES=gpu в .env.dev
  ./run.sh dev set-gpu off      # Выключить профили (очистить переменную)
  ./run.sh dev set-gpu auto     # Авто: если есть nvidia, включит gpu

  # Дополнительно
  ./run.sh dev health           # Пинг http://localhost:7050/health
  ./run.sh prod pull            # docker compose pull
  ./run.sh prod build           # docker compose build
  ./run.sh prod down-v          # down -v (сносит volume’ы)
EOF
}

if [[ -z "$MODE" ]]; then usage; exit 1; fi

case "$MODE" in
  dev)
    COMPOSE_FILE="docker-compose.dev.yml"
    ENV_FILE=".env.dev"
    URL_HINT="http://localhost:7050"
    APP_SERVICE="app"
    ;;
  prod)
    COMPOSE_FILE="docker-compose.prod.yml"
    ENV_FILE=".env.prod"
    URL_HINT="http://localhost:8050"
    APP_SERVICE="app"
    ;;
  *)
    echo "❌ Неизвестный режим: $MODE (нужно dev или prod)"; exit 1 ;;
esac

[[ -f "$COMPOSE_FILE" ]] || { echo "❌ Не найден $COMPOSE_FILE"; exit 1; }
[[ -f "$ENV_FILE"     ]] || { echo "❌ Не найден $ENV_FILE"; exit 1; }

# autodetect docker compose CLI
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  DCMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1 && docker-compose version >/dev/null 2>&1; then
  DCMD=(docker-compose)
else
  echo "❌ Docker Compose не найден. Установи 'docker compose' (v2) или 'docker-compose' (v1)."
  exit 1
fi

# Проверим, поддерживается ли флаг --ansi (в некоторых версиях его нет)
compose_supports_ansi() {
  "${DCMD[@]}" up --help 2>/dev/null | grep -q -- '--ansi' || return 1
}
ANSI_FLAGS=()
if compose_supports_ansi; then
  ANSI_FLAGS=(--ansi=always)
fi

# ---------- helpers для .env ----------
_sed_in_place() {
  local file="$1"; shift
  if sed --version >/dev/null 2>&1; then
    sed -i "$@" "$file"        # GNU sed
  else
    sed -i '' "$@" "$file"     # BSD/macOS sed
  fi
}

_escape_regex() { printf '%s' "$1" | sed -e 's/[.[\*^$\/|&()-]/\\&/g'; }

# безопасная подстановка key=val (создаём ключ, если его нет)
set_kv() {
  local file="$1" key="$2" val="$3"
  local key_re="$(_escape_regex "$key")"
  if grep -Eq "^[[:space:]]*${key_re}[[:space:]]*=" "$file"; then
    _sed_in_place "$file" "s|^[[:space:]]*${key_re}[[:space:]]*=.*$|${key}=${val}|g"
  else
    echo "${key}=${val}" >> "$file"
  fi
}

# чтение значения (игнор комментов, пробелы вокруг '='); берём последнее
get_kv() {
  local file="$1" key="$2"
  awk -F= -v k="$key" '
    $0 !~ /^[[:space:]]*#/ && $1==k { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); v=$2 }
    END{ if (v!="") print v }
  ' "$file"
}

print_embed_config() {
  local file="$1"
  local backend hf_model
  backend="$(get_kv "$file" "EMB_BACKEND" || true)"
  hf_model="$(get_kv "$file" "HF_MODEL" || true)"
  echo "🔧 EMB_BACKEND = ${backend:-<не задан>}"
  echo "   HF_MODEL    = ${hf_model:-BAAI/bge-m3 (по умолчанию)}"
}

ensure_defaults() {
  local file="$1"
  local backend hf_model
  backend="$(get_kv "$file" "EMB_BACKEND" || true)"
  if [[ -z "${backend:-}" ]]; then
    set_kv "$file" "EMB_BACKEND" "hf"
  fi
  hf_model="$(get_kv "$file" "HF_MODEL" || true)"
  [[ -n "$hf_model" ]] || set_kv "$file" "HF_MODEL" "BAAI/bge-m3"
}

set_emb_backend() {
  local file="$1" backend="$2"
  case "$backend" in
    hf)
      set_kv "$file" "EMB_BACKEND" "hf"
      set_kv "$file" "HF_MODEL" "BAAI/bge-m3"
      echo "✅ Установлен EMB_BACKEND=hf и HF_MODEL=BAAI/bge-m3 в ${file}"
      ;;
    *)
      echo "❌ set-emb: сейчас поддерживается только 'hf'"; exit 1 ;;
  esac
}

detect_gpu() {
  # 0 = успех (есть nvidia), 1 = нет
  if command -v nvidia-smi >/dev/null 2>&1; then return 0; fi
  if command -v docker >/dev/null 2>&1; then
    if docker info --format '{{json .Runtimes.nvidia}}' 2>/dev/null | grep -qv 'null'; then
      return 0
    fi
  fi
  return 1
}

set_gpu_profile() {
  local file="$1" mode="$2"
  case "$mode" in
    on)
      set_kv "$file" "COMPOSE_PROFILES" "gpu"
      echo "✅ Включён GPU-профиль (COMPOSE_PROFILES=gpu) в ${file}"
      ;;
    off)
      set_kv "$file" "COMPOSE_PROFILES" ""
      echo "✅ Выключены compose-профили (COMPOSE_PROFILES очищен) в ${file}"
      ;;
    auto)
      if detect_gpu; then
        set_kv "$file" "COMPOSE_PROFILES" "gpu"
        echo "✅ Авто: обнаружена NVIDIA, включён COMPOSE_PROFILES=gpu в ${file}"
      else
        set_kv "$file" "COMPOSE_PROFILES" ""
        echo "ℹ️  Авто: NVIDIA не найдена, профили очищены (CPU-режим)"
      fi
      ;;
    *)
      echo "❌ set-gpu: используй on|off|auto"; exit 1 ;;
  esac
}

print_profiles_hint() {
  local file="$1"
  local prof="$(get_kv "$file" "COMPOSE_PROFILES" || true)"
  echo "   COMPOSE_PROFILES = ${prof:-<пусто>} (gpu-профиль включай через: ./run.sh ${MODE} set-gpu on)"
}

# ---------- действия чисто для env / set-emb / set-gpu ----------
case "$ACTION" in
  env)
    echo "📄 Просмотр конфигурации эмбеддинга для $MODE ($ENV_FILE)"
    ensure_defaults "$ENV_FILE"
    print_embed_config "$ENV_FILE"
    print_profiles_hint "$ENV_FILE"
    exit 0
    ;;
  set-emb)
    [[ -n "$ARG3" ]] || { echo "❌ Укажи бэкенд: hf"; exit 1; }
    set_emb_backend "$ENV_FILE" "$ARG3"
    echo "ℹ️  Текущая конфигурация:"
    print_embed_config "$ENV_FILE"
    print_profiles_hint "$ENV_FILE"
    exit 0
    ;;
  set-gpu)
    [[ -n "$ARG3" ]] || { echo "❌ Укажи режим: on|off|auto"; exit 1; }
    set_gpu_profile "$ENV_FILE" "$ARG3"
    print_profiles_hint "$ENV_FILE"
    exit 0
    ;;
esac

# Перед запуском — проставим умолчания
ensure_defaults "$ENV_FILE"

echo "🔎 Эмбеддинг-конфиг ($MODE):"
print_embed_config "$ENV_FILE"
print_profiles_hint "$ENV_FILE"
echo

# Прочитаем COMPOSE_PROFILES из env-файла и экспортнём в окружение CLI,
# чтобы профиль гарантированно применился.
CURRENT_PROFILES="$(get_kv "$ENV_FILE" "COMPOSE_PROFILES" || true || printf '')"
if [[ -n "${CURRENT_PROFILES:-}" ]]; then
  export COMPOSE_PROFILES="${CURRENT_PROFILES}"
else
  unset COMPOSE_PROFILES || true
fi

# URL для health
case "$MODE" in
  dev)  APP_URL="${APP_URL:-http://localhost:7050}" ;;
  prod) APP_URL="${APP_URL:-http://localhost:8050}" ;;
esac

_has_jq() { command -v jq >/dev/null 2>&1; }

case "$ACTION" in
  up)
    echo "🚀 Запуск $MODE-среды..."
    if [[ "$MODE" == "dev" ]]; then
      "${DCMD[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --no-build --remove-orphans "${ANSI_FLAGS[@]}"
    else
      "${DCMD[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --build --remove-orphans "${ANSI_FLAGS[@]}"
    fi
    echo "⏳ Проверка здоровья сервиса..."
    if command -v curl >/dev/null 2>&1; then
      if _has_jq; then
        curl -fsS "${APP_URL%/}/health" | jq . || true
      else
        curl -fsS "${APP_URL%/}/health" || true
      fi
    fi
    echo "✅ Готово. Сервис: ${APP_URL}"
    ;;
  down)
    echo "🛑 Останавливаем контейнеры ($MODE)..."
    "${DCMD[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down --remove-orphans
    ;;
  down-v)
    echo "🧨 Останавливаем и удаляем volumes ($MODE)..."
    "${DCMD[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down -v --remove-orphans
    ;;
  rebuild)
    echo "🔄 Пересборка и перезапуск ($MODE)..."
    "${DCMD[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --build --remove-orphans "${ANSI_FLAGS[@]}"
    ;;
  restart)
    echo "♻️  Перезапуск ($MODE)..."
    "${DCMD[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" restart
    ;;
  logs)
    echo "📜 Логи ($MODE)... (Ctrl+C для выхода)"
    "${DCMD[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f
    ;;
  logs-app)
    echo "📜 Логи сервиса $APP_SERVICE ($MODE)... (Ctrl+C для выхода)"
    "${DCMD[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f "$APP_SERVICE"
    ;;
  ps)
    "${DCMD[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps
    ;;
  sh)
    echo "🧰 Входим в контейнер $APP_SERVICE ($MODE)..."
    "${DCMD[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec "$APP_SERVICE" bash
    ;;
  pull)
    echo "📥 docker compose pull ($MODE)..."
    "${DCMD[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull
    ;;
  build)
    echo "🛠  docker compose build ($MODE)..."
    "${DCMD[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build
    ;;
  health)
    echo "🩺 Проверка сервисов ($MODE)..."
    if command -v curl >/dev/null 2>&1; then
      echo "— app health:"
      if _has_jq; then
        (curl -fsS "${APP_URL%/}/health" | jq .) || true
      else
        curl -fsS "${APP_URL%/}/health" || true
      fi
    fi
    ;;
  *)
    echo "❌ Неизвестное действие: $ACTION"
    usage
    exit 1
    ;;
esac
