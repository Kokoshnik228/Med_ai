#!/usr/bin/env python3
from __future__ import annotations
from config.runtime_settings import settings


import json
import os
import re
import socket
import subprocess
import threading
import time
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

# ================================
# .env -> runtime_settings (правильный порядок)
# ================================
try:
    from dotenv import load_dotenv
    env_mode = (os.getenv("APP_ENV") or "dev").strip().lower()
    env_file = Path(".env.dev" if env_mode == "dev" else ".env.prod")
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
        print(f"🔧 Loaded env: {env_file}")
except Exception as e:
    print(f"⚠️ dotenv load skipped: {e}")

# runtime settings: после .env
from config.runtime_settings import settings  # noqa: E402
try:
    # важное: продавливаем CONTROL в окружение и объект
    settings.apply_env(force=True)
except Exception:
    pass

# ================================
# Импорты после настройки окружения
# ================================
from glob import glob  # noqa: F401
import requests
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# pydantic v1/v2 совместимость
try:
    from pydantic import field_validator  # type: ignore
    _P_V2 = True
except Exception:
    from pydantic import validator as field_validator  # type: ignore
    _P_V2 = False

# RAG utils
from rag.bm25_utils import bm25_search, retrieve_hybrid, embed_query_hf  # noqa: F401

# Глобальная HTTP-сессия (keep-alive)
_HTTP = requests.Session()
_HTTP.headers.update({"Connection": "keep-alive"})

# ================================
# FastAPI
# ================================
app = FastAPI(title="med_ai RAG API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# [api_app.py] — положи это рядом с остальными вспомогательными функциями
def _as_int(v, default):
    try:
        return int(str(v))
    except Exception:
        return default

def _as_float(v, default):
    try:
        return float(str(v))
    except Exception:
        return default

def _llm_conf_from_settings() -> dict:
    """
    Централизованная конфигурация вызова LLM.
    Источники приоритезированы: ENV > settings > дефолт.
    Гарантирует наличие всех ключей, чтобы не было KeyError.
    """
    # базовые дефолты
    d_model       = getattr(settings, "LLM_MODEL", "llama3.1:8b")
    d_base_url    = getattr(settings, "LLM_BASE_URL", "http://host.docker.internal:11434")
    d_num_ctx     = getattr(settings, "LLM_NUM_CTX", 4096)
    d_max_tokens  = getattr(settings, "LLM_MAX_TOKENS", 300)
    d_timeout_s   = getattr(settings, "LLM_TIMEOUT", 60)
    d_temp        = 0.4
    d_top_p       = 0.95
    d_repeat_pen  = 1.05
    d_gpu_layers  = -1
    d_keep_alive  = "30m"

    conf = {
        "model":        os.getenv("LLM_MODEL", d_model),
        "base_url":     os.getenv("LLM_BASE_URL", d_base_url),

        # лимиты
        "num_ctx_cap":  _as_int(os.getenv("LLM_NUM_CTX"), d_num_ctx),
        "max_tokens":   _as_int(os.getenv("LLM_MAX_TOKENS"), d_max_tokens),
        "timeout_s":    _as_int(os.getenv("LLM_TIMEOUT"), d_timeout_s),

        # сэмплинг
        "temperature":      _as_float(os.getenv("LLM_TEMPERATURE"), d_temp),
        "top_p":            _as_float(os.getenv("LLM_TOP_P"), d_top_p),
        "repeat_penalty":   _as_float(os.getenv("LLM_REPEAT_PENALTY"), d_repeat_pen),

        # GPU/сеанс
        "gpu_layers":   _as_int(os.getenv("LLM_NUM_GPU_LAYERS"), d_gpu_layers),
        "keep_alive":   os.getenv("LLM_KEEP_ALIVE", d_keep_alive),
    }
    return conf
# ================================
# Config (yaml + runtime overrides)
# ================================
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
print(f"📂 CWD set to: {Path.cwd()}")
CONF_DIR = ROOT / "config"
DEFAULT_YAML = CONF_DIR / "default.yaml"
LOCAL_YAML = CONF_DIR / "local.yaml"


def load_runtime_overrides() -> Dict[str, Any]:
    """Подхватывает config/runtime_settings.py (dict RUNTIME), можно менять без пересборки."""
    try:
        import config.runtime_settings as rs  # type: ignore
        importlib.reload(rs)
        data = getattr(rs, "RUNTIME", None)
        if isinstance(data, dict):
            print("🔁 runtime_settings.py loaded")
            return data
    except Exception as e:
        print(f"⚠️ runtime overrides not loaded: {e}")
    return {}

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off", ""):
        return False
    # на всякий случай: попробуем как int
    try:
        return bool(int(v))
    except Exception:
        return bool(default)

def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def load_config() -> Dict[str, Any]:
    
    def _load_yaml(p: Path) -> Dict[str, Any]:
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    DEFAULTS = {
        "app": {"data_dir": "data", "bm25_index_dir": "index/bm25_idx"},
        "qdrant": {
            "url": os.getenv("QDRANT_URL", "http://qdrant:6333"),
            "collection": os.getenv("QDRANT_COLLECTION", "med_kb_v3"),
        },
        "ollama": {
            "base_url": os.getenv("LLM_BASE_URL", "http://host.docker.internal:11434"),
            "model": os.getenv("LLM_MODEL", os.getenv("MODEL_ID", "llama3.1:8b")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
            "timeout_s": int(os.getenv("LLM_TIMEOUT", "60")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.4")),
            "top_p": float(os.getenv("LLM_TOP_P", "0.95")),
            "num_ctx": int(os.getenv("LLM_NUM_CTX", "6144")),
        },


        "retrieval": {"k": settings.RETR_TOP_K},
        "embedding": {
            "backend": os.getenv("EMB_BACKEND", settings.EMB_BACKEND or "hf"),
            "model": os.getenv("HF_MODEL", settings.HF_MODEL or "BAAI/bge-m3"),
            "device": os.getenv("HF_DEVICE", settings.HF_DEVICE or "auto"),
            "fp16": _env_flag("HF_FP16", bool(getattr(settings, "HF_FP16", True))),
        },

        
        "chunking": {"child_w": 200, "child_overlap": 35, "parent_w": 800},
        "prompt": {
            "system": (
                "Ты — медицинский ассистент. ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. "
                "Анализируй текст как врачебный кейс: в нём могут быть жалобы, анамнез, осмотр, диагноз и назначения. "
                "Распознай диагноз и предложенные меры, сопоставь их с контекстом базы знаний. "
                "Не повторяй за врачем слово в слово, ты должен только дополнять его речь. "
                "Верни СТРОГО ВАЛИДНЫЙ JSON со схемой:\n"
                "{score, subscores, critical_errors[], recommendations[], citations[], disclaimer}\n"
                "- score: число 0..100 (точность назначения).\n"
                "- subscores: карта подоценок (например: dosing, diagnosis_match, interactions…).\n"
                "- critical_errors: список объектов {type, explain}.\n"
                "- recommendations: список объектов {what_to_change, rationale}.\n"
                "- citations: список строк-источников только из переданного КОНТЕКСТА.\n"
                "- disclaimer: короткое предупреждение на русском.\n"
                "Если уверенности нет — снижай score, добавляй пояснение в disclaimer. ВНЕШНИЕ источники не используй."
            ),
            "user_template": (
                "[КЕЙС]\n{case_text}\n\n"
                "[КОНТЕКСТ]\n{ctx}\n\n"
                "Верни ТОЛЬКО один валидный JSON по указанной схеме.Без Markdown. Все тексты внутри — на русском. "
                "Источники указывай только из контекста. Без поясняющего текста вокруг."
            ),
        },
    }

    base = _load_yaml(DEFAULT_YAML)
    local = _load_yaml(LOCAL_YAML)
    runtime = load_runtime_overrides()

    def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in (b or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out

    return merge(DEFAULTS, merge(base, merge(local, runtime)))

CONFIG = load_config()

def cfg(*path: str, default: Any = None) -> Any:
    cur: Any = CONFIG
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def cfg_int(*path, default: int, allow_zero: bool = False) -> int:
    v = cfg(*path, default=None)
    try:
        v = int(v)
        if (not allow_zero and v <= 0) or (allow_zero and v < 0):
            raise ValueError
        return v
    except Exception:
        return int(default)

def cfg_float(*path, default: float) -> float:
    v = cfg(*path, default=None)
    try:
        return float(v)
    except Exception:
        return float(default)

def cfg_str(*path, default: str) -> str:
    v = cfg(*path, default=None)
    return str(v) if (v is not None and str(v).strip() != "") else str(default)

# ================================
# Warmup BM25 (один раз)
# ================================
WARMUP_DONE = False

@app.on_event("startup")
def warmup_bm25():
    global WARMUP_DONE
    if WARMUP_DONE or os.getenv("BM25_WARMUP_DISABLED") == "1":
        return
    try:
        idx = settings.BM25_INDEX_DIR
        bm25_search(idx, "тест", topk=1)  # прогрев JVM + индекса
        print("🔥 BM25 warmed up")
        WARMUP_DONE = True
    except Exception as e:
        print(f"⚠️ BM25 warmup skipped: {e}")

# ================================
# Utils
# ================================
def looks_meaningless(text: str) -> bool:
    t = (text or "").strip().lower()
    if len(t) < 3:
        return True
    if re.fullmatch(r"[a-z]\d{1,2}(\.\d+)?", t):
        return False
    if not re.search(r"[a-zа-яё0-9]", t):
        return True
    letters_only = re.fullmatch(r"[a-zа-яё\s]+", t)
    if letters_only and len(set(t)) < 5 and len(t) < 20:
        return True
    return False

def build_context_citations(ctx_items, max_out: int = 5):
    return [f"{it['doc_id']} стр.{it['page_start']}-{it['page_end']}" for it in ctx_items[:max_out]]

def build_ctx_string(ctx_items, max_chars: int = 8000, per_text_limit: int = 800) -> str:
    parts, total = [], 0
    for i, it in enumerate(ctx_items, 1):
        txt = (it.get("text", "") or "")[:per_text_limit]
        chunk = f"### [{i}] DOC {it['doc_id']} P{it['page_start']}-{it['page_end']}\n{txt}\n\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts)

def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)

# ================================
# Qdrant client (REST)
# ================================
def _qdrant_client_rest(url_override: Optional[str] = None):
    from qdrant_client import QdrantClient
    url = (url_override or cfg("qdrant", "url", default=settings.QDRANT_URL))
    if "qdrant:" in url:
        try:
            socket.gethostbyname("qdrant")
        except socket.gaierror:
            url = "http://localhost:7779"
    return QdrantClient(url=url, timeout=10, prefer_grpc=False, grpc_port=None)

# ================================
# LLM через Ollama
# ================================
def _trim_code_fences(txt: str) -> str:
    txt = re.sub(r"^\s*```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s*```\s*$", "", txt)
    return txt.strip()

def safe_json_extract(s: str) -> Dict[str, Any]:
    import json as _json, re

    def _default():
        return {
            "score": None, "subscores": {}, "critical_errors": [],
            "recommendations": [], "citations": [],
            "disclaimer": "Парсинг ответа не удался."
        }

    if not s:
        return _default()

    s1 = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).replace("```", "").strip()

    try:
        obj = _json.loads(s1)
        if isinstance(obj, str):
            try:
                return _json.loads(obj)
            except Exception:
                pass
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start, depth, best = None, 0, None
    for i, ch in enumerate(s1):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                cand = (start, i + 1)
                if not best or (cand[1] - cand[0]) > (best[1] - best[0]):
                    best = cand
    if best:
        chunk = s1[best[0]:best[1]]
        try:
            return _json.loads(chunk)
        except Exception:
            chunk2 = re.sub(r",\s*([}\]])", r"\1", chunk)
            try:
                return _json.loads(chunk2)
            except Exception:
                pass

    try:
        unescaped = s1.encode("utf-8", "ignore").decode("unicode_escape")
        return _json.loads(unescaped)
    except Exception:
        pass

    return _default()

def normalize_result(r: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "score": 0, "subscores": {}, "critical_errors": [],
        "recommendations": [], "citations": [], "disclaimer": ""
    }
    try:
        sc = r.get("score", 0) if isinstance(r, dict) else 0
        out["score"] = max(0, min(100, float(sc))) if isinstance(sc, (int, float)) else 0.0
    except Exception:
        out["score"] = 0.0

    subs = r.get("subscores") if isinstance(r, dict) else {}
    if isinstance(subs, dict):
        clean = {}
        for k, v in subs.items():
            try:
                clean[str(k)] = max(0, min(100, float(v)))
            except Exception:
                pass
        out["subscores"] = clean

    ce = r.get("critical_errors") if isinstance(r, dict) else []
    clean_ce = []
    if isinstance(ce, list):
        for it in ce:
            if isinstance(it, dict):
                clean_ce.append({"type": str(it.get("type", "general")),
                                 "explain": str(it.get("explain", it.get("message", "")))})
            elif isinstance(it, str):
                clean_ce.append({"type": "general", "explain": it})
    out["critical_errors"] = clean_ce

    recs = r.get("recommendations") if isinstance(r, dict) else []
    clean_recs = []
    if isinstance(recs, list):
        for it in recs:
            if isinstance(it, dict):
                w = str(it.get("what_to_change") or it.get("action") or it.get("recommendation") or it.get("text", ""))
                ra = str(it.get("rationale") or it.get("reason", ""))
                if w or ra:
                    clean_recs.append({"what_to_change": w, "rationale": ra})
            elif isinstance(it, str):
                clean_recs.append({"what_to_change": it, "rationale": ""})
    out["recommendations"] = clean_recs

    cits = r.get("citations") if isinstance(r, dict) else []
    if isinstance(cits, list):
        out["citations"] = [str(x) for x in cits if isinstance(x, (str, int, float))]
    elif isinstance(cits, (str, int, float)):
        out["citations"] = [str(cits)]

    disc = r.get("disclaimer") if isinstance(r, dict) else ""
    out["disclaimer"] = str(disc) if disc is not None else ""
    return out

def _ns_to_ms(ns: int) -> int:
    try:
        return int(round(float(ns) / 1_000_000.0))
    except Exception:
        return 0

# --- Ollama HTTP helpers (stream-first) ---

# --- Ollama HTTP helpers: stream-first, no JSON format on stream ---

# == ollama_io.py == (или где у тебя живут эти функции)
import json
import requests
from typing import Optional, Dict, Any

# общий http-сесс
_HTTP = requests.Session()

def _trim_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("` \n")
        # убираем возможный префикс типа json
        s = s.split("\n", 1)[-1]
    return s.strip()

def safe_json_extract(s: str) -> Dict[str, Any]:
    s = _trim_code_fences(s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        # если список – завернём в объект
        return {"result": obj}
    except Exception:
        # вернём «обёртку» с дисклеймером, чтобы не падать
        return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [],
                "disclaimer": "LLM вернул не-JSON (stream relax)."}

def _ollama_generate_stream(
    ollama_url: str,
    payload: Dict[str, Any],
    *,
    per_chunk_timeout_s: float = 30.0,
    connect_timeout_s: float = 3.0,
    enforce_json_on_stream: bool = False,
) -> str:
    """
    Стримим токены. ВНИМАНИЕ: по умолчанию НЕ ставим format=json на стриме,
    чтобы не ждать «целый JSON» и получать токены сразу.
    """
    # Готовим payload ДЛЯ СТРИМА: УБИРАЕМ format, если не требуется жёсткая валидация.
    stream_payload = dict(payload)
    stream_payload["stream"] = True
    if not enforce_json_on_stream and "format" in stream_payload:
        stream_payload.pop("format", None)

    with _HTTP.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json=stream_payload,
        timeout=(float(connect_timeout_s), float(per_chunk_timeout_s)),
        stream=True,
    ) as r:
        r.raise_for_status()
        buf = []
        for ln in r.iter_lines(decode_unicode=True):
            if not ln:
                continue
            try:
                chunk = json.loads(ln)
                # Стандартный формат Ollama stream: {"response": "..." , "done": false}
                if "response" in chunk and chunk["response"]:
                    buf.append(chunk["response"])
                # можно логировать «первый токен» для диагностики латентности
            except Exception:
                # если прилетело что-то не JSON (редко) — просто пропускаем
                continue
        return "".join(buf)


def call_ollama_json(
    ollama_url: Optional[str],
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    connect_timeout_s: float = 3.0,
    read_timeout_s: float = 90.0,
    stream_chunk_timeout_s: float = 30.0,
    num_ctx: int = 6144,
    num_predict: int = 160,
    temperature: float = 0.2,
    extra_options: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    keep_alive: Optional[str] = None,
    **_ignored_kwargs,
) -> Dict[str, Any]:
    """
    1) Пытаемся стримить БЕЗ format=json (быстрое появление токенов).
    2) Если таймаут/ошибка — делаем короткую блокирующую попытку с format=json.
    """
    try:
        if not ollama_url:
            # твой дефолт, как раньше
            ollama_url = "http://host.docker.internal:11434"

        # собираем options (фильтруем «опасные» ключи)
        bad = {"gpu_layers", "num_gpu", "main_gpu"}  # мы сами явно пробрасываем gpu_layers, если надо
        opts = {
            "num_ctx": int(num_ctx),
            "num_predict": int(num_predict),
            "temperature": float(temperature),
        }
        if options:
            opts.update({k: v for k, v in options.items() if k not in bad})
        if extra_options:
            opts.update({k: v for k, v in extra_options.items() if k not in bad})

        base_payload = {
            "model": model,
            "prompt": user_prompt,
            "system": system_prompt,
            # ВНИМАНИЕ: format='json' НЕ ставим на стриме; добавим только в блокирующей попытке
            "options": opts,
        }
        if keep_alive is not None:
            base_payload["keep_alive"] = keep_alive

        # --- 1) STREAM-FIRST (без format=json) ---
        try:
            print("LLM STREAM: start")
            text = _ollama_generate_stream(
                ollama_url,
                base_payload,
                per_chunk_timeout_s=stream_chunk_timeout_s,
                connect_timeout_s=connect_timeout_s,
                enforce_json_on_stream=False,  # критично
            )
            text = _trim_code_fences(text)
            if text:
                # Пытаемся распарсить как JSON — если не выйдет, safe_json_extract вернёт вежливую обёртку.
                return safe_json_extract(text)
            else:
                print("LLM STREAM: empty stream result")
        except requests.exceptions.ReadTimeout:
            print(f"LLM STREAM: ReadTimeout (chunk {stream_chunk_timeout_s}s), fallback to short blocking call")
        except Exception as e:
            print(f"LLM STREAM: error={type(e).__name__}: {e}, fallback to blocking")

        # --- 2) Блокирующая короткая попытка c format=json ---
        short_payload = dict(base_payload)
        short_opts = dict(opts)
        # сильно ужмём длину предсказания, чтобы не ждать
        short_opts["num_predict"] = min(80, int(opts.get("num_predict", 120)))
        short_payload["options"] = short_opts
        short_payload["format"] = "json"  # тут уже можно требовать «валидный JSON» целиком
        try:
            resp = _HTTP.post(
                f"{ollama_url.rstrip('/')}/api/generate",
                json=short_payload,
                timeout=(float(connect_timeout_s), float(read_timeout_s)),
            )
            resp.raise_for_status()
            if str(resp.headers.get("content-type", "")).startswith("application/json"):
                obj = resp.json()
                # Ответ Ollama в blocking режиме: {"response": "<строка>", "done": true, ...}
                s = obj.get("response", "") if isinstance(obj, dict) else ""
            else:
                s = resp.text or ""
            s = _trim_code_fences(s)
            if not s:
                return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [],
                        "disclaimer": "LLM вернул пустой ответ (blocking fallback)."}
            return safe_json_extract(s)
        except requests.exceptions.ReadTimeout as e:
            return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [],
                    "disclaimer": f"LLM timeout: {e} (blocking fallback)"}
        except Exception as e:
            return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [],
                    "disclaimer": f"Ошибка LLM ({type(e).__name__}): {e}"}

    except requests.exceptions.ConnectTimeout:
        return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [],
                "disclaimer": f"LLM timeout: соединение >{connect_timeout_s} c."}
    except Exception as e:
        return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [],
                "disclaimer": f"Ошибка LLM ({type(e).__name__}): {e}"}

# ================================
# API models
# ================================
class AnalyzeReq(BaseModel):
    case_text: str
    query: Optional[str] = None
    k: Optional[int] = Field(default=None)
    model: str = Field(default_factory=lambda: cfg("ollama", "model", default="llama3.1:8b"))
    ollama_url: Optional[str] = Field(default_factory=lambda: cfg("ollama", "base_url", default="http://host.docker.internal:11434"))

    if _P_V2:
        @field_validator("k", mode="before")
        def _coerce_k_v2(cls, v):
            if v in (None, "", "null"):
                return settings.RETR_TOP_K
            try:
                return int(v)
            except Exception:
                return settings.RETR_TOP_K
    else:
        @field_validator("k", pre=True)
        def _coerce_k_v1(cls, v):
            if v in (None, "", "null"):
                return settings.RETR_TOP_K
            try:
                return int(v)
            except Exception:
                return settings.RETR_TOP_K

# ================================
# Helpers
# ================================
def _resolve(name: str, default: str) -> str:
    return (os.getenv(name) or cfg(*name.lower().split("_"), default=None) or default)

# ================================
# Routes
# ================================
@app.get("/health")
def health():
    qdrant_url = settings.QDRANT_URL
    collection = settings.QDRANT_COLLECTION
    emb_backend = settings.EMB_BACKEND
    hf_model = settings.HF_MODEL
    device = settings.HF_DEVICE or "auto"
    return {
        "status": "ok",
        "app_env": os.getenv("APP_ENV", "dev"),
        "qdrant": qdrant_url,
        "qdrant_collection": collection,
        "llm_model": cfg("ollama", "model", default="llama3.1:8b"),
        "embed_backend": emb_backend,
        "embed_model": hf_model,
        "embed_device": device,
    }

@app.get("/debug/config")
def debug_config():
    return {
        "ollama": {
            "base_url": cfg_str("ollama", "base_url", default="http://host.docker.internal:11434"),
            "model": cfg_str("ollama", "model", default="llama3.1:8b"),
            "max_tokens": cfg_int("ollama", "max_tokens", default=2048),
            "timeout_s": cfg_int("ollama", "timeout_s", default=60),
            "temperature": cfg_float("ollama", "temperature", default=0.4),
            "top_p": cfg_float("ollama", "top_p", default=0.95),
            "num_ctx": cfg_int("ollama", "num_ctx", default=6144),
        },
        "qdrant": {
            "url": settings.QDRANT_URL,
            "collection": settings.QDRANT_COLLECTION,
        },
        "retrieval": {"k": settings.RETR_TOP_K},
        "chunking": {
            "child_w": cfg_int("chunking", "child_w", default=200),
            "child_overlap": cfg_int("chunking", "child_overlap", default=35),
            "parent_w": cfg_int("chunking", "parent_w", default=800),
        },
    }

@app.post("/config/reload")
def config_reload():
    global CONFIG
    CONFIG = load_config()
    try:
        settings.apply_env(force=True)
    except Exception:
        pass
    return {"status": "reloaded"}

def _compact_case_text(txt: str, target_chars: int = 1400) -> str:
    if not txt:
        return ""
    t = txt

    # Нормализация пробелов/переводов
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    # Убираем часто повторяющиеся отрицания/штампы
    t = re.sub(r"\b(отрицает|не\s+обнаружено|не\s+находился|не\s+отягощен)\b[.,;:\s]*", "нет. ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(безболезненная|кожа обычной окраски|умеренно влажная|свободн\w*|по средней линии)\b[.,;:\s]*", "", t, flags=re.IGNORECASE)

    # Сжимаем длинные числовые хвосты/повторы пунктуации
    t = re.sub(r"[.;:,]\s*(?:[.;:,]\s*)+", ". ", t)
    t = re.sub(r"\s{2,}", " ", t)

    # Грубая дедупликация предложений
    seen = set()
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", t):
        s = sent.strip()
        key = re.sub(r"\W+", "", s.lower())
        if len(key) < 5:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    t = " ".join(out)

    # Урізаем до целевого окна, но стараемся не резать посреди предложения
    if len(t) > target_chars:
        cut = t[:target_chars]
        last_dot = cut.rfind(".")
        if last_dot > target_chars * 0.6:
            t = cut[:last_dot+1]
        else:
            t = cut

    return t.strip()

@app.post("/analyze")
def analyze_ep(req: AnalyzeReq):
    try:
        print("🚀 /analyze")
        import os, json, time, re
        t0 = time.perf_counter()
        # таймеры безопасно инициализируем
        t_r0 = t_r1 = t_l0 = t_l1 = t0

        if looks_meaningless(req.case_text):
            return {"result": {
                "score": 0, "subscores": {}, "critical_errors": [],
                "recommendations": [], "citations": [],
                "disclaimer": "Текст кейса не содержит осмысленных данных.",
            }}

        # --- Сбор поискового запроса ---
        def _smart_query(case_text: str) -> str:
            m = re.search(r"\b([A-Za-z]\d{1,2}(?:\.\d+)?)\b", case_text)
            if m:
                return m.group(1)
            t = re.sub(r"\s+", " ", (case_text or "")).strip()
            return t[:200]

        diag_query = (getattr(req, "query", "") or "").strip()
        user_input_text = (getattr(req, "case_text", "") or "").strip()
        k = req.k if isinstance(req.k, int) and 0 <= req.k <= 20 else settings.RETR_TOP_K

        if diag_query:
            search_q = f"{diag_query}\n{user_input_text[:10000]}".strip() if user_input_text else diag_query
        else:
            base = _smart_query(user_input_text)
            search_q = f"{base}\n{user_input_text[:10000]}".strip() if user_input_text else base

        print("🔍 query =", search_q)

        # --- Ретрив ---
        t_r0 = time.perf_counter()
        ctx_items = retrieve_hybrid(
            search_q, k,
            bm25_index_dir=settings.BM25_INDEX_DIR,
            qdrant_url=settings.QDRANT_URL,
            qdrant_collection=settings.QDRANT_COLLECTION,
            pages_dir=settings.PAGES_DIR,
            hf_model=settings.HF_MODEL,
            hf_device=settings.HF_DEVICE,
            hf_fp16=settings.HF_FP16,
            per_doc_limit=settings.RETR_PER_DOC_LIMIT,
            reranker_enabled=settings.RERANKER_ENABLED,
            rerank_top_k=settings.RERANK_TOP_K,
        )
        t_r1 = time.perf_counter()

        if not ctx_items:
            return {"result": {
                "score": 0, "subscores": {}, "critical_errors": [],
                "recommendations": [], "citations": [],
                "disclaimer": "Контекст не найден в базе знаний — невозможно оценить кейс.",
            }}

        ctx = build_ctx_string(
            ctx_items,
            max_chars=min(6000, settings.LLM_NUM_CTX * 3),  # общий «потолок» контекста (≈3 симв/токен)
            per_text_limit=settings.CTX_SNIPPET_LIMIT       # берём из runtime_settings.py
        )
        print(f"📏 lengths: case={len(req.case_text)} ctx={len(ctx)} k={k}")

        # --- Промпт ---
        DEFAULT_SYSTEM = (
            "Ты — медицинский ассистент. ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. "
            "Анализируй текст как врачебный кейс и верни СТРОГО ВАЛИДНЫЙ JSON согласно схеме."
        )
        DEFAULT_USER_TPL = (
            "[КЕЙС]\n{case_text}\n\n[КОНТЕКСТ]\n{ctx}\n\n"
            "Верни ТОЛЬКО один валидный JSON по указанной схеме."
        )
        system = cfg("prompt", "system", default=DEFAULT_SYSTEM) or DEFAULT_SYSTEM
        user_t = cfg("prompt", "user_template", default=DEFAULT_USER_TPL) or DEFAULT_USER_TPL
        user = user_t.format(case_text=req.case_text, ctx=ctx)

        # --- LLM-параметры из централизованных настроек ---
        conf = _llm_conf_from_settings()
        total_est = _approx_tokens(system) + _approx_tokens(user)

        num_ctx_cap   = int(conf["num_ctx_cap"])               # напр., 16384
        min_ctx_env   = int(os.getenv("LLM_MIN_CTX", "4096"))  # даём управлять через ENV
        ctx_margin    = int(os.getenv("LLM_CTX_MARGIN", "768"))

        MIN_CTX = min(num_ctx_cap, max(1024, min_ctx_env))
        num_ctx = max(MIN_CTX, min(num_ctx_cap, total_est + ctx_margin))

        print(
            f"🤖 LLM url={req.ollama_url or cfg('ollama','base_url', default='N/A')} "
            f"model={req.model} used_ctx={num_ctx} cap_ctx={num_ctx_cap} "
            f"min_ctx={MIN_CTX} max_tokens={conf['max_tokens']} timeout={conf['timeout_s']}s"
        )
        # лог полезной части payload
        _sanitized = {
            "model": req.model,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": conf["max_tokens"],
                "temperature": conf["temperature"],
                "top_p": conf["top_p"],
                "repeat_penalty": conf["repeat_penalty"],
                "num_gpu_layers": conf["gpu_layers"],
            },
            "keep_alive": conf["keep_alive"],
        }
        print("LLM PAYLOAD (sanitized):", json.dumps(_sanitized, ensure_ascii=False))

        # --- Вызов LLM (стрим по умолчанию — см. Patch 2) ---
        t_l0 = time.perf_counter()
        resp = call_ollama_json(
            req.ollama_url, req.model, system, user,
            read_timeout_s=float(conf["timeout_s"]),
            num_ctx=num_ctx,
            num_predict=int(conf["max_tokens"]),
            temperature=float(conf["temperature"]),
            options={
                "top_p": float(conf["top_p"]),
                "repeat_penalty": float(conf["repeat_penalty"]),
                "num_gpu_layers": int(conf["gpu_layers"]) if conf["gpu_layers"] is not None else -1,
            },
            keep_alive=conf["keep_alive"],
            force_stream=True,                                  # стрим сразу
            per_chunk_timeout_s=float(settings.LLM_STREAM_CHUNK_TIMEOUT),  # таймаут «тишины» между чанками
        )

        data = normalize_result(resp)
        t_l1 = time.perf_counter()

        def _is_empty(d):
            return (
                (d.get("score") in (None, 0)) and
                not d.get("subscores") and
                not d.get("critical_errors") and
                not d.get("recommendations")
            )
        timed_out = isinstance(resp, dict) and isinstance(resp.get("disclaimer"), str) and "timeout" in resp["disclaimer"]

        # --- Fast-retry при таймауте/пустоте ---
        if timed_out or _is_empty(data):
            print("⏩ fast-retry: shrinking context and num_predict")

            ctx_small = build_ctx_string(
                ctx_items[:min(3, len(ctx_items))],
                max_chars=6000,
                per_text_limit=700
            )
            user_small = user_t.format(case_text=req.case_text, ctx=ctx_small)

            total_est_small = _approx_tokens(system) + _approx_tokens(user_small)
            num_ctx_small = min(num_ctx_cap, max(1024, total_est_small + 128))
            retry_tokens = min(60, conf["max_tokens"])

            retry_opts = {
                "top_p": float(conf["top_p"]),
                "repeat_penalty": float(conf["repeat_penalty"]),
            }
            if conf.get("gpu_layers") is not None:
                retry_opts["num_gpu_layers"] = int(conf["gpu_layers"])

            # повтор — тоже стримим
            t_l0 = time.perf_counter()
            resp2 = call_ollama_json(
                req.ollama_url, req.model, system, user_small,
                read_timeout_s=float(conf["timeout_s"]),
                num_ctx=int(num_ctx_small),
                num_predict=int(retry_tokens),
                temperature=max(0.0, float(conf["temperature"]) * 0.9),
                options=retry_opts,
                keep_alive=conf["keep_alive"],
                force_stream=True,
                per_chunk_timeout_s=float(settings.LLM_STREAM_CHUNK_TIMEOUT),
            )

            data2 = normalize_result(resp2)
            if not _is_empty(data2):
                data = data2
            t_l1 = time.perf_counter()

            if "disclaimer" in data and isinstance(data["disclaimer"], str):
                if "timeout" in data["disclaimer"]:
                    data["disclaimer"] += " (выполнен fast-retry, сократили контекст/ответ)"
                else:
                    data["disclaimer"] = (data["disclaimer"] + " ") if data["disclaimer"] else ""
                    data["disclaimer"] += "Выполнен fast-retry: контекст и длина ответа уменьшены."

            if _is_empty(data) and not data.get("recommendations"):
                data["recommendations"] = [{
                    "what_to_change": "Уменьшите K (например, до 4–6) или укоротите кейс",
                    "rationale": "Слишком тяжёлый промпт замедляет ответ модели."
                }]

        # --- Корректировка по длине контекста ---
        ctx_len = sum(len(it.get("text", "")) for it in ctx_items)
        if ctx_len < 500:
            data["score"] = max(0, data.get("score", 0) * 0.5)
            data["disclaimer"] += " (Недостаточно контекста из базы знаний — достоверность снижена.)"
        elif ctx_len < 1500:
            data["score"] = max(0, data.get("score", 0) * 0.8)
            data["disclaimer"] += " (Контекст ограничен — достоверность частично снижена.)"

        # --- Цитаты ---
        data["citations"] = build_context_citations(ctx_items, max_out=5) or [
            f"{it['doc_id']} стр.{it['page_start']}-{it['page_end']}" for it in ctx_items[:5]
        ]

        # --- Штраф за критические ошибки ---
        crit_count = len(data.get("critical_errors", []))
        if crit_count > 0:
            data["score"] = max(0, data["score"] - 10 * crit_count)
            data["disclaimer"] += f" (Обнаружено {crit_count} критических ошибок.)"

        # --- Перфоманс-лог ---
        t2 = time.perf_counter()
        def _ms(a, b): return int((b - a) * 1000) if a is not None and b is not None else 0
        print(f"⏱️ perf: retrieval={_ms(t_r0, t_r1)}ms, llm={_ms(t_l0, t_l1)}ms, total={_ms(t0, t2)}ms")

        return {"result": data, "citations_used": [x["doc_id"] for x in ctx_items]}

    except Exception as e:
        import traceback
        print("❌ Ошибка analyze_ep:\n", traceback.format_exc())
        return {
            "result": {
                "score": None, "subscores": {}, "critical_errors": [],
                "recommendations": [], "citations": [],
                "disclaimer": f"Ошибка API: {e}",
            }
        }

# ================================
# UI
# ================================
UI_HTML = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI-ассистент врача (MVP21)</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f6f7fb;margin:0;color:#101828}
.wrap{max-width:1100px;margin:20px auto;padding:16px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 1px 3px rgba(16,24,40,.08);padding:16px;margin-bottom:16px}
h1{font-size:20px;margin:0 0 8px}
label{font-weight:600;font-size:14px;margin:6px 0;display:block}
input,select,textarea{width:100%;border:1px solid #d0d5dd;border-radius:10px;padding:10px;font-size:14px}
textarea{min-height:180px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.btn{background:#2563eb;color:#fff;border:none;border-radius:10px;padding:10px 14px;font-weight:600;cursor:pointer}
.btn:disabled{opacity:.6;cursor:not-allowed}
.badge{display:inline-block;border:1px solid #d0d5dd;border-radius:999px;padding:2px 8px;font-size:12px;margin-left:8px}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
.small{font-size:12px;color:#475467}
.err{color:#b91c1c}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.help{font-size:12px;color:#667085;margin-top:4px}
</style>
<div class="wrap">
  <div class="card">
    <h1>AI-ассистент врача (MVP12) <span id="score" class="badge">оценка: —</span></h1>
    <div class="small">API: <span id="api"></span></div>
  </div>
  <div class="card">
    <label>Текст кейса</label>
    <textarea id="case" placeholder="Вставьте кейс: жалобы, анамнез, диагноз, назначения..."></textarea>
    <div class="row">
      <div>
        <label>Запрос для поиска (необязательно)</label>
        <input id="query" placeholder="например: гипертоническая болезнь лечение эналаприл">
      </div>
      <div>
        <label>Модель / K</label>
        <div class="row" style="grid-template-columns:2fr 1fr;gap:8px">
          <select id="model"><option>llama3.1:8b</option><option>llama3.1:70b</option></select>
          <input id="k" type="number" value="" min="0" max="20" placeholder="по умолч.">
        </div>
        <div class="help">Оставьте K пустым — возьмётся значение из настроек сервера</div>
      </div>
    </div>
    <div style="margin-top:10px;display:flex;gap:8px;align-items:center">
      <button id="run" class="btn">Проанализировать</button>
      <button id="reindex" class="btn" style="background:#059669">🔄 Обновить базу</button>
      <span id="busy" class="small" style="display:none">⏳ выполняется…</span>
      <span id="error" class="small err"></span>
    </div>
  </div>
  <div class="card">
    <h3 style="margin:0 0 6px">Результат</h3>
    <div class="grid2" id="subs"></div>
    <div><h4>Критические ошибки</h4><ul id="crit"></ul></div>
    <div><h4>Рекомендации</h4><ul id="recs"></ul></div>
    <div><h4>Источники (цитаты)</h4><ul id="cits"></ul></div>
    <details><summary class="small">Сырой JSON</summary><pre id="raw" class="mono"></pre></details>
  </div>
</div>
<script>
const API = window.location.origin;
document.getElementById('api').textContent = API;

const el = id => document.getElementById(id);
const show = (n,on) => n.style.display = on ? '' : 'none';

function colorForScore(s){ if(typeof s!=='number') return ''; if(s>=85) return '#dcfce7'; if(s>=65) return '#fef9c3'; return '#fee2e2'; }

function renderResult(r){
  const sc = r.score ?? '—';
  const sb = el('score');
  sb.textContent = 'оценка: ' + sc;
  sb.style.background = colorForScore(sc);

  const subs = el('subs'); subs.innerHTML = '';
  Object.entries(r.subscores||{}).forEach(([k,v])=>{
    const d=document.createElement('div'); d.className='card'; d.style.margin=0;
    d.innerHTML=`<div class="small">${labelMap[k] || k}</div><div style="font-weight:700">${v??'—'}</div>`;
    subs.appendChild(d);
  });

  const crit=el('crit'); crit.innerHTML=''; (r.critical_errors||[]).forEach(x=>{
    const li=document.createElement('li'); li.textContent=`${x.type}: ${x.explain}`; crit.appendChild(li);
  });

  const recs=el('recs'); recs.innerHTML=''; (r.recommendations||[]).forEach(x=>{
    const li=document.createElement('li'); li.textContent=`${x.what_to_change} — ${x.rationale}`; recs.appendChild(li);
  });

  const cits=el('cits'); cits.innerHTML=''; (r.citations||[]).forEach(x=>{
    const li=document.createElement('li'); li.textContent=String(x); cits.appendChild(li);
  });

  el('raw').textContent = JSON.stringify(r,null,2);
}

const labelMap = {
  "diagnosis": "Диагноз",
  "diagnosis_match": "Соответствие диагнозу",
  "therapy": "Терапия",
  "med_choice": "Выбор препарата",
  "dosage": "Дозировка",
  "dosing": "Дозировка",
  "interactions": "Лекарственные взаимодействия",
  "contraindications": "Противопоказания",
  "monitoring": "Мониторинг",
  "evidence": "Доказательность"
};

async function run(){
  el('error').textContent = '';
  show(el('busy'), true);
  el('run').disabled = true;

  try{
    const body = {
      case_text: (el('case').value || '').trim(),
      query:     (el('query').value || '').trim() || null,
      model:     el('model').value || 'llama3.1:8b'
    };

    const kRaw = (el('k').value || '').trim();
    if (kRaw !== '') {
      const kParsed = parseInt(kRaw, 10);
      if (Number.isFinite(kParsed)) body.k = kParsed;
    }

    const res = await fetch(API + '/analyze', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });

    const txt = await res.text();
    let data; try{ data = JSON.parse(txt); }catch(e){
      throw new Error('Не удалось разобрать JSON ответа: ' + txt.slice(0,200));
    }
    renderResult(data.result || data);
  }catch(e){
    el('error').textContent = 'Ошибка: ' + (e?.message || e);
  }finally{
    show(el('busy'), false);
    el('run').disabled = false;
  }
}
el('run').onclick = run;

el('reindex').onclick = async () => {
  show(el('busy'), true);
  el('error').textContent = '';
  try {
    const res = await fetch(API + '/reindex', { method: 'POST' });
    const data = await res.json();
    el('error').textContent = data.message || '';
  } catch(e) {
    el('error').textContent = 'Ошибка запуска: ' + e.message;
  } finally {
    show(el('busy'), false);
  }
};

async function checkReindexStatus() {
  try {
    const res = await fetch(API + '/reindex/status');
    const data = await res.json();
    const msg = data.message || '';
    const state = data.state;
    if (state === 'running') {
      el('busy').textContent = '🔄 ' + msg;
      show(el('busy'), true);
    } else if (state === 'done') {
      el('busy').textContent = '✅ Индексация завершена';
      setTimeout(() => show(el('busy'), false), 4000);
    } else if (state === 'error') {
      el('busy').textContent = '❌ ' + msg;
    }
  } catch(e) { console.error(e); }
}
setInterval(checkReindexStatus, 3000);
</script>
"""

@app.get("/", response_class=HTMLResponse)
def ui_root():
    return HTMLResponse(UI_HTML)

# ================================
# Reindex
# ================================
index_status = {"state": "idle", "message": "Ожидание"}

@app.get("/runtime/defaults")
def runtime_defaults():
    return {
        "RETR_TOP_K": settings.RETR_TOP_K,
        "RERANKER_ENABLED": settings.RERANKER_ENABLED,
        "RERANK_TOP_K": settings.RERANK_TOP_K,
        "HF_MODEL": settings.HF_MODEL,
    }

@app.get("/reindex/status")
def reindex_status():
    return index_status

def run_reindex(*, full: bool = False):
    import os as _os
    import time as _time
    import socket as _socket
    import subprocess as _subprocess
    from pathlib import Path

    # Продавим значения из runtime_settings.CONTROL в окружение,
    # чтобы они стали источником правды для всех подпроцессов.
    try:
        settings.apply_env(force=True)
    except Exception:
        pass

    global index_status

    base = ROOT
    raw_dir = str(base / "raw_docs")
    data_dir = str(base / "data")
    ingest_py = str(base / "ingest_from_raw.py")
    build_bm25_py = str(base / "build_bm25.py")
    chunk_and_index_py = str(base / "chunk_and_index.py")

    # ---------- helpers ----------
    def _nz(val, default):
        s = (val or "").strip() if isinstance(val, str) else val
        return s if s not in (None, "", "None") else default

    def _as_int(val, fallback: int) -> int:
        try:
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return int(val)
            if isinstance(val, str) and val.strip() and val.strip().lower() != "none":
                return int(float(val.strip()))
        except Exception:
            pass
        return int(fallback)

    def _normalize_qdrant_url(url: str) -> str:
        try:
            if "qdrant:" in url:
                _socket.gethostbyname("qdrant")
        except Exception:
            return "http://localhost:7779"
        return url

    # --- штамп свежести BM25 ---
    STAMP_BM25 = Path("index/.bm25_last_build")

    def _latest_pages_mtime() -> float:
        pages = list(Path("data").glob("*.pages.jsonl"))
        return max((p.stat().st_mtime for p in pages), default=0.0)

    def _bm25_needs_rebuild() -> bool:
        last_pages = _latest_pages_mtime()
        if last_pages == 0.0:
            return False
        if not STAMP_BM25.exists():
            return True
        return last_pages > STAMP_BM25.stat().st_mtime

    def _touch_bm25_stamp():
        STAMP_BM25.parent.mkdir(parents=True, exist_ok=True)
        STAMP_BM25.write_text(str(_time.time()), encoding="utf-8")

    try:
        env = _os.environ.copy()
        env["QDRANT__PREFER_GRPC"] = "false"

        # --- Шаг 1: Ingest ---
        index_status.update({"state": "running", "message": "📄 Шаг 1: парсинг RAW → JSONL (инкрементально)..."})
        print("▶️ ingest_from_raw.py ...")

        cmd_ingest = ["python", ingest_py, "--input-dir", raw_dir, "--out-dir", data_dir]

        man = Path(data_dir) / "manifest.json"
        try:
            first_run = not man.exists() or not (json.loads(man.read_text(encoding="utf-8") or "{}").get("docs") or [])
        except Exception:
            first_run = True

        if full or first_run:
            cmd_ingest.append("--force")

        _subprocess.run(cmd_ingest, check=True, env=env)

        # --- Резолв параметров (всё через env, выставленное settings.apply_env) ---
        qdrant_url = _normalize_qdrant_url(
            _nz(_os.getenv("QDRANT_URL") or cfg("qdrant", "url", default=settings.QDRANT_URL),
                settings.QDRANT_URL)
        )
        collection  = _nz(_os.getenv("QDRANT_COLLECTION") or cfg("qdrant", "collection", default=settings.QDRANT_COLLECTION),
                          settings.QDRANT_COLLECTION)
        emb_backend = _nz(_os.getenv("EMB_BACKEND") or cfg("embedding", "backend", default=settings.EMB_BACKEND), "hf")
        hf_model    = _nz(_os.getenv("HF_MODEL") or cfg("embedding", "model", default=settings.HF_MODEL), settings.HF_MODEL)

        # Dense-чанкинг
        child_w       = _as_int(_os.getenv("CHILD_W"),       cfg("chunking", "child_w",       default=200))
        child_overlap = _as_int(_os.getenv("CHILD_OVERLAP"), cfg("chunking", "child_overlap", default=35))
        parent_w      = _as_int(_os.getenv("PARENT_W"),      cfg("chunking", "parent_w",      default=800))

        # BM25-чанкинг/язык (управляется runtime_settings через ENV)
        bm25_child_w  = _as_int(_os.getenv("BM25_CHILD_W"),         _as_int(_os.getenv("CHILD_W"), 200))
        bm25_overlap  = _as_int(_os.getenv("BM25_CHILD_OVERLAP"),   _as_int(_os.getenv("CHILD_OVERLAP"), 40))
        bm25_lang     = _nz(_os.getenv("BM25_LANGUAGE"),            "ru")

        print(
            "🔧 RESOLVED → "
            f"QDRANT_URL={qdrant_url}  QDRANT_COLLECTION={collection}  "
            f"EMB_BACKEND={emb_backend}  HF_MODEL={hf_model}  "
            f"child_w={child_w} child_overlap={child_overlap} parent_w={parent_w}  "
            f"[BM25 child_w={bm25_child_w} overlap={bm25_overlap} lang={bm25_lang}]"
        )

        # --- Шаг 2: BM25 ---
        if full or _bm25_needs_rebuild():
            index_status["message"] = "📚 Шаг 2: построение/обновление BM25 индекса..."
            print("▶️ build_bm25.py ...")
            _subprocess.run(
                [
                    "python", build_bm25_py,
                    "--pages-glob", "data/*.pages.jsonl",
                    "--out-json",   "index/bm25_json",
                    "--index-dir",  "index/bm25_idx",
                    "--child-w",    str(bm25_child_w),
                    "--child-overlap", str(bm25_overlap),
                    "--language",   bm25_lang,
                ],
                check=True, env=env
            )
            _touch_bm25_stamp()
        else:
            index_status["message"] = "⏭️  Шаг 2 пропущен: новых страниц для BM25 нет"
            print(index_status["message"])

        # --- Шаг 3: Dense → Qdrant ---
        index_status["message"] = "🧠 Шаг 3: индексация в Qdrant (dense)..."
        cmd_qdr = [
            "python", chunk_and_index_py,
            "--pages-glob",    "data/*.pages.jsonl",
            "--collection",    collection,
            "--qdrant-url",    qdrant_url,
            "--emb-backend",   emb_backend,
            "--hf-model",      hf_model,
            "--batch",         "128",
            "--child-w",       str(child_w),
            "--child-overlap", str(child_overlap),
            "--parent-w",      str(parent_w),
        ]
        cmd_qdr.append("--recreate" if full else "--only-new")
        print("▶️ CMD:", " ".join(cmd_qdr))
        _subprocess.run(cmd_qdr, check=True, env=env)

        index_status.update({"state": "done", "message": "✅ Индексация завершена."})
        print("✅ Индексация завершена.")

    except _subprocess.CalledProcessError as e:
        index_status.update({"state": "error", "message": f"❌ Процесс упал: {e}"})
        print(index_status["message"])
    except Exception as e:
        index_status.update({"state": "error", "message": f"❌ Ошибка: {e}"})
        print(index_status["message"])

@app.post("/reindex")
def reindex_ep(full: bool = False):
    threading.Thread(target=run_reindex, kwargs={"full": bool(full)}, daemon=True).start()
    return {"status": "started", "message": "Индексация запущена", "full": bool(full)}
