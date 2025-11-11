#!/usr/bin/env python3
from __future__ import annotations

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

# runtime settings: импорт ПОСЛЕ загрузки .env
from config.runtime_settings import settings  # noqa: E402
try:
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

# ----- Локальные хелперы по моделям (из централизованного runtime) -----
def llm_get_allowed():
    return list(getattr(settings, "LLM_ALLOWED", []) or [])

def llm_get_active():
    return str(getattr(settings, "LLM_ACTIVE", "") or "")

def llm_get_preset(model_id: str) -> dict:
    return dict((getattr(settings, "LLM_PRESETS", {}) or {}).get(model_id, {}))

def llm_get_labels() -> dict:
    pretty = {
        "llama3.1:8b": "Llama 3.1 (8B)",
        "llama3.1:70b": "Llama 3.1 (70B)",
        "deepseek-r1:32b": "DeepSeek R1 (32B)",
    }
    return {m: pretty.get(m, m) for m in llm_get_allowed()}

# ================================
# FastAPI
# ================================
app = FastAPI(title="med_ai RAG API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

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
    d_model       = getattr(settings, "LLM_MODEL", "llama3.1:8b")
    d_base_url    = getattr(settings, "LLM_BASE_URL", "http://ollama:11434")
    d_num_ctx     = getattr(settings, "LLM_NUM_CTX", 4096)
    d_max_tokens  = getattr(settings, "LLM_MAX_TOKENS", 300)
    d_timeout_s   = getattr(settings, "LLM_TIMEOUT", 60)
    d_temp        = 0.4
    d_top_p       = 0.95
    d_repeat_pen  = 1.05
    d_gpu_layers  = -1
    d_keep_alive  = "30m"

    return {
        "model":        os.getenv("LLM_MODEL", d_model),
        "base_url":     os.getenv("LLM_BASE_URL", d_base_url),
        "num_ctx_cap":  _as_int(os.getenv("LLM_NUM_CTX"), d_num_ctx),
        "max_tokens":   _as_int(os.getenv("LLM_MAX_TOKENS"), d_max_tokens),
        "timeout_s":    _as_int(os.getenv("LLM_TIMEOUT"), d_timeout_s),
        "temperature":      _as_float(os.getenv("LLM_TEMPERATURE"), d_temp),
        "top_p":            _as_float(os.getenv("LLM_TOP_P"), d_top_p),
        "repeat_penalty":   _as_float(os.getenv("LLM_REPEAT_PENALTY"), d_repeat_pen),
        "gpu_layers":   _as_int(os.getenv("LLM_NUM_GPU_LAYERS"), d_gpu_layers),
        "keep_alive":   os.getenv("LLM_KEEP_ALIVE", d_keep_alive),
    }

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
            "base_url": os.getenv("LLM_BASE_URL", "http://ollama:11434"),
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
            
                "system": os.getenv("PROMPT_SYSTEM", getattr(settings, "PROMPT_SYSTEM", "")),
                "user_template": os.getenv("PROMPT_USER_TPL", getattr(settings, "PROMPT_USER_TPL", "")),
            
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
# LLM через Ollama — helpers
# ================================
def _trim_code_fences(txt: str) -> str:
    txt = re.sub(r"^\s*```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s*```\s*$", "", txt)
    return txt.strip()

def safe_json_extract(s: str) -> Dict[str, Any]:
    import json as _json, re as _re

    def _default():
        return {
            "score": None, "subscores": {}, "critical_errors": [], "recommendations": [],
            "citations": [], "disclaimer": "Парсинг ответа не удался."
        }

    if not s:
        return _default()

    s1 = _re.sub(r"```(?:json)?", "", s, flags=_re.IGNORECASE).replace("```", "").strip()

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
            chunk2 = _re.sub(r",\s*([}\]])", r"\1", chunk)
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

# --- helpers для очистки и извлечения текста/JSON --------------------------------

_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\s*([\s\S]*?)```", flags=re.DOTALL)

def _strip_code_fences_strict(s: str) -> str:
    """Удаляет полноценный ```блок``` если он целиком; иначе возвращает как есть."""
    if not s:
        return ""
    m = _CODE_BLOCK_RE.search(s)
    return m.group(1).strip() if m else s.strip()

def _strip_code_fences_loose(s: str) -> str:
    """
    Удаляет даже НЕЗАКРЫТЫЕ начала кода: '```json\\n...' → текст без заголовка,
    а также срезает хвостовые случайные обратные кавычки.
    """
    if not s:
        return ""
    s = s.replace("\r", "").lstrip()
    if s.startswith("```"):
        p = s.find("\n")
        s = s[p + 1:] if p != -1 else ""
    s = s.rstrip("`").rstrip()
    return s

def _clean_free_text(s: str) -> str:
    """Чистый человеческий текст: без ```фенсов```, без мусора по краям."""
    if not s:
        return ""
    t = _strip_code_fences_strict(s)
    if "```" in t:
        t = _strip_code_fences_loose(t)
    # убираем одиночные кавычки/бектики по краям
    t = t.strip().strip("`").strip()
    return t

def _try_parse_json_from_text(s: str):
    """Пробуем извлечь JSON из текста (в т.ч. внутри ```json ... ``` или просто { ... })."""
    if not s:
        return None
    t = _strip_code_fences_strict(s)
    if t == s:
        t = _strip_code_fences_loose(t)
    t = t.strip()

    # прямой parse
    try:
        return json.loads(t)
    except Exception:
        pass

    # вырезать первую {...} «скобочную» область
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end > start:
        candidate = t[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None

def _extract_any_text(resp) -> str:
    """Достаём любой текст из словаря-ответа модели (raw_stream/raw_block/response/... )."""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for key in ("raw_stream", "raw_block", "response", "text", "message", "content"):
            v = resp.get(key)
            if isinstance(v, str) and v.strip():
                return v
    return ""

def _norm_critical_errors(ce) -> list[dict]:
    """Нормализация списка крит.ошибок к [{type, explain}]"""
    out = []
    if isinstance(ce, list):
        for item in ce:
            if isinstance(item, dict):
                t = str(item.get("type", "")).strip()
                e = str(item.get("explain", "")).strip()
                if t or e:
                    out.append({"type": t, "explain": e})
            elif isinstance(item, str) and item.strip():
                out.append({"type": "", "explain": item.strip()})
    elif isinstance(ce, str) and ce.strip():
        out.append({"type": "", "explain": ce.strip()})
    return out

def _norm_recommendations(rec) -> list[str]:
    """
    Нормализуем рекомендации в список строк.
    Чистим кодовые блоки/мусор внутри строк. Не дублируем пустяки.
    """
    items: list[str] = []
    if isinstance(rec, str):
        r = _clean_free_text(rec)
        if r:
            items = [r]
    elif isinstance(rec, list):
        for x in rec:
            if isinstance(x, str):
                r = _clean_free_text(x)
                if r:
                    items.append(r)
            elif isinstance(x, dict):
                wt = str(x.get("what_to_change", "")).strip()
                rn = str(x.get("rationale", "")).strip()
                s = f"{wt} — {rn}" if wt and rn else (wt or rn)
                s = _clean_free_text(s)
                if s:
                    items.append(s)
            else:
                try:
                    items.append(json.dumps(x, ensure_ascii=False))
                except Exception:
                    pass
    elif rec is not None:
        try:
            s = json.dumps(rec, ensure_ascii=False)
            s = _clean_free_text(s)
            if s:
                items = [s]
        except Exception:
            pass

    # dedup + фильтр пустых
    seen = set()
    out = []
    for it in items:
        t = (it or "").strip()
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out

# --- основная функция -----------------------------------------------------------

def normalize_result_loose(resp: dict | str) -> dict:
    """
    Мягкая схема:
      - critical_errors: list[{type, explain}]
      - recommendations: [str] (если были в ответе/JSON; не тянем их из «свободного текста»)
      - free_text: str (любой не-JSON контент модели, очищенный от ``` и мусора)
      - meta: dict (опционально), например meta.role = 'медицинский ассистент'
      - citations/disclaimer могут быть, но не обязательны
    """
    out = {
        "critical_errors": [],
        "recommendations": [],
        "citations": [],
        "disclaimer": "",
        "meta": {},
        "free_text": ""
    }

    # --- вариант: пришла просто строка
    if isinstance(resp, str):
        parsed = _try_parse_json_from_text(resp)
        if isinstance(parsed, dict):
            out["critical_errors"] = _norm_critical_errors(parsed.get("critical_errors"))
            out["recommendations"] = _norm_recommendations(parsed.get("recommendations"))
            if isinstance(parsed.get("meta"), dict):
                out["meta"] = dict(parsed["meta"])
            if isinstance(parsed.get("citations"), list):
                out["citations"] = [str(x) for x in parsed["citations"] if x]
            if isinstance(parsed.get("disclaimer"), str):
                out["disclaimer"] = parsed["disclaimer"]
        else:
            out["free_text"] = _clean_free_text(resp)
        return out

    # --- вариант: пришёл словарь
    if not isinstance(resp, dict):
        return out

    # 1) критические ошибки / рекомендации — только из «структурных» полей
    out["critical_errors"] = _norm_critical_errors(resp.get("critical_errors"))
    out["recommendations"] = _norm_recommendations(resp.get("recommendations"))

    # 2) meta/citations/disclaimer — по возможности
    if isinstance(resp.get("meta"), dict):
        out["meta"] = dict(resp["meta"])
    if isinstance(resp.get("citations"), list):
        out["citations"] = [str(x) for x in resp["citations"] if x]
    if isinstance(resp.get("disclaimer"), str):
        out["disclaimer"] = resp["disclaimer"]

    # 3) свободный текст — из любых «сырьевых» ключей
    raw_txt = _extract_any_text(resp)
    if raw_txt:
        # Если там на самом деле лежит JSON — попробуем вытащить структурные поля
        parsed = _try_parse_json_from_text(raw_txt)
        if isinstance(parsed, dict):
            # дозаполняем рекомендации/крит.ошибки только если их не было
            if not out["recommendations"]:
                out["recommendations"] = _norm_recommendations(parsed.get("recommendations"))
            if not out["critical_errors"]:
                out["critical_errors"] = _norm_critical_errors(parsed.get("critical_errors"))
            # free_text всё равно оставим «человеческим»
            free = (
                str(parsed.get("answer") or parsed.get("text") or parsed.get("message") or "").strip()
            )
            out["free_text"] = _clean_free_text(free) if free else _clean_free_text(raw_txt)
        else:
            out["free_text"] = _clean_free_text(raw_txt)

    return out

def _ns_to_ms(ns: int) -> int:
    try:
        return int(round(float(ns) / 1_000_000.0))
    except Exception:
        return 0

# --- Ollama HTTP helpers: stream-first, no JSON format on stream ---
import requests as _requests

_HTTP2 = _requests.Session()

def _trim_code_fences2(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("` \n")
        s = s.split("\n", 1)[-1]
    return s.strip()

def safe_json_extract2(s: str) -> Dict[str, Any]:
    s = _trim_code_fences2(s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        return {"result": obj}
    except Exception:
        return {
            "score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [],
            "disclaimer": "LLM вернул не-JSON (stream relax)."
        }

def _ollama_generate_stream(
    ollama_url: str,
    payload: Dict[str, Any],
    *,
    per_chunk_timeout_s: float = 30.0,
    connect_timeout_s: float = 3.0,
    enforce_json_on_stream: bool = False,
) -> str:
    """
    Если enforce_json_on_stream=True — добавляем "format":"json" прямо в стрим-пейлоад.
    Это не мешает инкрементальному выводу, но итоговая строка 'response' будет JSON.
    """
    import json as _json

    stream_payload = dict(payload)
    stream_payload["stream"] = True
    if enforce_json_on_stream:
        stream_payload["format"] = "json"
    else:
        # если не навязываем JSON — удаляем возможный format из payload
        stream_payload.pop("format", None)

    with _HTTP2.post(
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
                chunk = _json.loads(ln)
                if "response" in chunk and chunk["response"]:
                    buf.append(chunk["response"])
            except Exception:
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
    try:
        if not ollama_url:
            ollama_url = "http://ollama:11434"

        bad = {"gpu_layers", "num_gpu", "main_gpu"}
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
            "options": opts,
        }
        if keep_alive is not None:
            base_payload["keep_alive"] = keep_alive

        raw_stream_text = ""

        # 1) STREAM-FIRST (без format=json)
        try:
            print("LLM STREAM: start")
            text = _ollama_generate_stream(
                ollama_url,
                base_payload,
                per_chunk_timeout_s=stream_chunk_timeout_s,
                connect_timeout_s=connect_timeout_s,
                enforce_json_on_stream=False,
            )
            text = _trim_code_fences2(text)
            raw_stream_text = text or ""
            if text:
                parsed = safe_json_extract2(text)
                if isinstance(parsed, dict) and parsed:
                    # даже если JSON распарсился, оставим оригинал
                    parsed.setdefault("raw_stream", raw_stream_text)
                    return parsed
                else:
                    # не JSON: попробуем блокирующий fallback, но raw_stream обязательно вернём
                    print("LLM STREAM: got non-JSON or unparseable JSON → fallback to blocking format=json")
            else:
                print("LLM STREAM: empty stream result")
        except _requests.exceptions.ReadTimeout:
            print(f"LLM STREAM: ReadTimeout (chunk {stream_chunk_timeout_s}s), fallback to short blocking call")
        except Exception as e:
            print(f"LLM STREAM: error={type(e).__name__}: {e}, fallback to blocking")

        # 2) Короткая блокирующая попытка c format=json
        short_payload = dict(base_payload)
        short_opts = dict(opts)
        short_opts["num_predict"] = min(80, int(opts.get("num_predict", 120)))
        short_payload["options"] = short_opts
        short_payload["format"] = "json"
        try:
            resp = _HTTP2.post(
                f"{ollama_url.rstrip('/')}/api/generate",
                json=short_payload,
                timeout=(float(connect_timeout_s), float(read_timeout_s)),
            )
            resp.raise_for_status()
            if str(resp.headers.get("content-type", "")).startswith("application/json"):
                obj = resp.json()
                s = obj.get("response", "") if isinstance(obj, dict) else ""
            else:
                s = resp.text or ""
            s = _trim_code_fences2(s)

            if not s:
                out = {
                    "score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [],
                    "disclaimer": "LLM вернул пустой ответ (blocking fallback)."
                }
                if raw_stream_text:
                    out["raw_stream"] = raw_stream_text
                return out

            parsed2 = safe_json_extract2(s)
            if isinstance(parsed2, dict) and parsed2:
                if raw_stream_text:
                    parsed2.setdefault("raw_stream", raw_stream_text)
                return parsed2

            # даже блокирующий ответ не JSON → вернём как raw_block + raw_stream
            return {
                "critical_errors": [],
                "recommendations": [],
                "citations": [],
                "disclaimer": "LLM вернул не-JSON (blocking fallback).",
                "raw_block": s,
                **({"raw_stream": raw_stream_text} if raw_stream_text else {}),
            }

        except _requests.exceptions.ReadTimeout as e:
            out = {
                "critical_errors": [], "recommendations": [], "citations": [],
                "disclaimer": f"LLM timeout: {e} (blocking fallback)"
            }
            if raw_stream_text:
                out["raw_stream"] = raw_stream_text
            return out
        except Exception as e:
            out = {
                "critical_errors": [], "recommendations": [], "citations": [],
                "disclaimer": f"Ошибка LLM ({type(e).__name__}): {e}"
            }
            if raw_stream_text:
                out["raw_stream"] = raw_stream_text
            return out

    except _requests.exceptions.ConnectTimeout:
        return {
            "critical_errors": [], "recommendations": [], "citations": [],
            "disclaimer": f"LLM timeout: соединение >{connect_timeout_s} c."
        }
    except Exception as e:
        return {
            "critical_errors": [], "recommendations": [], "citations": [],
            "disclaimer": f"Ошибка LLM ({type(e).__name__}): {e}"
        }


# ================================
# API models
# ================================
class AnalyzeReq(BaseModel):
    case_text: str
    query: Optional[str] = None
    k: Optional[int] = Field(default=None)
    model: str = Field(default_factory=lambda: cfg("ollama", "model", default="llama3.1:8b"))
    ollama_url: Optional[str] = Field(default_factory=lambda: cfg("ollama", "base_url", default="http://ollama:11434"))

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
            "base_url": cfg_str("ollama", "base_url", default="http://ollama:11434"),
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
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    t = re.sub(r"\b(отрицает|не\s+обнаружено|не\s+находился|не\s+отягощен)\b[.,;:\s]*", "нет. ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(безболезненная|кожа обычной окраски|умеренно влажная|свободн\w*|по средней линии)\b[.,;:\s]*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[.;:,]\s*(?:[.;:,]\s*)+", ". ", t)
    t = re.sub(r"\s{2,}", " ", t)

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

    if len(t) > target_chars:
        cut = t[:target_chars]
        last_dot = cut.rfind(".")
        if last_dot > target_chars * 0.6:
            t = cut[:last_dot+1]
        else:
            t = cut

    return t.strip()
# ---- helpers: safe formatting for prompt templates ----
class _SafeDict(dict):
    def __missing__(self, key):
        # оставляем неизвестные плейсхолдеры как есть
        return "{" + key + "}"

def safe_format(template: str, **kwargs) -> str:
    """
    Форматирует строку, не падая на чужих {скобках}.
    Оставляет только {case_text} и {ctx} как настоящие плейсхолдеры,
    остальные фигурные скобки экранируются.
    """
    if not isinstance(template, str):
        return template
    # временно прячем допустимые плейсхолдеры
    t = template.replace("{case_text}", "<<<__CASE__>>>").replace("{ctx}", "<<<__CTX__>>>")
    # экранируем все остальные фигурные скобки
    t = t.replace("{", "{{").replace("}", "}}")
    # возвращаем допустимые плейсхолдеры
    t = t.replace("<<<__CASE__>>>", "{case_text}").replace("<<<__CTX__>>>", "{ctx}")
    # форматируем безопасно
    try:
        return t.format_map(_SafeDict(**kwargs))
    except Exception:
        # на крайний случай – подставим только известные поля
        return t.format(case_text=kwargs.get("case_text", ""), ctx=kwargs.get("ctx", ""))

CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\s*([\s\S]*?)```", flags=re.DOTALL)

def strip_code_fences_strict(s: str) -> str:
    """Удаляет полноценный ```блок``` если он целиком. Иначе возвращает как есть."""
    if not s: return s
    m = CODE_BLOCK_RE.search(s)
    return m.group(1).strip() if m else s.strip()

def strip_code_fences_loose(s: str) -> str:
    """Удаляет даже НЕЗАКРЫТЫЕ начала кода: '```json\\n...' → текст без первой строки."""
    if not s: return s
    s = s.replace("\r", "")
    s = s.lstrip()
    if s.startswith("```"):
        p = s.find("\n")
        if p != -1:
            s = s[p+1:]
        else:
            s = ""  # весь текст был одной строкой с ``` — убираем
    # убираем хвостовые «случайные» ```
    s = s.rstrip("`").rstrip()
    return s

def smart_trim(s: str, max_len: int = 1800) -> str:
    """Аккуратно обрезает по концу предложения/строки/слова, чтобы не резать посреди слова."""
    if not s or len(s) <= max_len:
        return (s or "").strip()
    cut = s[:max_len]
    # 1) попробуем найти конец предложения в последних 200 символах
    tail = cut[-200:]
    off = max(tail.rfind(". "), tail.rfind("! "), tail.rfind("? "), tail.rfind("… "))
    if off != -1:
        return (cut[:max_len-200 + off + 2]).rstrip()
    # 2) иначе конец строки
    nl = cut.rfind("\n")
    if nl >= max_len - 200:
        return cut[:nl].rstrip()
    # 3) иначе последний пробел
    sp = cut.rfind(" ")
    if sp >= max_len - 120:
        return cut[:sp].rstrip()
    return cut.rstrip()



# ================================
# UI
# ================================
UI_HTML = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI-ассистент врача (MVP)</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f6f7fb;margin:0;color:#101828}
.wrap{max-width:1100px;margin:20px auto;padding:16px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 1px 3px rgba(16,24,40,.08);padding:16px;margin-bottom:16px}
h1{font-size:20px;margin:0 0 8px}
h3{margin:0 0 6px}
h4{margin:12px 0 6px}
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
.muted{color:#667085}
li{margin:6px 0}
.chk{display:flex;align-items:center;gap:8px;margin-top:6px}
</style>

<div class="wrap">
  <div class="card">
    <h1>
      AI-ассистент врача (MVP)
      <span id="score" class="badge">оценка: —</span>
      <span id="mode" class="badge" style="background:#eef2ff">режим: мед</span>
    </h1>
    <div class="small">API: <span id="api"></span></div>
    <div class="small" id="role" style="margin-top:4px"></div>
  </div>

  <div class="card">
    <label>Текст кейса</label>
    <textarea id="case" placeholder="Вставьте кейс: жалобы, анамнез, диагноз, назначения..."></textarea>

    <div class="row">
      <div>
        <label class="chk" title="Если включено — ответит свободная модель без поиска по базе (цитат не будет).">
          <input type="checkbox" id="use_free">
          <span class="small">Свободная модель</span>
        </label>

        <label>Запрос для поиска (опционально, только для режима «мед»)</label>
        <input id="query" placeholder="например: ИБС лечение...">
        <div class="help">В режиме «свободная модель» поиск по базе не выполняется.</div>
      </div>

      <div>
        <label>Модель / K (только для «мед»)</label>
        <div class="row" style="grid-template-columns:2fr 1fr;gap:8px">
          <select id="model"></select>
          <input id="k" type="number" value="" min="0" max="20" placeholder="по умолч.">
        </div>
        <div class="help">Оставьте K пустым — возьмётся значение из настроек сервера</div>
      </div>
    </div>

    <div style="margin-top:10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <button id="run" class="btn">Отправить</button>
      <button id="reindex" class="btn" style="background:#059669">🔄 Обновить базу</button>
      <span id="busy" class="small" style="display:none">⏳ выполняется…</span>
      <span id="error" class="small err"></span>
    </div>
  </div>

  <div class="card">
    <h3>Результат</h3>
    <div class="grid2" id="subs"></div>

    <div id="crit_wrap">
      <h4>Критические ошибки</h4>
      <ul id="crit"></ul>
    </div>

    <div id="recs_wrap">
      <h4>Рекомендации</h4>
      <ul id="recs"></ul>
    </div>

    <h4>Свободный ответ</h4>
    <div id="free" style="white-space:pre-wrap;border:1px dashed #e5e7eb;border-radius:10px;padding:10px;background:#fafafa;display:none"></div>

    <div id="cits_wrap">
      <h4>Источники (цитаты)</h4>
      <ul id="cits"></ul>
    </div>

    <div id="disc" class="small muted" style="margin-top:8px"></div>

    <details style="margin-top:8px">
      <summary class="small">Сырой JSON</summary>
      <pre id="raw" class="mono"></pre>
    </details>
  </div>
</div>

<script>
const API = window.location.origin;
document.getElementById('api').textContent = API;

const el = id => document.getElementById(id);
const show = (n,on) => n.style.display = on ? '' : 'none';

function colorForScore(s){
  if (typeof s !== 'number') return '';
  if (s >= 85) return '#dcfce7';
  if (s >= 65) return '#fef9c3';
  return '#fee2e2';
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

function renderList(ul, items, renderItem, emptyText="Нет"){
  ul.innerHTML = '';
  if (!Array.isArray(items) || items.length === 0){
    const li=document.createElement('li');
    li.className='muted';
    li.textContent = emptyText;
    ul.appendChild(li);
    return;
  }
  items.forEach(it => {
    const li=document.createElement('li');
    renderItem(li, it);
    ul.appendChild(li);
  });
}

function normalizeRecs(R){
  const arr = Array.isArray(R) ? R : (
    typeof R === 'string' && R.trim() ? [R.trim()] :
    (R && typeof R === 'object') ? [R] : []
  );
  return arr.filter(x => {
    if (typeof x === 'string') return x.trim().length > 0;
    if (x && typeof x === 'object') {
      const wt = (x.what_to_change || '').trim();
      const rn = (x.rationale || '').trim();
      return wt.length > 0 || rn.length > 0;
    }
    return !!x;
  });
}

function renderResult(r){
  const metaMode = (r.meta && r.meta.mode) ? String(r.meta.mode) : null;
  const modeBadge = el('mode');
  const isFree = metaMode === 'free';

  // бейдж режима
  modeBadge.textContent = 'режим: ' + (isFree ? 'свободная модель' : 'мед');
  modeBadge.style.background = isFree ? '#e0f2fe' : '#eef2ff';

  // бейдж оценки (в free не красим)
  const scBadge = el('score');
  const sc = (typeof r.score === 'number') ? r.score : '—';
  scBadge.textContent = 'оценка: ' + sc;
  scBadge.style.background = isFree ? '' : colorForScore(r.score);

  // роль
  el('role').textContent = (r.meta && r.meta.role) ? ('Роль: ' + String(r.meta.role)) : '';

  // свободный ответ:
  //  - в FREE показываем ТОЛЬКО free_text
  //  - в MED показываем free_text, если он есть (иначе прячем)
  const freeBox = el('free');
  const freeTxt = (r.free_text || '').toString().trim();
  if (isFree) {
    if (freeTxt) { freeBox.textContent = freeTxt; show(freeBox, true); }
    else { freeBox.textContent = ''; show(freeBox, false); }
  } else {
    if (freeTxt) { freeBox.textContent = freeTxt; show(freeBox, true); }
    else { freeBox.textContent = ''; show(freeBox, false); }
  }

  // сабскоринг
  const subs = el('subs');
  subs.innerHTML = '';
  const entries = Object.entries(r.subscores || {});
  if (entries.length === 0) subs.style.display = 'none';
  else {
    subs.style.display = 'grid';
    entries.forEach(([k,v])=>{
      const d=document.createElement('div'); d.className='card'; d.style.margin=0;
      d.innerHTML=`<div class="small">${labelMap[k] || k}</div><div style="font-weight:700">${v??'—'}</div>`;
      subs.appendChild(d);
    });
  }

  // критические ошибки
  renderList(el('crit'), r.critical_errors, (li,x)=>{
    const typ = (x && x.type) ? String(x.type) : 'Ошибка';
    const exp = (x && x.explain) ? String(x.explain) : '';
    li.textContent = exp ? (typ + ': ' + exp) : typ;
  });

  // рекомендации
  renderList(el('recs'), normalizeRecs(r.recommendations), (li,x)=>{
    if (typeof x === 'string') li.textContent = x;
    else if (x && typeof x === 'object') {
      const wt = (x.what_to_change || '').trim();
      const rn = (x.rationale || '').trim();
      li.textContent = wt && rn ? `${wt} — ${rn}` : (wt || rn || JSON.stringify(x));
    } else li.textContent = String(x);
  });

  // цитаты: в FREE полностью скрываем; в MED — по наличию
  const citsWrap = el('cits_wrap');
  const cits = el('cits');
  if (isFree) {
    citsWrap.style.display = 'none';
    cits.innerHTML = '';
  } else {
    const hasCits = Array.isArray(r.citations) && r.citations.length > 0;
    if (hasCits) {
      citsWrap.style.display = '';
      renderList(cits, r.citations, (li,x)=>{ li.textContent = String(x); });
    } else {
      citsWrap.style.display = 'none';
      cits.innerHTML = '';
    }
  }

  // дисклеймер — не показываем в FREE
  el('disc').textContent = isFree ? '' : (r.disclaimer ? String(r.disclaimer) : '');
  el('raw').textContent = JSON.stringify(r,null,2);
}

function prettyModelName(id){
  if (id === 'llama3.1:8b')      return 'Llama 3.1 (8B)';
  if (id === 'llama3.1:70b')     return 'Llama 3.1 (70B)';
  if (id === 'deepseek-r1:32b')  return 'DeepSeek R1 (32B)';
  return id;
}

function updateModeUI(){
  const isFree = el('use_free').checked;
  el('query').disabled = isFree;
  el('model').disabled = isFree;
  el('k').disabled = isFree;

  const modeBadge = el('mode');
  modeBadge.textContent = 'режим: ' + (isFree ? 'свободная модель' : 'мед');
  modeBadge.style.background = isFree ? '#e0f2fe' : '#eef2ff';
}

async function fillModels(){
  try{
    const res = await fetch(API + '/runtime/models');
    const m = await res.json();
    const sel = el('model');
    sel.innerHTML = '';
    (m.allowed || []).forEach(id => {
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = prettyModelName(id);
      if (id === m.active) opt.selected = true;
      sel.appendChild(opt);
    });
  }catch(e){
    console.error('models fetch failed', e);
    const sel = el('model');
    ['llama3.1:8b','llama3.1:70b','deepseek-r1:32b'].forEach(id=>{
      const opt = document.createElement('option');
      opt.value = id; opt.textContent = prettyModelName(id);
      sel.appendChild(opt);
    });
  } finally {
    updateModeUI();
  }
}

async function run(){
  el('error').textContent = '';
  el('busy').textContent = '⏳ анализ…';
  show(el('busy'), true);
  el('run').disabled = true;

  try{
    const body = {
      case_text: (el('case').value || '').trim(),
      query:     (el('query').value || '').trim() || null,
      model:     el('model').value || 'llama3.1:8b',
      use_free:  !!el('use_free').checked
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
    let data;
    try { data = JSON.parse(txt); }
    catch(e){ throw new Error('Не удалось разобрать JSON ответа: ' + txt.slice(0,200)); }

    const payload = (data && data.result) ? data.result : data;
    renderResult(payload || {});
  }catch(e){
    el('error').textContent = 'Ошибка: ' + (e?.message || e);
  }finally{
    show(el('busy'), false);
    el('run').disabled = false;
  }
}

el('use_free').addEventListener('change', updateModeUI);
el('run').onclick = run;

el('reindex').onclick = async () => {
  el('error').textContent = '';
  el('busy').textContent = '🔄 запуск переиндексации…';
  show(el('busy'), true);
  try {
    const res = await fetch(API + '/reindex', { method: 'POST' });
    const data = await res.json();
    el('error').textContent = data.message || 'Запущено.';
  } catch(e) {
    el('error').textContent = 'Ошибка запуска: ' + (e?.message || e);
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

setInterval(checkReindexStatus, 60000);
fillModels();
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
        "LLM_ACTIVE": settings.LLM_ACTIVE,
        "LLM_ALLOWED": settings.LLM_ALLOWED,
        "LLM_PRESETS": settings.LLM_PRESETS,
        "LLM_BASE_URL": settings.LLM_BASE_URL,
    }

@app.get("/runtime/models")
def runtime_models():
    allowed = llm_get_allowed()
    labels = llm_get_labels()
    return {
        "active": llm_get_active(),
        "allowed": allowed,
        "labels": {m: labels.get(m, m) for m in allowed},
        "presets": {m: llm_get_preset(m) for m in allowed},
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

        qdrant_url = _normalize_qdrant_url(
            _nz(_os.getenv("QDRANT_URL") or cfg("qdrant", "url", default=settings.QDRANT_URL),
                settings.QDRANT_URL)
        )
        collection  = _nz(_os.getenv("QDRANT_COLLECTION") or cfg("qdrant", "collection", default=settings.QDRANT_COLLECTION),
                          settings.QDRANT_COLLECTION)
        emb_backend = _nz(_os.getenv("EMB_BACKEND") or cfg("embedding", "backend", default=settings.EMB_BACKEND), "hf")
        hf_model    = _nz(_os.getenv("HF_MODEL") or cfg("embedding", "model", default=settings.HF_MODEL), settings.HF_MODEL)

        child_w       = _as_int(_os.getenv("CHILD_W"),       cfg("chunking", "child_w",       default=200))
        child_overlap = _as_int(_os.getenv("CHILD_OVERLAP"), cfg("chunking", "child_overlap", default=35))
        parent_w      = _as_int(_os.getenv("PARENT_W"),      cfg("chunking", "parent_w",      default=800))

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

        emb_batch = str(_as_int(_os.getenv("EMB_BATCH"), 128))  # читаем из ENV, дефолт 128

        cmd_qdr = [
            "python", chunk_and_index_py,
            "--pages-glob",    "data/*.pages.jsonl",
            "--collection",    collection,
            "--qdrant-url",    qdrant_url,
            "--emb-backend",   emb_backend,
            "--hf-model",      hf_model,
            "--batch",         emb_batch,          # <── вот оно
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
