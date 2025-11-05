#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from glob import glob
import importlib  # ← для горячей подгрузки runtime_settings.py

# RAG utils
from rag.bm25_utils import bm25_search, retrieve_hybrid, embed_query_hf

import requests
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ================================
# .env загрузка (dev/prod)
# ================================
try:
    from dotenv import load_dotenv
    _env_mode = (os.getenv("APP_ENV") or "dev").strip().lower()
    _env_file = Path(".env.dev" if _env_mode == "dev" else ".env.prod")
    if _env_file.exists():
        load_dotenv(dotenv_path=_env_file)
        print(f"🔧 Loaded env: {_env_file}")
    else:
        print(f"⚠️ Env file not found: {_env_file} (fallback to process env)")
except Exception as _e:
    print(f"⚠️ dotenv load skipped: {_e}")

# Нормализация env
if os.getenv("EMBEDDING_MODEL") and not os.getenv("HF_MODEL"):
    os.environ["HF_MODEL"] = os.getenv("EMBEDDING_MODEL", "")
if not os.getenv("EMB_BACKEND"):
    os.environ["EMB_BACKEND"] = "hf"

# Автовыбор устройства для HF
if not os.getenv("HF_DEVICE"):
    try:
        import torch
        os.environ["HF_DEVICE"] = "cuda" if torch.cuda.is_available() else "auto"
    except Exception:
        os.environ["HF_DEVICE"] = "auto"

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

# ================================
# Config
# ================================
ROOT = Path(__file__).resolve().parent
CONF_DIR = ROOT / "config"
DEFAULT_YAML = CONF_DIR / "default.yaml"
LOCAL_YAML = CONF_DIR / "local.yaml"

def load_runtime_overrides() -> Dict[str, Any]:
    """
    Подхватывает config/runtime_settings.py (dict RUNTIME), можно менять без пересборки.
    """
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
            "url": os.getenv("QDRANT_URL", "http://localhost:7779"),
            "collection": os.getenv("QDRANT_COLLECTION", "med_kb_v3"),
        },
        "ollama": {
            "base_url": os.getenv("LLM_BASE_URL", "http://host.docker.internal:11434"),
            "model": os.getenv("MODEL_ID", os.getenv("LLM_MODEL", "llama3.1:8b")),
            # ↓ параметры «как долго/много думает»
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
            "timeout_s": int(os.getenv("LLM_TIMEOUT", "180")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.4")),
            "top_p": float(os.getenv("LLM_TOP_P", "0.95")),
            "num_ctx": int(os.getenv("LLM_NUM_CTX", "6144")),
        },
        "retrieval": {"k": 8},
        "embedding": {
            "backend": os.getenv("EMB_BACKEND", "hf"),
            "model": os.getenv("HF_MODEL", os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")),
            "device": os.getenv("HF_DEVICE", "auto"),
            "fp16": False,
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
    runtime = load_runtime_overrides()  # ← слой Python-настроек

    def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in (b or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out

    # Порядок важен: DEFAULTS <- default.yaml <- local.yaml <- runtime_settings.py
    return merge(DEFAULTS, merge(base, merge(local, runtime)))

CONFIG = load_config()

# после CONFIG = load_config()

WARMUP_DONE = False

@app.on_event("startup")
def warmup_bm25():
    """Открываем Lucene индекс один раз, чтобы ускорить первые запросы."""
    global WARMUP_DONE
    if WARMUP_DONE or os.getenv("BM25_WARMUP_DISABLED") == "1":
        return
    try:
        from rag.bm25_utils import bm25_search
        idx = cfg("app", "bm25_index_dir", default="index/bm25_idx")
        bm25_search(idx, "тест", topk=1)  # прогрев JVM + индекса
        print("🔥 BM25 warmed up")
        WARMUP_DONE = True
    except Exception as e:
        print(f"⚠️ BM25 warmup skipped: {e}")

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

def load_pages_text(pages_dir: Path, doc_id: str, p_start: int, p_end: int) -> str:
    jf = pages_dir / f"{doc_id}.pages.jsonl"
    if not jf.exists():
        return ""
    out: List[str] = []
    for line in jf.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rec = json.loads(line)
            pg = int(rec.get("page", 0))
            if p_start <= pg <= p_end:
                out.append(rec.get("text", "") or "")
        except Exception:
            continue
    return "\n".join(out)

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
    # грубо: 1 токен ≈ 4 символа
    return max(1, len(s) // 4)

# ================================
# Qdrant client (REST)
# ================================
def _qdrant_client_rest(url_override: Optional[str] = None):
    from qdrant_client import QdrantClient
    url = (url_override or cfg("qdrant", "url", default="http://qdrant:6333"))
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

def _ollama_generate_stream(ollama_url, payload, connect_timeout_s=3.0, read_timeout_s=50.0) -> str:
    # стрим JSON-объектов построчно; собираем только поле "response"
    with _HTTP.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={**payload, "stream": True},
        timeout=(float(connect_timeout_s), float(read_timeout_s)),
        stream=True
    ) as r:
        r.raise_for_status()
        buf = []
        for ln in r.iter_lines(decode_unicode=True):
            if not ln:
                continue
            try:
                chunk = json.loads(ln)
                if "response" in chunk:
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
    read_timeout_s: float = 50.0,
    num_ctx: int = 6144,
    num_predict: int = 160,
    temperature: float = 0.2,
    extra_options: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    **_ignored_kwargs,
) -> Dict[str, Any]:
    """Вызов Ollama /api/generate с поддержкой таймаутов, стрим-fallback и метрик."""
    import json as _json
    try:
        if not ollama_url:
            ollama_url = cfg("ollama", "base_url", default="http://host.docker.internal:11434") or "http://host.docker.internal:11434"

        opts = {
            "num_ctx": int(num_ctx),
            "num_predict": int(num_predict),
            "temperature": float(temperature),
        }
        if options:
            bad = {"gpu_layers", "num_gpu", "main_gpu"}
            opts.update({k:v for k,v in options.items() if k not in bad})
        if extra_options:
            opts.update({k:v for k,v in extra_options.items() if k not in bad})

        payload = {
            "model": model,
            "prompt": user_prompt,
            "system": system_prompt,
            "format": "json",
            "options": opts,
            "keep_alive": -1,
            "stream": False,
        }

        try:
            resp = _HTTP.post(
                f"{ollama_url.rstrip('/')}/api/generate",
                json=payload,
                timeout=(float(connect_timeout_s), float(read_timeout_s)),
            )
            resp.raise_for_status()
        except requests.exceptions.ReadTimeout:
            # ⬇️ стрим-fallback
            try:
                print("⏩ switch to streaming fallback")
                raw_stream = _ollama_generate_stream(
                    ollama_url, payload,
                    connect_timeout_s=connect_timeout_s, read_timeout_s=read_timeout_s
                )
                raw_stream = _trim_code_fences(raw_stream or "")
                if not raw_stream:
                    return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [], "disclaimer": f"LLM timeout: чтение >{read_timeout_s} с. (stream fallback empty)"}
                return safe_json_extract(raw_stream)
            except Exception as e2:
                return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [], "disclaimer": f"LLM timeout (stream fallback failed): {e2}"}

        # лог метрик (если не стрим)
        meta_logged = False
        if str(resp.headers.get("content-type", "")).startswith("application/json"):
            obj = resp.json()
            if isinstance(obj, dict):
                meta = {
                    "load_ms":   _ns_to_ms(obj.get("load_duration", 0)),
                    "prompt_ms": _ns_to_ms(obj.get("prompt_eval_duration", 0)),
                    "gen_ms":    _ns_to_ms(obj.get("eval_duration", 0)),
                    "total_ms":  _ns_to_ms(obj.get("total_duration", 0)),
                    "prompt_tok": obj.get("prompt_eval_count"),
                    "gen_tok":    obj.get("eval_count"),
                }
                print(f"🧪 OLLAMA META: {meta}")
                response_field = obj.get("response", "")
                meta_logged = True
            else:
                response_field = ""
        else:
            response_field = resp.text or ""

        raw = _json.dumps(response_field, ensure_ascii=False) if isinstance(response_field, (dict, list)) else f"{response_field}".strip()
        if not raw:
            return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [], "disclaimer": "LLM вернул пустой ответ."}
        if not meta_logged:
            print("🧪 OLLAMA META: (no meta in response headers)")
        raw = _trim_code_fences(raw)
        return safe_json_extract(raw)

    except requests.exceptions.ConnectTimeout:
        return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [], "disclaimer": f"LLM timeout: соединение >{connect_timeout_s} с."}
    except Exception as e:
        return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [], "disclaimer": f"Ошибка LLM ({type(e).__name__}): {e}"}

# ================================
# API models
# ================================
class AnalyzeReq(BaseModel):
    case_text: str
    query: Optional[str] = None
    k: int = Field(default_factory=lambda: cfg("retrieval", "k", default=8))
    model: str = Field(default_factory=lambda: cfg("ollama", "model", default="llama3.1:8b"))
    ollama_url: Optional[str] = Field(default_factory=lambda: cfg("ollama", "base_url", default="http://host.docker.internal:11434"))

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
    qdrant_url = _resolve("QDRANT_URL", "http://localhost:7779")
    collection = _resolve("QDRANT_COLLECTION", "med_kb_v3")
    emb_backend = os.getenv("EMB_BACKEND") or cfg("embedding", "backend", default="hf")
    hf_model = os.getenv("HF_MODEL") or cfg("embedding", "model", default="BAAI/bge-m3")
    device = os.getenv("HF_DEVICE") or cfg("embedding", "device", default="auto")
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
            "timeout_s": cfg_int("ollama", "timeout_s", default=180),
            "temperature": cfg_float("ollama", "temperature", default=0.4),
            "top_p": cfg_float("ollama", "top_p", default=0.95),
            "num_ctx": cfg_int("ollama", "num_ctx", default=6144),
        },
        "qdrant": {
            "url": cfg_str("qdrant", "url", default="http://qdrant:6333"),
            "collection": cfg_str("qdrant", "collection", default="med_kb_v3"),
        },
        "retrieval": {"k": cfg_int("retrieval", "k", default=8)},
        "chunking": {
            "child_w": cfg_int("chunking", "child_w", default=200),
            "child_overlap": cfg_int("chunking", "child_overlap", default=35),
            "parent_w": cfg_int("chunking", "parent_w", default=800),
        },
    }

@app.post("/config/reload")
def config_reload():
    global CONFIG
    CONFIG = load_config()  # ← подтянет обновлённый runtime_settings.py
    return {"status": "reloaded"}

@app.post("/analyze")
def analyze_ep(req: AnalyzeReq):
    try:
        print("🚀 /analyze")
        t0 = time.perf_counter()

        if looks_meaningless(req.case_text):
            return {"result": {
                "score": 0, "subscores": {}, "critical_errors": [],
                "recommendations": [], "citations": [],
                "disclaimer": "Текст кейса не содержит осмысленных данных.",
            }}

        # --- Поисковый запрос ---
        def _smart_query(case_text: str) -> str:
            m = re.search(r"\b([A-Za-z]\d{1,2}(?:\.\d+)?)\b", case_text)
            if m:
                return m.group(1)
            t = re.sub(r"\s+", " ", (case_text or "")).strip()
            return t[:200]

        query = req.query or _smart_query(req.case_text)
        print("🔍 query =", query)

        # --- Извлечение контекста ---
        t_r0 = time.perf_counter()
        ctx_items = retrieve_hybrid(
            query, req.k,
            bm25_index_dir = cfg("app", "bm25_index_dir", default="index/bm25_idx"),
            qdrant_url     = cfg("qdrant", "url",        default="http://qdrant:6333"),
            qdrant_collection = cfg("qdrant", "collection", default="med_kb_v3"),
            pages_dir      = cfg("app", "data_dir",      default="data"),

            # эмбеддинги
            hf_model  = cfg("embedding", "model",  default=os.getenv("HF_MODEL", "BAAI/bge-m3")),
            hf_device = cfg("embedding", "device", default=os.getenv("HF_DEVICE", "auto")),
            hf_fp16   = bool(cfg("embedding", "fp16",   default=False)),

            # сколько фрагментов из одного документа максимум
            per_doc_limit = int(os.getenv("PER_DOC_LIMIT", cfg("app", "per_doc_limit", default=2))),

            # РЕРАНКЕР — берём из runtime_settings
            reranker_enabled = bool(cfg("reranker", "enabled", default=False)),
            reranker_model   = cfg("reranker", "model",   default=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")),
            reranker_topn    = int(cfg("reranker", "top_n", default=50)),
            reranker_device  = cfg("reranker", "device", default="cpu"),
)

        t_r1 = time.perf_counter()
        if not ctx_items:
            return {"result": {
                "score": 0, "subscores": {}, "critical_errors": [],
                "recommendations": [], "citations": [],
                "disclaimer": "Контекст не найден в базе знаний — невозможно оценить кейс.",
            }}

        ctx = build_ctx_string(ctx_items, max_chars=8000, per_text_limit=800)
        print(f"📏 lengths: case={len(req.case_text)} ctx={len(ctx)} k={req.k}")

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

        # ── динамический бюджет контекста (но не выше num_ctx из конфига)
        num_ctx_cap = cfg_int("ollama", "num_ctx", default=6144)
        total_est = _approx_tokens(system) + _approx_tokens(user)
        num_ctx = min(num_ctx_cap, max(3072, total_est + 256))

        # ── параметры «как долго и как много думаем»
        llm_timeout   = cfg_int("ollama", "timeout_s", default=180)
        llm_max_tok   = cfg_int("ollama", "max_tokens", default=2048)
        llm_temp      = cfg_float("ollama", "temperature", default=0.4)
        llm_top_p     = cfg_float("ollama", "top_p", default=0.95)

        print(f"🤖 LLM url={req.ollama_url or cfg('ollama','base_url', default='N/A')} "
              f"model={req.model} num_ctx={num_ctx} max_tokens={llm_max_tok} timeout={llm_timeout}s")

        # --- Вызов LLM (попытка 1) ---
        t_l0 = time.perf_counter()
        resp = call_ollama_json(
            req.ollama_url, req.model, system, user,
            read_timeout_s=float(llm_timeout),
            num_ctx=num_ctx,
            num_predict=int(llm_max_tok),
            temperature=float(llm_temp),
            options={"top_p": float(llm_top_p), "repeat_penalty": 1.05},
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

        # --- Если таймаут/пусто — fast-retry ---
        if timed_out or _is_empty(data):
            print("⏩ fast-retry: shrinking context and num_predict")
            ctx_small = build_ctx_string(ctx_items[:min(3, len(ctx_items))], max_chars=6000, per_text_limit=700)
            user_small = user_t.format(case_text=req.case_text, ctx=ctx_small)
            total_est_small = _approx_tokens(system) + _approx_tokens(user_small)
            num_ctx_small = min(num_ctx_cap, max(3072, total_est_small + 128))
            retry_tokens = min(180, llm_max_tok)

            resp2 = call_ollama_json(
                req.ollama_url, req.model, system, user_small,
                read_timeout_s=float(llm_timeout),
                num_ctx=num_ctx_small,
                num_predict=int(retry_tokens),
                temperature=max(0.0, float(llm_temp) * 0.9),
                options={"top_p": float(llm_top_p), "repeat_penalty": 1.05},
            )
            data2 = normalize_result(resp2)
            if not _is_empty(data2):
                data = data2

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

        t2 = time.perf_counter()
        print(f"⏱️ perf: retrieval={int((t_r1-t_r0)*1000)}ms, llm={int((t_l1-t_l0)*1000)}ms, total={int((t2-t0)*1000)}ms")

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
<title>AI-ассистент врача (MVP)</title>
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
</style>
<div class="wrap">
  <div class="card">
    <h1>AI-ассистент врача (MVP) <span id="score" class="badge">оценка: —</span></h1>
    <div class="small">API: <span id="api"></span></div>
  </div>
  <div class="card">
    <label>Текст кейса</label>
    <textarea id="caseText" placeholder="Вставьте кейс: жалобы, анамнез, диагноз, назначения..."></textarea>
    <div class="row">
      <div>
        <label>Запрос для поиска (необязательно)</label>
        <input id="query" placeholder="например: гипертоническая болезнь лечение эналаприл">
      </div>
      <div>
        <label>Модель / K</label>
        <div class="row" style="grid-template-columns:2fr 1fr;gap:8px">
          <select id="model"><option>llama3.1:8b</option><option>llama3.1:70b</option></select>
          <input id="k" type="number" value="6" min="0" max="20">
        </div>
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
const API = window.location.origin; document.getElementById('api').textContent = API;
const el=id=>document.getElementById(id); const show=(n,on)=>n.style.display=on?'':'none';
function colorForScore(s){ if(typeof s!=='number') return ''; if(s>=85) return '#dcfce7'; if(s>=65) return '#fef9c3'; return '#fee2e2'; }
function renderResult(r){
  const sc=r.score??'—'; const sb=document.getElementById('score'); sb.textContent='оценка: '+sc; sb.style.background=colorForScore(sc);
  const subs=el('subs'); subs.innerHTML=''; Object.entries(r.subscores||{}).forEach(([k,v])=>{
    const d=document.createElement('div'); d.className='card'; d.style.margin=0;
    d.innerHTML=`<div class="small">${labelMap[k] || k}</div><div style="font-weight:700">${v??'—'}</div>`;
    subs.appendChild(d);
  });
  const crit=el('crit'); crit.innerHTML=''; (r.critical_errors||[]).forEach(x=>{ const li=document.createElement('li'); li.textContent=`${x.type}: ${x.explain}`; crit.appendChild(li); });
  const recs=el('recs'); recs.innerHTML=''; (r.recommendations||[]).forEach(x=>{ const li=document.createElement('li'); li.textContent=`${x.what_to_change} — ${x.rationale}`; recs.appendChild(li); });
  const cits=el('cits'); cits.innerHTML=''; (r.citations||[]).forEach(x=>{ const li=document.createElement('li'); li.textContent=String(x); cits.appendChild(li); });
  el('raw').textContent=JSON.stringify(r,null,2);
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
async function run(){ el('error').textContent=''; show(el('busy'),true); el('run').disabled=true;
  try{
    const body={ case_text: el('caseText').value||'', query: el('query').value||null, k: parseInt(el('k').value||'6',10), model: el('model').value||'llama3.1:8b' };
    const res=await fetch(API+'/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const txt=await res.text(); let data; try{ data=JSON.parse(txt); }catch(e){ throw new Error('Не удалось разобрать JSON ответа: '+txt.slice(0,200)); }
    renderResult(data.result || data);
  }catch(e){ el('error').textContent='Ошибка: '+(e?.message||e); }
  finally{ show(el('busy'),false); el('run').disabled=false; }
}
el('run').onclick=run;

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


@app.get("/reindex/status")
def reindex_status():
    # всегда объект, чтобы фронт не падал
    return index_status

def run_reindex(*, full: bool = False):
    import os as _os
    import time as _time
    import socket as _socket
    import subprocess as _subprocess
    from pathlib import Path

    global index_status

    # --- helpers ---
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

    # штамп свежести BM25
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

        # --- Шаг 1: Ingest (всегда) ---
        index_status.update({"state": "running", "message": "📄 Шаг 1: парсинг RAW → JSONL (инкрементально)..."})
        print("▶️ ingest_from_raw.py ...")
        cmd_ingest = ["python", "ingest_from_raw.py", "--input-dir", "raw_docs", "--out-dir", "data"]
        if full:
            cmd_ingest.append("--force")
        _subprocess.run(cmd_ingest, check=True, env=env)

        # --- Резолв параметров для следующих шагов ---
        qdrant_url = _normalize_qdrant_url(
            _nz(_os.getenv("QDRANT_URL") or cfg("qdrant", "url", default="http://qdrant:6333"),
                "http://localhost:7779")
        )
        collection = _nz(_os.getenv("QDRANT_COLLECTION") or cfg("qdrant", "collection", default="med_kb_v3"), "med_kb_v3")
        emb_backend = _nz(_os.getenv("EMB_BACKEND") or cfg("embedding", "backend", default="hf"), "hf")
        hf_model = _nz(
            _os.getenv("HF_MODEL") or
            cfg("embedding", "hf_model", default=cfg("embedding", "model", default="BAAI/bge-m3")),
            "BAAI/bge-m3"
        )

        child_w       = _as_int(_os.getenv("CHILD_W"),       cfg("chunking", "child_w",       default=200))
        child_overlap = _as_int(_os.getenv("CHILD_OVERLAP"), cfg("chunking", "child_overlap", default=35))
        parent_w      = _as_int(_os.getenv("PARENT_W"),      cfg("chunking", "parent_w",      default=800))

        print(
            "🔧 RESOLVED → "
            f"QDRANT_URL={qdrant_url}  QDRANT_COLLECTION={collection}  "
            f"EMB_BACKEND={emb_backend}  HF_MODEL={hf_model}  "
            f"child_w={child_w} child_overlap={child_overlap} parent_w={parent_w}"
        )

        # --- Шаг 2: BM25 (только при необходимости или при full) ---
        if full or _bm25_needs_rebuild():
            index_status["message"] = "📚 Шаг 2: построение/обновление BM25 индекса..."
            print("▶️ build_bm25.py ...")
            _subprocess.run(
                [
                    "python", "build_bm25.py",
                    "--pages-glob", "data/*.pages.jsonl",
                    "--out-json",   "index/bm25_json",
                    "--index-dir",  "index/bm25_idx",
                    # если добавишь в build_bm25.py инкрементальный режим, сюда можно докинуть флаг:
                    # "--only-new",
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
            "python", "chunk_and_index.py",
            "--pages-glob",    "data/*.pages.jsonl",
            "--collection",    collection,
            "--qdrant-url",    qdrant_url,
            "--emb-backend",   emb_backend,
            "--hf-model",      hf_model,
            "--batch",         "512",
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
    # Запускаем в фоне и ВСЕГДА возвращаем объект с message
    threading.Thread(target=run_reindex, kwargs={"full": bool(full)}, daemon=True).start()
    return {"status": "started", "message": "Индексация запущена", "full": bool(full)}