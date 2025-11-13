#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import socket
import threading
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import requests
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ================================
# .env -> runtime_settings
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

# runtime settings AFTER env
from config.runtime_settings import settings  # noqa: E402
try:
    settings.apply_env(force=True)
except Exception:
    pass

# pydantic v1/v2 compat
try:
    from pydantic import field_validator  # type: ignore
    _P_V2 = True
except Exception:
    from pydantic import validator as field_validator  # type: ignore
    _P_V2 = False

# RAG utils
from rag.bm25_utils import bm25_search, retrieve_hybrid  # noqa: F401

# Global HTTP sessions (keep-alive)
_HTTP = requests.Session()
_HTTP.headers.update({"Connection": "keep-alive"})
_HTTP2 = requests.Session()

# ----- Local helpers (models from runtime) -----
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

def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def _load_yaml(p: Path) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def load_config() -> Dict[str, Any]:
    DEFAULTS = {
        "app": {"data_dir": "data", "bm25_index_dir": "index/bm25_idx"},
        "qdrant": {
            "url": os.getenv("QDRANT_URL", "http://qdrant:6333"),
            "collection": os.getenv("QDRANT_COLLECTION", "med_kb_v3"),
        },
        "ollama": {
            "base_url": os.getenv("LLM_BASE_URL", "http://ollama:11434"),
            "model": os.getenv("LLM_MODEL", os.getenv("MODEL_ID", settings.LLM_ACTIVE)),
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
# Warmup BM25 (one-time)
# ================================
WARMUP_DONE = False

@app.on_event("startup")
def warmup_bm25():
    global WARMUP_DONE
    if WARMUP_DONE or os.getenv("BM25_WARMUP_DISABLED") == "1":
        return
    try:
        idx = settings.BM25_INDEX_DIR
        bm25_search(idx, "тест", topk=1)  # warms JVM + index
        print("🔥 BM25 warmed up")
        WARMUP_DONE = True
    except Exception as e:
        print(f"⚠️ BM25 warmup skipped: {e}")

# ================================
# Text utils
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

def build_sources(ctx_items, max_out: int = 10):
    out = []
    for it in ctx_items[:max_out]:
        out.append({
            "doc_id": it.get("doc_id") or "unknown",
            "page_start": int(it.get("page_start") or it.get("page") or 1),
            "page_end": int(it.get("page_end") or it.get("page") or 1),
        })
    return out

def build_context_citations(ctx_items, max_out: int = 5):
    cits = []
    for it in ctx_items[:max_out]:
        ps = int(it.get("page_start", it.get("page", 1)) or 1)
        pe = int(it.get("page_end", ps) or ps)
        cits.append(f"{it.get('doc_id','unknown')} стр.{ps}-{pe}")
    return cits

def build_ctx_string(ctx_items, max_chars: int = 8000, per_text_limit: int = 800) -> str:
    parts, total = [], 0
    for i, it in enumerate(ctx_items, 1):
        txt = (it.get("text", "") or "")[:per_text_limit]
        ps = int(it.get("page_start", it.get("page", 1)) or 1)
        pe = int(it.get("page_end", ps) or ps)
        chunk = f"### [{i}] DOC {it.get('doc_id','unknown')} P{ps}-{pe}\n{txt}\n\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts)

CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\s*([\s\S]*?)```", flags=re.DOTALL)
def strip_code_fences_strict(s: str) -> str:
    if not s: return s
    m = CODE_BLOCK_RE.search(s)
    return m.group(1).strip() if m else s.strip()

def strip_code_fences_loose(s: str) -> str:
    if not s: return s
    s = s.replace("\r", "")
    s = s.lstrip()
    if s.startswith("```"):
        p = s.find("\n")
        s = s[p+1:] if p != -1 else ""
    s = s.rstrip("`").rstrip()
    return s

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
        t = cut[:last_dot+1] if last_dot > target_chars * 0.6 else cut
    return t.strip()

# ---- helpers: formatting for prompt templates ----
class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

def safe_format(template: str, **kwargs) -> str:
    if not isinstance(template, str):
        return template
    t = template.replace("{case_text}", "<<<__CASE__>>>").replace("{ctx}", "<<<__CTX__>>>")
    t = t.replace("{", "{{").replace("}", "}}")
    t = t.replace("<<<__CASE__>>>", "{case_text}").replace("<<<__CTX__>>>", "{ctx}")
    try:
        return t.format_map(_SafeDict(**kwargs))
    except Exception:
        return t.format(case_text=kwargs.get("case_text", ""), ctx=kwargs.get("ctx", ""))

# ================================
# Static UI
# ================================
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def _ui_index_path() -> Optional[Path]:
    name = os.getenv("UI_INDEX_FILE", "").strip()
    candidates = [name] if name else []
    candidates += ["med_ui.html", "med_ai.html", "index.html"]
    for n in candidates:
        if not n:
            continue
        p = STATIC_DIR / n
        if p.exists():
            return p
    for p in sorted(STATIC_DIR.glob("*.html")):
        return p
    return None

@app.get("/", response_class=HTMLResponse)
def ui_root():
    p = _ui_index_path()
    if not p:
        return HTMLResponse(
            "<h3>UI файл не найден в ./static (ожидались med_ui.html / med_ai.html / index.html)</h3>",
            status_code=404
        )
    return FileResponse(str(p))

# ================================
# /health & debug
# ================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "app_env": os.getenv("APP_ENV", "dev"),
        "qdrant": settings.QDRANT_URL,
        "qdrant_collection": settings.QDRANT_COLLECTION,
        "llm_model": llm_get_active(),
        "embed_backend": settings.EMB_BACKEND,
        "embed_model": settings.HF_MODEL,
        "embed_device": settings.HF_DEVICE or "auto",
    }

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

# ================================
# Indexing (background)
# ================================
index_status = {"state": "idle", "message": "Ожидание"}

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

    def _as_int_local(val, fallback: int) -> int:
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

        index_status.update({"state": "running", "message": "📄 Шаг 1: парсинг RAW → JSONL (инкрементально)..."} )
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
            _nz(_os.getenv("QDRANT_URL") or cfg("qdrant", "url", default=settings.QDRANT_URL), settings.QDRANT_URL)
        )
        collection  = _nz(_os.getenv("QDRANT_COLLECTION") or cfg("qdrant", "collection", default=settings.QDRANT_COLLECTION),
                          settings.QDRANT_COLLECTION)
        emb_backend = _nz(_os.getenv("EMB_BACKEND") or cfg("embedding", "backend", default=settings.EMB_BACKEND), "hf")
        hf_model    = _nz(_os.getenv("HF_MODEL") or cfg("embedding", "model", default=settings.HF_MODEL), settings.HF_MODEL)

        child_w       = _as_int_local(_os.getenv("CHILD_W"),       cfg("chunking", "child_w",       default=200))
        child_overlap = _as_int_local(_os.getenv("CHILD_OVERLAP"), cfg("chunking", "child_overlap", default=35))
        parent_w      = _as_int_local(_os.getenv("PARENT_W"),      cfg("chunking", "parent_w",      default=800))

        bm25_child_w  = _as_int_local(_os.getenv("BM25_CHILD_W"),         _as_int_local(_os.getenv("CHILD_W"), 200))
        bm25_overlap  = _as_int_local(_os.getenv("BM25_CHILD_OVERLAP"),   _as_int_local(_os.getenv("CHILD_OVERLAP"), 40))
        bm25_lang     = _nz(_os.getenv("BM25_LANGUAGE"), "ru")

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

        # --- Step 3: Dense → Qdrant ---
        index_status["message"] = "🧠 Шаг 3: индексация в Qdrant (dense)..."

        emb_batch = str(_as_int_local(_os.getenv("EMB_BATCH"), 128))

        cmd_qdr = [
            "python", chunk_and_index_py,
            "--pages-glob",    "data/*.pages.jsonl",
            "--collection",    collection,
            "--qdrant-url",    qdrant_url,
            "--emb-backend",   emb_backend,
            "--hf-model",      hf_model,
            "--batch",         emb_batch,
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

# ================================
# Retrieval preview / citations
# ================================
class CitationsReq(BaseModel):
    case_text: str
    query: Optional[str] = None
    k: Optional[int] = None

@app.post("/citations")
def citations_ep(req: CitationsReq):
    k = int(req.k or settings.RETR_TOP_K)
    query = (req.query or "").strip()
    if not query or looks_meaningless(query):
        query = _compact_case_text(req.case_text, target_chars=220)

    ctx_items = retrieve_hybrid(
        query=query,
        k=k,
        bm25_index_dir=cfg("app","bm25_index_dir", default=settings.BM25_INDEX_DIR),
        qdrant_url=settings.QDRANT_URL,
        qdrant_collection=settings.QDRANT_COLLECTION,
        pages_dir=cfg("app","data_dir", default=settings.PAGES_DIR),
        hf_model=settings.HF_MODEL,
        hf_device=(settings.HF_DEVICE or "auto"),
        hf_fp16=bool(getattr(settings,"HF_FP16", True)),
        per_doc_limit=int(getattr(settings,"RETR_PER_DOC_LIMIT", 1)),
        reranker_enabled=bool(getattr(settings,"RERANKER_ENABLED", False)),
        rerank_top_k=int(getattr(settings,"RERANK_TOP_K", 50)),
    )
    return {"citations": build_context_citations(ctx_items)}

class RetrievalPreviewReq(BaseModel):
    case_text: str
    query: Optional[str] = None
    k: Optional[int] = None
    per_doc_limit: Optional[int] = None
    with_text: Optional[bool] = True  # вернуть тексты фрагментов

@app.post("/retrieval/preview")
def retrieval_preview(req: RetrievalPreviewReq):
    k = int(req.k or settings.RETR_TOP_K)
    query = (req.query or "").strip()
    if not query or looks_meaningless(query):
        query = _compact_case_text(req.case_text, target_chars=220)

    per_doc = int(req.per_doc_limit or int(getattr(settings, "RETR_PER_DOC_LIMIT", 1)))

    ctx_items = retrieve_hybrid(
        query=query,
        k=k,
        bm25_index_dir=cfg("app", "bm25_index_dir", default=settings.BM25_INDEX_DIR),
        qdrant_url=settings.QDRANT_URL,
        qdrant_collection=settings.QDRANT_COLLECTION,
        pages_dir=cfg("app", "data_dir", default=settings.PAGES_DIR),
        hf_model=settings.HF_MODEL,
        hf_device=(settings.HF_DEVICE or "auto"),
        hf_fp16=bool(getattr(settings, "HF_FP16", True)),
        per_doc_limit=per_doc,
        reranker_enabled=bool(getattr(settings, "RERANKER_ENABLED", False)),
        rerank_top_k=int(getattr(settings, "RERANK_TOP_K", 50)),
    )

    chunks: List[Dict[str, Any]] = []
    for it in ctx_items:
        chunks.append({
            "doc_id": str(it.get("doc_id", "")),
            "page_start": int(it.get("page_start", 1)),
            "page_end": int(it.get("page_end", it.get("page_start", 1))),
            "page_range": f"{int(it.get('page_start',1))}-{int(it.get('page_end', it.get('page_start',1)))}",
            "text": (it.get("text") or "")[:1500] if req.with_text else None,
        })

    return {
        "citations": build_context_citations(ctx_items),
        "chunks": chunks,
        "debug": {
            "bm25_index_dir": cfg("app", "bm25_index_dir", default=settings.BM25_INDEX_DIR),
            "pages_dir": cfg("app", "data_dir", default=settings.PAGES_DIR),
            "qdrant_url": settings.QDRANT_URL,
            "qdrant_collection": settings.QDRANT_COLLECTION,
            "hf_model": settings.HF_MODEL,
            "per_doc_limit": per_doc,
            "k": k,
        }
    }

# ================================
# LLM generate (JSON-safe util)
# ================================
def _trim_code_fences2(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("` \n")
        s = s.split("\n", 1)[-1]
    return s.strip()

# ================================
# CHAT — RAG-CONTROLLED
# ================================
class ChatTurn(BaseModel):
    role: Literal["system","user","assistant"]
    content: str

class ChatReq(BaseModel):
    model: str
    messages: List["ChatTurn"]
    stream: Optional[bool] = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    k: Optional[int] = None
    debug: Optional[bool] = False  # NEW

def _build_case_from_history(msgs: List["ChatTurn"]) -> str:
    # Берём последние 6 пользовательских сообщений и сжимаем
    last_user = [m.content for m in msgs if m.role == "user"][-6:]
    # реальный regexp, а не s.replace(r"\W","")
    last_user = [s for s in last_user if s and re.sub(r"\W+", "", s).strip()]
    txt = "\n\n".join(last_user).strip()
    # даём ретривалу побольше «мяса»
    return _compact_case_text(txt, target_chars=2000)

def _split_case_and_plan(raw: str) -> tuple[str, str]:
    t = (raw or "").strip()
    if not t:
        return "", ""
    m = re.search(
        r"(?:^|\n)\s*(?:Назначени[ея]|План(?:\s+лечения)?|Рекомендац[ии]|Терапия|Лечение)\s*[:\-]?\s*\n([\s\S]+)$",
        t, flags=re.IGNORECASE
    )
    block = m.group(1) if m else t
    lines = [ln.rstrip() for ln in block.splitlines()]
    plan_lines = []
    for ln in lines:
        s = ln.strip()
        if (re.match(r"^[-•∙–]\s*", s) or
            re.match(r"^\d+[.)]\s*", s) or
            re.search(r"\b(мг|г|мл|р/д|таб|\bсвеч|\bмазь|\bкрем|\bгель)\b", s, flags=re.IGNORECASE)):
            plan_lines.append(s)
    plan = "\n".join(plan_lines).strip()
    if plan:
        src_lines = t.splitlines()
        rm = set(plan_lines)
        case_clean = "\n".join([x for x in src_lines if x.strip() not in rm]).strip()
    else:
        case_clean = t
    if len(plan) < 15:
        plan = ""
    return case_clean, plan

def _make_system_prompt(ctx_text: str, citations: List[str]) -> str:
    """
    Жёсткий системный промпт: отвечать ТОЛЬКО по КОНТЕКСТУ.
    Если в контексте нет однозначного ответа — вернуть фиксированную фразу.
    Без Markdown и кода.
    """
    base = (cfg("prompt", "system", default="") or "").strip()

    hard = (
        "Ты — ассистент поиска по медицинской базе. Отвечай ТОЛЬКО по тексту из КОНТЕКСТА ниже. "
        "Никаких предположений и «общих знаний». Если в КОНТЕКСТЕ нет однозначного ответа, "
        "верни дословно: «Нет ответа в базе по предоставленному контексту.» "
        "Отвечай кратко, на русском, без Markdown и без код-блоков.\n\n"
        "=== КОНТЕКСТ (RAG) ===\n"
        f"{ctx_text}\n"
        "=== КОНЕЦ КОНТЕКСТА ===\n\n"
        "Справочные ссылки (их можно перечислить в конце ответа как Источники):\n"
        + ("\n".join(f"- {c}" for c in (citations or []))) +
        ("\n" if citations else "")
    )

    return (base + "\n\n" + hard).strip() if base else hard


def _make_user_prompt(case_text: str, doctor_plan: str) -> str:
    """
    Пользовательский промпт: сформулировать задачу «найти в контексте», без пересказа плана.
    Источники просим перечислить в конце строки.
    """
    add = f"\n\n[План врача]\n{doctor_plan}" if doctor_plan else ""
    return (
        "ВОПРОС/ЗАПРОС:\n"
        f"{case_text}{add}\n\n"
        "ЗАДАЧА:\n"
        "- Найди в КОНТЕКСТЕ выше точный и короткий ответ (термин, число, краткая фраза).\n"
        "- Если однозначного ответа в КОНТЕКСТЕ нет — напиши: «Нет ответа в базе по предоставленному контексту.»\n"
        "- В конце добавь строку вида: Источники: DOC_ID стр.A-B; DOC_ID стр.C-D\n"
        "- Краткость важнее. Без Markdown и без код-блоков."
    )

@app.post("/chat")
def chat(req: ChatReq):
    model_id = (req.model or llm_get_active()).strip()
    k = int(req.k or settings.RETR_TOP_K)

    # RAG
    mini_case = _build_case_from_history(req.messages)
    case_clean, doctor_plan = _split_case_and_plan(mini_case or "")
    # ключевая правка: используем полноценный сжатый текст, а не [:220]
    query = _compact_case_text(mini_case, target_chars=800) or "анальная трещина"  # запасной релевантный сид
    ctx_items = retrieve_hybrid(
        query=query if not looks_meaningless(query) else "анальная трещина",
        k=k,
        bm25_index_dir=cfg("app", "bm25_index_dir", default=settings.BM25_INDEX_DIR),
        qdrant_url=settings.QDRANT_URL,
        qdrant_collection=settings.QDRANT_COLLECTION,
        pages_dir=cfg("app", "data_dir", default=settings.PAGES_DIR),
        hf_model=settings.HF_MODEL,
        hf_device=(settings.HF_DEVICE or "auto"),
        hf_fp16=bool(getattr(settings, "HF_FP16", True)),
        per_doc_limit=int(getattr(settings, "RETR_PER_DOC_LIMIT", 1)),
        reranker_enabled=bool(getattr(settings, "RERANKER_ENABLED", False)),
        rerank_top_k=int(getattr(settings, "RERANK_TOP_K", 50)),
    )
    ctx_text = build_ctx_string(ctx_items, max_chars=int(getattr(settings, "CTX_SNIPPET_LIMIT", 4000)))
    citations = build_context_citations(ctx_items)

    system_prompt = _make_system_prompt(ctx_text, citations)
    user_prompt = _make_user_prompt(_compact_case_text(case_clean), doctor_plan)

    ollama = cfg_str("ollama", "base_url", default="http://ollama:11434").rstrip("/")
    preset = llm_get_preset(model_id)
    payload = {
        "model": model_id,
        "system": system_prompt,
        "prompt": user_prompt,
        "options": {
            "num_ctx": int(preset.get("num_ctx", getattr(settings, "LLM_NUM_CTX", 8192))),
            "temperature": float(req.temperature or preset.get("temperature", getattr(settings, "LLM_TEMPERATURE", 0.2))),
            "top_p": float(req.top_p or preset.get("top_p", getattr(settings, "LLM_TOP_P", 0.95))),
            "repeat_penalty": float(preset.get("repeat_penalty", 1.05)),
            "num_predict": int(preset.get("max_tokens", getattr(settings, "LLM_MAX_TOKENS", 600))),
        }
    }
    r = _HTTP2.post(f"{ollama}/api/generate", json=payload, timeout=(3.0, 180.0))
    r.raise_for_status()
    text = r.json().get("response","") if str(r.headers.get("content-type","")).startswith("application/json") else (r.text or "")
    msg = _trim_code_fences2(text)

    if req.debug:
        return {
            "message": msg,
            "debug": {
                "query": query,
                "ctx_items": [{"doc_id": it.get("doc_id"), "p": [it.get("page_start"), it.get("page_end")]} for it in ctx_items[:8]],
                "citations": citations,
                "ctx_chars": len(ctx_text),
            }
        }
    return {"message": msg}

@app.post("/chat/stream")
def chat_stream(req: ChatReq):
    model_id = (req.model or llm_get_active()).strip()
    k = int(req.k or settings.RETR_TOP_K)

    mini_case = _build_case_from_history(req.messages)
    case_clean, doctor_plan = _split_case_and_plan(mini_case or "")
    query = _compact_case_text(mini_case, target_chars=800) or "анальная трещина"
    ctx_items = retrieve_hybrid(
        query=query if not looks_meaningless(query) else "анальная трещина",
        k=k,
        bm25_index_dir=cfg("app", "bm25_index_dir", default=settings.BM25_INDEX_DIR),
        qdrant_url=settings.QDRANT_URL,
        qdrant_collection=settings.QDRANT_COLLECTION,
        pages_dir=cfg("app", "data_dir", default=settings.PAGES_DIR),
        hf_model=settings.HF_MODEL,
        hf_device=(settings.HF_DEVICE or "auto"),
        hf_fp16=bool(getattr(settings, "HF_FP16", True)),
        per_doc_limit=int(getattr(settings, "RETR_PER_DOC_LIMIT", 1)),
        reranker_enabled=bool(getattr(settings, "RERANKER_ENABLED", False)),
        rerank_top_k=int(getattr(settings, "RERANK_TOP_K", 50)),
    )
    ctx_text = build_ctx_string(ctx_items, max_chars=int(getattr(settings, "CTX_SNIPPET_LIMIT", 4000)))
    citations = build_context_citations(ctx_items)

    system_prompt = _make_system_prompt(ctx_text, citations)
    user_prompt = _make_user_prompt(_compact_case_text(case_clean), doctor_plan)

    ollama = cfg_str("ollama", "base_url", default="http://ollama:11434").rstrip("/")
    preset = llm_get_preset(model_id)
    payload = {
        "model": model_id,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": True,
        "options": {
            "num_ctx": int(preset.get("num_ctx", getattr(settings, "LLM_NUM_CTX", 8192))),
            "temperature": float(req.temperature or preset.get("temperature", getattr(settings, "LLM_TEMPERATURE", 0.2))),
            "top_p": float(req.top_p or preset.get("top_p", getattr(settings, "LLM_TOP_P", 0.95))),
            "repeat_penalty": float(preset.get("repeat_penalty", 1.05)),
            "num_predict": int(preset.get("max_tokens", getattr(settings, "LLM_MAX_TOKENS", 600))),
        }
    }

    def gen():
        # при debug=true первым кадром отдаём служебную инфу
        if getattr(req, "debug", False):
            dbg = {
                "type": "debug",
                "query": query,
                "citations": citations,
                "ctx_chars": len(ctx_text),
                "ctx_first_ids": [str(it.get("doc_id")) for it in ctx_items[:6]],
            }
            yield json.dumps(dbg, ensure_ascii=False) + "\n"

        try:
            with _HTTP2.post(f"{ollama}/api/generate", json=payload, stream=True, timeout=(3.0, 600.0)) as r:
                r.raise_for_status()
                for ln in r.iter_lines(decode_unicode=True):
                    if not ln:
                        continue
                    try:
                        evt = json.loads(ln)
                    except Exception:
                        continue
                    delta = str(evt.get("response", "") or "")
                    if delta:
                        yield json.dumps({"type": "delta", "delta": delta}, ensure_ascii=False) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "status", "message": f"stream error: {type(e).__name__}: {e}"}, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
# ================================
# Analyze (оставлен для совместимости/тестов)
# ================================
class AnalyzeReq(BaseModel):
    case_text: str
    query: Optional[str] = None
    k: Optional[int] = Field(default=None)
    model: str = Field(default_factory=lambda: llm_get_active())
    ollama_url: Optional[str] = Field(default_factory=lambda: cfg("ollama", "base_url", default="http://ollama:11434"))
    use_free: Optional[bool] = Field(default=False)

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

def _extract_any_text(resp) -> str:
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for key in ("raw_block", "response", "text", "message", "content", "answer"):
            v = resp.get(key)
            if isinstance(v, str) and v.strip():
                return v
    return ""

def _norm_critical_errors(ce) -> list[dict]:
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

def _clean_free_text(s: str) -> str:
    if not s:
        return ""
    t = strip_code_fences_strict(s)
    if "```" in t:
        t = strip_code_fences_loose(t)
    return t.strip().strip("`").strip()

def _norm_recommendations(rec) -> list[str]:
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

    seen = set()
    out = []
    for it in items:
        t = (it or "").strip()
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out

def _try_parse_json_from_text(s: str):
    if not s:
        return None
    t = strip_code_fences_strict(s)
    if t == s:
        t = strip_code_fences_loose(t)
    t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end > start:
        candidate = t[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None

def normalize_result_loose(resp: dict | str) -> dict:
    out = {
        "critical_errors": [], "recommendations": [], "citations": [],
        "disclaimer": "", "meta": {}, "free_text": ""
    }

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

    if not isinstance(resp, dict):
        return out

    out["critical_errors"] = _norm_critical_errors(resp.get("critical_errors"))
    out["recommendations"] = _norm_recommendations(resp.get("recommendations"))

    if isinstance(resp.get("meta"), dict):
        out["meta"] = dict(resp["meta"])
    if isinstance(resp.get("citations"), list):
        out["citations"] = [str(x) for x in resp["citations"] if x]
    if isinstance(resp.get("disclaimer"), str):
        out["disclaimer"] = resp["disclaimer"]

    raw_txt = _extract_any_text(resp)
    if raw_txt:
        parsed = _try_parse_json_from_text(raw_txt)
        if isinstance(parsed, dict):
            if not out["recommendations"]:
                out["recommendations"] = _norm_recommendations(parsed.get("recommendations"))
            if not out["critical_errors"]:
                out["critical_errors"] = _norm_critical_errors(parsed.get("critical_errors"))
            free = (
                str(parsed.get("answer") or parsed.get("text") or parsed.get("message") or "").strip()
            )
            out["free_text"] = _clean_free_text(free) if free else _clean_free_text(raw_txt)
        else:
            out["free_text"] = _clean_free_text(raw_txt)

    return out

@app.post("/analyze")
def analyze(req: AnalyzeReq):
    # оставлено для обратной совместимости (не основной путь)
    model_id   = (req.model or cfg("ollama", "model", default="llama3.1:8b")).strip()
    preset     = llm_get_preset(model_id)
    ollama_url = (req.ollama_url or cfg("ollama", "base_url", default="http://ollama:11434")).strip()

    k     = int(req.k or settings.RETR_TOP_K)
    query = (req.query or "").strip()
    if not query or looks_meaningless(query):
        query = _compact_case_text(req.case_text, target_chars=220)

    ctx_items = retrieve_hybrid(
        query=query,
        k=k,
        bm25_index_dir=cfg("app", "bm25_index_dir", default=settings.BM25_INDEX_DIR),
        qdrant_url=settings.QDRANT_URL,
        qdrant_collection=settings.QDRANT_COLLECTION,
        pages_dir=cfg("app", "data_dir", default=settings.PAGES_DIR),
        hf_model=settings.HF_MODEL,
        hf_device=(settings.HF_DEVICE or "auto"),
        hf_fp16=bool(getattr(settings, "HF_FP16", True)),
        per_doc_limit=int(getattr(settings, "RETR_PER_DOC_LIMIT", 1)),
        reranker_enabled=bool(getattr(settings, "RERANKER_ENABLED", False)),
        rerank_top_k=int(getattr(settings, "RERANK_TOP_K", 50)),
    )

    ctx_text  = build_ctx_string(ctx_items, max_chars=int(getattr(settings, "CTX_SNIPPET_LIMIT", 4000)))
    citations = build_context_citations(ctx_items)

    system = (cfg("prompt", "system", default="") or "").strip()
    user_t = (cfg("prompt", "user_template", default="") or "").strip()
    if not user_t:
        user_t = "{case_text}\n\n(используй предоставленный контекст)"
    user_f = safe_format(user_t, case_text=_compact_case_text(req.case_text), ctx=ctx_text)

    payload = {
        "model": model_id,
        "system": system,
        "prompt": user_f,
        "options": {
            "num_ctx": int(preset.get("num_ctx", getattr(settings, "LLM_NUM_CTX", 6144))),
            "num_predict": int(preset.get("max_tokens", getattr(settings, "LLM_MAX_TOKENS", 600))),
            "temperature": float(preset.get("temperature", getattr(settings, "LLM_TEMPERATURE", 0.2))),
            "top_p": float(preset.get("top_p", getattr(settings, "LLM_TOP_P", 0.95))),
        }
    }

    r = _HTTP2.post(f"{ollama_url}/api/generate", json=payload, timeout=(3.0, 180.0))
    r.raise_for_status()
    if str(r.headers.get("content-type","")).startswith("application/json"):
        obj = r.json()
        text = obj.get("response","") if isinstance(obj, dict) else ""
    else:
        text = r.text or ""

    out = normalize_result_loose(text)
    out.setdefault("meta", {})
    out["meta"].setdefault("role", "медицинский ассистент")
    out["meta"]["mode"] = "med"
    if citations and not out.get("citations"):
        out["citations"] = citations
    return {"result": out}
