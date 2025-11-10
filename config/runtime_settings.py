# config/runtime_settings.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Optional, Dict, Any

# ---------------- helpers ----------------

def _str2bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

def _to_int(v: Optional[str], default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default

def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _normalize_qdrant_url(url_in: Optional[str]) -> str:
    url = (url_in or os.getenv("QDRANT_URL") or os.getenv("QDRANT") or "http://qdrant:6333").strip()
    if "://" not in url:
        return f"http://{url}"
    if url.lower().startswith("qdrant://"):
        return "http://" + url[len("qdrant://"):]
    if url.lower().startswith("qdrant:"):
        return "http://" + url[len("qdrant:"):]
    return url


# ================== ГЛАВНЫЙ БЛОК УПРАВЛЕНИЯ ==================
# Меняешь здесь — эти значения перекроют .env
CONTROL: Dict[str, Any] = {
    # LLM
    "LLM_MODEL": "llama3.1:8b",

    # Чанкинг (dense и BM25)
    "CHILD_W": 200,
    "CHILD_OVERLAP": 40,
    "PARENT_W": 800,

    # BM25-специфика
    "BM25_CHILD_W": 200,
    "BM25_CHILD_OVERLAP": 40,
    "BM25_LANGUAGE": "ru",

    # API / App
    "APP_ENV":        "dev",
    "APP_HOST":       "0.0.0.0",
    "APP_PORT":       8000,
    "LLM_BASE_URL":   "http://host.docker.internal:11434",

    # Retrieval
    "RETR_TOP_K":         4,
    "RETR_PER_DOC_LIMIT": 1,          
    "CTX_SNIPPET_LIMIT":  400,        # сколько символов контекста отдавать в LLM

    # Paths & indexes
    "PAGES_DIR":          "data",
    "BM25_INDEX_DIR":     "index/bm25_idx",
    "QDRANT_COLLECTION":  "med_kb_v3",
    "QDRANT_URL":         "http://qdrant:6333",

    # Embeddings
    "EMB_BACKEND": "hf",
    "HF_MODEL":    "BAAI/bge-m3",
    "HF_DEVICE":   None,               # 'cuda' | 'cpu' | None(авто)
    "HF_FP16":     True,

    # Reranker
    "RERANKER_ENABLED": True,
    "RERANK_TOP_K":     20,
    "RERANKER_MODEL":   "BAAI/bge-reranker-v2-m3",

    # EasyOCR
    "EASYOCR_DIR":             "/root/.EasyOCR",
    "EASYOCR_ALLOW_DOWNLOADS": True,

    # HF cache (чтобы не писать в '/.cache/...'),
    "TRANSFORMERS_CACHE": "/root/.cache/huggingface",

    # Logs
    "LOG_LEVEL": "INFO",
    

    # LLM лимиты/таймауты
    "LLM_NUM_CTX":    4096,
    "LLM_MAX_TOKENS": 300,
    "LLM_TIMEOUT":    250, # сек
    "LLM_CTX_MARGIN": 768,
    "LLM_MIN_CTX": 2048, 
    "LLM_KEEP_ALIVE": "30m",     # чтобы держать модель тёплой
    "LLM_NUM_GPU_LAYERS": -1,
    "LLM_STREAM_CHUNK_TIMEOUT": 30,
    "LLM_TEMPERATURE": 0.2,
}

# Для обратной совместимости:
RUNTIME = CONTROL


# ================== КЛАСС НАСТРОЕК ==================

class Settings:
    # Инициализация читает сначала ENV, потом мы принудительно перекроем ENV значениями CONTROL через apply_env(force=True).

    # --- App ---
    APP_ENV: str = os.getenv("APP_ENV", CONTROL["APP_ENV"])
    APP_HOST: str = os.getenv("APP_HOST", CONTROL["APP_HOST"])
    APP_PORT: int = _to_int(os.getenv("APP_PORT"), CONTROL["APP_PORT"])
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", CONTROL["LLM_BASE_URL"])

    # --- Paths & DB ---
    PAGES_DIR: str = os.getenv("PAGES_DIR", CONTROL["PAGES_DIR"])
    BM25_INDEX_DIR: str = os.getenv("BM25_INDEX_DIR", CONTROL["BM25_INDEX_DIR"])
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", CONTROL["QDRANT_COLLECTION"])
    QDRANT_URL: str = _normalize_qdrant_url(os.getenv("QDRANT_URL") or CONTROL["QDRANT_URL"])

    # --- Retrieval ---
    RETR_TOP_K: int = _to_int(os.getenv("RETR_TOP_K"), CONTROL["RETR_TOP_K"])
    RETR_PER_DOC_LIMIT: int = _to_int(os.getenv("RETR_PER_DOC_LIMIT"), CONTROL["RETR_PER_DOC_LIMIT"])
    CTX_SNIPPET_LIMIT: int = _to_int(os.getenv("CTX_SNIPPET_LIMIT"), CONTROL["CTX_SNIPPET_LIMIT"])

    # --- Embeddings ---
    EMB_BACKEND: str = os.getenv("EMB_BACKEND", CONTROL["EMB_BACKEND"])
    HF_MODEL: str = os.getenv("HF_MODEL", CONTROL["HF_MODEL"])
    HF_DEVICE: Optional[str] = os.getenv("HF_DEVICE", CONTROL["HF_DEVICE"] or "") or None
    HF_FP16: bool = _str2bool(os.getenv("HF_FP16"), CONTROL["HF_FP16"])

    # --- Reranker ---
    RERANKER_ENABLED: bool = _str2bool(os.getenv("RERANKER_ENABLED"), CONTROL["RERANKER_ENABLED"])
    RERANK_TOP_K: int = _to_int(os.getenv("RERANK_TOP_K"), CONTROL["RERANK_TOP_K"])
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", CONTROL["RERANKER_MODEL"])

    # --- OCR / Caches ---
    EASYOCR_DIR: str = os.getenv("EASYOCR_DIR", CONTROL["EASYOCR_DIR"])
    EASYOCR_ALLOW_DOWNLOADS: bool = _str2bool(os.getenv("EASYOCR_ALLOW_DOWNLOADS"), CONTROL["EASYOCR_ALLOW_DOWNLOADS"])
    TRANSFORMERS_CACHE: str = os.getenv("TRANSFORMERS_CACHE", CONTROL["TRANSFORMERS_CACHE"])

    # --- Logs & LLM ---
    LLM_KEEP_ALIVE: str = os.getenv("LLM_KEEP_ALIVE", CONTROL["LLM_KEEP_ALIVE"])
    LLM_STREAM_CHUNK_TIMEOUT: int = _to_int(os.getenv("LLM_STREAM_CHUNK_TIMEOUT"), CONTROL["LLM_STREAM_CHUNK_TIMEOUT"])
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", CONTROL["LOG_LEVEL"])
    LLM_NUM_CTX: int = _to_int(os.getenv("LLM_NUM_CTX"), CONTROL["LLM_NUM_CTX"])
    LLM_MAX_TOKENS: int = _to_int(os.getenv("LLM_MAX_TOKENS"), CONTROL["LLM_MAX_TOKENS"])
    LLM_TIMEOUT: int = _to_int(os.getenv("LLM_TIMEOUT"), CONTROL["LLM_TIMEOUT"])
    # ── в class Settings (раздел "Logs & LLM") добавь поля:
    LLM_CTX_MARGIN: int = _to_int(os.getenv("LLM_CTX_MARGIN"), CONTROL["LLM_CTX_MARGIN"])
    LLM_MIN_CTX:   int = _to_int(os.getenv("LLM_MIN_CTX"),   CONTROL["LLM_MIN_CTX"])
    # и эти два, чтобы быть полностью централизованным:
    LLM_KEEP_ALIVE: str = os.getenv("LLM_KEEP_ALIVE", CONTROL["LLM_KEEP_ALIVE"])
    LLM_NUM_GPU_LAYERS: int = _to_int(os.getenv("LLM_NUM_GPU_LAYERS"), CONTROL["LLM_NUM_GPU_LAYERS"])


    def __init__(self) -> None:
        # Подготовим каталоги
        _ensure_dir(Path(self.TRANSFORMERS_CACHE))
        os.environ.setdefault("TRANSFORMERS_CACHE", self.TRANSFORMERS_CACHE)

        _ensure_dir(Path(self.EASYOCR_DIR))
        _ensure_dir(Path(self.EASYOCR_DIR) / "model")

        # ── в __init__ (ниже прочих setdefault) по желанию пробрось в окружение:
        os.environ.setdefault("LLM_KEEP_ALIVE", self.LLM_KEEP_ALIVE)
        os.environ.setdefault("LLM_NUM_GPU_LAYERS", str(self.LLM_NUM_GPU_LAYERS))
        os.environ.setdefault("LLM_CTX_MARGIN", str(self.LLM_CTX_MARGIN))
        os.environ.setdefault("LLM_MIN_CTX",   str(self.LLM_MIN_CTX))


        # Пробросим базовые переменные окружения для кода, который их ожидает
        os.environ.setdefault("QDRANT_URL", self.QDRANT_URL)
        os.environ.setdefault("BM25_INDEX_DIR", self.BM25_INDEX_DIR)
        os.environ.setdefault("PAGES_DIR", self.PAGES_DIR)
        os.environ.setdefault("HF_MODEL", self.HF_MODEL)
        os.environ.setdefault("LLM_BASE_URL", self.LLM_BASE_URL)
        os.environ.setdefault("EASYOCR_DIR", self.EASYOCR_DIR)
        os.environ.setdefault("EASYOCR_ALLOW_DOWNLOADS", "1" if self.EASYOCR_ALLOW_DOWNLOADS else "0")
        if self.HF_DEVICE:
            os.environ.setdefault("HF_DEVICE", self.HF_DEVICE)

    def apply_env(self, force: bool = False) -> None:
        """
        Применяет значения из CONTROL к объекту и окружению.
        Если force=True — перекрывает даже уже существующие переменные из .env.
        """
        for k, v in CONTROL.items():
            # 1) выставить в os.environ
            if force or os.getenv(k) is None:
                os.environ[str(k)] = "" if v is None else str(v)

            # 2) обновить атрибуты объекта settings, если такой есть
            if hasattr(self, k):
                if k == "HF_DEVICE":
                    setattr(self, k, v if v else None)
                else:
                    setattr(self, k, v)

        # Спец: нормализуем QDRANT_URL после возможной подстановки
        self.QDRANT_URL = _normalize_qdrant_url(os.environ.get("QDRANT_URL", CONTROL["QDRANT_URL"]))
        os.environ["QDRANT_URL"] = self.QDRANT_URL

    def pretty_print(self) -> None:
        print("🔁 runtime_settings.py loaded")
        print(f"  LLM     = num_ctx={self.LLM_NUM_CTX}, max_tokens={self.LLM_MAX_TOKENS}, timeout={self.LLM_TIMEOUT}s")
        print(f"  LLM.ex  = min_ctx={self.LLM_MIN_CTX}, ctx_margin={self.LLM_CTX_MARGIN}, keep_alive={self.LLM_KEEP_ALIVE}, gpu_layers={self.LLM_NUM_GPU_LAYERS}")
        print(f"  APP_ENV = {self.APP_ENV}")
        print(f"  APP     = {self.APP_HOST}:{self.APP_PORT}")
        print(f"  QDRANT  = {self.QDRANT_URL} (collection={self.QDRANT_COLLECTION})")
        print(f"  BM25    = {self.BM25_INDEX_DIR}")
        print(f"  PAGES   = {self.PAGES_DIR}")
        print(f"  HF_MODEL= {self.HF_MODEL} (fp16={self.HF_FP16}, device={self.HF_DEVICE or 'auto'})")
        print(f"  RETR    = top_k={self.RETR_TOP_K}, per_doc_limit={self.RETR_PER_DOC_LIMIT}")
        print(f"  RERANK  = enabled={self.RERANKER_ENABLED}, top_k={self.RERANK_TOP_K}, model={self.RERANKER_MODEL or '-'}")
        print(f"  SNIP    = CTX_SNIPPET_LIMIT={self.CTX_SNIPPET_LIMIT}")
        print(f"  LLM     = num_ctx={self.LLM_NUM_CTX}, max_tokens={self.LLM_MAX_TOKENS}, timeout={self.LLM_TIMEOUT}s")


# Глобальный singleton
settings = Settings()
settings.apply_env(force=True)  # <-- ключевое: CONTROL принудительно перекрывает .env

try:
    if settings.APP_ENV == "dev":
        settings.pretty_print()
except Exception:
    pass
