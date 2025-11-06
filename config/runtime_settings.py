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
    # API / App
    "APP_ENV":        "dev",
    "APP_HOST":       "0.0.0.0",
    "APP_PORT":       8000,
    "LLM_BASE_URL":   "http://host.docker.internal:11434",

    # Retrieval
    "RETR_TOP_K":         4,     
    "RETR_PER_DOC_LIMIT": 2,

    # Paths & indexes
    "PAGES_DIR":       "data",
    "BM25_INDEX_DIR":  "index/bm25_idx",
    "QDRANT_COLLECTION": "med_kb_v3",
    "QDRANT_URL":      "http://qdrant:6333",

    # Embeddings
    "EMB_BACKEND": "hf",
    "HF_MODEL":    "BAAI/bge-m3",
    "HF_DEVICE":   None,   # 'cuda' | 'cpu' | None(авто)
    "HF_FP16":     True,

    # Reranker
    "RETR_PER_DOC_LIMIT":  1,
    "RERANKER_ENABLED": True,
    "RERANK_TOP_K":     50,
    "RERANKER_MODEL":   "",

    # EasyOCR
    "EASYOCR_DIR":            "/root/.EasyOCR",
    "EASYOCR_ALLOW_DOWNLOADS": True,

    # HF cache (чтобы не писать в '/.cache/...')
    "TRANSFORMERS_CACHE": "/root/.cache/huggingface",

    # Logs
    "LOG_LEVEL": "INFO",

    # LLM лимиты/таймауты (если используешь их из settings в api_app)
    "LLM_NUM_CTX": 4096,
    "LLM_MAX_TOKENS": 768,
    "LLM_TIMEOUT": 80,
        # сек
}

# Для обратной совместимости со старой функцией загрузки:
RUNTIME = CONTROL

# ================== КЛАСС НАСТРОЕК ==================

class Settings:
    # Инициализация читает сначала ENV (на случай prod), но потом мы принудительно
    # перекроем ENV значениями из CONTROL через apply_env(force=True).

    # --- Retrieval / Prompt size ---
    #RETR_TOP_K         = _to_int(os.getenv("RETR_TOP_K"), 8)
    RETR_PER_DOC_LIMIT = _to_int(os.getenv("RETR_PER_DOC_LIMIT"), 1)  # <= советую 1 при k>=6
    #RERANKER_ENABLED   = _str2bool(os.getenv("RERANKER_ENABLED", "true"), True)
    #RERANK_TOP_K       = _to_int(os.getenv("RERANK_TOP_K"), 50)
    #K_HARD_MAX         = _to_int(os.getenv("K_HARD_MAX"), 12)

    # главный параметр, который ты хотел двигать из runtime
    CTX_SNIPPET_LIMIT  = _to_int(os.getenv("CTX_SNIPPET_LIMIT"), 900)  # начни с 900

    # --- LLM ---
    #LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.1:8b")
    #LLM_BASE_URL    = os.getenv("LLM_BASE_URL", "http://host.docker.internal:11434")
    #LLM_NUM_CTX     = _to_int(os.getenv("LLM_NUM_CTX"), 4096)
    #LLM_NUM_PREDICT = _to_int(os.getenv("LLM_NUM_PREDICT"), 768)
    #LLM_TIMEOUT_S   = _to_int(os.getenv("LLM_TIMEOUT_S"), 80)


    APP_ENV: str = os.getenv("APP_ENV", CONTROL["APP_ENV"])
    APP_HOST: str = os.getenv("APP_HOST", CONTROL["APP_HOST"])
    APP_PORT: int = _to_int(os.getenv("APP_PORT"), CONTROL["APP_PORT"])
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", CONTROL["LLM_BASE_URL"])

    PAGES_DIR: str = os.getenv("PAGES_DIR", CONTROL["PAGES_DIR"])
    BM25_INDEX_DIR: str = os.getenv("BM25_INDEX_DIR", CONTROL["BM25_INDEX_DIR"])
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", CONTROL["QDRANT_COLLECTION"])
    QDRANT_URL: str = _normalize_qdrant_url(os.getenv("QDRANT_URL") or CONTROL["QDRANT_URL"])

    RETR_TOP_K: int = _to_int(os.getenv("RETR_TOP_K"), CONTROL["RETR_TOP_K"])
    RETR_PER_DOC_LIMIT: int = _to_int(os.getenv("RETR_PER_DOC_LIMIT"), CONTROL["RETR_PER_DOC_LIMIT"])

    EMB_BACKEND: str = os.getenv("EMB_BACKEND", CONTROL["EMB_BACKEND"])
    HF_MODEL: str = os.getenv("HF_MODEL", CONTROL["HF_MODEL"])
    HF_DEVICE: Optional[str] = os.getenv("HF_DEVICE", CONTROL["HF_DEVICE"] or "")
    HF_FP16: bool = _str2bool(os.getenv("HF_FP16", "1" if CONTROL["HF_FP16"] else "0"), CONTROL["HF_FP16"])

    RERANKER_ENABLED: bool = _str2bool(os.getenv("RERANKER_ENABLED"), CONTROL["RERANKER_ENABLED"])
    RERANK_TOP_K: int = _to_int(os.getenv("RERANK_TOP_K"), CONTROL["RERANK_TOP_K"])
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", CONTROL["RERANKER_MODEL"])

    EASYOCR_DIR: str = os.getenv("EASYOCR_DIR", CONTROL["EASYOCR_DIR"])
    EASYOCR_ALLOW_DOWNLOADS: bool = _str2bool(os.getenv("EASYOCR_ALLOW_DOWNLOADS"), CONTROL["EASYOCR_ALLOW_DOWNLOADS"])

    TRANSFORMERS_CACHE: str = os.getenv("TRANSFORMERS_CACHE", CONTROL["TRANSFORMERS_CACHE"])

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", CONTROL["LOG_LEVEL"])

    # LLM tuneables (если используешь их дальше в коде)
    LLM_NUM_CTX: int = _to_int(os.getenv("LLM_NUM_CTX"), CONTROL["LLM_NUM_CTX"])
    LLM_MAX_TOKENS: int = _to_int(os.getenv("LLM_MAX_TOKENS"), CONTROL["LLM_MAX_TOKENS"])
    LLM_TIMEOUT: int = _to_int(os.getenv("LLM_TIMEOUT"), CONTROL["LLM_TIMEOUT"])

    def __init__(self) -> None:
        # Подготовим каталоги
        _ensure_dir(Path(self.TRANSFORMERS_CACHE))
        os.environ.setdefault("TRANSFORMERS_CACHE", self.TRANSFORMERS_CACHE)
        _ensure_dir(Path(self.EASYOCR_DIR))
        _ensure_dir(Path(self.EASYOCR_DIR) / "model")

        # Пробросим базовые переменные
        os.environ.setdefault("QDRANT_URL", self.QDRANT_URL)
        os.environ.setdefault("BM25_INDEX_DIR", self.BM25_INDEX_DIR)
        os.environ.setdefault("PAGES_DIR", self.PAGES_DIR)
        os.environ.setdefault("HF_MODEL", self.HF_MODEL)

    def apply_env(self, force: bool = False) -> None:
        """
        Применяет значения из CONTROL к объекту и окружению.
        Если force=True — перекрывает даже уже существующие переменные из .env.
        """
        for k, v in CONTROL.items():
            # 1) выставить в os.environ
            if force or os.getenv(k) is None:
                os.environ[str(k)] = str(v)

            # 2) обновить атрибуты объекта settings, если такой есть
            if hasattr(self, k):
                setattr(self, k, v if k != "HF_DEVICE" or v else None)

        # Спец: нормализуем QDRANT_URL после возможной подстановки
        self.QDRANT_URL = _normalize_qdrant_url(os.environ.get("QDRANT_URL", CONTROL["QDRANT_URL"]))
        os.environ["QDRANT_URL"] = self.QDRANT_URL

    def pretty_print(self) -> None:
        print("🔁 runtime_settings.py loaded")
        print(f"  APP_ENV = {self.APP_ENV}")
        print(f"  QDRANT  = {self.QDRANT_URL}")
        print(f"  BM25    = {self.BM25_INDEX_DIR}")
        print(f"  PAGES   = {self.PAGES_DIR}")
        print(f"  HF_MODEL= {self.HF_MODEL} (fp16={self.HF_FP16}, device={self.HF_DEVICE or 'auto'})")
        print(f"  RETR    = top_k={self.RETR_TOP_K}, per_doc_limit={self.RETR_PER_DOC_LIMIT}")
        print(f"  RERANK  = enabled={self.RERANKER_ENABLED}, top_k={self.RERANK_TOP_K}")
        print(f"  LLM     = num_ctx={self.LLM_NUM_CTX}, max_tokens={self.LLM_MAX_TOKENS}, timeout={self.LLM_TIMEOUT}s")
        print(f"  RETR   = top_k={self.RETR_TOP_K}, per_doc={self.RETR_PER_DOC_LIMIT}")
        print(f"  SNIP   = CTX_SNIPPET_LIMIT={self.CTX_SNIPPET_LIMIT}")
        print(f"  LLM    = num_ctx={self.LLM_NUM_CTX}, max_tokens={self.LLM_NUM_PREDICT}, timeout={self.LLM_TIMEOUT_S}s")


# Глобальный singleton
settings = Settings()

try:
    print("🔁 runtime_settings.py loaded")
except Exception:
    pass
