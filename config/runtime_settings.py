# -*- coding: utf-8 -*-
"""
Единая точка настройки приложения (без правок .env/.compose).
Файл управляёт:
  - путями к данным/индексам/кэшу
  - подключением к Qdrant
  - параметрами retriever'a (k, per_doc_limit, reranker)
  - параметрами LLM (модель, контекст, длина ответа, таймаут и т.д.)

Ключевая идея: правим числа/строки ТУТ, и это перекрывает .env.
"""

import os
from pathlib import Path
from typing import Optional


# ---------------- Вспомогательное ----------------

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
    """
    Делает URL пригодным для клиента qdrant:
      - пусто → env(QDRANT_URL|QDRANT) → http://qdrant:6333
      - 'qdrant:6333'        → 'http://qdrant:6333'
      - 'qdrant://host:6333' → 'http://host:6333'
    """
    url = (url_in or os.getenv("QDRANT_URL") or os.getenv("QDRANT") or "http://qdrant:6333").strip()
    if "://" not in url:
        return f"http://{url}"
    if url.lower().startswith("qdrant://"):
        return "http://" + url[len("qdrant://"):]
    if url.lower().startswith("qdrant:"):
        return "http://" + url[len("qdrant:"):]
    return url


# ---------------- Настройки ----------------

class Settings:
    # Кто «главнее»: значения из файла или .env?
    # True  → правки в этом файле перекрывают .env
    # False → .env может переопределить то, что здесь
    PRIORITY_FILE: bool = True

    # --------- App / API ----------
    APP_ENV: str  = os.getenv("APP_ENV", "dev")
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = _to_int(os.getenv("APP_PORT"), 8000)

    # --------- Данные/индексы ----------
    PAGES_DIR: str       = os.getenv("PAGES_DIR", "data")
    BM25_INDEX_DIR: str  = os.getenv("BM25_INDEX_DIR", "index/bm25_idx")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "med_kb_v3")
    QDRANT_URL: str        = _normalize_qdrant_url(os.getenv("QDRANT_URL"))

    # --------- Эмбеддер / HF ----------
    EMB_BACKEND: str    = os.getenv("EMB_BACKEND", "hf")  # 'hf' | 'none'
    HF_MODEL: str       = os.getenv("HF_MODEL", "BAAI/bge-m3")
    HF_DEVICE: Optional[str] = os.getenv("HF_DEVICE")      # 'cuda' | 'cpu' | None(авто)
    HF_FP16: bool       = _str2bool(os.getenv("HF_FP16", "true"), True)

    # HF cache — уводим из недоступного '/.cache/...'
    TRANSFORMERS_CACHE: str = os.getenv("TRANSFORMERS_CACHE", "/root/.cache/huggingface")

    # --------- Retriever ----------
    RETR_TOP_K: int         = _to_int(os.getenv("RETR_TOP_K"), 4)   # было 8 — сделаем быстрее
    RETR_PER_DOC_LIMIT: int = _to_int(os.getenv("RETR_PER_DOC_LIMIT"), 2)

    # Переранкер (косинус на том же эмбеддере)
    RERANKER_ENABLED: bool = _str2bool(os.getenv("RERANKER_ENABLED", "false"), False)
    RERANK_TOP_K: int      = _to_int(os.getenv("RERANK_TOP_K"), 50)
    RERANKER_MODEL: str    = os.getenv("RERANKER_MODEL", "")

    # --------- LLM (Ollama/совместимые HTTP) ----------
    # Всё управление «нейронкой» тут:
    LLM_BASE_URL: str   = os.getenv("LLM_BASE_URL", "http://host.docker.internal:11434")
    LLM_MODEL: str      = os.getenv("LLM_MODEL", "llama3.1:8b")
    LLM_NUM_CTX: int    = _to_int(os.getenv("LLM_NUM_CTX"), 2048)   # было 3072 → быстрее
    LLM_MAX_TOKENS: int = _to_int(os.getenv("LLM_MAX_TOKENS"), 800) # было 2048 → быстрее
    LLM_TIMEOUT: int    = _to_int(os.getenv("LLM_TIMEOUT"), 60)     # было 180 → чтобы не ждать вечно
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_STREAM: bool    = _str2bool(os.getenv("LLM_STREAM", "1"), True)

    # --------- EasyOCR ----------
    EASYOCR_DIR: str = os.getenv("EASYOCR_DIR", "/root/.EasyOCR")
    EASYOCR_ALLOW_DOWNLOADS: bool = _str2bool(os.getenv("EASYOCR_ALLOW_DOWNLOADS", "1"), True)

    # --------- Логи ----------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ====== Жизненный цикл ======
    def __init__(self) -> None:
        # Готовим кэши
        _ensure_dir(Path(self.TRANSFORMERS_CACHE))
        # важные переменные окружения (ставим по-любому)
        os.environ["TRANSFORMERS_CACHE"] = self.TRANSFORMERS_CACHE

        # EasyOCR каталоги
        easy_p = Path(self.EASYOCR_DIR)
        _ensure_dir(easy_p)
        _ensure_dir(easy_p / "model")

        # Базовые env для других модулей
        os.environ.setdefault("QDRANT_URL", self.QDRANT_URL)
        os.environ.setdefault("BM25_INDEX_DIR", self.BM25_INDEX_DIR)
        os.environ.setdefault("PAGES_DIR", self.PAGES_DIR)
        os.environ.setdefault("HF_MODEL", self.HF_MODEL)

        # Если файл — главный источник, сразу «вталкиваем» ключевые LLM/env
        if self.PRIORITY_FILE:
            self.apply_env(force=True)

    # --- Синхронизация с окружением (.env) ---
    def reload_from_env(self) -> None:
        """Обновить атрибуты settings из текущего os.environ (если вдруг нужно)."""
        self.PAGES_DIR = os.getenv("PAGES_DIR", self.PAGES_DIR)
        self.BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", self.BM25_INDEX_DIR)
        self.QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", self.QDRANT_COLLECTION)
        self.QDRANT_URL = _normalize_qdrant_url(os.getenv("QDRANT_URL", self.QDRANT_URL))

        self.EMB_BACKEND = os.getenv("EMB_BACKEND", self.EMB_BACKEND)
        self.HF_MODEL = os.getenv("HF_MODEL", self.HF_MODEL)
        self.HF_DEVICE = os.getenv("HF_DEVICE", self.HF_DEVICE)
        self.HF_FP16 = _str2bool(os.getenv("HF_FP16"), self.HF_FP16)

        self.RETR_TOP_K = _to_int(os.getenv("RETR_TOP_K"), self.RETR_TOP_K)
        self.RETR_PER_DOC_LIMIT = _to_int(os.getenv("RETR_PER_DOC_LIMIT"), self.RETR_PER_DOC_LIMIT)
        self.RERANKER_ENABLED = _str2bool(os.getenv("RERANKER_ENABLED"), self.RERANKER_ENABLED)
        self.RERANK_TOP_K = _to_int(os.getenv("RERANK_TOP_K"), self.RERANK_TOP_K)
        self.RERANKER_MODEL = os.getenv("RERANKER_MODEL", self.RERANKER_MODEL)

        self.LLM_BASE_URL = os.getenv("LLM_BASE_URL", self.LLM_BASE_URL)
        self.LLM_MODEL = os.getenv("LLM_MODEL", self.LLM_MODEL)
        self.LLM_NUM_CTX = _to_int(os.getenv("LLM_NUM_CTX"), self.LLM_NUM_CTX)
        self.LLM_MAX_TOKENS = _to_int(os.getenv("LLM_MAX_TOKENS"), self.LLM_MAX_TOKENS)
        self.LLM_TIMEOUT = _to_int(os.getenv("LLM_TIMEOUT"), self.LLM_TIMEOUT)
        try:
            self.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", self.LLM_TEMPERATURE))
        except Exception:
            pass
        self.LLM_STREAM = _str2bool(os.getenv("LLM_STREAM"), self.LLM_STREAM)

    def apply_env(self, force: bool = False) -> None:
        """
        Прописать важные значения назад в os.environ, чтобы их увидели клиенты,
        которые читают ТОЛЬКО env.
        force=True — перезапишет существующие значения в окружении.
        """
        def _set(k: str, v: str) -> None:
            if force or (k not in os.environ):
                os.environ[k] = v

        _set("QDRANT_URL", self.QDRANT_URL)
        _set("BM25_INDEX_DIR", self.BM25_INDEX_DIR)
        _set("PAGES_DIR", self.PAGES_DIR)
        _set("HF_MODEL", self.HF_MODEL)

        _set("LLM_BASE_URL", self.LLM_BASE_URL)
        _set("LLM_MODEL", self.LLM_MODEL)
        _set("LLM_NUM_CTX", str(self.LLM_NUM_CTX))
        _set("LLM_MAX_TOKENS", str(self.LLM_MAX_TOKENS))
        _set("LLM_TIMEOUT", str(self.LLM_TIMEOUT))
        _set("LLM_TEMPERATURE", str(self.LLM_TEMPERATURE))
        _set("LLM_STREAM", "1" if self.LLM_STREAM else "0")

        _set("TRANSFORMERS_CACHE", self.TRANSFORMERS_CACHE)
        _set("EASYOCR_DIR", self.EASYOCR_DIR)

    def pretty_print(self) -> None:
        print("🔁 runtime_settings.py loaded")
        print(f"  APP_ENV = {self.APP_ENV}")
        print(f"  QDRANT  = {self.QDRANT_URL}")
        print(f"  BM25    = {self.BM25_INDEX_DIR}")
        print(f"  PAGES   = {self.PAGES_DIR}")
        print(f"  HF_MODEL= {self.HF_MODEL} (fp16={self.HF_FP16}, device={self.HF_DEVICE or 'auto'})")
        print(f"  RETR    = top_k={self.RETR_TOP_K}, per_doc={self.RETR_PER_DOC_LIMIT}, rerank={self.RERANKER_ENABLED}/{self.RERANK_TOP_K}")
        print(f"  LLM     = url={self.LLM_BASE_URL}, model={self.LLM_MODEL}, ctx={self.LLM_NUM_CTX}, max_out={self.LLM_MAX_TOKENS}, timeout={self.LLM_TIMEOUT}s, T={self.LLM_TEMPERATURE}")


# Экземпляр
settings = Settings()

try:
    # отметка в логах при импорте
    print("🔁 runtime_settings.py loaded")
except Exception:
    pass
