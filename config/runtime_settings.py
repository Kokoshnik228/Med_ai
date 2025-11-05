# runtime_settings.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Optional


def _str2bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _to_int(v: Optional[str], default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _to_float(v: Optional[str], default: float) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # не падаем из-за прав — просто продолжаем
        pass


def _normalize_qdrant_url(url_in: Optional[str]) -> str:
    """
    Делает URL пригодным для клиента:
      - пусто → берём из env (QDRANT_URL|QDRANT) или http://qdrant:6333
      - 'qdrant:6333' → 'http://qdrant:6333'
      - 'qdrant://qdrant:6333' → 'http://qdrant:6333'
    Разрешённые схемы у клиента: http/https/grpc/grpcs.
    """
    url = (url_in or os.getenv("QDRANT_URL") or os.getenv("QDRANT") or "http://qdrant:6333").strip()

    if "://" not in url:
        return f"http://{url}"
    if url.lower().startswith("qdrant://"):
        return "http://" + url[len("qdrant://"):]
    if url.lower().startswith("qdrant:"):
        return "http://" + url[len("qdrant:"):]
    return url


class Settings:
    # ---------------- App / API ----------------
    APP_ENV: str = os.getenv("APP_ENV", "dev")
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = _to_int(os.getenv("APP_PORT"), 8000)
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://host.docker.internal:11434")

    # ---------------- Данные и индексы ----------------
    PAGES_DIR: str = os.getenv("PAGES_DIR", "data")                      # где лежат *.pages.jsonl
    BM25_INDEX_DIR: str = os.getenv("BM25_INDEX_DIR", "index/bm25_idx")  # pyserini индекс
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "med_kb_v3")
    QDRANT_URL: str = _normalize_qdrant_url(os.getenv("QDRANT_URL"))

    # ---------------- Ретривер (RAG) ----------------
    # Удобно править прямо здесь: дефолты сделаны "лёгкими", чтобы не ловить таймауты LLM.
    RETR_TOP_K: int = _to_int(os.getenv("RETR_TOP_K"), 4)             # сколько фрагментов вернуть в LLM
    RETR_PER_DOC_LIMIT: int = _to_int(os.getenv("RETR_PER_DOC_LIMIT"), 1)  # не брать слишком много из одного дока
    TEXT_SNIPPET_LIMIT: int = _to_int(os.getenv("TEXT_SNIPPET_LIMIT"), 1200)  # длина одного фрагмента в символах
    USER_TEXT_MAX_CHARS: int = _to_int(os.getenv("USER_TEXT_MAX_CHARS"), 800) # сколько из текста пользователя класть в запрос к ретриверу

    # ---------------- Эмбеддер ----------------
    EMB_BACKEND: str = os.getenv("EMB_BACKEND", "hf")                    # 'hf' | 'none'
    HF_MODEL: str = os.getenv("HF_MODEL", "BAAI/bge-m3")
    HF_DEVICE: Optional[str] = os.getenv("HF_DEVICE")                    # 'cuda' | 'cpu' | None(авто)
    HF_FP16: bool = _str2bool(os.getenv("HF_FP16", "true"), True)

    # Кэш HuggingFace (не кладём в недоступный '/.cache/...'):
    TRANSFORMERS_CACHE: str = os.getenv("TRANSFORMERS_CACHE", "/root/.cache/huggingface")

    # ---------------- Переранкер (опционально) ----------------
    RERANKER_ENABLED: bool = _str2bool(os.getenv("RERANKER_ENABLED", "false"), False)
    RERANK_TOP_K: int = _to_int(os.getenv("RERANK_TOP_K"), 50)
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "")  # зарезервировано, можно не задавать

    # ---------------- LLM (Ollama / HTTP) ----------------
    # Всё, что влияет на «тяжесть» запроса к модели:
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b-instruct")    # можно поменять на 'llama3.1:8b'
    LLM_NUM_CTX: int = _to_int(os.getenv("LLM_NUM_CTX"), 1536)         # общий контекст токенов
    LLM_MAX_TOKENS: int = _to_int(os.getenv("LLM_MAX_TOKENS"), 512)    # длина генерируемого ответа
    LLM_TEMPERATURE: float = _to_float(os.getenv("LLM_TEMPERATURE"), 0.2)
    LLM_TOP_P: float = _to_float(os.getenv("LLM_TOP_P"), 0.9)
    LLM_TIMEOUT: float = _to_float(os.getenv("LLM_TIMEOUT"), 60.0)     # секунды; было 180

    # ---------------- EasyOCR ----------------
    EASYOCR_DIR: str = os.getenv("EASYOCR_DIR", "/root/.EasyOCR")
    EASYOCR_ALLOW_DOWNLOADS: bool = _str2bool(os.getenv("EASYOCR_ALLOW_DOWNLOADS", "1"), True)

    # ---------------- Разное ----------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def __init__(self) -> None:
        # Создадим каталоги для кэшей, чтобы не ловить PermissionError
        _ensure_dir(Path(self.TRANSFORMERS_CACHE))
        os.environ.setdefault("TRANSFORMERS_CACHE", self.TRANSFORMERS_CACHE)
        # Для новых версий transformers лучше также выставлять HF_HOME
        os.environ.setdefault("HF_HOME", self.TRANSFORMERS_CACHE)
        # Уберём ворнинг по токенизаторам
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # EasyOCR: подготовим каталог и подпапку 'model'
        easy_p = Path(self.EASYOCR_DIR)
        _ensure_dir(easy_p)
        _ensure_dir(easy_p / "model")

        # Пробросим несколько переменных обратно в окружение — удобно для библиотек/утилит
        os.environ.setdefault("QDRANT_URL", self.QDRANT_URL)
        os.environ.setdefault("BM25_INDEX_DIR", self.BM25_INDEX_DIR)
        os.environ.setdefault("PAGES_DIR", self.PAGES_DIR)
        os.environ.setdefault("HF_MODEL", self.HF_MODEL)

        # Параметры ретривера, чтобы их мог прочитать код, который смотрит на env:
        os.environ.setdefault("RETR_TOP_K", str(self.RETR_TOP_K))
        os.environ.setdefault("RETR_PER_DOC_LIMIT", str(self.RETR_PER_DOC_LIMIT))
        os.environ.setdefault("TEXT_SNIPPET_LIMIT", str(self.TEXT_SNIPPET_LIMIT))
        os.environ.setdefault("USER_TEXT_MAX_CHARS", str(self.USER_TEXT_MAX_CHARS))

        # Параметры LLM — на случай, если HTTP-клиент/обёртка читает из env:
        os.environ.setdefault("LLM_MODEL", self.LLM_MODEL)
        os.environ.setdefault("LLM_NUM_CTX", str(self.LLM_NUM_CTX))
        os.environ.setdefault("LLM_MAX_TOKENS", str(self.LLM_MAX_TOKENS))
        os.environ.setdefault("LLM_TEMPERATURE", str(self.LLM_TEMPERATURE))
        os.environ.setdefault("LLM_TOP_P", str(self.LLM_TOP_P))
        os.environ.setdefault("LLM_TIMEOUT", str(self.LLM_TIMEOUT))
        os.environ.setdefault("LLM_BASE_URL", self.LLM_BASE_URL)

        # Также дублируем в кастомные имена (если где-то в проекте они ожидаются):
        os.environ.setdefault("MEDAI_TEXT_SNIPPET_LIMIT", str(self.TEXT_SNIPPET_LIMIT))
        os.environ.setdefault("MEDAI_USER_TEXT_MAX_CHARS", str(self.USER_TEXT_MAX_CHARS))

    def pretty_print(self) -> None:
        print("🔁 runtime_settings.py loaded")
        print(f"  APP_ENV = {self.APP_ENV}")
        print(f"  QDRANT  = {self.QDRANT_URL} (collection={self.QDRANT_COLLECTION})")
        print(f"  BM25    = {self.BM25_INDEX_DIR}")
        print(f"  PAGES   = {self.PAGES_DIR}")
        print(f"  RAG     = top_k={self.RETR_TOP_K}, per_doc_limit={self.RETR_PER_DOC_LIMIT}, "
              f"snippet_limit={self.TEXT_SNIPPET_LIMIT}, user_text_max={self.USER_TEXT_MAX_CHARS}")
        print(f"  HF_MODEL= {self.HF_MODEL} (fp16={self.HF_FP16}, device={self.HF_DEVICE or 'auto'})")
        print(f"  LLM     = model={self.LLM_MODEL}, num_ctx={self.LLM_NUM_CTX}, "
              f"max_tokens={self.LLM_MAX_TOKENS}, temp={self.LLM_TEMPERATURE}, "
              f"top_p={self.LLM_TOP_P}, timeout={self.LLM_TIMEOUT}s")


# Глобальный singleton для импорта: from config.runtime_settings import settings
settings = Settings()

# при импорте сразу коротко отметимся в логах
try:
    print("🔁 runtime_settings.py loaded")
except Exception:
    pass
