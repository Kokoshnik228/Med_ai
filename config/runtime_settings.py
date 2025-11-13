# config/runtime_settings.py
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

# ---------------- helpers ----------------

def _str2bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

def _to_int(v: Optional[str | int | float], default: int) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        try:
            return int(v)  # если уже число
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
# Меняешь здесь — эти значения перекроют .env / docker env
CONTROL: Dict[str, Any] = {
    "PROMPT_SYSTEM": (
        "Ты — ассистент поиска по медицинской базе. Отвечай ТОЛЬКО по тексту из КОНТЕКСТА (фрагменты базы). "
        "Никаких собственных знаний и предположений. Если точного ответа в КОНТЕКСТЕ нет — прямо напиши: "
        "\"Нет ответа в базе по предоставленному контексту.\" "
        "Отвечай кратко, на русском, без Markdown и без код-блоков."
    ),
    "PROMPT_USER_TPL": (
        "ВОПРОС/ЗАПРОС:\n"
        "{case_text}\n\n"
        "КОНТЕКСТ (фрагменты из базы):\n"
        "{ctx}\n\n"
        "ЗАДАЧА:\n"
        "- Найди в КОНТЕКСТЕ точный ответ (термин, число, краткую фразу) и выдай его кратко.\n"
        "- Если в КОНТЕКСТЕ нет однозначного ответа — напиши: \"Нет ответа в базе по предоставленному контексту.\"\n"
        "- В конце строки перечисли источники в формате: Источники: DOC_ID стр.A-B; DOC_ID стр.C-D\n"
        "- Краткость важнее. Без Markdown и код-блоков."
    ),


    # --- LLM выбор модели ---
    "LLM_ACTIVE": "deepseek-r1:32b",  # активная по умолчанию
    "LLM_ALLOWED": ["llama3.1:8b", "llama3.1:70b", "deepseek-r1:32b"],
    "LLM_LABELS": {
        "llama3.1:8b": "Llama 3.1 (8B)",
        "llama3.1:70b": "Llama 3.1 (70B)",
        "deepseek-r1:32b": "DeepSeek R1 (32B) — Reasoning",
    },
    # Пер-модельные пресеты (подходят для Ollama)
    "LLM_PRESETS": {
        "llama3.1:8b": {
            "num_ctx": 4096, "max_tokens": 600, "timeout_s": 150,
            "temperature": 0.2, "top_p": 0.95, "repeat_penalty": 1.05,
            "gpu_layers": -1, "keep_alive": "30m"
        },
        "llama3.1:70b": {
            "num_ctx": 8192, "max_tokens": 600, "timeout_s": 180,
            "temperature": 0.2, "top_p": 0.95, "repeat_penalty": 1.05,
            "gpu_layers": -1, "keep_alive": "30m"
        },
        "deepseek-r1:32b": {
            "num_ctx": 12288, "max_tokens": 1800, "timeout_s": 180,
            "temperature": 0.3, "top_p": 0.90, "repeat_penalty": 1.05,
            "gpu_layers": -1, "keep_alive": "60m"
        },
    },

    # Free-mode
    "MED_GUARD_MODE": "soft",
    "FREECHAT_ENABLED": True,
    "FREECHAT_MODEL": "deepseek-r1:32b",
    "FREECHAT_NUM_CTX": 8192,
    "FREECHAT_MAX_TOKENS": 800,
    "FREECHAT_TEMPERATURE": 0.9,
    "FREECHAT_TOP_P": 0.95,
    "FREECHAT_REPEAT_PENALTY": 1.0,
    "FREECHAT_NUM_GPU_LAYERS": -1,
    "FREECHAT_KEEP_ALIVE": "60m",
    "FREE_PROMPT_SYSTEM": "Ты дружелюбный помощник. Отвечай кратко, только на русском, без Markdown и код-блоков.",
    "FREE_PROMPT_USER_TPL": "{case_text}",

    # Fast-retry управления
    "FAST_RETRY_ENABLED": True,
    "FAST_RETRY_ON_EMPTY": False,
    "FAST_RETRY_CTX_SHRINK_RATIO": 0.65,
    "FAST_RETRY_MAX_TOKENS": 300,

    # режим свободного чата (триггеры)
    "FREE_CHAT_ENABLED": True,
    "FREE_CHAT_MAX_LEN": 64,
    "FREE_CHAT_TRIGGERS": ["кто ты", "кто-ты", "скажи кто ты", "ты кто", "кто такой"],

    # Глобальные LLM-параметры (дефолты, если где-то понадобятся)
    "LLM_KEEP_ALIVE": "30m",
    "LLM_NUM_CTX": 8192,
    "LLM_MAX_TOKENS": 600,
    "LLM_TIMEOUT": 150,
    "LLM_CTX_MARGIN": 256,
    "LLM_MIN_CTX": 2048,
    "LLM_NUM_GPU_LAYERS": -1,

    # Чанкинг (dense и BM25)
    "CHILD_W": 180,
    "CHILD_OVERLAP": 30,
    "PARENT_W": 500,

    # BM25-специфика
    "BM25_CHILD_W": 180,
    "BM25_CHILD_OVERLAP": 30,
    "BM25_LANGUAGE": "ru",

    # API / App
    "APP_ENV":  "dev",
    "APP_HOST": "0.0.0.0",
    "APP_PORT": 8000,
    "LLM_BASE_URL": "http://ollama:11434",

    # Retrieval
    "RETR_TOP_K": 8,
    "RETR_PER_DOC_LIMIT": 1,
    "CTX_SNIPPET_LIMIT": 4000,

    # Paths & indexes
    "PAGES_DIR": "data",
    "BM25_INDEX_DIR": "index/bm25_idx",
    "QDRANT_COLLECTION": "med_kb_v3",
    "QDRANT_URL": "http://qdrant:6333",

    # Embeddings
    "EMB_BACKEND": "hf",
    "HF_MODEL": "BAAI/bge-m3",
    "HF_DEVICE": 'cuda',
    "HF_FP16": True,
    "EMB_BATCH": 128,

    # Reranker
    "RERANKER_ENABLED": True,
    "RERANK_TOP_K": 50,
    "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",

    # EasyOCR
    "EASYOCR_DIR": "/root/.EasyOCR",
    "EASYOCR_ALLOW_DOWNLOADS": True,

    # HF cache
    "TRANSFORMERS_CACHE": "/root/.cache/huggingface",

    # Logs
    "LOG_LEVEL": "INFO",
}

# Для обратной совместимости:
RUNTIME = CONTROL


# ================== Утилиты реестра моделей (единый источник правды) ==================

def _env_json_or_default(name: str, default: Any) -> Any:
    """Прочитать JSON из ENV, иначе вернуть default."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default

def llm_get_allowed() -> List[str]:
    return list(_env_json_or_default("LLM_ALLOWED", CONTROL["LLM_ALLOWED"]))

def llm_get_labels() -> Dict[str, str]:
    return dict(_env_json_or_default("LLM_LABELS", CONTROL.get("LLM_LABELS", {})))

def llm_get_preset(model_id: str) -> Dict[str, Any]:
    presets = _env_json_or_default("LLM_PRESETS", CONTROL["LLM_PRESETS"])
    return dict(presets.get(model_id, {}))

def llm_get_active() -> str:
    env_active = os.getenv("LLM_ACTIVE", CONTROL["LLM_ACTIVE"])
    allowed = llm_get_allowed()
    return env_active if env_active in allowed else (allowed[0] if allowed else "")

def llm_resolve(model_in: Optional[str]) -> str:
    m = (model_in or "").strip()
    allowed = llm_get_allowed()
    if m and m in allowed:
        return m
    return llm_get_active()


# ================== КЛАСС НАСТРОЕК ==================

class Settings:
    # Значения читаются сначала из ENV, а затем мы принудительно перекроем ENV значениями CONTROL через apply_env(force=True).

    # --- LLM (сводные дефолты) ---
    LLM_ACTIVE: str = os.getenv("LLM_ACTIVE", CONTROL["LLM_ACTIVE"])
    LLM_ALLOWED: list = _env_json_or_default("LLM_ALLOWED", CONTROL["LLM_ALLOWED"])
    LLM_LABELS: dict = _env_json_or_default("LLM_LABELS", CONTROL.get("LLM_LABELS", {}))
    LLM_PRESETS: dict = _env_json_or_default("LLM_PRESETS", CONTROL["LLM_PRESETS"])
    PROMPT_SYSTEM: str = os.getenv("PROMPT_SYSTEM", CONTROL["PROMPT_SYSTEM"])
    PROMPT_USER_TPL: str = os.getenv("PROMPT_USER_TPL", CONTROL["PROMPT_USER_TPL"])

    LLM_KEEP_ALIVE: str = os.getenv("LLM_KEEP_ALIVE", CONTROL["LLM_KEEP_ALIVE"])
    LLM_NUM_CTX: int = _to_int(os.getenv("LLM_NUM_CTX"), CONTROL["LLM_NUM_CTX"])
    LLM_MAX_TOKENS: int = _to_int(os.getenv("LLM_MAX_TOKENS"), CONTROL["LLM_MAX_TOKENS"])
    LLM_TIMEOUT: int = _to_int(os.getenv("LLM_TIMEOUT"), CONTROL["LLM_TIMEOUT"])
    LLM_CTX_MARGIN: int = _to_int(os.getenv("LLM_CTX_MARGIN"), CONTROL["LLM_CTX_MARGIN"])
    LLM_MIN_CTX: int = _to_int(os.getenv("LLM_MIN_CTX"), CONTROL["LLM_MIN_CTX"])
    LLM_NUM_GPU_LAYERS: int = _to_int(os.getenv("LLM_NUM_GPU_LAYERS"), CONTROL["LLM_NUM_GPU_LAYERS"])

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
    HF_DEVICE: Optional[str] = (os.getenv("HF_DEVICE", "") or (CONTROL["HF_DEVICE"] or "")) or None
    HF_FP16: bool = _str2bool(os.getenv("HF_FP16"), CONTROL["HF_FP16"])

    # --- Reranker ---
    RERANKER_ENABLED: bool = _str2bool(os.getenv("RERANKER_ENABLED"), CONTROL["RERANKER_ENABLED"])
    RERANK_TOP_K: int = _to_int(os.getenv("RERANK_TOP_K"), CONTROL["RERANK_TOP_K"])
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", CONTROL["RERANKER_MODEL"])

    # --- OCR / Caches ---
    EASYOCR_DIR: str = os.getenv("EASYOCR_DIR", CONTROL["EASYOCR_DIR"])
    EASYOCR_ALLOW_DOWNLOADS: bool = _str2bool(os.getenv("EASYOCR_ALLOW_DOWNLOADS"), CONTROL["EASYOCR_ALLOW_DOWNLOADS"])
    TRANSFORMERS_CACHE: str = os.getenv("TRANSFORMERS_CACHE", CONTROL["TRANSFORMERS_CACHE"])

    # --- Logs ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", CONTROL["LOG_LEVEL"])

    def __init__(self) -> None:
        # Подготовим каталоги
        _ensure_dir(Path(self.TRANSFORMERS_CACHE))
        os.environ.setdefault("TRANSFORMERS_CACHE", self.TRANSFORMERS_CACHE)

        _ensure_dir(Path(self.EASYOCR_DIR))
        _ensure_dir(Path(self.EASYOCR_DIR) / "model")

        # Базовые переменные окружения для кода, который их ожидает
        os.environ.setdefault("QDRANT_URL", self.QDRANT_URL)
        os.environ.setdefault("PROMPT_SYSTEM", self.PROMPT_SYSTEM)
        os.environ.setdefault("PROMPT_USER_TPL", self.PROMPT_USER_TPL)
        os.environ.setdefault("BM25_INDEX_DIR", self.BM25_INDEX_DIR)
        os.environ.setdefault("PAGES_DIR", self.PAGES_DIR)
        os.environ.setdefault("HF_MODEL", self.HF_MODEL)
        os.environ.setdefault("LLM_BASE_URL", self.LLM_BASE_URL)
        os.environ.setdefault("EASYOCR_DIR", self.EASYOCR_DIR)
        os.environ.setdefault("EASYOCR_ALLOW_DOWNLOADS", "1" if self.EASYOCR_ALLOW_DOWNLOADS else "0")
        if self.HF_DEVICE:
            os.environ.setdefault("HF_DEVICE", self.HF_DEVICE)

        # Проброс некоторых LLM-настроек
        os.environ.setdefault("LLM_KEEP_ALIVE", self.LLM_KEEP_ALIVE)
        os.environ.setdefault("LLM_NUM_GPU_LAYERS", str(self.LLM_NUM_GPU_LAYERS))
        os.environ.setdefault("LLM_CTX_MARGIN", str(self.LLM_CTX_MARGIN))
        os.environ.setdefault("LLM_MIN_CTX", str(self.LLM_MIN_CTX))

    def apply_env(self, force: bool = False) -> None:
        """
        Применяет значения из CONTROL к объекту и окружению.
        Если force=True — перекрывает даже уже существующие переменные из .env.
        Списки/словари кладём в ENV как JSON.
        """
        for k, v in CONTROL.items():
            # 1) выставить в os.environ
            if isinstance(v, (dict, list)):
                enc = json.dumps(v, ensure_ascii=False)
                if force or os.getenv(k) is None:
                    os.environ[k] = enc
            else:
                if force or os.getenv(k) is None:
                    os.environ[k] = "" if v is None else str(v)

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
        print(f"  APP     = {self.APP_ENV} @ {self.APP_HOST}:{self.APP_PORT}")
        print(f"  LLM.active = {llm_get_active()}")
        print(f"  LLM.allowed= {', '.join(llm_get_allowed())}")
        print(f"  LLM.globals= num_ctx={self.LLM_NUM_CTX}, max_tokens={self.LLM_MAX_TOKENS}, timeout={self.LLM_TIMEOUT}s")
        print(f"  LLM.extra  = min_ctx={self.LLM_MIN_CTX}, ctx_margin={self.LLM_CTX_MARGIN}, keep_alive={self.LLM_KEEP_ALIVE}, gpu_layers={self.LLM_NUM_GPU_LAYERS}")
        print(f"  QDRANT  = {self.QDRANT_URL} (collection={self.QDRANT_COLLECTION})")
        print(f"  BM25    = {self.BM25_INDEX_DIR} (lang={CONTROL['BM25_LANGUAGE']})")
        print(f"  PAGES   = {self.PAGES_DIR}")
        print(f"  HF_EMB  = {self.HF_MODEL} (fp16={self.HF_FP16}, device={self.HF_DEVICE or 'auto'})")
        print(f"  RETR    = top_k={self.RETR_TOP_K}, per_doc_limit={self.RETR_PER_DOC_LIMIT}, snippet={self.CTX_SNIPPET_LIMIT}")
        print(f"  RERANK  = enabled={self.RERANKER_ENABLED}, top_k={self.RERANK_TOP_K}, model={self.RERANKER_MODEL or '-'}")


# Глобальный singleton
settings = Settings()
settings.apply_env(force=True)  # <-- ключевое: CONTROL принудительно перекрывает .env

try:
    if settings.APP_ENV == "dev":
        settings.pretty_print()
except Exception:
    pass
