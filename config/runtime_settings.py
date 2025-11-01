"""
config/runtime_settings.py

Меняй параметры здесь. Применение без пересборки:
1) Сохрани файл
2) POST /config/reload (или curl -X POST http://localhost:7050/config/reload)
"""

RUNTIME = {
    # ── LLM / Ollama ────────────────────────────────────────────────
    "ollama": {
        # если пусто, возьмётся из .env (LLM_BASE_URL)
        # "base_url": "http://host.docker.internal:11434",
        "model": "llama3.1:8b",
        "max_tokens": 300,     # сколько токенов генерировать (длина ответа)
        "timeout_s": 60,       # таймаут HTTP запроса к LLM (секунды)
        "temperature": 0.1,
        "top_p": 0.9,
        "num_ctx": 6144,        # контекст (лимит токенов входа)
    },

    # ── Ретраивл ────────────────────────────────────────────────────
    "retrieval": {
        "k": 8,                 # сколько кусочков контекста забирать
    },

    # ── Чанкинг документов (индексация) ─────────────────────────────
    "chunking": {
        "child_w": 200,
        "child_overlap": 35,
        "parent_w": 800,
    },

    # ── Qdrant (можно не трогать, если .env уже ок) ────────────────
    # "qdrant": {
    #     "url": "http://qdrant:6333",
    #     "collection": "med_kb_v3",
    # },

    # ── Эмбеддинги / Реранкер (при желании можно править) ──────────
    # "embedding": {
    #     "backend": "hf",
    #     "model": "BAAI/bge-m3",
    #     "device": "cpu",
    # },
    # "reranker": {
    #     "model": "BAAI/bge-reranker-v2-m3",
    # },
}
