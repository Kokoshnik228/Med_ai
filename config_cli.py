#!/usr/bin/env python3
"""
Единый менеджер настроек проекта (CLI + интерактивный мастер).

Возможности:
- --init            → создаёт config/default.yaml и config/local.yaml
- show              → печатает итоговую конфигурацию (default ⊕ local)
- get <key>         → читает значение по ключу (dot-path), напр. chunking.child_w
- set <key> <val>   → пишет в local.yaml (авто-каст типов: int/float/bool/json)
- list              → показывает все ключи и значения
- write-env [.env]  → генерирует .env для сервисов
- wizard            → интерактивный мастер настройки

Примеры:
  python config_cli.py --init
  python config_cli.py show
  python config_cli.py get chunking.child_w
  python config_cli.py set retrieval.k 12
  python config_cli.py list
  python config_cli.py wizard
  python config_cli.py write-env .env
"""
from __future__ import annotations
import argparse, json, copy
from pathlib import Path
from typing import Any, Dict

# --- YAML обработка ---
try:
    from ruamel.yaml import YAML
except ImportError:
    raise SystemExit("Установи ruamel.yaml: pip install ruamel.yaml")

_yaml = YAML()
_yaml.indent(mapping=2, sequence=4)

# --- Цветной вывод ---
def info(msg): print(f"\033[36mℹ {msg}\033[0m")
def ok(msg): print(f"\033[32m✅ {msg}\033[0m")
def warn(msg): print(f"\033[33m⚠ {msg}\033[0m")
def err(msg): print(f"\033[31m❌ {msg}\033[0m")

# --- Пути ---
ROOT = Path(__file__).resolve().parent
CONF_DIR = ROOT / "config"
DEFAULT_YAML = CONF_DIR / "default.yaml"
LOCAL_YAML = CONF_DIR / "local.yaml"

# --- Базовый конфиг ---
# ВАЖНО: эмбеддинги теперь получаем через Ollama embeddings.
# embedding.model — имя эмбеддинг-модели Ollama (например, zylonai/multilingual-e5-large)
DEFAULT_CONFIG: Dict[str, Any] = {
    "app": {
        "cases_dir": "cases",
        "data_dir": "data",
        "out_dir": "out",
        "pages_glob": "data/*.pages.jsonl",
        "bm25_json_dir": "index/bm25_json",
        "bm25_index_dir": "index/bm25_idx",
    },
    "qdrant": {
        "url": "http://localhost:7777",
        "collection": "med_kb_v3",
    },
    "embedding": {
        # используем Ollama embeddings
        "model": "zylonai/multilingual-e5-large",
        # поля ниже оставлены для обратной совместимости; в режиме Ollama не используются
        "device": "ollama",
        "batch": 16,
    },
    "chunking": {"child_w": 200, "child_overlap": 40, "parent_w": 800},
    "retrieval": {"k": 15, "kq": 80, "kb": 80, "combine_strategy": "weighted"},
    "debug": {"verbose": False},
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama3.1:8b",
        "options": {"temperature": 0.1, "num_ctx": 8192, "top_p": 0.9},
    },
    "prompt": {
        "system": (
            "Ты – медицинский ассистент. Твоя задача: ПРОВЕРИТЬ врача по предоставленному кейсу и "
            "фрагментам клинических источников. Работай ТОЛЬКО по контексту, без выдумок.\n"
            "1) Дай общий балл 1–100 и подпункты. 2) Критические ошибки. 3) Рекомендации со ссылками.\n"
            "Ответ строго JSON по схеме."
        ),
        "user_template": (
            "[КЕЙС]\n{case_text}\n\n[КОНТЕКСТ]\n{ctx}\n\nСформируй один JSON по схеме."
        ),
    },
}

# --- Утилиты ---
def load_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = _yaml.load(f)
        return data or {}

def save_yaml(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        _yaml.dump(obj, f)

def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def deep_set(d: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split('.')
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out

def auto_cast(val: str) -> Any:
    v = val.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        if v.startswith("{") or v.startswith("["):
            return json.loads(v)
    except Exception:
        pass
    try:
        return float(v) if "." in v else int(v)
    except Exception:
        return v

# --- Валидация ---
def validate(conf: Dict[str, Any]) -> None:
    ch = conf.get("chunking", {})
    if ch.get("child_w", 0) < 50:
        warn("chunking.child_w слишком мал (<50)")
    if ch.get("parent_w", 0) <= ch.get("child_w", 0):
        warn("parent_w должен быть больше child_w")
    # мягкая проверка модели эмбеддингов
    em = deep_get(conf, "embedding.model", "")
    if not em:
        warn("embedding.model не задан — по умолчанию будет использован zylonai/multilingual-e5-large")

# --- Команды ---
def cmd_init(_: argparse.Namespace) -> None:
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_YAML.exists():
        save_yaml(DEFAULT_YAML, DEFAULT_CONFIG)
        ok(f"Создан {DEFAULT_YAML}")
    else:
        info(f"{DEFAULT_YAML} уже существует")
    if not LOCAL_YAML.exists():
        save_yaml(LOCAL_YAML, {})
        ok(f"Создан {LOCAL_YAML}")
    else:
        info(f"{LOCAL_YAML} уже существует")

def cmd_show(_: argparse.Namespace) -> None:
    conf = deep_merge(load_yaml(DEFAULT_YAML), load_yaml(LOCAL_YAML))
    validate(conf)
    print(json.dumps(conf, ensure_ascii=False, indent=2))

def cmd_list(_: argparse.Namespace) -> None:
    conf = deep_merge(load_yaml(DEFAULT_YAML), load_yaml(LOCAL_YAML))
    for k, v in flatten(conf).items():
        print(f"{k}: {v}")

def cmd_get(ns: argparse.Namespace) -> None:
    conf = deep_merge(load_yaml(DEFAULT_YAML), load_yaml(LOCAL_YAML))
    val = deep_get(conf, ns.key)
    if val is None:
        err(f"Ключ не найден: {ns.key}")
        raise SystemExit(1)
    print(json.dumps(val, ensure_ascii=False, indent=2) if isinstance(val, (dict, list)) else val)

def cmd_set(ns: argparse.Namespace) -> None:
    local = load_yaml(LOCAL_YAML)
    value = auto_cast(ns.value)
    deep_set(local, ns.key, value)
    save_yaml(LOCAL_YAML, local)
    ok(f"{ns.key} = {value}")

def cmd_write_env(ns: argparse.Namespace) -> None:
    conf = deep_merge(load_yaml(DEFAULT_YAML), load_yaml(LOCAL_YAML))
    env_lines = [
        f"QDRANT_URL={conf['qdrant']['url']}",
        f"QDRANT_COLLECTION={conf['qdrant']['collection']}",
        # эмбеддинги через Ollama
        f"OLLAMA_BASE_URL={conf['ollama']['base_url']}",
        f"OLLAMA_EMB_MODEL={conf['embedding']['model']}",
        # LLM для генерации
        f"LLM_BASE_URL={conf['ollama']['base_url']}",
        f"LLM_MODEL={conf['ollama']['model']}",
        f"LLM_TEMPERATURE={conf['ollama']['options']['temperature']}",
        f"LLM_NUM_CTX={conf['ollama']['options']['num_ctx']}",
        # ретривал
        f"RETRIEVE_K={conf['retrieval']['k']}",
        f"RETRIEVE_KQ={conf['retrieval']['kq']}",
        f"RETRIEVE_KB={conf['retrieval']['kb']}",
    ]
    out = Path(ns.path or ".env")
    out.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    ok(f"Сгенерирован {out}")

# --- Wizard ---
def ask(prompt: str, default: Any = None, cast=str) -> Any:
    suffix = f" [{default}]" if default is not None else ""
    s = input(f"{prompt}{suffix}: ").strip()
    if not s:
        return default
    if cast is bool:
        return s.lower() in {"1", "y", "yes", "true", "t", "да", "д"}
    if cast is int:
        return int(s)
    if cast is float:
        return float(s)
    if s.startswith("{") or s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            pass
    return s

def cmd_wizard(_: argparse.Namespace) -> None:
    conf = deep_merge(load_yaml(DEFAULT_YAML), load_yaml(LOCAL_YAML))
    print("\n=== Интерактивный мастер настройки ===\n")
    deep_set(conf, "retrieval.k", ask("Количество фрагментов (retrieval.k)", conf["retrieval"]["k"], int))
    deep_set(conf, "embedding.model", ask("Модель эмбеддингов (Ollama)", conf["embedding"]["model"]))
    deep_set(conf, "ollama.base_url", ask("Ollama base URL", conf["ollama"]["base_url"]))
    deep_set(conf, "ollama.model", ask("Модель LLM (chat)", conf["ollama"]["model"]))
    deep_set(conf, "qdrant.url", ask("Qdrant URL", conf["qdrant"]["url"]))
    deep_set(conf, "qdrant.collection", ask("Qdrant коллекция", conf["qdrant"]["collection"]))
    save_yaml(LOCAL_YAML, conf)
    ok("config/local.yaml обновлён")

# --- Main ---
def main() -> None:
    ap = argparse.ArgumentParser(description="MedAI Config Manager")
    ap.add_argument("--init", action="store_true", help="Создать конфиги")
    sp = ap.add_subparsers(dest="cmd")

    for name, func, help_text in [
        ("show", cmd_show, "Показать итоговую конфигурацию"),
        ("get", cmd_get, "Получить значение по ключу"),
        ("set", cmd_set, "Изменить значение по ключу"),
        ("list", cmd_list, "Показать все ключи и значения"),
        ("write-env", cmd_write_env, "Сгенерировать .env"),
        ("wizard", cmd_wizard, "Интерактивный мастер"),
    ]:
        p = sp.add_parser(name, help=help_text)
        if name == "get":
            p.add_argument("key")
        if name == "set":
            p.add_argument("key")
            p.add_argument("value")
        if name == "write-env":
            p.add_argument("path", nargs="?")
        p.set_defaults(func=func)

    ns, _ = ap.parse_known_args()

    # Алиасы
    aliases = {"s": "set", "g": "get", "sh": "show", "w": "wizard", "ls": "list"}
    if ns.cmd in aliases:
        ns.cmd = aliases[ns.cmd]

    if ns.__dict__.get("init"):
        cmd_init(ns)
        if not ns.cmd:
            return
    if not ns.cmd:
        ap.print_help()
        return
    ns.func(ns)

if __name__ == "__main__":
    main()
