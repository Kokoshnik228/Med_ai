#!/usr/bin/env python3
"""
Единый менеджер настроек проекта (CLI + интерактивный мастер).

Возможности:
- --init            → создаёт config/default.yaml и config/local.yaml
- show              → печатает итоговую конфигурацию (default ⊕ local)
- get <key>         → читает значение по ключу (dot‑path), напр. chunking.child_w
- set <key> <val>   → пишет в local.yaml (авто‑каст типов: int/float/bool/json)
- write-env [.env]  → генерирует .env для сервисов (QDRANT_URL, LLM_*, ...)
- wizard            → интерактивный мастер изменения всех ключевых параметров

Примеры:
  python config_cli.py --init
  python config_cli.py show
  python config_cli.py get chunking.child_w
  python config_cli.py set retrieval.k 12
  python config_cli.py wizard
  python config_cli.py write-env .env

Интеграция с существующими скриптами:
  python chunk_and_index.py \
    --pages-glob "$(python config_cli.py get app.pages_glob)" \
    --qdrant-url "$(python config_cli.py get qdrant.url)" \
    --collection "$(python config_cli.py get qdrant.collection)" \
    --child-w "$(python config_cli.py get chunking.child_w)" \
    --child-overlap "$(python config_cli.py get chunking.child_overlap)" \
    --parent-w "$(python config_cli.py get chunking.parent_w)" \
    --device "$(python config_cli.py get embedding.device)" \
    --batch "$(python config_cli.py get embedding.batch)"
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # PyYAML
except Exception as e:
    raise SystemExit("Установи pyyaml: pip install pyyaml")

ROOT = Path(__file__).resolve().parent
CONF_DIR = ROOT / "config"
DEFAULT_YAML = CONF_DIR / "default.yaml"
LOCAL_YAML = CONF_DIR / "local.yaml"

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
        "collection": "med_kb",
    },
    "embedding": {
        "model": "BAAI/bge-m3",
        "device": "cuda",  # cpu|cuda|auto
        "batch": 16,
    },
    "chunking": {
        "child_w": 200,
        "child_overlap": 40,
        "parent_w": 800,
    },
    "retrieval": {
        "k": 8,
        "kq": 80,
        "kb": 80,
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama3.1:8b",
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
            "top_p": 0.9,
        },
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

# ---------- утилиты ----------

def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
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


def load_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding='utf-8')) or {}


def save_yaml(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding='utf-8')


def auto_cast(val: str) -> Any:
    v = val.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        if v.startswith('{') or v.startswith('['):
            return json.loads(v)
    except Exception:
        pass
    try:
        if '.' in v:
            return float(v)
        return int(v)
    except Exception:
        return v


def validate(conf: Dict[str, Any]) -> None:
    def warn(msg: str):
        print(f"[WARN] {msg}")
    ch = conf.get('chunking', {})
    if ch.get('child_w', 0) < 50:
        warn('chunking.child_w слишком мал (<50) — увеличь до 150–250 слов')
    if ch.get('parent_w', 0) <= ch.get('child_w', 0):
        warn('parent_w должен быть больше child_w')
    opt = conf.get('ollama', {}).get('options', {})
    t = opt.get('temperature', 0.1)
    try:
        if not (0.0 <= float(t) <= 1.0):
            warn('ollama.options.temperature вне [0,1]')
    except Exception:
        warn('ollama.options.temperature не число')

# ---------- команды ----------

def cmd_init(_: argparse.Namespace) -> None:
    if not DEFAULT_YAML.exists():
        save_yaml(DEFAULT_YAML, DEFAULT_CONFIG)
        print(f"Создан {DEFAULT_YAML}")
    else:
        print(f"{DEFAULT_YAML} уже существует")
    if not LOCAL_YAML.exists():
        save_yaml(LOCAL_YAML, {})
        print(f"Создан {LOCAL_YAML}")
    else:
        print(f"{LOCAL_YAML} уже существует")


def cmd_show(_: argparse.Namespace) -> None:
    conf = deep_merge(load_yaml(DEFAULT_YAML), load_yaml(LOCAL_YAML))
    validate(conf)
    print(yaml.safe_dump(conf, sort_keys=False, allow_unicode=True))


def cmd_get(ns: argparse.Namespace) -> None:
    conf = deep_merge(load_yaml(DEFAULT_YAML), load_yaml(LOCAL_YAML))
    val = deep_get(conf, ns.key)
    if val is None:
        raise SystemExit(f"Ключ не найден: {ns.key}")
    if isinstance(val, (dict, list)):
        print(json.dumps(val, ensure_ascii=False))
    else:
        print(val)


def cmd_set(ns: argparse.Namespace) -> None:
    local = load_yaml(LOCAL_YAML)
    value = auto_cast(ns.value)
    deep_set(local, ns.key, value)
    save_yaml(LOCAL_YAML, local)
    print(f"OK: {ns.key} = {value}")


def cmd_write_env(ns: argparse.Namespace) -> None:
    conf = deep_merge(load_yaml(DEFAULT_YAML), load_yaml(LOCAL_YAML))
    env_lines = [
        f"QDRANT_URL={conf.get('qdrant',{}).get('url','')}",
        f"QDRANT_COLLECTION={conf.get('qdrant',{}).get('collection','')}",
        f"EMBEDDING_MODEL={conf.get('embedding',{}).get('model','')}",
        f"EMBEDDING_DEVICE={conf.get('embedding',{}).get('device','')}",
        f"EMBEDDING_BATCH={conf.get('embedding',{}).get('batch','')}",
        f"LLM_BASE_URL={conf.get('ollama',{}).get('base_url','')}",
        f"LLM_MODEL={conf.get('ollama',{}).get('model','')}",
        f"LLM_TEMPERATURE={conf.get('ollama',{}).get('options',{}).get('temperature','')}",
        f"LLM_NUM_CTX={conf.get('ollama',{}).get('options',{}).get('num_ctx','')}",
        f"RETRIEVE_K={conf.get('retrieval',{}).get('k','')}",
        f"RETRIEVE_KQ={conf.get('retrieval',{}).get('kq','')}",
        f"RETRIEVE_KB={conf.get('retrieval',{}).get('kb','')}",
    ]
    out = Path(ns.path or ".env")
    out.write_text("\n".join(env_lines) + "\n", encoding='utf-8')
    print(f"OK: записал {out}")

# ---------- интерактивный мастер ----------

def ask(prompt: str, default: Any = None, cast=str) -> Any:
    suffix = f" [{default}]" if default is not None else ""
    s = input(f"{prompt}{suffix}: ").strip()
    if not s:
        return default
    if cast is bool:
        return s.lower() in {"1","y","yes","true","t","да","д"}
    if cast is int:
        return int(s)
    if cast is float:
        return float(s)
    # json попытка
    if s.startswith('{') or s.startswith('['):
        try:
            return json.loads(s)
        except Exception:
            pass
    return s


def cmd_wizard(_: argparse.Namespace) -> None:
    default = load_yaml(DEFAULT_YAML)
    local = load_yaml(LOCAL_YAML)
    conf = deep_merge(default, local)

    print("\n=== Мастер настроек (Enter — оставить текущее) ===\n")

    # Qdrant
    print("[Qdrant]")
    deep_set(conf, 'qdrant.url', ask('Qdrant URL', conf['qdrant']['url']))
    deep_set(conf, 'qdrant.collection', ask('Коллекция', conf['qdrant']['collection']))

    # Embeddings
    print("\n[Эмбеддинги]")
    deep_set(conf, 'embedding.model', ask('Модель эмбеддингов', conf['embedding']['model']))
    deep_set(conf, 'embedding.device', ask('Девайс (cpu/cuda/auto)', conf['embedding']['device']))
    deep_set(conf, 'embedding.batch', ask('Batch', conf['embedding']['batch'], int))

    # Chunking
    print("\n[Чанкинг]")
    deep_set(conf, 'chunking.child_w', ask('Child (слова)', conf['chunking']['child_w'], int))
    deep_set(conf, 'chunking.child_overlap', ask('Overlap (слова)', conf['chunking']['child_overlap'], int))
    deep_set(conf, 'chunking.parent_w', ask('Parent (слова)', conf['chunking']['parent_w'], int))

    # Retrieval
    print("\n[Ретривер]")
    deep_set(conf, 'retrieval.k', ask('k (фрагментов)', conf['retrieval']['k'], int))
    deep_set(conf, 'retrieval.kq', ask('kq (Qdrant)', conf['retrieval']['kq'], int))
    deep_set(conf, 'retrieval.kb', ask('kb (BM25)', conf['retrieval']['kb'], int))

    # Ollama / LLM
    print("\n[LLM (Ollama)]")
    deep_set(conf, 'ollama.base_url', ask('Ollama URL', conf['ollama']['base_url']))
    deep_set(conf, 'ollama.model', ask('Модель (llama3.1:8b/70b …)', conf['ollama']['model']))
    deep_set(conf, 'ollama.options.temperature', ask('Temperature', conf['ollama']['options']['temperature'], float))
    deep_set(conf, 'ollama.options.num_ctx', ask('Context tokens (num_ctx)', conf['ollama']['options']['num_ctx'], int))
    deep_set(conf, 'ollama.options.top_p', ask('top_p', conf['ollama']['options']['top_p'], float))

    # Prompt
    print("\n[Промпты]")
    print("System prompt (текущее значение ниже):\n---\n" + conf['prompt']['system'] + "\n---")
    sp = ask('Изменить system prompt? (y/N)', False, bool)
    if sp:
        deep_set(conf, 'prompt.system', input('Введи новый system prompt (многострочно, закончить Ctrl+D/Ctrl+Z):\n') or conf['prompt']['system'])
    print("\nUser template (текущее значение ниже):\n---\n" + conf['prompt']['user_template'] + "\n---")
    up = ask('Изменить user template? (y/N)', False, bool)
    if up:
        deep_set(conf, 'prompt.user_template', input('Введи новый user template (многострочно, закончить Ctrl+D/Ctrl+Z):\n') or conf['prompt']['user_template'])

    # Пути
    print("\n[Пути]")
    deep_set(conf, 'app.cases_dir', ask('Папка кейсов', conf['app']['cases_dir']))
    deep_set(conf, 'app.data_dir', ask('Папка данных', conf['app']['data_dir']))
    deep_set(conf, 'app.out_dir', ask('Папка результатов', conf['app']['out_dir']))
    deep_set(conf, 'app.pages_glob', ask('Глоб для pages.jsonl', conf['app']['pages_glob']))
    deep_set(conf, 'app.bm25_json_dir', ask('Папка BM25 JSON', conf['app']['bm25_json_dir']))
    deep_set(conf, 'app.bm25_index_dir', ask('Папка BM25 индекс', conf['app']['bm25_index_dir']))

    # Сохранение
    print("\nСохраняю изменения → config/local.yaml …")
    save_yaml(LOCAL_YAML, conf)  # сохраняем полную слитую; можно сохранить только дельту при желании
    print("OK: config/local.yaml обновлён.\n")

# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Config manager for med_ai")
    ap.add_argument("--init", action="store_true", help="Создать default.yaml и local.yaml, если их нет")
    sp = ap.add_subparsers(dest="cmd")

    sp_show = sp.add_parser("show", help="Показать итоговую конфигурацию")
    sp_show.set_defaults(func=cmd_show)

    sp_get = sp.add_parser("get", help="Получить значение по ключу (dot-path)")
    sp_get.add_argument("key")
    sp_get.set_defaults(func=cmd_get)

    sp_set = sp.add_parser("set", help="Записать значение по ключу в local.yaml")
    sp_set.add_argument("key")
    sp_set.add_argument("value")
    sp_set.set_defaults(func=cmd_set)

    sp_env = sp.add_parser("write-env", help="Сгенерировать .env для сервисов")
    sp_env.add_argument("path", nargs="?")
    sp_env.set_defaults(func=cmd_write_env)

    sp_wiz = sp.add_parser("wizard", help="Интерактивный мастер настройки")
    sp_wiz.set_defaults(func=cmd_wizard)

    ns, extra = ap.parse_known_args()
    if ns.__dict__.get("init"):
        cmd_init(ns)
        if ns.cmd is None:
            return
    if not ns.cmd:
        ap.print_help()
        return
    ns.func(ns)


if __name__ == "__main__":
    main()
