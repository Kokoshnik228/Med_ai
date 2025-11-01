#!/usr/bin/env python3
"""
Единый менеджер настроек проекта (CLI + интерактивный мастер).

Команды:
  --init                → создаёт config/default.yaml и config/local.yaml (если нет)
  show                  → печатает итоговую конфигурацию (default ⊕ local)
  list                  → плоский список всех ключей/значений
  get <key>             → читает значение по ключу (dot-path), напр. chunking.child_w
  set <key> <val>       → пишет в local.yaml (автокаст: int/float/bool/json/str)
  write-env [path]      → генерирует .env (по умолчанию .env.dev); опция --profile dev|prod|local
  wizard                → интерактивная настройка основных параметров
  doctor                → проверка доступности Qdrant и Ollama
  migrate-keys          → миграция старых ключей (embedding.model → embedding.hf_model)

Примеры:
  python config_cli.py --init
  python config_cli.py show
  python config_cli.py set retrieval.k 12
  python config_cli.py write-env --profile local .env.dev
  python config_cli.py doctor
  python config_cli.py migrate-keys
"""
from __future__ import annotations
import argparse, json, copy, sys
from pathlib import Path
from typing import Any, Dict, Tuple
import urllib.request
import urllib.error
import socket

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

# --- Базовый конфиг (совместим c api_app.py) ---
DEFAULT_CONFIG: Dict[str, Any] = {
    "app": {
        "cases_dir": "cases",
        "data_dir": "data",
        "out_dir": "out",
        "pages_glob": "data/*.pages.jsonl",
        "bm25_json_dir": "index/bm25_json",
        "bm25_index_dir": "index/bm25_idx",
        "port_dev": 7050,
        "port_prod": 8050,
    },
    "qdrant": {
        # По умолчанию — Docker-сеть; write-env подменит для local → http://localhost:7779
        "url": "http://qdrant:6333",
        "collection": "med_kb_v3",
    },
    "embedding": {
        "backend": "hf",         # hf | ollama
        "hf_model": "BAAI/bge-m3",
        "device": "cuda",        # cuda | cpu | auto
        "fp16": False,
        "batch": 256,
    },
    "reranker": {
        "enabled": True,
        "model": "BAAI/bge-reranker-v2-m3"
    },
    "chunking": {
        "child_w": 200,
        "child_overlap": 35,
        "parent_w": 800
    },
    "retrieval": {
        "k": 15,
        "kq": 80,
        "kb": 80,
        "combine_strategy": "rrf"  # rrf | weighted
    },
    "ollama": {
        "base_url": "http://host.docker.internal:11434",
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
    "profiles": {
        # Подсказки для write-env
        "local": {"QDRANT_URL": "http://localhost:7779"},
        "dev":   {"QDRANT_URL": "http://qdrant:6333"},
        "prod":  {"QDRANT_URL": "http://qdrant:6333"},
    }
}

# --- Утилиты YAML ---
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

# --- Миграция ключей ---
def migrate_keys(conf: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    changed = False
    emb = conf.get("embedding", {})
    # Перенос embedding.model → embedding.hf_model, если нового нет
    if "model" in emb and "hf_model" not in emb:
        emb["hf_model"] = emb["model"]
        changed = True
    # Бэкенд по умолчанию hf
    if "backend" not in emb:
        emb["backend"] = "hf"
        changed = True
    conf["embedding"] = emb
    return conf, changed

# --- Валидация ---
def validate(conf: Dict[str, Any]) -> None:
    ch = conf.get("chunking", {})
    if ch.get("child_w", 0) < 50:
        warn("chunking.child_w слишком мал (<50)")
    if ch.get("parent_w", 0) <= ch.get("child_w", 0):
        warn("chunking.parent_w должен быть > chunking.child_w")
    emb = conf.get("embedding", {})
    if emb.get("backend") == "hf" and not emb.get("hf_model"):
        warn("embedding.hf_model не задан — используется дефолт BAAI/bge-m3")
    if emb.get("backend") not in {"hf", "ollama"}:
        warn("embedding.backend должен быть 'hf' или 'ollama'")
    # Reranker hint
    rr = conf.get("reranker", {})
    if rr.get("enabled") and not rr.get("model"):
        warn("reranker.enabled=true, но reranker.model не указан")

# --- Doctor (проверки) ---
def http_get(url: str, timeout: float = 3.5) -> Tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            code = r.getcode()
            body = r.read(200).decode("utf-8", "ignore")
            return code, body
    except urllib.error.HTTPError as e:
        return e.code, str(e)
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"

def ping_qdrant(url: str) -> bool:
    if "qdrant:" in url:
        try:
            socket.gethostbyname("qdrant")
        except socket.gaierror:
            warn("Имя 'qdrant' не резолвится — для локалки используй http://localhost:7779")
    code, _ = http_get(url.rstrip("/") + "/collections")
    return code == 200

def ping_ollama(base_url: str) -> bool:
    code, _ = http_get(base_url.rstrip("/") + "/api/tags")
    return code == 200

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
    base = load_yaml(DEFAULT_YAML) or {}
    local = load_yaml(LOCAL_YAML) or {}
    conf = deep_merge(base, local)
    conf, _ = migrate_keys(conf)
    validate(conf)
    print(json.dumps(conf, ensure_ascii=False, indent=2))

def cmd_list(_: argparse.Namespace) -> None:
    conf = deep_merge(load_yaml(DEFAULT_YAML), load_yaml(LOCAL_YAML))
    conf, _ = migrate_keys(conf)
    for k, v in flatten(conf).items():
        print(f"{k}: {v}")

def cmd_get(ns: argparse.Namespace) -> None:
    conf = deep_merge(load_yaml(DEFAULT_YAML), load_yaml(LOCAL_YAML))
    conf, _ = migrate_keys(conf)
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

def _profile_to_qdrant_url(conf: Dict[str, Any], profile: str) -> str:
    prof = deep_get(conf, f"profiles.{profile}", {}) or {}
    return prof.get("QDRANT_URL") or conf["qdrant"]["url"]

def cmd_write_env(ns: argparse.Namespace) -> None:
    """
    Генерируем .env с ключами, которые реально читает приложение:
      APP_ENV, APP_PORT, QDRANT_URL, QDRANT_COLLECTION,
      EMB_BACKEND, HF_MODEL, HF_DEVICE, HF_FP16,
      CHILD_W, CHILD_OVERLAP, PARENT_W,
      RETRIEVE_K, RETRIEVE_KQ, RETRIEVE_KB,
      LLM_BASE_URL, MODEL_ID, QDRANT__PREFER_GRPC=false
    """
    profile = ns.profile or "dev"  # dev|prod|local
    conf = deep_merge(load_yaml(DEFAULT_YAML), load_yaml(LOCAL_YAML))
    conf, _ = migrate_keys(conf)

    # Определяем порт
    if profile in {"local", "dev"}:
        app_port = conf["app"].get("port_dev", 7050)
    else:
        app_port = conf["app"].get("port_prod", 8050)

    # Qdrant URL c учётом профиля
    qdrant_url = _profile_to_qdrant_url(conf, profile)

    env_lines = [
        f"APP_ENV={profile if profile!='local' else 'dev'}",
        f"APP_PORT={app_port}",
        f"QDRANT_URL={qdrant_url}",
        f"QDRANT_COLLECTION={conf['qdrant']['collection']}",
        "QDRANT__PREFER_GRPC=false",
        f"EMB_BACKEND={conf['embedding'].get('backend','hf')}",
        f"HF_MODEL={conf['embedding'].get('hf_model','BAAI/bge-m3')}",
        f"HF_DEVICE={conf['embedding'].get('device','auto')}",
        f"HF_FP16={'1' if conf['embedding'].get('fp16', False) else '0'}",
        f"CHILD_W={conf['chunking']['child_w']}",
        f"CHILD_OVERLAP={conf['chunking']['child_overlap']}",
        f"PARENT_W={conf['chunking']['parent_w']}",
        f"RETRIEVE_K={conf['retrieval']['k']}",
        f"RETRIEVE_KQ={conf['retrieval']['kq']}",
        f"RETRIEVE_KB={conf['retrieval']['kb']}",
        f"LLM_BASE_URL={conf['ollama']['base_url']}",
        f"MODEL_ID={conf['ollama']['model']}",
    ]

    out = Path(ns.path or (".env.dev" if profile in {"dev", "local"} else ".env.prod"))
    out.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    ok(f"Сгенерирован {out}")

def cmd_wizard(_: argparse.Namespace) -> None:
    conf = deep_merge(load_yaml(DEFAULT_YAML), load_yaml(LOCAL_YAML))
    conf, _ = migrate_keys(conf)
    print("\n=== Интерактивный мастер настройки ===\n")
    # embedding
    emb_backend = ask("Бэкенд эмбеддингов (hf/ollama)", conf["embedding"].get("backend","hf"))
    deep_set(conf, "embedding.backend", emb_backend or "hf")
    if emb_backend == "hf":
        deep_set(conf, "embedding.hf_model", ask("HF модель эмбеддингов", conf["embedding"].get("hf_model","BAAI/bge-m3")))
        deep_set(conf, "embedding.device", ask("Устройство (cuda/cpu/auto)", conf["embedding"].get("device","cuda")))
        deep_set(conf, "embedding.fp16", ask("Включить fp16 (да/нет)", conf["embedding"].get("fp16",False), bool))
    else:
        # на будущее — если решишь вернуться к Ollama embeddings
        deep_set(conf, "embedding.hf_model", ask("Имя эмбеддинг-модели (строка)", conf["embedding"].get("hf_model","BAAI/bge-m3")))

    # qdrant / ollama
    deep_set(conf, "qdrant.url", ask("Qdrant URL (docker: http://qdrant:6333; локалка: http://localhost:7779)",
                                     conf["qdrant"]["url"]))
    deep_set(conf, "qdrant.collection", ask("Qdrant коллекция", conf["qdrant"]["collection"]))
    deep_set(conf, "ollama.base_url", ask("Ollama base URL", conf["ollama"]["base_url"]))
    deep_set(conf, "ollama.model", ask("LLM модель (chat)", conf["ollama"]["model"]))

    # retrieval / chunking
    deep_set(conf, "retrieval.k", ask("Количество фрагментов (retrieval.k)", conf["retrieval"]["k"], int))
    deep_set(conf, "chunking.parent_w", ask("Размер parent чанка (слов)", conf["chunking"]["parent_w"], int))
    deep_set(conf, "chunking.child_w", ask("Размер child чанка (слов)", conf["chunking"]["child_w"], int))
    deep_set(conf, "chunking.child_overlap", ask("Перекрытие child чанков (слов)", conf["chunking"]["child_overlap"], int))

    save_yaml(LOCAL_YAML, conf)
    ok("config/local.yaml обновлён")

def cmd_doctor(_: argparse.Namespace) -> None:
    conf = deep_merge(load_yaml(DEFAULT_YAML), load_yaml(LOCAL_YAML))
    conf, _ = migrate_keys(conf)

    # Пробуем оба варианта Qdrant (из конфига и локалку)
    qd_cfg = conf["qdrant"]["url"]
    qd_local = "http://localhost:7779"
    for label, url in [("config.qdrant.url", qd_cfg), ("localhost", qd_local)]:
        ok_str = "OK" if ping_qdrant(url) else "FAIL"
        print(f"Qdrant [{label}] {url} → {ok_str}")

    # Ollama
    ollama_url = conf["ollama"]["base_url"]
    print(f"Ollama {ollama_url} → {'OK' if ping_ollama(ollama_url) else 'FAIL'}")

def cmd_migrate_keys(_: argparse.Namespace) -> None:
    base = load_yaml(DEFAULT_YAML)
    local = load_yaml(LOCAL_YAML)
    conf = deep_merge(base, local)
    conf2, changed = migrate_keys(conf)
    if not changed:
        info("Миграция не требуется — ключи уже в актуальном виде.")
        return
    # записываем только изменения в local.yaml
    save_yaml(LOCAL_YAML, conf2)
    ok("Ключи мигрированы (embedding.model → embedding.hf_model, добавлен embedding.backend)")

# --- I/O helpers ---
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

# --- Главный вход ---
def main() -> None:
    ap = argparse.ArgumentParser(description="MedAI Config Manager")
    ap.add_argument("--init", action="store_true", help="Создать конфиги (default/local)")
    sp = ap.add_subparsers(dest="cmd")

    # Подкоманды
    for name, help_text in [
        ("show", "Показать итоговую конфигурацию"),
        ("get", "Получить значение по ключу"),
        ("set", "Изменить значение по ключу"),
        ("list", "Показать все ключи и значения"),
        ("write-env", "Сгенерировать .env (используй --profile)"),
        ("wizard", "Интерактивный мастер"),
        ("doctor", "Проверка Qdrant/Ollama"),
        ("migrate-keys", "Миграция ключей к актуальному формату"),
    ]:
        sp.add_parser(name, help=help_text)

    # Аргументы для некоторых команд
    p_get = sp.choices["get"]; p_get.add_argument("key")
    p_set = sp.choices["set"]; p_set.add_argument("key"); p_set.add_argument("value")
    p_env = sp.choices["write-env"]; p_env.add_argument("path", nargs="?"); p_env.add_argument("--profile", choices=["local","dev","prod"], default="dev")

    ns, _ = ap.parse_known_args()

    # Алиасы
    aliases = {"s": "set", "g": "get", "sh": "show", "w": "wizard", "ls": "list", "doc": "doctor", "mig": "migrate-keys"}
    if ns.cmd in aliases:
        ns.cmd = aliases[ns.cmd]

    if ns.__dict__.get("init"):
        cmd_init(ns)
        if not ns.cmd:
            return

    if not ns.cmd:
        ap.print_help()
        return

    # Диспатч
    {
        "show": cmd_show,
        "list": cmd_list,
        "get": cmd_get,
        "set": cmd_set,
        "write-env": cmd_write_env,
        "wizard": cmd_wizard,
        "doctor": cmd_doctor,
        "migrate-keys": cmd_migrate_keys,
    }[ns.cmd](ns)

if __name__ == "__main__":
    main()
