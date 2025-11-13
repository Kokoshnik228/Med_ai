#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Шаг 6. Чанкинг + индексация в Qdrant (dense: HF/GPU или Ollama)

Поддерживает два бэкенда эмбеддингов:
  - --emb-backend hf       → локально через FlagEmbedding (GPU, быстро)
  - --emb-backend ollama   → через Ollama /api/embeddings

Ключевые правки:
  • В payload теперь пишется chunk_text (обрезанный текст чанка) и стабильный chunk_id,
    чтобы на этапе retrieve_hybrid можно было фьюзить и возвращать лучшие ЧАНКИ, а не документы.
  • По умолчанию child_w/overlap приведены к более «узким» окнам (180/40), parent_w уменьшён до 500,
    чтобы повысить точность попаданий. Параметры можно переопределять флагами/ENV.

Примеры:
  # HF / GPU (bge-m3)
  python chunk_and_index.py \
    --emb-backend hf --hf-model BAAI/bge-m3 \
    --pages-glob "data/*.pages.jsonl" --collection med_kb_v3 \
    --qdrant-url http://localhost:7779 --recreate --batch 512

  # Ollama embeddings
  python chunk_and_index.py \
    --emb-backend ollama --emb-model zylonai/multilingual-e5-large:latest \
    --ollama-url http://localhost:11435 \
    --pages-glob "data/*.pages.jsonl" --collection med_kb_v3 \
    --qdrant-url http://localhost:7779 --only-new --batch 128
"""
from __future__ import annotations
import os
import sys
# ----------------- imports -----------------
import argparse
import numpy as np
import hashlib
import json
import re
import time
import uuid
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

try:
    import torch  # type: ignore
except Exception:
    torch = None  # допустимо, если используем ollama

# ----------------- Константы -----------------
MAX_EMB_CHARS_OLLAMA = 2000  # безопасный лимит для /api/embeddings
seen_hashes_global: Set[str] = set()

# ----------------- Утилиты ENV/флагов -----------------
def _env_truthy(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

# ----------------- Текстовые утилиты -----------------
def _clean_jsonl_line(s: str) -> str:
    """
    Мягкая очистка: убираем NUL и нестандартные контролы, нормализуем Юникод.
    Не чинит логические ошибки JSON, но убирает мусорные символы из OCR.
    """
    if not s:
        return s
    s = s.replace("\x00", "")
    s = "".join(ch for ch in s if ch.isprintable() or ch in "\t\r\n")
    s = unicodedata.normalize("NFKC", s)
    return s

def _read_pages_robust(fp: Path) -> tuple[list[dict], int]:
    """
    Читает .pages.jsonl построчно. Возвращает (pages, skipped_count).
    Плохие строки пропускаем с предупреждением (не валим весь процесс).
    """
    pages: list[dict] = []
    skipped = 0
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            line = _clean_jsonl_line(line).strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    pages.append({
                        "page": int(rec.get("page", 0) or 0),
                        "text": rec.get("text", "") or "",
                    })
            except Exception as e:
                skipped += 1
                print(f"⚠️  {fp.name}: битая JSONL-строка #{i}: {e}", flush=True)
    return pages, skipped

def normalize_for_hash(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^0-9a-zа-яё %/.,;:()\-\n]+", " ", t, flags=re.IGNORECASE)
    return t.strip()

def words(text: str) -> List[str]:
    return (text or "").split()

def chunk_words(tokens: List[str], max_len: int, overlap: int) -> List[List[str]]:
    chunks: List[List[str]] = []
    i, n = 0, len(tokens)
    while i < n:
        j = min(i + max_len, n)
        chunks.append(tokens[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def build_parent_child(
    pages: List[Dict[str, Any]],
    child_w: int,
    child_overlap: int,
    parent_w: int
) -> List[Dict[str, Any]]:
    """Возвращает список child-чанков с полями: text, parent_id, page_range."""
    parents: List[Dict[str, Any]] = []
    cur_words: List[str] = []
    cur_pages: List[int] = []

    for p in pages:
        w = words(p.get("text", ""))
        if not w:
            continue
        # Режем длинную страницу кусками по parent_w слов
        for i in range(0, len(w), parent_w):
            part = w[i:i + parent_w]
            if not part:
                continue
            cur_words.extend(part)
            cur_pages.append(int(p.get("page", 0)))
            if len(cur_words) >= parent_w:
                pid = f"P{len(parents) + 1}"
                parents.append({
                    "parent_id": pid,
                    "text": " ".join(cur_words[:parent_w]),
                    "page_range": (min(cur_pages), max(cur_pages)),
                })
                cur_words, cur_pages = [], []

    if cur_words:
        pid = f"P{len(parents) + 1}"
        parents.append({
            "parent_id": pid,
            "text": " ".join(cur_words),
            "page_range": (min(cur_pages) if cur_pages else 0, max(cur_pages) if cur_pages else 0),
        })

    childs: List[Dict[str, Any]] = []
    for parent in parents:
        toks = words(parent["text"])
        for win in chunk_words(toks, child_w, child_overlap):
            childs.append({
                "parent_id": parent["parent_id"],
                "text": " ".join(win),
                "page_range": parent["page_range"],
            })
    return childs

# ----------------- Общие утилиты -----------------
def l2_unit(vec: List[float]) -> List[float]:
    s = (sum(x * x for x in vec) ** 0.5) or 1.0
    return [x / s for x in vec]

def batched(it: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    buf: List[Any] = []
    for x in it:
        buf.append(x)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf

# ----------------- OLLAMA backend -----------------
def make_session(total_retries: int = 5, backoff: float = 0.5, timeout: int = 60) -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=16, pool_maxsize=32)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.request_timeout = timeout
    return sess

def _safe_trim(text: str, max_chars: int = MAX_EMB_CHARS_OLLAMA) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    sp = cut.rfind(" ")
    return (cut if sp < 200 else cut[:sp]) + " …"

def ollama_embed_one(session: requests.Session, base_url: str, model: str, text: str, timeout: int) -> List[float]:
    url = base_url.rstrip("/") + "/api/embeddings"
    payload = {"model": model, "prompt": _safe_trim(text)}
    r = session.post(url, json=payload, timeout=timeout or getattr(session, "request_timeout", 60))
    if r.status_code >= 500:
        raise RuntimeError(f"Ollama HTTP {r.status_code}")
    r.raise_for_status()
    data = r.json()
    vec = data.get("embedding") or []
    if not isinstance(vec, list) or not vec:
        raise RuntimeError("Empty embedding vector")
    return l2_unit([float(x) for x in vec])

def ollama_get_dim(session: requests.Session, base_url: str, model: str, timeout: int) -> int:
    v = ollama_embed_one(session, base_url, model, "dimension probe", timeout)
    return len(v)

def ollama_embed_batch(session: requests.Session, base_url: str, model: str, texts: List[str], timeout: int) -> List[List[float]]:
    out: List[List[float]] = []
    for t in texts:
        out.append(ollama_embed_one(session, base_url, model, t, timeout))
        time.sleep(0.02)  # лёгкий троттлинг
    return out

# ----------------- HF backend (FlagEmbedding) -----------------
class HFEmbedder:
    def __init__(self, model_name: str, device_hint: Optional[str] = None, use_fp16: bool = False):
        from FlagEmbedding import BGEM3FlagModel

        # Прецизионка — для новых карт даёт хороший буст и экономию ОЗУ
        if torch is not None:
            try:
                torch.set_float32_matmul_precision("medium")
            except Exception:
                pass

        # Источник дефолтов: аргумент → ENV(HF_DEVICE) → auto
        env_dev = os.getenv("HF_DEVICE", "").strip() or None
        requested = (device_hint or env_dev or "auto").lower()
        if requested in ("auto", ""):
            device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        else:
            device = requested

        # fp16: аргумент → ENV(HF_FP16) → auto(вкл. на cuda)
        fp16_env = _env_truthy(os.getenv("HF_FP16"), default=False)
        fp16 = bool(use_fp16 or fp16_env or device.startswith("cuda"))

        # Лог девайса
        gpu_note = ""
        if device.startswith("cuda"):
            if torch and torch.cuda.is_available():
                try:
                    gpu_note = f" | GPU={torch.cuda.get_device_name(0)}  CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES','<unset>')}"
                except Exception:
                    gpu_note = " | GPU=?"
            else:
                print("⚠️  Запрошен CUDA, но torch.cuda недоступен — переключаюсь на CPU.", flush=True)
                device, fp16 = "cpu", False

        print(f"🧠 HF embedder: {model_name} on {device} (fp16={fp16}){gpu_note}", flush=True)

        # Инициализация с фолбэком на CPU при проблемах CUDA
        try:
            self.model = BGEM3FlagModel(model_name, use_fp16=fp16, device=device)
        except Exception as e:
            if device.startswith("cuda"):
                print(f"⚠️  Не удалось инициализировать модель на CUDA ({e}). Фолбэк на CPU.", flush=True)
                self.model = BGEM3FlagModel(model_name, use_fp16=False, device="cpu")
            else:
                raise

        self.default_batch = _env_int("EMB_BATCH", 128)

    def embed_texts(self, texts, batch_size: int | None = None) -> np.ndarray:
        """
        Возвращает np.ndarray (N, D) с плотными эмбеддингами.
        Нормализует разные форматы выдачи библиотек (dict / list / np.array).
        """
        bs = int(batch_size or getattr(self, "default_batch", 128))
        # Пытаемся попросить только dense-выдачу, чтобы экономить память
        try:
            out = self.model.encode(
                texts,
                batch_size=max(1, bs),
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
                normalize_embeddings=True,
            )
        except TypeError:
            # Некоторые реализации не знают этих флагов
            out = self.model.encode(texts, batch_size=max(1, bs))

        # Нормализация формата
        if isinstance(out, dict):
            if "dense_vecs" in out and out["dense_vecs"] is not None:
                vecs = out["dense_vecs"]
            elif "embeddings" in out and out["embeddings"] is not None:
                vecs = out["embeddings"]
            elif "sentence_embeddings" in out and out["sentence_embeddings"] is not None:
                vecs = out["sentence_embeddings"]
            else:
                raise ValueError(f"Неожиданный формат encode(): ключи {list(out.keys())}")
        else:
            vecs = out

        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        # страховочная нормализация
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs = (vecs / norms).astype(np.float32)
        return vecs

    def get_dim(self) -> int:
        vecs = self.embed_texts(["probe"], batch_size=4)
        return int(vecs.shape[-1])

# ----------------- Qdrant helpers -----------------
def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool = False) -> None:
    if recreate and client.collection_exists(name):
        print(f"⚠️ Пересоздаём коллекцию {name}...", flush=True)
        client.delete_collection(name)
        time.sleep(2)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"✅ Создана коллекция {name} (dim={dim}, metric=COSINE)", flush=True)
    else:
        print(f"ℹ️  Коллекция {name} уже существует (dim={dim})", flush=True)

def fetch_existing_hashes(client: QdrantClient, collection: str, doc_id: str) -> set[str]:
    existing: set[str] = set()
    scroll_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=1000,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            th = (p.payload or {}).get("text_hash")
            if isinstance(th, str):
                existing.add(th)
        if next_offset is None:
            break
    return existing

# ----------------- MAIN -----------------
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pages-glob", required=True)
    ap.add_argument("--qdrant-url", default="http://qdrant:6333")
    ap.add_argument("--collection", default="med_kb_v3")

    # Синхронизированы дефолты (более «узкие» окна)
    ap.add_argument("--child-w", type=int, default=180)
    ap.add_argument("--child-overlap", type=int, default=40)
    ap.add_argument("--parent-w", type=int, default=500)

    # batch из ENV EMB_BATCH (деф. 256 если не указан в ENV)
    ap.add_argument("--batch", type=int, default=_env_int("EMB_BATCH", 256))

    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--only-new", action="store_true")
    ap.add_argument("--qdrant-wait", action="store_true")

    # Embedding backend
    ap.add_argument("--emb-backend", choices=["hf", "ollama"], default="hf")

    # HF (дефолты из ENV)
    ap.add_argument("--hf-model", default=os.getenv("HF_MODEL", "BAAI/bge-m3"))
    ap.add_argument("--hf-device", default=os.getenv("HF_DEVICE", None))      # например, cuda или cuda:0
    ap.add_argument("--hf-fp16", action="store_true", default=_env_truthy(os.getenv("HF_FP16"), default=False))

    # Ollama
    ap.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11435"))
    ap.add_argument("--emb-model", default=os.getenv("EMB_MODEL", "zylonai/multilingual-e5-large:latest"))
    ap.add_argument("--timeout", type=int, default=int(os.getenv("EMB_TIMEOUT", "180")))

    args = ap.parse_args()

    files = sorted(Path().glob(args.pages_glob))
    if not files:
        raise SystemExit(f"❌ Не найдено файлов по маске: {args.pages_glob}")

    # Проверка доступности Qdrant
    print(f"🔗 Qdrant URL: {args.qdrant_url}", flush=True)
    try:
        test = requests.get(args.qdrant_url.rstrip("/") + "/readyz", timeout=5)
        test.raise_for_status()
        print("✅ Qdrant доступен.", flush=True)
    except Exception as e:
        raise SystemExit(f"❌ Не удалось подключиться к Qdrant ({args.qdrant_url}): {e}")

    # Инициализация эмбеддингов
    if args.emb_backend == "hf":
        hf = HFEmbedder(args.hf_model, device_hint=args.hf_device, use_fp16=args.hf_fp16)
        dim = hf.get_dim()
        print(f"🔤 HF embeddings: {args.hf_model} (dim={dim}) | batch={args.batch}", flush=True)
        embed_fn = lambda texts: hf.embed_texts(texts, batch_size=args.batch)
    else:
        session = make_session(timeout=args.timeout)
        dim = ollama_get_dim(session, args.ollama_url, args.emb_model, args.timeout)
        print(f"🔤 Ollama embeddings: {args.emb_model} (dim={dim}) @ {args.ollama_url} | batch={args.batch}", flush=True)
        embed_fn = lambda texts: ollama_embed_batch(session, args.ollama_url, args.emb_model, texts, args.timeout)

    # Qdrant коллекция
    client = QdrantClient(url=args.qdrant_url, prefer_grpc=False, grpc_port=None, timeout=60)
    ensure_collection(client, args.collection, dim=dim, recreate=args.recreate)

    total_points_upserted = 0
    total_childs_after_dedup = 0

    for fp in files:
        doc_id = fp.stem.replace(".pages", "")
        pages, bad = _read_pages_robust(fp)
        if bad:
            print(f"⚠️  {doc_id}: пропущено битых строк: {bad}", flush=True)

        if not any(p.get("text") for p in pages):
            print(f"📄 {doc_id}: нет текста (пропуск)", flush=True)
            continue

        # Чанкинг
        childs = build_parent_child(pages, args.child_w, args.child_overlap, args.parent_w)
        if not childs:
            print(f"📄 {doc_id}: нет чанков (пропуск)", flush=True)
            continue

        # Хэши для дедупликации
        for c in childs:
            c["text_hash"] = hashlib.sha1(normalize_for_hash(c["text"]).encode("utf-8")).hexdigest()

        # already-in-DB дедуп (по doc_id и text_hash)
        existing_hashes = fetch_existing_hashes(client, args.collection, doc_id) if args.only_new and not args.recreate else set()

        filtered: List[Dict[str, Any]] = []
        skipped_existing = 0
        for c in childs:
            h = c["text_hash"]
            if h in seen_hashes_global:
                continue
            if h in existing_hashes:
                skipped_existing += 1
                continue
            seen_hashes_global.add(h)
            filtered.append(c)
        childs = filtered
        total_childs_after_dedup += len(childs)

        print(f"📄 {doc_id}: новых чанков {len(childs)}, уже существующих {skipped_existing}", flush=True)
        if not childs:
            continue

        # Батчевый эмбеддинг + upsert
        for chunk in tqdm(batched(childs, args.batch), desc=f"{doc_id}", unit="batch"):
            texts = [c["text"] for c in chunk]
            if not texts:
                continue

            try:
                emb = embed_fn(texts)  # np.ndarray (N, D) или List[List[float]]
            except Exception as e:
                print(f"⚠️ Ошибка эмбеддинга батча ({len(texts)}): {e}", flush=True)
                continue

            # Нормализация к List[float] для Qdrant
            if isinstance(emb, np.ndarray):
                emb_rows = [row.astype(np.float32).tolist() for row in emb]
            else:
                emb_rows = [[float(x) for x in row] for row in emb]

            points: List[PointStruct] = []
            for i, c in enumerate(chunk):
                # стабильный id точки (uuid5 на основе doc_id и text_hash)
                unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{c['text_hash']}"))
                # стабильный идентификатор чанка для фьюжна/дедупа на этапе retrieve
                chunk_id = f"{doc_id}:{c['parent_id']}:{c['text_hash']}"

                payload = {
                    "doc_id": doc_id,
                    "parent_id": c["parent_id"],
                    "page_start": c["page_range"][0],
                    "page_end": c["page_range"][1],
                    "len_words": len(c["text"].split()),
                    "child_len_words": len(c["text"].split()),  # алиас для наглядности
                    "text_hash": c["text_hash"],
                    "chunk_text": c["text"][:800],               # <— точный текст чанка (усечённый)
                    "chunk_id": chunk_id,                         # <— стабильный chunk-id
                }
                points.append(PointStruct(id=unique_id, vector=emb_rows[i], payload=payload))

            client.upsert(
                collection_name=args.collection,
                points=points,
                wait=bool(args.qdrant_wait),
            )
            total_points_upserted += len(points)

    print("📊 Итог:", flush=True)
    print(f"  Файлов обработано: {len(files)}", flush=True)
    print(f"  Новых чанков: {total_childs_after_dedup}", flush=True)
    print(f"  Добавлено/обновлено векторов: {total_points_upserted}", flush=True)
    print("✅ Индексация в Qdrant завершена.", flush=True)

if __name__ == "__main__":
    main()
