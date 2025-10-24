#!/usr/bin/env python3
"""
Шаг 6. Чанкинг + индексация в Qdrant (dense, через Ollama embeddings)

Берёт выход из шага ingest (data/*.pages.jsonl), собирает parent/child чанки,
считает эмбеддинги через Ollama (по умолчанию zylonai/multilingual-e5-large)
и загружает child-чанки в Qdrant.

✅ Поддерживает:
- --only-new  — добавляет только новые чанки (по text_hash) и не трогает уже проиндексированные
- --recreate  — пересоздаёт коллекцию с нуля

Оптимизации:
- prefer_grpc=True для Qdrant (быстрее)
- upsert большими батчами, wait=False (не ждём каждую запись)
- единый requests.Session с ретраями к Ollama

Зависимости:
  pip install qdrant-client tqdm requests

Пример:
  python chunk_and_index.py \
    --pages-glob "data/*.pages.jsonl" \
    --collection med_kb_v3 \
    --qdrant-url http://localhost:7777 \
    --ollama-url http://localhost:11434 \
    --emb-model zylonai/multilingual-e5-large \
    --batch 256 --only-new
"""

from __future__ import annotations
import argparse
import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple, Optional

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

# ----------------- Глобальные -----------------

seen_hashes_global: Set[str] = set()


# ----------------- УТИЛИТЫ ТЕКСТА -----------------

def normalize_for_hash(text: str) -> str:
    """Нормализация текста для хэширования (совместимо с прежними запусками)."""
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^0-9a-zа-яё %/.,;:()\-\n]+", " ", t, flags=re.IGNORECASE)
    return t.strip()


def words(text: str) -> List[str]:
    return (text or "").split()


def chunk_words(tokens: List[str], max_len: int, overlap: int) -> List[List[str]]:
    """Разбивает список токенов на перекрывающиеся чанки (скользящее окно)."""
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
    parent_w: int,
) -> List[Dict[str, Any]]:
    """Возвращает список child-чанков c полями: text, parent_id, page_range."""
    parents: List[Dict[str, Any]] = []
    cur_words: List[str] = []
    cur_pages: List[int] = []

    for p in pages:
        w = words(p.get("text", ""))
        if not w:
            continue
        for i in range(0, len(w), parent_w):
            part = w[i : i + parent_w]
            cur_words.extend(part)
            cur_pages.append(int(p.get("page", 0)))
            if len(cur_words) >= parent_w:
                pid = f"P{len(parents) + 1}"
                parents.append(
                    {
                        "parent_id": pid,
                        "text": " ".join(cur_words[:parent_w]),
                        "page_range": (min(cur_pages), max(cur_pages)),
                    }
                )
                cur_words, cur_pages = [], []

    if cur_words:
        pid = f"P{len(parents) + 1}"
        parents.append(
            {
                "parent_id": pid,
                "text": " ".join(cur_words),
                "page_range": (
                    min(cur_pages) if cur_pages else 0,
                    max(cur_pages) if cur_pages else 0,
                ),
            }
        )

    childs: List[Dict[str, Any]] = []
    for parent in parents:
        toks = words(parent["text"])
        for _, win in enumerate(chunk_words(toks, child_w, child_overlap), start=1):
            childs.append(
                {
                    "parent_id": parent["parent_id"],
                    "text": " ".join(win),
                    "page_range": parent["page_range"],
                }
            )
    return childs


# ----------------- OLLAMA EMBEDDINGS -----------------

def l2_unit(vec: List[float]) -> List[float]:
    s = (sum(x * x for x in vec) ** 0.5) or 1.0
    return [x / s for x in vec]


def make_session(total_retries: int = 5, backoff: float = 0.5, timeout: int = 60) -> requests.Session:
    """HTTP сессия с ретраями для Ollama."""
    sess = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=16, pool_maxsize=32)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.request_timeout = timeout  # просто для хранения
    return sess


def _ollama_embed_one(session: requests.Session, ollama_url: str, model: str, text: str, timeout: Optional[int]=None) -> List[float]:
    """Один вызов /api/embeddings (через общий Session)."""
    url = ollama_url.rstrip("/") + "/api/embeddings"
    payload = {"model": model, "prompt": text}
    r = session.post(url, json=payload, timeout=timeout or getattr(session, "request_timeout", 60))
    r.raise_for_status()
    data = r.json()
    vec = data.get("embedding") or []
    if not isinstance(vec, list) or not vec:
        raise RuntimeError("Empty embedding vector")
    return l2_unit([float(x) for x in vec])


def get_ollama_vector_dim(session: requests.Session, ollama_url: str, model: str) -> int:
    """Определяет размерность вектора по одному пробному эмбеддингу."""
    vec = _ollama_embed_one(session, ollama_url, model, "dimension probe")
    return len(vec)


def embed_texts_ollama(session: requests.Session, ollama_url: str, model: str, texts: List[str]) -> List[List[float]]:
    """
    Получить эмбеддинги для списка текстов.
    Ollama embeddings, как правило, принимает по одному prompt,
    поэтому вызываем поштучно (но с сессией и ретраями).
    """
    out: List[List[float]] = []
    for t in texts:
        out.append(_ollama_embed_one(session, ollama_url, model, t))
    return out


# ----------------- QDRANT -----------------

def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool = False) -> None:
    """Создаёт коллекцию Qdrant, если её нет (или пересоздаёт)."""
    if recreate and client.collection_exists(name):
        print(f"⚠️ Пересоздаём коллекцию {name}...")
        client.delete_collection(name)
        import time as _t; _t.sleep(2)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def fetch_existing_hashes(client: QdrantClient, collection: str, doc_id: str) -> set[str]:
    """Возвращает множество text_hash, уже присутствующих в коллекции для doc_id."""
    existing: set[str] = set()
    scroll_filter = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )
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
            if isinstance(th, str) and th:
                existing.add(th)
        if next_offset is None:
            break
    return existing


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    """Разбивает iterable на батчи."""
    batch: List[Any] = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ----------------- MAIN -----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-glob", required=True)
    ap.add_argument("--qdrant-url", default="http://localhost:7777")
    ap.add_argument("--collection", default="med_kb_v3")
    ap.add_argument("--child-w", type=int, default=200)
    ap.add_argument("--child-overlap", type=int, default=40)
    ap.add_argument("--parent-w", type=int, default=800)
    ap.add_argument("--batch", type=int, default=256)  # больше батч = быстрее
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--only-new", action="store_true")
    ap.add_argument("--qdrant-wait", action="store_true", help="Ждать подтверждения upsert (по умолчанию False)")

    # Ollama embeddings настройки
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--emb-model", default="zylonai/multilingual-e5-large")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout к Ollama")

    args = ap.parse_args()

    files = sorted(Path().glob(args.pages_glob))
    if not files:
        raise SystemExit(f"❌ Не найдено файлов по маске: {args.pages_glob}")

    # Единая HTTP-сессия для Ollama
    session = make_session(timeout=args.timeout)

    # Узнаём размерность эмбеддинга у Ollama-модели (один пробный вызов)
    dim = get_ollama_vector_dim(session, args.ollama_url, args.emb_model)
    print(f"🔤 Ollama embeddings: {args.emb_model} (dim={dim}) @ {args.ollama_url}")

    # Qdrant (gRPC быстрее)
    client = QdrantClient(url=args.qdrant_url, prefer_grpc=True)
    ensure_collection(client, args.collection, dim=dim, recreate=args.recreate)

    total_points_upserted = 0
    total_childs_after_dedup = 0

    for fp in files:
        doc_id = fp.stem.replace(".pages", "")
        pages: List[Dict[str, Any]] = []
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pages.append({"page": rec.get("page", 0), "text": rec.get("text", "")})

        childs = build_parent_child(pages, args.child_w, args.child_overlap, args.parent_w)
        if not childs:
            print(f"📄 {doc_id}: нет текста для индексации (пропуск)")
            continue

        # Добавляем text_hash каждому чанку (совместимо со старыми)
        for c in childs:
            c["text_hash"] = hashlib.sha1(normalize_for_hash(c["text"]).encode("utf-8")).hexdigest()

        # Пропускаем уже существующие чанки (в коллекции по этому doc_id)
        if args.only_new and not args.recreate:
            existing_hashes = fetch_existing_hashes(client, args.collection, doc_id)
        else:
            existing_hashes = set()

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

        print(f"📄 {doc_id}: новых чанков {len(childs)}, уже существующих {skipped_existing}")
        if not childs:
            continue

        # Эмбеддинг и апсерты батчами
        for chunk in tqdm(batched(childs, args.batch), desc=f"{doc_id}"):
            texts = [c["text"] for c in chunk]
            if not texts:
                continue

            t0 = time.time()
            emb = embed_texts_ollama(session, args.ollama_url, args.emb_model, texts)
            t1 = time.time()

            points: List[PointStruct] = []
            for i, c in enumerate(chunk):
                unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{c['text_hash']}"))
                payload = {
                    "doc_id": doc_id,
                    "parent_id": c["parent_id"],
                    "page_start": c["page_range"][0],
                    "page_end": c["page_range"][1],
                    "len_words": len(c["text"].split()),
                    "text_hash": c["text_hash"],
                }
                points.append(PointStruct(id=unique_id, vector=emb[i], payload=payload))

            # быстрый upsert: крупные батчи, gRPC, wait=False по умолчанию
            client.upsert(
                collection_name=args.collection,
                points=points,
                wait=bool(args.qdrant_wait)
            )
            t2 = time.time()

            total_points_upserted += len(points)
            print(f"    [emb {len(texts)} за {t1-t0:.2f}s] [upsert {len(points)} за {t2-t1:.2f}s]")

    print(f"📊 Файлов обработано: {len(files)}")
    print(f"🧩 Новых чанков: {total_childs_after_dedup}")
    print(f"📦 Добавлено/обновлено векторов в Qdrant: {total_points_upserted}")
    print("✅ Индексация в Qdrant завершена.")


if __name__ == "__main__":
    main()
