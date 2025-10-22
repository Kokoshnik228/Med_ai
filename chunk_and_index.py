#!/usr/bin/env python3
"""
Шаг 6. Чанкинг + индексация в Qdrant (dense, BGE‑M3)

Берёт выход из шага ingest (data/*.pages.jsonl), собирает parent/child чанки,
считает эмбеддинги BGE‑M3 (через FlagEmbedding) и загружает child‑чанки в Qdrant.

Коллекция по умолчанию: med_kb (size=1024, cosine).

Зависимости:
  pip install FlagEmbedding qdrant-client tqdm

Запуск (GPU):
  python chunk_and_index.py --pages-glob "data/*.pages.jsonl" \
      --collection med_kb --qdrant-url http://localhost:7777 \
      --device cuda --batch 16

Если не хватает памяти на GPU — уменьшай --batch (например, 8).
"""

from __future__ import annotations
import hashlib, re
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Iterable


from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Эмбеддинги BGE‑M3
try:
    from FlagEmbedding import BGEM3FlagModel
except Exception as e:
    raise RuntimeError("Требуется пакет FlagEmbedding. Установи: pip install FlagEmbedding") from e
seen_hashes_global:set[str] = set()

# ----------------- УТИЛИТЫ -----------------


def normalize_for_hash(text: str) -> str:
    # простая нормализация: нижний регистр, убираем не-буквенно-цифровые, схлопываем пробелы
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^0-9a-zа-яё %/.,;:()\\-]+", " ", t, flags=re.IGNORECASE)
    return t.strip()


def l2_unit(v: List[float]) -> List[float]:
    """L2-нормализация вектора."""
    s = (sum(x * x for x in v) ** 0.5) or 1.0
    return [x / s for x in v]


def words(text: str) -> List[str]:
    return text.split()


def chunk_words(tokens: List[str], max_len: int, overlap: int) -> List[List[str]]:
    chunks: List[List[str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        j = min(i + max_len, n)
        chunks.append(tokens[i:j])
        if j == n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return chunks


def build_parent_child(pages: List[Dict[str, Any]], child_w: int, child_overlap: int, parent_w: int) -> List[Dict[str, Any]]:
    """Возвращает список child‑чанков с полями: text, parent_id, page_range."""
    # Собираем parent-блоки ~parent_w слов
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

    # Нарезаем каждого родителя на child-окна
    childs: List[Dict[str, Any]] = []
    for parent in parents:
        toks = words(parent["text"])
        for idx, win in enumerate(chunk_words(toks, child_w, child_overlap), start=1):
            childs.append(
                {
                    "parent_id": parent["parent_id"],
                    "text": " ".join(win),
                    "page_range": parent["page_range"],
                }
            )
    return childs


# ----------------- Индексация -----------------

def ensure_collection(client: QdrantClient, name: str, dim: int = 1024) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-glob", required=True, help="Путь-глоб к *.pages.jsonl (например, data/*.pages.jsonl)")
    ap.add_argument("--qdrant-url", default="http://localhost:7777")
    ap.add_argument("--collection", default="med_kb")
    ap.add_argument("--child-w", type=int, default=200, help="Размер child чанка (в словах)")
    ap.add_argument("--child-overlap", type=int, default=40, help="Перекрытие child (в словах)")
    ap.add_argument("--parent-w", type=int, default=800, help="Размер parent блока (в словах)")
    ap.add_argument("--batch", type=int, default=64, help="Размер батча при эмбеддинге/апсерте")
    ap.add_argument("--device", default="auto", help="cuda / cpu / auto")
    args = ap.parse_args()

    files = sorted(Path().glob(args.pages_glob))
    if not files:
        raise SystemExit(f"Не найдено файлов по маске: {args.pages_glob}")

    # Модель эмбеддингов
    device = None if args.device == "auto" else args.device
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)

    client = QdrantClient(url=args.qdrant_url)
    ensure_collection(client, args.collection, dim=1024)

    point_id = 0
    for fp in files:
        doc_id = fp.stem.replace(".pages", "")
        pages: List[Dict[str, Any]] = []
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pages.append({"page": rec.get("page", 0), "text": rec.get("text", "")})

        childs = build_parent_child(pages, args.child_w, args.child_overlap, args.parent_w)
# --- дедупликация: убираем повторяющиеся чанки по нормализованному тексту ---
        filtered = []
        for c in childs:
          h = hashlib.sha1(normalize_for_hash(c["text"]).encode("utf-8")).hexdigest()
          if h in seen_hashes_global:
             continue
        seen_hashes_global.add(h)
        c["text_hash"] = h
        filtered.append(c)
        childs = filtered
# --- конец дедупликации ---

        # Эмбеддим и записываем батчами
        for chunk in tqdm(batched(childs, args.batch), desc=f"{doc_id}"):
            texts = [c["text"] for c in chunk]
            emb = model.encode(
                texts,
                batch_size=args.batch,
                max_length=8192,
                return_dense=True,
                return_sparse=False,
            )["dense_vecs"]
            emb = [l2_unit(v) for v in emb]  # L2-нормализация вручную

            points: List[PointStruct] = []
            for i, c in enumerate(chunk):
                point_id += 1
                payload = {
                    "doc_id": doc_id,
                    "parent_id": c["parent_id"],
                    "page_start": c["page_range"][0],
                    "page_end": c["page_range"][1],
                    "len_words": len(c["text"].split()),
                    "text_hash": c.get("text_hash"),
                }
                points.append(PointStruct(id=point_id, vector=emb[i], payload=payload))
            client.upsert(collection_name=args.collection, points=points)

    print("Готово: индексация в Qdrant завершена.")


if __name__ == "__main__":
    main()
