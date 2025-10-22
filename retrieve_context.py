#!/usr/bin/env python3
"""
Шаг 8 (исправлено). Гибридный ретривер контекста для LLM
— устойчив к отсутствию текста/метаданных: всегда возвращает строку.
Qdrant (dense, BGE-M3) + BM25 (Pyserini) → RRF по CHUNK_ID, возврат top-k фрагментов.

Выход: JSON-массив объектов [{doc_id, page_start, page_end, text}, ...]
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel
from pyserini.search.lucene import LuceneSearcher


# ----------------- utils -----------------

def l2_unit(v: List[float]) -> List[float]:
    s = (sum(x * x for x in v) ** 0.5) or 1.0
    return [x / s for x in v]


def rrf_fusion(rank_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion по спискам chunk_id."""
    scores: Dict[str, float] = {}
    for ranks in rank_lists:
        for r, cid in enumerate(ranks, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r)
    return scores


# ----------------- search backends -----------------

def qdrant_search(
    client: QdrantClient,
    model: BGEM3FlagModel,
    query: str,
    collection: str,
    topk: int,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Возвращает список кортежей: (chunk_id, doc_id, payload)
    chunk_id формируем как {doc_id}_p{page_start}_e{page_end}.
    """
    vec = model.encode([query], return_dense=True, return_sparse=False)["dense_vecs"][0]
    vec = l2_unit(vec)

    # ВНИМАНИЕ: если в коллекции именованный вектор ("dense"), используйте query_vector=("dense", vec)
    res = client.query_points(collection_name=collection, query=vec, limit=topk).points

    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for p in res:
        payload = p.payload or {}
        did = payload.get("doc_id", "unknown")
        ps = int(payload.get("page_start", 1) or 1)
        pe = int(payload.get("page_end", ps) or ps)
        chunk_id = f"{did}_p{ps}_e{pe}"
        out.append((chunk_id, did, payload))
    return out


def bm25_search(
    searcher: LuceneSearcher,
    query: str,
    topk: int,
) -> List[Tuple[str, str, Optional[int], str]]:
    """
    Возвращает список кортежей: (chunk_id, doc_id, page, text)
    chunk_id берём как doc.id() из Pyserini (обычно '{doc_id}_p{page}_c{chunk}').
    """
    hits = searcher.search(query, k=topk)
    out: List[Tuple[str, str, Optional[int], str]] = []
    for h in hits:
        doc = searcher.doc(h.docid)
        contents = doc.contents() or ""
        ext_id = doc.id()  # это уникальный chunk_id в индексе

        # Безопасно выделяем страницу с конца строки
        did: Optional[str] = None
        page: Optional[int] = None
        m = re.search(r"_p(\d+)_c(\d+)$", ext_id)
        if m:
            page = int(m.group(1))
            did = ext_id[: m.start()]  # всё до '_p{page}_c{chunk}'
        else:
            raw = doc.raw()
            if raw:
                try:
                    meta = json.loads(raw)
                    did = meta.get("doc_id")
                    page = meta.get("page")
                except Exception:
                    pass

        out.append((ext_id, did or "unknown", page, contents))
    return out


# ----------------- IO -----------------

def load_pages_text(pages_dir: Path, doc_id: str, p_start: int, p_end: int) -> str:
    jf = pages_dir / f"{doc_id}.pages.jsonl"
    if not jf.exists():
        return ""
    texts: List[str] = []
    for line in jf.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        page = int(rec.get("page", 0))
        if p_start <= page <= p_end:
            texts.append(rec.get("text", "") or "")
    return "\n".join(texts)


# ----------------- main -----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid context retriever (robust)")
    ap.add_argument("--q", required=True, help="Текст запроса")
    ap.add_argument("--qdrant", default="http://localhost:7777")
    ap.add_argument("--collection", default="med_kb")
    ap.add_argument("--bm25-index", default="index/bm25_idx")
    ap.add_argument("--pages-dir", default="data")
    ap.add_argument("--k", type=int, default=8, help="Сколько фрагментов вернуть")
    ap.add_argument("--kq", type=int, default=80, help="Сколько кандидатов из Qdrant")
    ap.add_argument("--kb", type=int, default=80, help="Сколько кандидатов из BM25")
    ap.add_argument("--per-doc-limit", type=int, default=2, help="Макс. фрагментов с одного документа")
    ap.add_argument("--out", default="out/context.json")
    args = ap.parse_args()

    pages_dir = Path(args.pages_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Модели/клиенты
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    client = QdrantClient(url=args.qdrant)
    searcher = LuceneSearcher(args.bm25_index)

    # Кандидаты
    qd = qdrant_search(client, model, args.q, args.collection, args.kq)            # (cid, did, payload)
    bm = bm25_search(searcher, args.q, args.kb)                                     # (cid, did, page, text)

    # RRF по CHUNK_ID (а не по doc_id)
    qd_chunkids = [cid for cid, _, _ in qd]
    bm_chunkids = [cid for cid, _, _, _ in bm]
    fused = rrf_fusion([qd_chunkids, bm_chunkids])
    top_chunks = [cid for (cid, _) in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    # Быстрый доступ по cid
    bm_map: Dict[str, Tuple[str, str, Optional[int], str]] = {cid: (did, cid, page, contents) for cid, did, page, contents in bm}
    qd_map: Dict[str, Tuple[str, str, Dict[str, Any]]] = {cid: (did, cid, payload) for cid, did, payload in qd}

    # Сбор результатов
    results: List[Dict[str, Any]] = []
    doc_counts: Dict[str, int] = {}

    for cid in top_chunks:
        if len(results) >= args.k:
            break

        item: Dict[str, Any] = {}

        if cid in bm_map:
            did, _, page, contents = bm_map[cid]
            item.update({
                "doc_id": did,
                "page_start": page or 1,
                "page_end": page or 1,
                "text": contents or ""
            })
        elif cid in qd_map:
            did, _, payload = qd_map[cid]
            p_start = int(payload.get("page_start", 1) or 1)
            p_end = int(payload.get("page_end", p_start) or p_start)
            txt = load_pages_text(pages_dir, did, p_start, p_end) or ""
            if not txt:
                # fallback на первую страницу если диапазон пуст
                txt = load_pages_text(pages_dir, did, 1, 1) or ""
                p_start = p_end = 1
            item.update({
                "doc_id": did,
                "page_start": p_start,
                "page_end": p_end,
                "text": txt
            })
        else:
            continue

        # ограничим число фрагментов с одного документа
        if args.per_doc_limit > 0:
            if doc_counts.get(item["doc_id"], 0) >= args.per_doc_limit:
                continue
            doc_counts[item["doc_id"]] = doc_counts.get(item["doc_id"], 0) + 1

        # безопасная усечка
        t = item.get("text") or ""
        ws = t.split()
        if len(ws) > 1000:
            item["text"] = " ".join(ws[:1000]) + " …"

        results.append(item)

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: записано {len(results)} фрагментов → {args.out}")


if __name__ == "__main__":
    main()
