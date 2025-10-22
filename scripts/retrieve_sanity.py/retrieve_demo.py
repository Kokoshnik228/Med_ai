#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from typing import Dict, Any, List, Tuple

from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel
from pyserini.search.lucene import LuceneSearcher

def l2_unit(v: List[float]) -> List[float]:
    s = (sum(x*x for x in v) ** 0.5) or 1.0
    return [x / s for x in v]

def rrf_fusion(rank_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ranks in rank_lists:
        for r, did in enumerate(ranks, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + r)
    return scores

def qdrant_search(client: QdrantClient, model: BGEM3FlagModel, query: str, collection: str, topk: int) -> List[Tuple[str, Dict[str, Any]]]:
    vec = model.encode([query], return_dense=True, return_sparse=False)["dense_vecs"][0]
    vec = l2_unit(vec)
    res = client.query_points(collection_name=collection, query=vec, limit=topk).points
    out: List[Tuple[str, Dict[str, Any]]] = []
    for p in res:
        payload = p.payload or {}
        did = payload.get("doc_id", "unknown")
        out.append((did, payload))
    return out

def bm25_search(searcher: LuceneSearcher, query: str, topk: int) -> List[Tuple[str, int | None, str]]:
    hits = searcher.search(query, k=topk)
    out: List[Tuple[str, int | None, str]] = []
    for h in hits:
        doc = searcher.doc(h.docid)
        contents = doc.contents()
        ext_id = doc.id()  # e.g. "{doc_id}_p{page}_c{chunk}"
        did, page = None, None
        m = re.match(r"(.+?)_p(\d+)_c(\d+)$", ext_id)
        if m:
            did, page = m.group(1), int(m.group(2))
        else:
            raw = doc.raw()
            if raw:
                try:
                    meta = json.loads(raw)
                    did = meta.get("doc_id")
                    page = meta.get("page")
                except Exception:
                    pass
        out.append((did or ext_id, page, contents))
    return out

def main():
    ap = argparse.ArgumentParser(description="Hybrid retrieval demo: Qdrant + BM25 → RRF")
    ap.add_argument("--q", required=True, help="Текст запроса")
    ap.add_argument("--qdrant", default="http://localhost:7777")
    ap.add_argument("--collection", default="med_kb")
    ap.add_argument("--bm25-index", default="index/bm25_idx")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--kq", type=int, default=50)
    ap.add_argument("--kb", type=int, default=50)
    args = ap.parse_args()

    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    client = QdrantClient(url=args.qdrant)
    searcher = LuceneSearcher(args.bm25_index)

    qd = qdrant_search(client, model, args.q, args.collection, args.kq)           # [(did, payload)]
    bm = bm25_search(searcher, args.q, args.kb)                                   # [(did, page, contents)]

    qd_docids = [d for d, _ in qd]
    bm_docids = [d for d, _, _ in bm]
    fused = rrf_fusion([qd_docids, bm_docids])

    # Быстрые маппинги для вывода
    qd_map: Dict[str, Dict[str, Any]] = {d: pl for d, pl in qd if d not in {}}
    bm_map: Dict[str, Tuple[int | None, str]] = {}
    for d, page, contents in bm:
        if d not in bm_map:  # первый пример
            bm_map[d] = (page, contents)

    top_docs = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:args.k]

    print("\n=== FUSED TOP DOCS ===")
    for rank, (did, score) in enumerate(top_docs, start=1):
        page = bm_map.get(did, (None, ""))[0]
        qd_payload = qd_map.get(did)
        print(f"{rank:02d}. doc_id={did}  RRF={score:.4f}  page={page}  qdrant_payload={qd_payload}")

if __name__ == "__main__":
    main()
