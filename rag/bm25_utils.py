# -*- coding: utf-8 -*-
"""
BM25 / Hybrid Retrieval утилиты для Med_ai.

Публичные функции:
  - bm25_search(index_dir: str, query: str, topk: int) -> List[dict]
  - embed_query_hf(query: str, model_name: str, device_hint: Optional[str], use_fp16: bool) -> List[float]
  - retrieve_hybrid(query: str, k: int, ..., per_doc_limit: int,
                    reranker_enabled: bool=False, rerank_top_k: int=50) -> List[dict]

Ключевые правки:
- Эмбеддинг запроса теперь через тот же BGEM3FlagModel (FlagEmbedding), что и при индексации.
- Qdrant возвращает кандидатов на уровне ЧАНКОВ (chunk_id + chunk_text из payload).
- RRF фьюжн выполняется по chunk_id, а не по doc_id.
- per_doc_limit применяется ПОСЛЕ фьюжна; берём действительно лучшие чанки.
- Сохранены бэкапы при отсутствии chunk_text (подтягиваем текст страниц из data/*.pages.jsonl).
"""

from __future__ import annotations

import json
import os
import re
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------- Pyserini (Lucene) ----------
try:
    from pyserini.search.lucene import LuceneSearcher  # pyserini>=0.20
except Exception:
    LuceneSearcher = None

# ---------- Qdrant ----------
try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None  # обработаем ниже

# ---------- FlagEmbedding (BGEM3) ----------
try:
    from FlagEmbedding import BGEM3FlagModel
except Exception as e:
    BGEM3FlagModel = None  # дадим понятную ошибку при вызове

# -------------------- Константы --------------------
_TEXT_SNIPPET_LIMIT = int(
    os.getenv("CTX_SNIPPET_LIMIT")
    or os.getenv("TEXT_SNIPPET_LIMIT")
    or os.getenv("MEDAI_TEXT_SNIPPET_LIMIT", "3500")
)

_PER_DOC_LIMIT_DEFAULT = int(os.getenv("RETR_PER_DOC_LIMIT", "2"))

# -------------------- Утилиты --------------------


def _normalize_qdrant_url(url_in: Optional[str]) -> str:
    """
    Приводим значение переменной/аргумента к валидному виду:
      - если пусто -> берем из env: QDRANT_URL или QDRANT
      - если без схемы (нет '://') -> подставляем http://
      - если что-то вроде 'qdrant:6333' или 'qdrant://qdrant:6333' -> переводим в http://qdrant:6333
    Разрешенные схемы: http, https, grpc, grpcs.
    """
    url = (url_in or os.getenv("QDRANT_URL") or os.getenv("QDRANT") or "http://qdrant:6333").strip()

    if "://" not in url:
        return f"http://{url}"

    if url.lower().startswith("qdrant://"):
        return "http://" + url[len("qdrant://"):]

    if url.lower().startswith("qdrant:"):
        return "http://" + url[len("qdrant:"):]

    return url


@lru_cache(maxsize=4)
def _get_lucene_searcher(index_dir: str):
    if LuceneSearcher is None:
        raise RuntimeError("Pyserini/LuceneSearcher не установлен. Проверь requirements и Dockerfile.")
    idx = Path(index_dir)
    if not idx.exists():
        raise FileNotFoundError(f"BM25 индекс не найден: {idx}")
    s = LuceneSearcher(index_dir)
    try:
        s.set_bm25(k1=0.9, b=0.4)
    except Exception:
        pass
    return s


@lru_cache(maxsize=4)
def _load_bm25_meta(index_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Строит маппинг id чанка -> {doc_id, page, text} из JSONL в index/bm25_json/*.json.
    """
    base = Path(index_dir)
    meta_dir = base.with_name("bm25_json")
    mapping: Dict[str, Dict[str, Any]] = {}
    if not meta_dir.exists():
        print(f"⚠️ bm25_json not found near {index_dir} — will rely on docid parsing/raw")
        return mapping

    for p in meta_dir.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    j = json.loads(line)
                    _id = j.get("id")
                    contents = j.get("contents", "") or ""

                    # raw -> внутри meta: {"doc_id","page","child_idx"}
                    doc_id, page = "unknown", 1
                    raw_meta = j.get("raw")
                    if isinstance(raw_meta, str):
                        try:
                            rm = json.loads(raw_meta)
                            doc_id = rm.get("doc_id", doc_id)
                            page = int(rm.get("page", page))
                        except Exception:
                            pass
                    elif isinstance(raw_meta, dict):
                        doc_id = raw_meta.get("doc_id", doc_id)
                        page = int(raw_meta.get("page", page))

                    if _id:
                        mapping[str(_id)] = {
                            "doc_id": str(doc_id),
                            "page": int(page),
                            "text": str(contents)[:_TEXT_SNIPPET_LIMIT],
                        }
        except Exception:
            continue
    return mapping


def _hit_to_record(searcher, hit, meta_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Возвращает BM25-кандидата уровня чанка.
    """
    id_str = getattr(hit, "docid", None)
    if id_str and id_str in meta_map:
        m = meta_map[id_str]
        return {
            "doc_id": m["doc_id"],
            "page": int(m["page"] or 1),
            "text": (m["text"] or "")[:_TEXT_SNIPPET_LIMIT],
            "chunk_id": str(id_str),
        }

    rec: Dict[str, Any] = {"doc_id": "unknown", "page": 1, "text": "", "chunk_id": str(id_str) if id_str else "unknown"}

    try:
        doc = searcher.doc(hit.docid)
        raw = doc.raw() if callable(getattr(doc, "raw", None)) else None
        if raw:
            j = json.loads(raw)  # объект {"id","contents","raw": "...meta..."}
            text = j.get("contents") or ""
            doc_id = j.get("id") or "unknown"
            page = 1

            raw_meta = j.get("raw")
            if isinstance(raw_meta, str):
                try:
                    rm = json.loads(raw_meta)
                    doc_id = rm.get("doc_id", doc_id)
                    page = int(rm.get("page", page))
                except Exception:
                    pass
            elif isinstance(raw_meta, dict):
                doc_id = raw_meta.get("doc_id", doc_id)
                page = int(raw_meta.get("page", page))

            if not text:
                contents = doc.contents() if callable(getattr(doc, "contents", None)) else None
                if contents:
                    text = str(contents)

            return {
                "doc_id": str(doc_id),
                "page": int(page),
                "text": str(text)[:_TEXT_SNIPPET_LIMIT],
                "chunk_id": str(id_str) if id_str else str(doc_id),
            }

        # fallback: хотя бы contents
        contents = doc.contents() if callable(getattr(doc, "contents", None)) else None
        if contents:
            rec["text"] = str(contents)[:_TEXT_SNIPPET_LIMIT]
    except Exception:
        pass

    # последний вариант — парсим id вида "doc_p12_c3"
    if id_str:
        m = re.match(r"^(?P<doc>.+)_p(?P<page>\d+)_c\d+$", str(id_str))
        if m:
            rec["doc_id"] = m.group("doc")
            try:
                rec["page"] = int(m.group("page"))
            except Exception:
                rec["page"] = 1
        else:
            rec["doc_id"] = str(id_str)

    return rec


# -------------------- Embeddings (FlagEmbedding BGEM3) --------------------


@lru_cache(maxsize=3)
def _get_flag_model(model_name: str, device_hint: Optional[str], use_fp16: bool):
    if BGEM3FlagModel is None:
        raise RuntimeError("FlagEmbedding (BGEM3FlagModel) не установлен. Установите пакет FlagEmbedding.")
    dev = (device_hint or "auto").strip().lower()
    if dev == "cpu":
        device = "cpu"
        fp16 = False
    elif dev.startswith("cuda") or dev == "gpu" or dev == "auto":
        # пусть BGEM3 сам разберётся; fp16 включаем только если cuda
        device = "cuda" if dev != "cpu" else "cpu"
        fp16 = bool(use_fp16 and device == "cuda")
    else:
        device = "cuda" if dev != "cpu" else "cpu"
        fp16 = bool(use_fp16 and device == "cuda")
    return BGEM3FlagModel(model_name, use_fp16=fp16, device=device)


def _encode_texts_flag(texts: List[str], model_name: str, device_hint: Optional[str], use_fp16: bool,
                       batch_size: int = 128) -> np.ndarray:
    m = _get_flag_model(model_name, device_hint, use_fp16)
    try:
        out = m.encode(
            texts,
            batch_size=max(1, batch_size),
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            normalize_embeddings=True,
        )
    except TypeError:
        out = m.encode(texts, batch_size=max(1, batch_size))
    vecs = None
    if isinstance(out, dict) and out.get("dense_vecs") is not None:
        vecs = out["dense_vecs"]
    elif isinstance(out, (list, tuple)):
        vecs = out
    if vecs is None:
        raise RuntimeError("BGEM3 encode() вернул неожиданный формат.")
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # уже L2-нормировано (normalize_embeddings=True), но на всякий:
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return (arr / norms).astype(np.float32)


def embed_query_hf(
    query: str,
    model_name: str = "BAAI/bge-m3",
    device_hint: Optional[str] = None,
    use_fp16: bool = False,
) -> List[float]:
    """
    Совместимая сигнатура, но теперь — BGEM3FlagModel (как при индексации).
    """
    if not query:
        return []
    return _encode_texts_flag([query], model_name, device_hint, use_fp16, batch_size=16)[0].tolist()


# -------------------- Qdrant helpers --------------------


@lru_cache(maxsize=2)
def _qdrant_client(url: str):
    if QdrantClient is None:
        raise RuntimeError("qdrant-client не установлен.")
    prefer_grpc = (os.getenv("QDRANT__PREFER_GRPC", "false").lower() == "true")
    return QdrantClient(url=url, prefer_grpc=prefer_grpc)


def _canon_doc_id(pl: Dict[str, Any], fallback: Optional[str] = None) -> str:
    for key in ("doc_id", "id", "source", "file", "document_id", "doc", "name"):
        v = pl.get(key)
        if v:
            return str(v)
    return fallback or "unknown"


def _rrf_fusion(runs: List[List[str]], k: int = 60, c: int = 60) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion по идентификаторам ЧАНКОВ.
    """
    scores: Dict[str, float] = {}
    for run in runs:
        for rank, cid in enumerate(run[:k], start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (c + rank)
    return scores


def _load_pages_text(pages_dir: Path, doc_id: str, a: int, b: int) -> str:
    """
    Собирает текст из data/{doc_id}.pages.jsonl по страницам a..b с лимитом длины.
    """
    pj = pages_dir / f"{doc_id}.pages.jsonl"
    if not pj.exists():
        return ""
    want = set(range(int(a), int(b) + 1))
    out: List[str] = []
    try:
        with pj.open("r", encoding="utf-8") as r:
            for line in r:
                j = json.loads(line)
                p = int(j.get("page") or 1)
                if p in want:
                    t = j.get("text") or ""
                    if t:
                        out.append(str(t))
        return "\n\n".join(out)[:_TEXT_SNIPPET_LIMIT]
    except Exception:
        return ""


# -------------------- Публичные функции --------------------


def bm25_search(index_dir: str, query: str, topk: int = 50) -> List[Dict[str, Any]]:
    """
    Поиск по Lucene (Pyserini). Возвращает список словарей уровня чанка:
    { "doc_id": str, "page": int, "text": str, "chunk_id": str }
    """
    if not query:
        return []
    try:
        s = _get_lucene_searcher(index_dir)
    except Exception as e:
        print(f"⚠️ BM25 недоступен: {e}")
        return []

    meta = _load_bm25_meta(index_dir)
    try:
        hits = s.search(query, topk)
    except Exception as e:
        print(f"⚠️ BM25 search error: {e}")
        return []

    out: List[Dict[str, Any]] = []
    for h in hits:
        rec = _hit_to_record(s, h, meta)
        # добавим приблизительный score, если есть
        try:
            rec["_bm25_score"] = float(getattr(h, "score", 0.0))
        except Exception:
            pass
        out.append(rec)
    return out


def retrieve_hybrid(
    query: str,
    k: int,
    *,
    bm25_index_dir: str = "index/bm25_idx",
    qdrant_url: str = "http://qdrant:6333",
    qdrant_collection: str = "med_kb_v3",
    pages_dir: str = "data",
    hf_model: str = "BAAI/bge-m3",
    hf_device: Optional[str] = None,
    hf_fp16: bool = False,
    per_doc_limit: int = _PER_DOC_LIMIT_DEFAULT,
    reranker_enabled: bool = False,
    rerank_top_k: int = 50,
) -> List[Dict[str, Any]]:
    """
    Гибридный поиск: Qdrant (dense, на уровне чанков) + BM25 (sparse, чанки) + RRF.
    Возвращает до k фрагментов вида:
      { "doc_id": str, "page_start": int, "page_end": int, "text": str }
    """
    results: List[Dict[str, Any]] = []
    if k <= 0 or not query:
        return results

    qdrant_url = _normalize_qdrant_url(qdrant_url)
    pages_dir_p = Path(pages_dir)

    # ---------- 1) Qdrant (dense) -> чанки ----------
    dense_runs_chunk_ids: List[str] = []
    dense_by_chunk: Dict[str, Dict[str, Any]] = {}
    try:
        q_vec = embed_query_hf(query, model_name=hf_model, device_hint=hf_device, use_fp16=hf_fp16)
        client = _qdrant_client(qdrant_url)
        pts = client.query_points(
            collection_name=qdrant_collection,
            query=q_vec,
            limit=max(k * 8, 80)  # побольше кандидатов, потом урежем
        ).points

        for p in pts:
            pl = p.payload or {}
            doc_id = _canon_doc_id(pl, fallback="unknown")
            a = int(pl.get("page_start", 1) or 1)
            b = int(pl.get("page_end", a) or a)
            chunk_id = str(pl.get("chunk_id") or pl.get("text_hash") or f"{doc_id}:{a}-{b}")
            text = (pl.get("chunk_text") or "").strip()

            # fallback если нет chunk_text (старый индекс)
            if not text:
                text = _load_pages_text(pages_dir_p, doc_id, a, b)

            text = (text or "")[:_TEXT_SNIPPET_LIMIT]
            item = {
                "doc_id": doc_id,
                "page_start": a,
                "page_end": b,
                "text": text,
                "_score_dense": float(getattr(p, "score", 0.0)),
                "chunk_id": chunk_id,
            }
            if chunk_id not in dense_by_chunk:
                dense_by_chunk[chunk_id] = item
                dense_runs_chunk_ids.append(chunk_id)
    except Exception as e:
        print("⚠️ Qdrant retrieve error:", e)

    # ---------- 2) BM25 (sparse) -> чанки ----------
    bm25_runs_chunk_ids: List[str] = []
    bm25_by_chunk: Dict[str, Dict[str, Any]] = {}
    try:
        hits = bm25_search(bm25_index_dir, query, topk=max(k * 8, 80))
        for h in hits:
            doc_id = str(h.get("doc_id") or "unknown")
            page = int(h.get("page") or 1)
            text = (h.get("text") or "")[:_TEXT_SNIPPET_LIMIT]
            chunk_id = str(h.get("chunk_id") or f"{doc_id}_p{page}")
            rec = {
                "doc_id": doc_id,
                "page_start": page,
                "page_end": page,
                "text": text,
                "_bm25_score": float(h.get("_bm25_score", 0.0)),
                "chunk_id": chunk_id,
            }
            if chunk_id not in bm25_by_chunk:
                bm25_by_chunk[chunk_id] = rec
                bm25_runs_chunk_ids.append(chunk_id)
    except Exception as e:
        print("⚠️ BM25 retrieve error:", e)

    # ---------- 3) RRF по chunk_id ----------
    fused_scores = _rrf_fusion([dense_runs_chunk_ids, bm25_runs_chunk_ids], k=60, c=60)
    # объединим мета: dense приоритетнее, затем bm25
    pool: Dict[str, Dict[str, Any]] = {}
    for cid in set(list(dense_by_chunk.keys()) + list(bm25_by_chunk.keys())):
        base = dense_by_chunk.get(cid) or bm25_by_chunk.get(cid)
        if base:
            pool[cid] = dict(base)

    # сортируем чанки по fused score
    sorted_chunk_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    # ---------- 4) Применяем per_doc_limit и собираем кандидатов ----------
    used_per_doc: Dict[str, int] = {}
    candidates: List[Dict[str, Any]] = []
    for cid in sorted_chunk_ids:
        info = pool.get(cid)
        if not info:
            continue
        did = info.get("doc_id", "unknown")
        if used_per_doc.get(did, 0) >= max(1, per_doc_limit):
            continue
        # если вдруг текст пустой — подгрузим страницы
        if not info.get("text"):
            a, b = int(info.get("page_start", 1)), int(info.get("page_end", 1))
            info["text"] = _load_pages_text(pages_dir_p, did, a, b)[:_TEXT_SNIPPET_LIMIT]
        if not info.get("text"):
            continue
        candidates.append({
            "doc_id": did,
            "page_start": int(info.get("page_start", 1)),
            "page_end": int(info.get("page_end", info.get("page_start", 1))),
            "text": str(info.get("text") or "")[:_TEXT_SNIPPET_LIMIT],
        })
        used_per_doc[did] = used_per_doc.get(did, 0) + 1
        if len(candidates) >= k:
            break

    # ---------- 5) Опциональный переранкер (cosine по BGEM3) ----------
    if reranker_enabled and candidates:
        try:
            n = min(rerank_top_k, len(candidates))
            if n > 0:
                q = np.asarray(embed_query_hf(query, model_name=hf_model, device_hint=hf_device, use_fp16=hf_fp16),
                               dtype=np.float32).reshape(1, -1)
                texts = [c["text"] for c in candidates[:n]]
                t = _encode_texts_flag(texts, model_name=hf_model, device_hint=hf_device, use_fp16=hf_fp16,
                                       batch_size=64)
                # cosine(q, T)
                sims = (q @ t.T).ravel().tolist()
                for i, s in enumerate(sims):
                    candidates[i]["_rerank_score"] = float(s)
                head = sorted(candidates[:n], key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
                tail = candidates[n:]
                candidates = head + tail
        except Exception as e:
            print("⚠️ Reranker (cosine/BGEM3) error:", e)
        finally:
            for it in candidates:
                it.pop("_rerank_score", None)

    return candidates[:k]
