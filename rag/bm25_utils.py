# -*- coding: utf-8 -*-
"""
Утилиты BM25/Hybrid Retrieval для Med_ai.

ДОСТУПНЫЕ ФУНКЦИИ:
  - bm25_search(index_dir: str, query: str, topk: int) -> List[dict]
  - embed_query_hf(query: str, model_name: str, device_hint: Optional[str], use_fp16: bool) -> List[float]
  - retrieve_hybrid(query: str, k: int, ..., per_doc_limit: int,
                    reranker_enabled: bool=False, rerank_top_k: int=50) -> List[dict]

Особенности:
- BM25 на Pyserini (Lucene). Если рядом с индексом есть meta-json (index/bm25_json/*.json),
  берём оттуда doc_id/page/text. Иначе пытаемся распарсить raw/contents из Lucene или docid.
- Qdrant: лёгкий клиент + query_points().
- RRF-фьюжн списков (dense+bm25).
- Опциональный переранкер: косинусная близость между эмбеддингом запроса и текста кандидата
  через тот же HF-эмбеддер (без дополнительных зависимостей).
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# ---------- HF embedding (Transformers) ----------
import torch
from transformers import AutoTokenizer, AutoModel


# -------------------- Константы --------------------

_TEXT_SNIPPET_LIMIT = 3500   # ограничим текст фрагментов
_PER_DOC_LIMIT_DEFAULT = 2   # по умолчанию не брать слишком много чанков из одного дока


# -------------------- Утилиты --------------------

def _normalize_qdrant_url(url_in: Optional[str]) -> str:
    """
    Приводим значение переменной/аргумента к валидному виду:
      - если пусто -> берем из env: QDRANT_URL или QDRANT
      - если без схемы (нет '://') -> подставляем http://
      - если что-то вроде 'qdrant:6333' или 'qdrant://qdrant:6333' -> переводим в http://qdrant:6333
    Разрешенные схемы для клиента: http, https, grpc, grpcs.
    """
    url = (url_in or os.getenv("QDRANT_URL") or os.getenv("QDRANT") or "http://qdrant:6333").strip()

    # 'qdrant:6333' -> добавим http://
    if "://" not in url:
        return f"http://{url}"

    # 'qdrant://...' -> заменим на http://...
    if url.lower().startswith("qdrant://"):
        return "http://" + url[len("qdrant://"):]

    # 'qdrant:...' (маловероятно с ://, но на всякий) -> http
    if url.lower().startswith("qdrant:"):
        return "http://" + url[len("qdrant:"):]

    # всё ок
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
    base = Path(index_dir)
    meta_dir = base.with_name("bm25_json")
    mapping: Dict[str, Dict[str, Any]] = {}
    if not meta_dir.exists():
        return mapping

    for p in meta_dir.glob("*.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            doc_id = j.get("doc_id") or "unknown"
            page = int(j.get("page") or 1)
            text = j.get("text") or j.get("contents") or ""
            _id = j.get("id") or f"{doc_id}#p{page}"
            mapping[str(_id)] = {"doc_id": doc_id, "page": page, "text": text}
        except Exception:
            continue
    return mapping


def _hit_to_record(searcher, hit, meta_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"doc_id": "unknown", "page": 1, "text": ""}

    id_str = getattr(hit, "docid", None)
    if id_str and id_str in meta_map:
        m = meta_map[id_str]
        return {"doc_id": m["doc_id"], "page": int(m["page"] or 1), "text": (m["text"] or "")[:_TEXT_SNIPPET_LIMIT]}

    try:
        doc = searcher.doc(hit.docid)
        raw = doc.raw() if callable(getattr(doc, "raw", None)) else None
        if raw:
            try:
                j = json.loads(raw)
                text = j.get("text") or j.get("contents") or ""
                doc_id = j.get("doc_id") or j.get("id") or "unknown"
                page = int(j.get("page") or 1)
                return {"doc_id": str(doc_id), "page": page, "text": str(text)[:_TEXT_SNIPPET_LIMIT]}
            except Exception:
                pass
        contents = doc.contents() if callable(getattr(doc, "contents", None)) else None
        if contents:
            rec["text"] = str(contents)[:_TEXT_SNIPPET_LIMIT]
    except Exception:
        pass

    if id_str:
        m = re.match(r"(.+?)(?:[#_:/-])p?(\d+)$", str(id_str))
        if m:
            rec["doc_id"] = m.group(1)
            try:
                rec["page"] = int(m.group(2))
            except Exception:
                rec["page"] = 1
        else:
            rec["doc_id"] = str(id_str)

    return rec


# -------------------- Embeddings (HF Transformers) --------------------

@lru_cache(maxsize=2)
def _load_hf(model_name: str) -> Tuple[AutoTokenizer, AutoModel]:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    mdl.eval()
    return tok, mdl


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _select_device(device_hint: Optional[str]) -> str:
    if device_hint:
        dh = device_hint.strip().lower()
        if dh in {"cuda", "gpu"} and torch.cuda.is_available():
            return "cuda"
        if dh in {"cpu"}:
            return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def embed_query_hf(
    query: str,
    model_name: str = "BAAI/bge-m3",
    device_hint: Optional[str] = None,
    use_fp16: bool = False,
) -> List[float]:
    if not query:
        return []

    tok, mdl = _load_hf(model_name)
    device = _select_device(device_hint)
    mdl = mdl.to(device)
    if use_fp16 and device == "cuda":
        mdl = mdl.half()

    with torch.no_grad():
        enc = tok(
            query,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = mdl(**enc)
        if hasattr(out, "last_hidden_state"):
            emb = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        elif isinstance(out, (list, tuple)):
            emb = _mean_pool(out[0], enc["attention_mask"])
        else:
            raise RuntimeError("Не удалось получить last_hidden_state из модели.")
        emb = torch.nn.functional.normalize(emb, p=2, dim=1).squeeze(0).float().cpu().tolist()
        return emb


def _embed_texts(
    texts: List[str],
    model_name: str = "BAAI/bge-m3",
    device_hint: Optional[str] = None,
    use_fp16: bool = False,
    batch_size: int = 16,
) -> torch.Tensor:
    tok, mdl = _load_hf(model_name)
    device = _select_device(device_hint)
    mdl = mdl.to(device)
    if use_fp16 and device == "cuda":
        mdl = mdl.half()

    all_vecs: List[torch.Tensor] = []
    mdl.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tok(
                chunk,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            out = mdl(**enc)
            if hasattr(out, "last_hidden_state"):
                vec = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            else:
                vec = _mean_pool(out[0], enc["attention_mask"])
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            all_vecs.append(vec.float().cpu())
    return torch.cat(all_vecs, dim=0) if all_vecs else torch.empty(0)


# -------------------- Qdrant --------------------

@lru_cache(maxsize=2)
def _qdrant_client(url: str):
    if QdrantClient is None:
        raise RuntimeError("qdrant-client не установлен.")
    prefer_grpc = (os.getenv("QDRANT__PREFER_GRPC", "false").lower() == "true")
    return QdrantClient(url=url, prefer_grpc=prefer_grpc)


def _rrf_fusion(runs: List[List[str]], k: int = 60, c: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for run in runs:
        for rank, did in enumerate(run[:k], start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (c + rank)
    return scores


def _load_pages_text(pages_dir: Path, doc_id: str, a: int, b: int) -> str:
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
    Гибридный поиск: Qdrant (dense) + BM25 (sparse) + RRF, возвращает до k фрагментов.
    Если включён переранкер — досортируем кандидатов косинусом к запросу.
    """
    results: List[Dict[str, Any]] = []
    if k <= 0 or not query:
        return results

    # Нормализуем URL (это уберёт 'Unknown scheme: qdrant' при 'qdrant:6333' и пр.)
    qdrant_url = _normalize_qdrant_url(qdrant_url)

    # ---------- 1) Qdrant (dense) ----------
    qd_docids: List[str] = []
    qd_payload_first: Dict[str, Dict[str, Any]] = {}
    try:
        vec = embed_query_hf(query, model_name=hf_model, device_hint=hf_device, use_fp16=hf_fp16)
        client = _qdrant_client(qdrant_url)
        pts = client.query_points(
            collection_name=qdrant_collection,
            query=vec,
            limit=max(k * 4, 40)
        ).points
        for p in pts:
            pl = p.payload or {}
            did = str(pl.get("doc_id") or "unknown")
            if did not in qd_payload_first:
                qd_payload_first[did] = pl
                qd_docids.append(did)
    except Exception as e:
        print("⚠️ Qdrant retrieve error:", e)

    # ---------- 2) BM25 (sparse) ----------
    bm_docids: List[str] = []
    bm_first: Dict[str, Dict[str, Any]] = {}
    try:
        hits = bm25_search(bm25_index_dir, query, topk=max(k * 4, 40))
        for h in hits:
            did = str(h.get("doc_id") or "unknown")
            if did not in bm_first:
                bm_first[did] = h
                bm_docids.append(did)
    except Exception as e:
        print("⚠️ BM25 retrieve error:", e)

    # ---------- 3) RRF ----------
    fused = _rrf_fusion([qd_docids, bm_docids])
    top_docids = [did for did, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    # ---------- 4) Сбор контента ----------
    pages_dir_p = Path(pages_dir)
    used_per_doc: Dict[str, int] = {}
    candidates: List[Dict[str, Any]] = []

    for did in top_docids:
        # BM25-чанк (узкий, обычно конкретная страница)
        if did in bm_first and used_per_doc.get(did, 0) < per_doc_limit:
            h = bm_first[did]
            candidates.append({
                "doc_id": did,
                "page_start": int(h.get("page") or 1),
                "page_end": int(h.get("page") or 1),
                "text": (h.get("text") or "")[:_TEXT_SNIPPET_LIMIT],
            })
            used_per_doc[did] = used_per_doc.get(did, 0) + 1

        # Qdrant «окно страниц»
        if did in qd_payload_first and used_per_doc.get(did, 0) < per_doc_limit:
            pl = qd_payload_first[did]
            a = int(pl.get("page_start", 1) or 1)
            b = int(pl.get("page_end", a) or a)
            text = _load_pages_text(pages_dir_p, did, a, b) or _load_pages_text(pages_dir_p, did, 1, 1)
            if text:
                candidates.append({
                    "doc_id": did,
                    "page_start": a,
                    "page_end": b,
                    "text": text[:_TEXT_SNIPPET_LIMIT],
                })
                used_per_doc[did] = used_per_doc.get(did, 0) + 1

    # ---------- 5) Опциональный переранкер ----------
    if reranker_enabled and candidates:
        try:
            q_vec = torch.tensor(embed_query_hf(query, model_name=hf_model, device_hint=hf_device, use_fp16=hf_fp16)).unsqueeze(0)
            t_vecs = _embed_texts([c["text"] for c in candidates[:max(rerank_top_k, len(candidates))]],
                                  model_name=hf_model, device_hint=hf_device, use_fp16=hf_fp16)
            if t_vecs.numel() > 0:
                sims = torch.matmul(torch.nn.functional.normalize(q_vec, p=2, dim=1),
                                    torch.nn.functional.normalize(t_vecs, p=2, dim=1).T).squeeze(0)
                for i, s in enumerate(sims.tolist()):
                    candidates[i]["_rerank_score"] = float(s)
                candidates.sort(key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
        except Exception as e:
            print("⚠️ Reranker (cosine) error:", e)

    # ---------- 6) Вернуть top-k ----------
    out: List[Dict[str, Any]] = []
    for it in candidates[:k]:
        it.pop("_rerank_score", None)
        out.append(it)
    return out
