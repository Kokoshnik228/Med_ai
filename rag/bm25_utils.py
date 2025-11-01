# rag/bm25_utils.py
import os
import re
import json
import socket
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------- Глобальные настройки (переопределяются ENV) ----------
_BM25_K1 = float(os.getenv("BM25_K1", "0.9"))
_BM25_B  = float(os.getenv("BM25_B",  "0.4"))
_PER_DOC_LIMIT = int(os.getenv("PER_DOC_LIMIT", "3"))
_TEXT_SNIPPET_LIMIT = int(os.getenv("TEXT_SNIPPET_LIMIT", "2000"))

# ---------- Быстрые регулярки / кэши ----------
_ID_RE = re.compile(r"(.+?)_p(\d+)_c(\d+)$")

_PAGES_CACHE: Dict[tuple, List[tuple]] = {}
_PAGES_CACHE_LOCK = threading.Lock()

_BM25 = {"searcher": None, "index_dir": None, "lock": threading.Lock()}

# ============================================================
# BM25: инициализация без потери качества (только скорость)
# ============================================================
def _get_bm25_searcher(index_dir: str):
    """
    Создаём LuceneSearcher один раз на каталог индекса и переиспользуем.
    Это НЕ влияет на качество — только убирает дорогое повторное открытие индекса.
    """
    from pyserini.search.lucene import LuceneSearcher
    with _BM25["lock"]:
        if _BM25["searcher"] is None or _BM25["index_dir"] != index_dir:
            s = LuceneSearcher(index_dir)
            try:
                # Настройка BM25 параметров (мягко, если доступно)
                s.set_bm25(k1=_BM25_K1, b=_BM25_B)
            except Exception:
                pass
            _BM25["searcher"] = s
            _BM25["index_dir"] = index_dir
    return _BM25["searcher"]

def bm25_search(index_dir: str, query: str, topk: int) -> List[Dict[str, Any]]:
    """
    Возвращает список найденных чанков BM25:
      {doc_id: str, page: int, text: str}

    Приоритеты извлечения:
      1) text        → doc.contents()
      2) doc_id/page → из doc.id(): {doc_id}_p{page}_c{child}
      3) fallback    → из doc.raw() (json-метаданные)
    """
    searcher = _get_bm25_searcher(index_dir)
    hits = searcher.search(query, k=topk)

    out: List[Dict[str, Any]] = []
    for h in hits:
        doc = searcher.doc(h.docid)
        if doc is None:
            continue

        # 1) текст чанка
        try:
            text = doc.contents() or ""
        except Exception:
            text = ""

        # 2) doc_id/page из doc.id()
        doc_id, page = "", 1
        try:
            ext_id = doc.id()
            m = _ID_RE.match(ext_id)
            if m:
                doc_id = m.group(1)
                page = int(m.group(2))
            else:
                doc_id = ext_id or ""
        except Exception:
            pass

        # 3) fallback к raw-метаданным
        if (not doc_id or page == 1) and hasattr(doc, "raw"):
            try:
                raw_json = doc.raw()
                if raw_json:
                    meta = json.loads(raw_json)
                    if not doc_id:
                        doc_id = str(meta.get("doc_id") or doc_id or "")
                    if page == 1:
                        p = meta.get("page")
                        if isinstance(p, int) and p > 0:
                            page = p
            except Exception:
                pass

        out.append({
            "doc_id": doc_id or "unknown",
            "page": page,
            "text": text
        })
    return out

# ============================================================
# Embeddings (BGE-M3)
# ============================================================
_HF = {"model": None, "device": None}

def _init_hf(model_name: str, device_hint: Optional[str] = None, use_fp16: bool = False):
    if _HF["model"] is not None:
        return
    from FlagEmbedding import BGEM3FlagModel
    try:
        import torch
        device = device_hint or ("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = device_hint or "cpu"
    _HF["model"] = BGEM3FlagModel(
        model_name,
        use_fp16=(use_fp16 and str(device).startswith("cuda")),
        device=device
    )

def _l2_unit(vec):
    s = (sum(x*x for x in vec) ** 0.5) or 1.0
    return [x/s for x in vec]

def embed_query_hf(text: str, model_name: str = "BAAI/bge-m3",
                   device_hint: Optional[str] = None, use_fp16: bool = False) -> List[float]:
    _init_hf(model_name, device_hint, use_fp16)
    out = _HF["model"].encode([text], batch_size=1)
    v = out["dense_vecs"][0].tolist()
    return _l2_unit([float(x) for x in v])

# ============================================================
# Qdrant client
# ============================================================
def _qdrant_client(url: str):
    from qdrant_client import QdrantClient
    # если в докер-сети имя qdrant не резолвится — подменим на localhost:7779 (локальная отладка)
    if "qdrant:" in url:
        try:
            socket.gethostbyname("qdrant")
        except socket.gaierror:
            url = "http://localhost:7779"
    # timeout уменьшаем, чтобы не коптиться при сетевых затыках
    return QdrantClient(url=url, timeout=6, prefer_grpc=False, grpc_port=None)

# ============================================================
# Кэш страниц: не читаем .pages.jsonl каждый раз
# ============================================================
def _get_doc_pages(pages_dir: Path, doc_id: str) -> List[tuple]:
    """
    Кэшируем содержимое {doc_id}.pages.jsonl как список (page, text), отсортированный по page.
    """
    key = (str(pages_dir), doc_id)
    v = _PAGES_CACHE.get(key)
    if v is not None:
        return v

    jf = pages_dir / f"{doc_id}.pages.jsonl"
    lst: List[tuple] = []
    if jf.exists():
        for line in jf.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                rec = json.loads(line)
                pg = int(rec.get("page", 0))
                txt = rec.get("text", "") or ""
                lst.append((pg, txt))
            except Exception:
                continue
        lst.sort(key=lambda x: x[0])

    with _PAGES_CACHE_LOCK:
        # Простейший контроль размера кэша
        if len(_PAGES_CACHE) > 4000:
            _PAGES_CACHE.clear()
        _PAGES_CACHE[key] = lst
    return lst

def _load_pages_text(pages_dir: Path, doc_id: str, p_start: int, p_end: int) -> str:
    pages = _get_doc_pages(pages_dir, doc_id)
    if not pages:
        return ""
    # склеиваем нужный диапазон страниц
    chunks = [txt for (pg, txt) in pages if p_start <= pg <= p_end]
    if not chunks:
        return ""
    joined = "\n".join(chunks)
    # ограничим длину для LLM
    return joined[:_TEXT_SNIPPET_LIMIT]

# ============================================================
# Reciprocal Rank Fusion (без изменений логики)
# ============================================================
def _rrf_fusion(rank_lists, k=60):
    scores: Dict[str, float] = {}
    for ranks in rank_lists:
        for r, did in enumerate(ranks, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0/(k+r)
    return scores

# ============================================================
# Гибридное извлечение: Qdrant (dense) + BM25 (sparse) + RRF
# ============================================================
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
    per_doc_limit: int = _PER_DOC_LIMIT,
) -> List[Dict[str, Any]]:
    """
    RRF: объединяет кандидатов Qdrant (dense) и BM25 (sparse). Возвращает до k фрагментов.
    КАЧЕСТВО НЕ МЕНЯЛ: только убран лишний I/O (кэш страниц и один раз открытый LuceneSearcher).
    """
    results: List[Dict[str, Any]] = []
    if k <= 0:
        return results

    # ---------- 1) Qdrant (dense) ----------
    qd_docids: List[str] = []
    qd_payload_first: Dict[str, Dict[str, Any]] = {}
    try:
        vec = embed_query_hf(query, model_name=hf_model, device_hint=hf_device, use_fp16=hf_fp16)
        client = _qdrant_client(qdrant_url)
        pts = client.query_points(
            collection_name=qdrant_collection,
            query=vec,
            limit=max(k*4, 40)
        ).points
        for p in pts:
            pl = p.payload or {}
            did = pl.get("doc_id") or "unknown"
            if did not in qd_payload_first:
                qd_payload_first[did] = pl
                qd_docids.append(did)
    except Exception as e:
        print("⚠️ Qdrant retrieve error:", e)

    # ---------- 2) BM25 (sparse) ----------
    bm_docids: List[str] = []
    bm_first: Dict[str, Dict[str, Any]] = {}
    try:
        hits = bm25_search(bm25_index_dir, query, topk=max(k*4, 40))
        for h in hits:
            did = h["doc_id"]
            if did not in bm_first:
                bm_first[did] = h
                bm_docids.append(did)
    except Exception as e:
        print("⚠️ BM25 retrieve error:", e)

    # ---------- 3) RRF ----------
    fused = _rrf_fusion([qd_docids, bm_docids])
    top_docids = [did for did, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    # ---------- 4) Сбор контента (с кэшем страниц) ----------
    pages_dir_p = Path(pages_dir)
    used_per_doc: Dict[str, int] = {}

    for did in top_docids:
        if len(results) >= k:
            break

        # BM25-чанк (обычно уже содержит конкретный page)
        if did in bm_first and used_per_doc.get(did, 0) < per_doc_limit:
            h = bm_first[did]
            results.append({
                "doc_id": did,
                "page_start": h["page"],
                "page_end": h["page"],
                "text": (h["text"] or "")[:_TEXT_SNIPPET_LIMIT],
            })
            used_per_doc[did] = used_per_doc.get(did, 0) + 1
            if len(results) >= k:
                break

        # Qdrant «окно страниц»
        if did in qd_payload_first and used_per_doc.get(did, 0) < per_doc_limit:
            pl = qd_payload_first[did]
            a = int(pl.get("page_start", 1) or 1)
            b = int(pl.get("page_end", a) or a)
            text = _load_pages_text(pages_dir_p, did, a, b) or _load_pages_text(pages_dir_p, did, 1, 1)
            if text:
                results.append({
                    "doc_id": did,
                    "page_start": a,
                    "page_end": b,
                    "text": text[:_TEXT_SNIPPET_LIMIT],
                })
                used_per_doc[did] = used_per_doc.get(did, 0) + 1

    return results
