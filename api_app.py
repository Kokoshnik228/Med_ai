#!/usr/bin/env python3
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml, requests
from pyserini.search.lucene import LuceneSearcher

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ----- app -----
app = FastAPI(title="med_ai RAG API", version="0.8")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ----- config -----
ROOT = Path(__file__).resolve().parent
CONF_DIR = ROOT / "config"
DEFAULT_YAML = CONF_DIR / "default.yaml"
LOCAL_YAML   = CONF_DIR / "local.yaml"

def build_context_citations(ctx_items, max_out: int = 5):
    return [f"{it['doc_id']} стр.{it['page_start']}-{it['page_end']}" for it in ctx_items[:max_out]]
def rrf_fusion(rank_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion по спискам doc_id."""
    scores: Dict[str, float] = {}
    for ranks in rank_lists:
        for r, did in enumerate(ranks, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + r)
    return scores

def bm25_search(index_dir: str, query: str, topk: int) -> List[Dict[str, Any]]:
    """Возвращает куски BM25: {doc_id, page, text}."""
    searcher = LuceneSearcher(index_dir)
    hits = searcher.search(query, k=topk)
    out: List[Dict[str, Any]] = []
    for h in hits:
        doc = searcher.doc(h.docid)
        contents = doc.contents() or ""
        ext_id = doc.id()  # "{doc_id}_p{page}_c{child}"
        m = re.match(r"(.+?)_p(\d+)_c(\d+)$", ext_id)
        doc_id = m.group(1) if m else ext_id
        page = int(m.group(2)) if m else 1
        out.append({"doc_id": doc_id, "page": page, "text": contents})
    return out



def looks_meaningless(text: str) -> bool:
    """Проверка на бессмысленный или слишком короткий текст."""
    text = text.strip().lower()
    if len(text) < 20:
        return True
    # если только буквы и мало разных символов (например, "аааааа" или "qwerty")
    if re.fullmatch(r"[a-zа-яё\s]+", text) and len(set(text)) < 5:
        return True
    return False


def load_config() -> Dict[str, Any]:
    def load(p: Path) -> Dict[str, Any]:
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    DEFAULTS = {
        "app": {"data_dir": "data", "bm25_index_dir": "index/bm25_idx"},
        "qdrant": {"url": "http://localhost:7777", "collection": "med_kb_v3"},
        "ollama": {"base_url": "http://localhost:11434", "model": "llama3.1:8b"},
        "retrieval": {"k": 8},
        "embedding": {"model": "BAAI/bge-m3"},
        "prompt": {
    "system": (
        "Ты — медицинский ассистент. ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. "
        "Анализируй текст как врачебный кейс: в нём могут быть жалобы, анамнез, осмотр, диагноз и назначения. "
        "Распознай диагноз и предложенные меры, сопоставь их с контекстом базы знаний."
        "Не повторяй за врачем слово в слово, ты должен только дополнять его речь"
        "Верни СТРОГО ВАЛИДНЫЙ JSON со схемой:\n"
        "{score, subscores, critical_errors[], recommendations[], citations[], disclaimer}\n"
        "- score: число 0..100 (точность назначения).\n"
        "- subscores: карта подоценок (например: dosing, diagnosis_match, interactions…).\n"
        "- critical_errors: список объектов {type, explain}.\n"
        "- recommendations: список объектов {what_to_change, rationale}.\n"
        "- citations: список строк-источников только из переданного КОНТЕКСТА.\n"
        "- disclaimer: короткое предупреждение на русском.\n"
        "Если уверенности нет — снижай score, добавляй пояснение в disclaimer. ВНЕШНИЕ источники не используй."
    ),
    "user_template": (
        "[КЕЙС]\n{case_text}\n\n"
        "[КОНТЕКСТ]\n{ctx}\n\n"
        "Верни ТОЛЬКО один валидный JSON по указанной схеме. Все тексты внутри — на русском. "
        "Источники указывай только из контекста."
    )
}

    }
    base  = load(DEFAULT_YAML)
    local = load(LOCAL_YAML)
    def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in (b or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out
    return merge(DEFAULTS, merge(base, local))

CONFIG = load_config()
def cfg(*path: str, default: Any=None) -> Any:
    cur: Any = CONFIG
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

# ----- utils -----
def l2_unit(vec: List[float]) -> List[float]:
    s = (sum(x*x for x in vec) ** 0.5) or 1.0
    return [x/s for x in vec]

def load_pages_text(pages_dir: Path, doc_id: str, p_start: int, p_end: int) -> str:
    jf = pages_dir / f"{doc_id}.pages.jsonl"
    if not jf.exists(): return ""
    out: List[str] = []
    for line in jf.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rec = json.loads(line); pg = int(rec.get("page", 0))
            if p_start <= pg <= p_end:
                out.append(rec.get("text","") or "")
        except Exception:
            continue
    return "\n".join(out)
def embed_query_ollama(text: str) -> List[float]:
    """Получить эмбеддинг запроса через Ollama и нормализовать."""
    base = cfg("ollama","base_url", default="http://localhost:11434") or "http://localhost:11434"
    url = base.rstrip("/") + "/api/embeddings"
    payload = {
        "model": cfg("embedding","model", default="zylonai/multilingual-e5-large"),
        "prompt": text,
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    vec = r.json().get("embedding", [])
    return l2_unit(vec)


def retrieve_light(query: str, k: int) -> List[Dict[str,Any]]:
    results: List[Dict[str,Any]] = []
    
    if k <= 0: return results
    try:
        from qdrant_client import QdrantClient
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(cfg("embedding","model", default="BAAI/bge-m3"), use_fp16=True)
        vec = model.encode([query], return_dense=True, return_sparse=False)["dense_vecs"][0]
        vec = l2_unit(vec)
        client = QdrantClient(url=cfg("qdrant","url", default="http://localhost:7777"))
        pts = client.query_points(collection_name=cfg("qdrant","collection", default="med_kb_v3"),
                                  query=vec, limit=max(k,20)).points
        pages_dir = Path(cfg("app","data_dir", default="data"))
        seen=set()
        for p in pts:
            pl = p.payload or {}
            did = pl.get("doc_id") or "unknown"
            if did in seen: continue
            seen.add(did)
            a = int(pl.get("page_start",1) or 1)
            b = int(pl.get("page_end",a) or a)
            text = load_pages_text(pages_dir, did, a, b) or load_pages_text(pages_dir, did, 1, 1)
            if not (pages_dir / f"{did}.pages.jsonl").exists():
               continue
            results.append({"doc_id": did, "page_start": a, "page_end": b, "text": (text or "")[:2000]})
            if len(results)>=k: break
    except Exception:
        pass
    return results

def retrieve_hybrid(query: str, k: int) -> List[Dict[str, Any]]:
    """RRF слияние кандидатов Qdrant и BM25. Не более 2 кусочков на документ."""
    results: List[Dict[str, Any]] = []
    if k <= 0:
        return results

    # --- Qdrant кандидаты (уникальные doc_id и их первый payload) ---
    qd_docids: List[str] = []
    qd_payload_first: Dict[str, Dict[str, Any]] = {}
    try:
        from qdrant_client import QdrantClient
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(cfg("embedding","model", default="BAAI/bge-m3"), use_fp16=True)
        vec = model.encode([query], return_dense=True, return_sparse=False)["dense_vecs"][0]
        vec = l2_unit(vec)
        client = QdrantClient(url=cfg("qdrant","url", default="http://localhost:7777"))
        pts = client.query_points(
            collection_name=cfg("qdrant","collection", default="med_kb_v3"),
            query=vec, limit=max(k*6, 50)
        ).points
        for p in pts:
            pl = p.payload or {}
            did = pl.get("doc_id") or "unknown"
            if did not in qd_payload_first:
                qd_payload_first[did] = pl
                qd_docids.append(did)
    except Exception:
        pass

    # --- BM25 кандидаты (уникальные doc_id и их первый hit) ---
    bm25_dir = cfg("app","bm25_index_dir", default="index/bm25_idx")
    bm_docids: List[str] = []
    bm_first: Dict[str, Dict[str, Any]] = {}
    try:
        bm_hits = bm25_search(bm25_dir, query, topk=max(k*6, 50))
        for h in bm_hits:
            did = h["doc_id"]
            if did not in bm_first:
                bm_first[did] = h
                bm_docids.append(did)
    except Exception:
        pass

    # --- RRF слияние списков doc_id ---
    fused = rrf_fusion([qd_docids, bm_docids])
    top_docids = [did for did,_ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    # --- Формируем выход (до k фрагментов, максимум 2 на doc_id) ---
    pages_dir = Path(cfg("app","data_dir", default="data"))
    per_doc_limit = 2
    used_per_doc: Dict[str, int] = {}

    for did in top_docids:
        if len(results) >= k:
            break

        # 1) Сначала — BM25 фрагмент (точнее по терминам)
        if did in bm_first and used_per_doc.get(did, 0) < per_doc_limit:
            h = bm_first[did]
            results.append({
                "doc_id": did,
                "page_start": h["page"],
                "page_end": h["page"],
                "text": (h["text"] or "")[:2000]
            })
            used_per_doc[did] = used_per_doc.get(did, 0) + 1
            if len(results) >= k:
                break

        # 2) Затем — Qdrant фрагмент (по embedding-окну страниц)
        if did in qd_payload_first and used_per_doc.get(did, 0) < per_doc_limit:
            pl = qd_payload_first[did]
            a = int(pl.get("page_start", 1) or 1)
            b = int(pl.get("page_end", a) or a)
            text = load_pages_text(pages_dir, did, a, b) or load_pages_text(pages_dir, did, 1, 1)
            results.append({
                "doc_id": did,
                "page_start": a,
                "page_end": b,
                "text": (text or "")[:2000]
            })
            used_per_doc[did] = used_per_doc.get(did, 0) + 1

    return results


def safe_json_extract(s: str) -> Dict[str,Any]:
    try:
        return json.loads(s)
    except Exception:
        pass
    i=s.find("{"); j=s.rfind("}")
    if 0<=i<j:
        try:
            return json.loads(s[i:j+1])
        except Exception:
            pass
    return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [], "disclaimer": "Парсинг ответа не удался."}

def normalize_result(r: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "score": 0, "subscores": {}, "critical_errors": [],
        "recommendations": [], "citations": [], "disclaimer": ""
    }
    # score
    try:
        sc = r.get("score", 0)
        out["score"] = max(0, min(100, float(sc))) if isinstance(sc, (int,float)) else 0.0
    except Exception:
        out["score"] = 0.0
    # subscores
    subs = r.get("subscores") or {}
    if isinstance(subs, dict):
        clean = {}
        for k,v in subs.items():
            try: clean[str(k)] = max(0, min(100, float(v)))
            except Exception: pass
        out["subscores"] = clean
    # critical_errors
    ce = r.get("critical_errors") or []
    clean_ce = []
    if isinstance(ce, list):
        for it in ce:
            if isinstance(it, dict):
                clean_ce.append({"type": str(it.get("type","general")),
                                 "explain": str(it.get("explain", it.get("message","")))})
            elif isinstance(it, str):
                clean_ce.append({"type":"general","explain":it})
    out["critical_errors"] = clean_ce
    # recommendations
    recs = r.get("recommendations") or []
    clean_recs = []
    if isinstance(recs, list):
        for it in recs:
            if isinstance(it, dict):
                w = str(it.get("what_to_change") or it.get("action") or it.get("recommendation") or it.get("text",""))
                ra = str(it.get("rationale") or it.get("reason",""))
                if w or ra: clean_recs.append({"what_to_change": w, "rationale": ra})
            elif isinstance(it, str):
                clean_recs.append({"what_to_change": it, "rationale": ""})
    out["recommendations"] = clean_recs
    # citations
    cits = r.get("citations") or []
    if isinstance(cits, list):
        out["citations"] = [str(x) for x in cits if isinstance(x,(str,int,float))]
    elif isinstance(cits,(str,int,float)):
        out["citations"] = [str(cits)]
    # disclaimer
    disc = r.get("disclaimer")
    out["disclaimer"] = str(disc) if disc is not None else ""
    return out
    
def extract_medical_query(case_text: str) -> str:
    """
    Преобразует врачебный текст в краткий поисковый запрос.
    Например:
    - из длинного анамнеза достаёт диагноз, жалобы, назначения.
    """
    case_text = case_text.lower()

    # Простое извлечение ключевых зон
    complaints = re.findall(r"жалоб[аи]\s*([^.\n]*)", case_text)
    diagnosis = re.findall(r"диагноз[:\s]*([a-zа-я0-9\s\.\-,]+)", case_text)
    treatment = re.findall(r"назначен[оы]?\s*([^.\n]*)", case_text)

    query_parts = []
    if complaints: query_parts.append(complaints[0])
    if diagnosis: query_parts.append(diagnosis[0])
    if treatment: query_parts.append(treatment[0])

    # fallback — ищем ключевые медицинские термины
    if not query_parts:
        key_terms = re.findall(r"(геморрой|трещин[аы]|простатит|диарея|запор|гипертензия|боль|кровотечен\w+)", case_text)
        query_parts.extend(key_terms)

    query = " ".join(query_parts).strip()
    return query or case_text[:200]


def call_ollama_json(ollama_url: Optional[str], model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    try:
        if not ollama_url:
            ollama_url = cfg("ollama","base_url", default="http://localhost:11434") or "http://localhost:11434"
        payload = {
            "model": str(model),
            "prompt": str(user_prompt),
            "system": str(system_prompt),
            "format": "json",
            "options": {},   # без кастомных options
            "stream": False,
        }
        resp = requests.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=180)
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [],
                    "citations": [], "disclaimer": f"LLM HTTP {resp.status_code}: {resp.text[:200]}"}
        if resp.headers.get("content-type","").startswith("application/json"):
            obj = resp.json()
            response_field = obj.get("response","") if isinstance(obj, dict) else ""
        else:
            response_field = resp.text
        raw = json.dumps(response_field, ensure_ascii=False) if isinstance(response_field,(dict,list)) else str(response_field or "").strip()
        if not raw:
            return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [],
                    "citations": [], "disclaimer": "LLM вернул пустой ответ."}
        return safe_json_extract(raw)
    except Exception as e:
        return {"score": None, "subscores": {}, "critical_errors": [], "recommendations": [],
                "citations": [], "disclaimer": f"Ошибка LLM ({type(e).__name__}): {e}"}

# ----- routes -----
class AnalyzeReq(BaseModel):
    case_text: str
    query: Optional[str] = None
    k: int = Field(default_factory=lambda: cfg("retrieval","k", default=8))
    model: str = Field(default_factory=lambda: cfg("ollama","model", default="llama3.1:8b"))
    ollama_url: Optional[str] = Field(default_factory=lambda: cfg("ollama","base_url", default="http://localhost:11434"))

@app.get("/health")
def health():
    return {"status":"ok", "qdrant": cfg("qdrant","url", default="http://localhost:7777"),
            "model": cfg("ollama","model", default="llama3.1:8b")}

@app.post("/config/reload")
def config_reload():
    global CONFIG
    CONFIG = load_config()
    return {"status":"reloaded"}

@app.post("/analyze")
def analyze_ep(req: AnalyzeReq):
    try:
        print("🚀 /analyze вызван")
        print(f"📋 Текст кейса: {req.case_text[:150]}...")

        # --- 1️⃣ Проверка на бессмысленный текст ---
        if looks_meaningless(req.case_text):
            print("⚠️ Текст бессмысленный — возврат без анализа")
            return {
                "result": {
                    "score": 0,
                    "subscores": {},
                    "critical_errors": [],
                    "recommendations": [],
                    "citations": [],
                    "disclaimer": "Текст кейса не содержит осмысленных данных.",
                }
            }

        # --- 2️⃣ Получаем контекст из базы ---
        if req.query:
            q = req.query
        else:
            q = extract_medical_query(req.case_text)
        print(f"🔍 Извлечён поисковый запрос: {q}")

        if not q:
            q = (req.case_text[:200] if req.case_text else "гипертензия лечение")

        print("📡 Запрос контекста (retrieve_hybrid)...")
        ctx_items = retrieve_hybrid(q, req.k)
        print(f"✅ Контекст найден: {len(ctx_items)} фрагментов")

        if not ctx_items:
            print("❌ Контекст не найден, выходим.")
            return {
                "result": {
                    "score": 0,
                    "subscores": {},
                    "critical_errors": [],
                    "recommendations": [],
                    "citations": [],
                    "disclaimer": "Контекст не найден в базе знаний — невозможно оценить кейс.",
                }
            }

        # --- 3️⃣ Формируем контекст ---
        ctx = "".join(
            [
                f"### [{i}] DOC {it['doc_id']} P{it['page_start']}-{it['page_end']}\n{it.get('text','')}\n\n"
                for i, it in enumerate(ctx_items, 1)
            ]
        ) or "(контекст не найден)"
        print("🧱 Контекст сформирован, длина:", len(ctx))

        system = cfg("prompt", "system", default="Анализируй кейс. Верни JSON.")
        user_t = cfg("prompt", "user_template", default="[КЕЙС]\n{case_text}\n\n[КОНТЕКСТ]\n{ctx}")
        user = user_t.format(case_text=req.case_text, ctx=ctx)

        # --- 4️⃣ Вызов модели ---
        print("🧠 Отправляем запрос в Ollama...")
        resp = call_ollama_json(req.ollama_url, req.model, system, user)
        print("🤖 Модель ответила:", str(resp)[:300])

        data = normalize_result(resp)
        print("📊 Результат нормализован:", data.get("score"))

        # --- 5️⃣ Понижаем оценку, если контекст короткий ---
        ctx_len = sum(len(it.get("text", "")) for it in ctx_items)
        print("📏 Длина контекста:", ctx_len)

        if ctx_len < 500:
            data["score"] = max(0, data.get("score", 0) * 0.5)
            data["disclaimer"] += " (Недостаточно контекста из базы знаний — достоверность снижена.)"
        elif ctx_len < 1500:
            data["score"] = max(0, data.get("score", 0) * 0.8)
            data["disclaimer"] += " (Контекст ограничен — достоверность частично снижена.)"

        # --- 6️⃣ Источники ---
        data["citations"] = build_context_citations(ctx_items, max_out=5)
        if not data.get("citations"):
            data["citations"] = [
                f"{it['doc_id']} стр.{it['page_start']}-{it['page_end']}"
                for it in ctx_items[:5]
            ]

        # --- 7️⃣ Наказание за ошибки ---
        crit_count = len(data.get("critical_errors", []))
        if crit_count > 0:
            data["score"] = max(0, data["score"] - 10 * crit_count)
            data["disclaimer"] += f" (Обнаружено {crit_count} критических ошибок.)"

        print("✅ Готово. Итоговая оценка:", data["score"])
        return {"result": data, "citations_used": [x["doc_id"] for x in ctx_items]}

    except Exception as e:
        print(f"❌ Ошибка в analyze_ep: {e}")
        return {
            "result": {
                "score": None,
                "subscores": {},
                "critical_errors": [],
                "recommendations": [],
                "citations": [],
                "disclaimer": f"Ошибка API: {e}",
            }
        }


# ----- simple UI at "/" -----
UI_HTML = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI-ассистент врача (MVP)</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f6f7fb;margin:0;color:#101828}
.wrap{max-width:1100px;margin:20px auto;padding:16px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 1px 3px rgba(16,24,40,.08);padding:16px;margin-bottom:16px}
h1{font-size:20px;margin:0 0 8px}
label{font-weight:600;font-size:14px;margin:6px 0;display:block}
input,select,textarea{width:100%;border:1px solid #d0d5dd;border-radius:10px;padding:10px;font-size:14px}
textarea{min-height:180px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.btn{background:#2563eb;color:#fff;border:none;border-radius:10px;padding:10px 14px;font-weight:600;cursor:pointer}
.btn:disabled{opacity:.6;cursor:not-allowed}
.badge{display:inline-block;border:1px solid #d0d5dd;border-radius:999px;padding:2px 8px;font-size:12px;margin-left:8px}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
.small{font-size:12px;color:#475467}
.err{color:#b91c1c}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
</style>
<div class="wrap">
  <div class="card">
    <h1>AI-ассистент врача (MVP) <span id="score" class="badge">оценка: —</span></h1>
    <div class="small">API: <span id="api"></span></div>
  </div>
  <div class="card">
    <label>Текст кейса</label>
    <textarea id="caseText" placeholder="Вставьте кейс: жалобы, анамнез, диагноз, назначения..."></textarea>
    <div class="row">
      <div>
        <label>Запрос для поиска (необязательно)</label>
        <input id="query" placeholder="например: гипертоническая болезнь лечение эналаприл">
      </div>
      <div>
        <label>Модель / K</label>
        <div class="row" style="grid-template-columns:2fr 1fr;gap:8px">
          <select id="model"><option>llama3.1:8b</option><option>llama3.1:70b</option></select>
          <input id="k" type="number" value="8" min="0" max="20">
        </div>
      </div>
    </div>
    <div style="margin-top:10px;display:flex;gap:8px;align-items:center">
      <button id="run" class="btn">Проанализировать</button>
      <button id="reindex" class="btn" style="background:#059669">🔄 Обновить базу</button>
      <span id="busy" class="small" style="display:none">⏳ выполняется…</span>
      <span id="error" class="small err"></span>
    </div>
  </div>
  <div class="card">
    <h3 style="margin:0 0 6px">Результат</h3>
    <div class="grid2" id="subs"></div>
    <div><h4>Критические ошибки</h4><ul id="crit"></ul></div>
    <div><h4>Рекомендации</h4><ul id="recs"></ul></div>
    <div><h4>Источники (цитаты)</h4><ul id="cits"></ul></div>
    <details><summary class="small">Сырой JSON</summary><pre id="raw" class="mono"></pre></details>
  </div>
</div>
<script>
const API = window.location.origin; document.getElementById('api').textContent = API;
const el=id=>document.getElementById(id); const show=(n,on)=>n.style.display=on?'':'none';
function colorForScore(s){ if(typeof s!=='number') return ''; if(s>=85) return '#dcfce7'; if(s>=65) return '#fef9c3'; return '#fee2e2'; }
function renderResult(r){
  const sc=r.score??'—'; const sb=document.getElementById('score'); sb.textContent='оценка: '+sc; sb.style.background=colorForScore(sc);
  const subs=el('subs'); subs.innerHTML=''; Object.entries(r.subscores||{}).forEach(([k,v])=>{ const d=document.createElement('div'); d.className='card'; d.style.margin=0; d.innerHTML=`<div class="small">${labelMap[k] || k}</div><div style="font-weight:700">${v??'—'}</div>`; subs.appendChild(d); });
  const crit=el('crit'); crit.innerHTML=''; (r.critical_errors||[]).forEach(x=>{ const li=document.createElement('li'); li.textContent=`${x.type}: ${x.explain}`; crit.appendChild(li); });
  const recs=el('recs'); recs.innerHTML=''; (r.recommendations||[]).forEach(x=>{ const li=document.createElement('li'); li.textContent=`${x.what_to_change} — ${x.rationale}`; recs.appendChild(li); });
  const cits=el('cits'); cits.innerHTML=''; (r.citations||[]).forEach(x=>{ const li=document.createElement('li'); li.textContent=String(x); cits.appendChild(li); });
  el('raw').textContent=JSON.stringify(r,null,2);
}
  const labelMap = {
    "diagnosis": "Диагноз",
    "diagnosis_match": "Соответствие диагнозу",
    "therapy": "Терапия",
    "med_choice": "Выбор препарата",
    "dosage": "Дозировка",
    "dosing": "Дозировка",
    "interactions": "Лекарственные взаимодействия",
    "contraindications": "Противопоказания",
    "monitoring": "Мониторинг",
    "evidence": "Доказательность"
  };

async function run(){ el('error').textContent=''; show(el('busy'),true); el('run').disabled=true;
  try{
    const body={ case_text: el('caseText').value||'', query: el('query').value||null, k: parseInt(el('k').value||'8',10), model: el('model').value||'llama3.1:8b' };
    const res=await fetch(API+'/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const txt=await res.text(); let data; try{ data=JSON.parse(txt); }catch(e){ throw new Error('Не удалось разобрать JSON ответа: '+txt.slice(0,200)); }
    renderResult(data.result || data);
  }catch(e){ el('error').textContent='Ошибка: '+(e?.message||e); }
  finally{ show(el('busy'),false); el('run').disabled=false; }
}
el('run').onclick=run;

el('reindex').onclick = async () => {
  show(el('busy'), true);
  el('error').textContent = '';
  try {
    const res = await fetch(API + '/reindex', { method: 'POST' });
    const data = await res.json();
    el('error').textContent = data.message || '';
  } catch(e) {
    el('error').textContent = 'Ошибка запуска: ' + e.message;
  } finally {
    show(el('busy'), false);
  }
};
// 🟢 Проверка статуса индексации каждые 3 сек
async function checkReindexStatus() {
  try {
    const res = await fetch(API + '/reindex/status');
    const data = await res.json();
    const msg = data.message || '';
    const state = data.state;
    if (state === 'running') {
      el('busy').textContent = '🔄 ' + msg;
      show(el('busy'), true);
    } else if (state === 'done') {
      el('busy').textContent = '✅ Индексация завершена';
      setTimeout(() => show(el('busy'), false), 4000);
    } else if (state === 'error') {
      el('busy').textContent = '❌ ' + msg;
    }
  } catch(e) {
    console.error(e);
  }
}
setInterval(checkReindexStatus, 3000);

</script>
"""
import subprocess, threading
index_status = {"state": "idle", "message": "Ожидание"}
import subprocess, threading, time

index_status = {"state": "idle", "message": "Ожидание"}

@app.post("/reindex")
def reindex_ep(full: bool = False):
    """Индексация: по умолчанию только новые; full=True — полная переиндексация."""
    def run_reindex():
        global index_status
        try:
            index_status.update({"state": "running", "message": "📘 Индексация запущена..."})
            print("⚙️ Запуск индексации документов...")

            index_status["message"] = "📄 Шаг 1: парсинг PDF и создание JSON..."
            subprocess.run([
                "python", "ingest_from_raw.py",
                "--input-dir", "raw_docs",
                "--out-dir", "data"
            ], check=True)

            index_status["message"] = "📚 Шаг 2: построение BM25 индекса..."
            subprocess.run([
                "python", "build_bm25.py",
                "--pages-glob", "data/*.pages.jsonl",
                "--out-json", "index/bm25_json",
                "--index-dir", "index/bm25_idx"
            ], check=True)

            index_status["message"] = "🧠 Шаг 3: индексация в Qdrant..."
            cmd = [
                "python", "chunk_and_index.py",
                "--pages-glob", "data/*.pages.jsonl",
                "--collection", "med_kb_v3",
                "--qdrant-url", "http://localhost:7777",
            ]
            if full:
                cmd.append("--recreate")
            else:
                cmd.append("--only-new")  # режим только новых чанков (по умолчанию и так включён)
            subprocess.run(cmd, check=True)

            index_status.update({"state": "done", "message": "✅ Индексация завершена."})
            print("✅ Индексация завершена.")
        except Exception as e:
            index_status.update({"state": "error", "message": f"❌ Ошибка: {e}"})
            print(f"❌ Ошибка при индексации: {e}")

    threading.Thread(target=run_reindex, daemon=True).start()
    return {"status": "started", "message": ""}




@app.get("/reindex/status")
def reindex_status():
    """Возвращает текущий статус индексации."""
    return index_status

@app.get("/", response_class=HTMLResponse)
def ui_root():
    return HTMLResponse(UI_HTML)

