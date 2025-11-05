#!/usr/bin/env python3


from __future__ import annotations
import argparse
import json
from pathlib import Path
import textwrap
import requests
from rag.bm25_utils import bm25_search, retrieve_hybrid



SCHEMA = {
  "type": "object",
  "properties": {
    "score": {"type": "integer", "minimum": 1, "maximum": 100},
    "subscores": {
      "type": "object",
      "properties": {
        "diagnosis": {"type": "integer", "minimum": 0, "maximum": 100},
        "med_choice": {"type": "integer", "minimum": 0, "maximum": 100},
        "dose": {"type": "integer", "minimum": 0, "maximum": 100},
        "interactions": {"type": "integer", "minimum": 0, "maximum": 100},
        "contraindications": {"type": "integer", "minimum": 0, "maximum": 100},
        "monitoring": {"type": "integer", "minimum": 0, "maximum": 100}
      },
      "required": ["diagnosis","med_choice","dose","interactions","contraindications","monitoring"]
    },
    "critical_errors": {"type": "array", "items": {
      "type": "object",
      "properties": {
        "type": {"type": "string"},
        "explain": {"type": "string"},
        "citations": {"type": "array", "items": {
          "type": "object",
          "properties": {
            "doc_id": {"type": "string"},
            "pages": {"type": "array", "items": {"type": "integer"}}
          },
          "required": ["doc_id","pages"]
        }}
      }, "required": ["type","explain","citations"]
    }},
    "recommendations": {"type": "array", "items": {
      "type": "object",
      "properties": {
        "what_to_change": {"type": "string"},
        "rationale": {"type": "string"},
        "citations": {"type": "array", "items": {
          "type": "object",
          "properties": {
            "doc_id": {"type": "string"},
            "pages": {"type": "array", "items": {"type": "integer"}}
          },
          "required": ["doc_id","pages"]
        }}
      }, "required": ["what_to_change","rationale","citations"]
    }},
    "citations": {"type": "array", "items": {
      "type": "object",
      "properties": {
        "doc_id": {"type": "string"},
        "pages": {"type": "array", "items": {"type": "integer"}},
        "quote": {"type": "string"}
      },
      "required": ["doc_id","pages","quote"]
    }},
    "disclaimer": {"type": "string"}
  },
  "required": ["score","subscores","critical_errors","recommendations","citations","disclaimer"]
}

SYSTEM_PROMPT = (
  "Ты – медицинский ассистент. Твоя задача: ПРОВЕРИТЬ врача по предоставленному кейсу"
  "и фрагментам клинических источников. Работай ТОЛЬКО по контексту, без выдумок.\n"
  "Анализируй текст как врачебный кейс: в нём могут быть жалобы, анамнез, осмотр, диагноз и назначения. "
  "Распознай диагноз и предложенные меры, сопоставь их с контекстом базы знаний."
  "Требования:\n"
  "1) Дай общий балл 1–100 и подсчёты по подпунктам.\n"
  "2) Отметь критические ошибки (несовместимость диагноза/препарата, превышение макс. дозы, противопоказания).\n"
  "3) Дай чёткие рекомендации врачу, обязательно со ссылками на источники (doc_id + страницы).\n"
  "4) Ссылайся ТОЛЬКО на переданный контекст; если данных мало – напиши, что доказательств недостаточно.\n"
  "5) Ответ строго в формате JSON по схеме. НЕЛЬЗЯ добавлять текст вне JSON.\n"
  
)

USER_PROMPT_TEMPLATE = (
  "Ниже – медицинский кейс и контекстные выдержки из базы.\n\n"
  "[КЕЙС]\n{case_text}\n\n"
  "[КОНТЕКСТ ИСТОЧНИКОВ]\n{ctx}\n\n"
  "Сформируй ОДИН JSON строго по схеме."
)


def load_case(path: Path, limit_chars: int = 8000) -> str:
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if len(txt) > limit_chars:
        txt = txt[:limit_chars] + "\n…"
    return txt


def build_context(ctx_path: Path, max_chunks: int = 12, chunk_char_limit: int = 1600) -> str:
    data = json.loads(ctx_path.read_text(encoding="utf-8"))
    lines = []
    for i, item in enumerate(data[:max_chunks], start=1):
        text = (item.get("text") or "").strip()
        if len(text) > chunk_char_limit:
            text = text[:chunk_char_limit] + " …"
        doc_id = item.get("doc_id")
        ps, pe = item.get("page_start"), item.get("page_end")
        lines.append(f"### [{i}] DOC {doc_id} P{ps}-{pe}\n{text}")
    return "\n\n".join(lines) if lines else "(контекст не найден)"


def call_ollama(ollama_url: str, model: str, prompt: str) -> dict:
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "format": SCHEMA,  # <-- было "json"
        "options": {"temperature": 0.1, "num_ctx": 8192, "top_p": 0.9},
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    raw = (r.json() or {}).get("response", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # безопасный фолбэк
        return {"score": 0, "subscores": {}, "critical_errors": [], "recommendations": [], "citations": [], "disclaimer": "Ответ не прошёл строгий JSON."}



def write_outputs(outdir: Path, case_name: str, result: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / f"{case_name}.rag.json"
    tpath = outdir / f"{case_name}.rag.txt"
    jpath.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # краткий txt
    parts = [
        f"SCORE: {result.get('score')}",
        "SUBSCORES: " + ", ".join(f"{k}={v}" for k, v in result.get("subscores", {}).items()),
        "CRITICAL ERRORS:" if result.get("critical_errors") else "CRITICAL ERRORS: none",
    ]
    for i, ce in enumerate(result.get("critical_errors", []), start=1):
        cites = "; ".join(f"{c.get('doc_id')}:{','.join(map(str,c.get('pages',[])))}" for c in ce.get("citations", []))
        parts.append(f"  {i}) {ce.get('type')}: {ce.get('explain')} [src: {cites}]")
    parts.append("RECOMMENDATIONS:")
    for i, rec in enumerate(result.get("recommendations", []), start=1):
        cites = "; ".join(f"{c.get('doc_id')}:{','.join(map(str,c.get('pages',[])))}" for c in rec.get("citations", []))
        parts.append(f"  {i}) {rec.get('what_to_change')} — {rec.get('rationale')} [src: {cites}]")
    parts.append("DISCLAIMER: " + (result.get("disclaimer") or ""))
    tpath.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, type=Path, help="Путь к кейсу .txt")
    ap.add_argument("--context", default=Path("out/context.json"), type=Path)
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--ollama", default="http://host.docker.internal:11434")
    ap.add_argument("--outdir", default=Path("out"), type=Path)
    args = ap.parse_args()

    case_text = load_case(args.case)
    ctx = build_context(Path(args.context))

    user_prompt = USER_PROMPT_TEMPLATE.format(case_text=case_text, ctx=ctx)
    # Сжимаем лишние отступы, чтобы экономить токены
    user_prompt = textwrap.dedent(user_prompt).strip()

    try:
        result = call_ollama(args.ollama, args.model, user_prompt)
    except Exception as e:
        # Если модель вернула невалидный JSON
        raise SystemExit(f"LLM call failed or invalid JSON: {e}")

    write_outputs(Path(args.outdir), args.case.stem, result)
    print(f"OK → {args.outdir}/{args.case.stem}.rag.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())