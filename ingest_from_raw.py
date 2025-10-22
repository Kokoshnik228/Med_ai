#!/usr/bin/env python3
"""
Извлекает данные из raw_docs и создает JSONL + manifest.json
(с пропуском уже обработанных и неизменённых файлов)
"""

from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path
from typing import Dict, Any, List
import chardet

# PDF
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# DOCX
try:
    import docx  # python-docx
except Exception:
    docx = None


# === УТИЛИТЫ ======================================================

def file_sha1(p: Path) -> str:
    """Вычислить SHA1-хэш файла (для отслеживания изменений)."""
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def detect_text_file(path: Path) -> str:
    data = path.read_bytes()
    enc = chardet.detect(data).get("encoding") or "utf-8"
    try:
        return data.decode(enc, errors="ignore")
    except Exception:
        return data.decode("utf-8", errors="ignore")


# === ИЗВЛЕЧЕНИЕ ТЕКСТА =============================================

def ingest_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Постраничное извлечение текста из PDF."""
    pages: List[Dict[str, Any]] = []
    page_num = 0
    for layout in extract_pages(str(pdf_path)):
        page_num += 1
        texts: List[str] = []
        for el in layout:
            if isinstance(el, LTTextContainer):
                texts.append(el.get_text())
        txt = "".join(texts).strip()
        pages.append({"page": page_num, "text": txt})
    return pages


def ingest_docx(docx_path: Path) -> List[Dict[str, Any]]:
    """Извлечение текста из DOCX (всё одной страницей)."""
    if not docx:
        raise RuntimeError("python-docx не установлен")
    d = docx.Document(str(docx_path))
    buf = [p.text for p in d.paragraphs]
    return [{"page": 1, "text": "\n".join(buf).strip()}]


def ingest_txt(txt_path: Path) -> List[Dict[str, Any]]:
    """Извлечение текста из TXT."""
    text = detect_text_file(txt_path).strip()
    return [{"page": 1, "text": text}]


# === ОСНОВНОЙ ПРОЦЕСС =============================================

def main():
    ap = argparse.ArgumentParser("Batch ingest RAW → data/*.pages.jsonl + data/manifest.json")
    ap.add_argument("--input-dir", default="raw_docs", help="Папка с PDF/DOCX/TXT")
    ap.add_argument("--out-dir", default="data", help="Куда сохранять JSONL и manifest.json")
    ap.add_argument("--force", action="store_true", help="Перезаписывать даже без изменений")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt"}])
    if not files:
        print(f"В {in_dir} нет pdf/docx/txt", file=sys.stderr)
        return 1

    manifest_path = out_dir / "manifest.json"
    manifest: Dict[str, Any] = {"docs": []}

    # --- Загружаем старый manifest, если есть ---
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or "docs" not in manifest:
                manifest = {"docs": []}
        except Exception:
            manifest = {"docs": []}

    docs = manifest.get("docs", [])
    existing_ids = {doc.get("doc_id") for doc in docs if isinstance(doc, dict)}
    by_source: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        sp = d.get("source_path")
        if isinstance(sp, str):
            by_source[sp] = d

    # --- Основной цикл по файлам ---
    processed = 0
    for f in files:
        src = str(f)
        sha = file_sha1(f)  # считаем хэш файла

        prev = by_source.get(src)
        if prev:
            prev_sha = prev.get("sha1")
            doc_id = prev.get("doc_id") or f.stem
            # если файл не изменился — пропускаем
            if prev_sha == sha and not args.force:
                print(f"→ Без изменений: {f.name}")
                continue
            out_pages = out_dir / f"{doc_id}.pages.jsonl"
        else:
            # новый файл
            stem = f.stem
            doc_id = stem
            suffix = 1
            while doc_id in existing_ids:
                suffix += 1
                doc_id = f"{stem}_{suffix}"
            out_pages = out_dir / f"{doc_id}.pages.jsonl"

        # --- Парсинг ---
        try:
            if f.suffix.lower() == ".pdf":
                pages = ingest_pdf(f)
            elif f.suffix.lower() == ".docx":
                pages = ingest_docx(f)
            else:
                pages = ingest_txt(f)
        except Exception as e:
            print(f"!! Ошибка парсинга {f.name}: {e}", file=sys.stderr)
            continue

        # --- Сохраняем постранично ---
        with out_pages.open("w", encoding="utf-8") as w:
            for p in pages:
                rec = {"doc_id": doc_id, "page": p.get("page", 1), "text": p.get("text", "")}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # --- Обновляем или добавляем запись в manifest ---
        entry = {
            "doc_id": doc_id,
            "source_path": src,
            "pages": len(pages),
            "lang": "ru",
            "sha1": sha,
        }

        if prev:
            prev.update(entry)
        else:
            manifest["docs"].append(entry)
            existing_ids.add(doc_id)

        processed += 1
        print(f"→ {f.name}  →  {out_pages.name}  ({len(pages)} стр.)")

    # --- Сохраняем manifest ---
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nГотово ✅: обработано файлов = {processed}. Обновлён {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
