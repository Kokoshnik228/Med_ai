#!/usr/bin/env python3
"""
Извлекает данные из raw_docs и создаёт JSONL + manifest.json
(с пропуском уже обработанных и неизменённых файлов)

Особенности:
- PDF через PyMuPDF (fitz). Если текста мало — OCR fallback (pytesseract).
- DOCX/TXT поддерживаются как раньше.
- Диагностика: где включался OCR и почему.
"""

from __future__ import annotations
import argparse
import json
import sys
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import chardet

# --- DOCX (опционально)
try:
    import docx  # python-docx
except Exception:
    docx = None

# --- PDF + OCR
try:
    import fitz  # PyMuPDF
except Exception as e:
    print("[ERR] Требуется PyMuPDF: pip install pymupdf", file=sys.stderr)
    raise

try:
    import pytesseract
    from PIL import Image
    TESS_AVAILABLE_IMPORT = True
except Exception:
    TESS_AVAILABLE_IMPORT = False


# ================== Утилиты ==================

def file_sha1(p: Path) -> str:
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


def pixmap_to_pil(pix: "fitz.Pixmap") -> "Image.Image":
    """Надёжная конверсия Pixmap → PIL.Image (учёт альфы и не-RGB пространств)."""
    try:
        needs_colorspace_convert = getattr(pix.colorspace, "n", 3) != 3  # не RGB
    except Exception:
        needs_colorspace_convert = False
    if pix.alpha or needs_colorspace_convert:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def setup_tesseract(verbose: bool = False) -> bool:
    """
    Настраивает pytesseract.pytesseract.tesseract_cmd, если бинарник найден.
    Возвращает True, если tesseract доступен; дополнительно подсказка по языкам.
    """
    if not TESS_AVAILABLE_IMPORT:
        if verbose:
            print("[WARN] pytesseract/Pillow не установлены — OCR недоступен.", file=sys.stderr)
        return False

    cand = shutil.which("tesseract")
    if not cand:
        for p in ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]:
            if Path(p).exists():
                cand = p
                break

    if not cand:
        if verbose:
            print("[WARN] Бинарник 'tesseract' не найден в PATH — OCR недоступен.", file=sys.stderr)
        return False

    pytesseract.pytesseract.tesseract_cmd = cand
    if verbose:
        print(f"[OK] Tesseract найден: {cand}")

    # Пытаемся предупредить, если языки rus/eng не установлены
    try:
        out = subprocess.run([cand, "--list-langs"], capture_output=True, text=True, timeout=5)
        langs = (out.stdout or "") + (out.stderr or "")
        if "rus" not in langs and verbose:
            print("[WARN] Похоже, языковой пакет 'rus' не установлен: sudo apt install tesseract-ocr-rus", file=sys.stderr)
        if "eng" not in langs and verbose:
            print("[WARN] Похоже, языковой пакет 'eng' не установлен: sudo apt install tesseract-ocr-eng", file=sys.stderr)
    except Exception:
        if verbose:
            print("[INFO] Не удалось проверить установленные языки tesseract.", file=sys.stderr)

    return True


# ================== Извлечение текста ==================

def ocr_page(page: "fitz.Page", dpi: int, lang: str) -> str:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = pixmap_to_pil(pix).convert("L")
    txt = pytesseract.image_to_string(img, lang=lang)
    return (txt or "").strip()


def ingest_pdf(pdf_path: Path, *,
               min_chars: int,
               ocr: bool,
               ocr_lang: str,
               dpi: int,
               verbose: bool) -> List[Dict[str, Any]]:

    pages: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[WARN] Не удалось открыть PDF {pdf_path.name}: {e}", file=sys.stderr)
        return pages

    need_ocr_but_unavailable = False

    for i, page in enumerate(doc, start=1):
        txt = (page.get_text("text") or "").strip()
        used_ocr = False

        if ocr and len(txt) < min_chars:
            if TESS_AVAILABLE_IMPORT and getattr(pytesseract.pytesseract, "tesseract_cmd", None):
                try:
                    txt_ocr = ocr_page(page, dpi=dpi, lang=ocr_lang)
                    if len(txt_ocr) > len(txt):
                        txt = txt_ocr
                        used_ocr = True
                        if verbose:
                            print(f"[OCR] {pdf_path.name} p.{i}: OCR применён (len={len(txt)})")
                    elif verbose:
                        print(f"[OCR] {pdf_path.name} p.{i}: OCR не улучшил текст.")
                except Exception as e:
                    print(f"[WARN] OCR ошибка на {pdf_path.name} p.{i}: {e}", file=sys.stderr)
            else:
                need_ocr_but_unavailable = True
                if verbose:
                    print(f"[WARN] {pdf_path.name} p.{i}: OCR необходим, но tesseract недоступен.", file=sys.stderr)

        if verbose and not used_ocr and len(txt) >= min_chars:
            print(f"[TXT]  {pdf_path.name} p.{i}: достаточно текста без OCR (len={len(txt)})")

        pages.append({"page": i, "text": txt})

    if need_ocr_but_unavailable:
        print(
            f"[ERROR] Для {pdf_path.name} требовался OCR, но tesseract не найден. "
            f"Установи: sudo apt install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng",
            file=sys.stderr
        )

    return pages


def ingest_docx(docx_path: Path) -> List[Dict[str, Any]]:
    if not docx:
        raise RuntimeError("python-docx не установлен (pip install python-docx)")
    d = docx.Document(str(docx_path))
    buf = [p.text for p in d.paragraphs]
    return [{"page": 1, "text": "\n".join(buf).strip()}]


def ingest_txt(txt_path: Path) -> List[Dict[str, Any]]:
    text = detect_text_file(txt_path).strip()
    return [{"page": 1, "text": text}]


# ================== Основной процесс ==================

def main():
    ap = argparse.ArgumentParser("RAW → data/*.pages.jsonl + data/manifest.json (с OCR-fallback)")
    ap.add_argument("--input-dir", default="raw_docs", help="Папка с PDF/DOCX/TXT")
    ap.add_argument("--out-dir", default="data", help="Куда сохранять JSONL и manifest.json")
    ap.add_argument("--force", action="store_true", help="Перепарсить даже без изменений")

    # OCR-настройки
    ap.add_argument("--ocr", action="store_true", default=True, help="Включить OCR для сканов (по умолчанию ВКЛ)")
    ap.add_argument("--no-ocr", action="store_false", dest="ocr", help="Выключить OCR")
    ap.add_argument("--ocr-lang", default="rus+eng", help="Языки OCR (например, 'rus+eng')")
    ap.add_argument("--min-chars", type=int, default=25, help="Порог символов, ниже которого включаем OCR")
    ap.add_argument("--dpi", type=int, default=250, help="DPI рендера страницы для OCR")
    ap.add_argument("--verbose", action="store_true", help="Подробный лог")

    args = ap.parse_args()

    # Настроим tesseract, если OCR включён
    if args.ocr:
        ok = setup_tesseract(verbose=args.verbose)
        if not ok:
            print(
                "[WARN] OCR включён, но tesseract не найден. Страницы-сканы останутся пустыми. "
                "Поставь: sudo apt install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng",
                file=sys.stderr
            )

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt"}])
    if not files:
        print(f"В {in_dir} нет pdf/docx/txt", file=sys.stderr)
        return 1

    manifest_path = out_dir / "manifest.json"
    manifest: Dict[str, Any] = {"docs": []}

    # Подтянем предыдущий manifest
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or "docs" not in manifest:
                manifest = {"docs": []}
        except Exception:
            manifest = {"docs": []}

    docs = manifest.get("docs", [])
    existing_ids = {d.get("doc_id") for d in docs if isinstance(d, dict)}
    by_source: Dict[str, Dict[str, Any]] = {d.get("source_path"): d for d in docs if isinstance(d, dict)}

    processed = 0
    for f in files:
        src = str(f)
        sha = file_sha1(f)

        prev = by_source.get(src)
        if prev:
            prev_sha = prev.get("sha1")
            doc_id = prev.get("doc_id") or f.stem
            if prev_sha == sha and not args.force:
                if args.verbose:
                    print(f"→ Без изменений: {f.name}")
                continue
            out_pages = out_dir / f"{doc_id}.pages.jsonl"
        else:
            stem = f.stem
            doc_id = stem
            suffix = 1
            while doc_id in existing_ids:
                suffix += 1
                doc_id = f"{stem}_{suffix}"
            out_pages = out_dir / f"{doc_id}.pages.jsonl"

        # Парсинг
        try:
            if f.suffix.lower() == ".pdf":
                pages = ingest_pdf(
                    f,
                    min_chars=args.min_chars,
                    ocr=args.ocr,
                    ocr_lang=args.ocr_lang,
                    dpi=args.dpi,
                    verbose=args.verbose,
                )
            elif f.suffix.lower() == ".docx":
                pages = ingest_docx(f)
            else:
                pages = ingest_txt(f)
        except Exception as e:
            print(f"!! Ошибка парсинга {f.name}: {e}", file=sys.stderr)
            continue

        # Сохраняем
        with out_pages.open("w", encoding="utf-8") as w:
            for p in pages:
                rec = {"doc_id": doc_id, "page": p.get("page", 1), "text": p.get("text", "")}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")

        empty_pages = sum(1 for p in pages if not (p.get("text") or "").strip())
        if empty_pages and args.ocr and not getattr(pytesseract.pytesseract, "tesseract_cmd", None):
            print(
                f"[ERROR] {f.name}: обнаружены пустые страницы, а OCR недоступен. "
                f"Установи tesseract и языки (rus/eng) и перезапусти с --force.",
                file=sys.stderr
            )

        entry = {
            "doc_id": doc_id,
            "source_path": src,
            "pages": len(pages),
            "lang": "ru",
            "sha1": sha,
            "ocr": bool(args.ocr),
            "ocr_lang": args.ocr_lang,
            "min_chars": args.min_chars,
        }

        if prev:
            prev.update(entry)
        else:
            manifest["docs"].append(entry)
            existing_ids.add(doc_id)

        processed += 1
        suffix_info = f", пустых страниц: {empty_pages}" if empty_pages else ""
        print(f"→ {f.name} → {out_pages.name} (стр.: {len(pages)}{suffix_info})")

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nГотово ✅: обработано файлов = {processed}. Обновлён {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
