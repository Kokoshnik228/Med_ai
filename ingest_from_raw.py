#!/usr/bin/env python3
"""
RAW (PDF/DOCX/TXT) -> data/*.pages.jsonl + data/manifest.json
С приоритетом на качественный OCR для сканов.

Особенности:
- OCR backends: tesseract (CPU) / easyocr (GPU при наличии CUDA).
- Режимы OCR: auto (только если мало текста), always (всегда), never (выкл).
- Предобработка изображений (OpenCV): CLAHE, бинаризация, шумоподавление.
- Параллельная обработка файлов.
- Чистка текста: дефисы на переносах, лишние пробелы, латиница→кириллица в частых ошибках.

Требования (в контейнере app):
  pip install chardet pymupdf pillow python-docx
  # для GPU-OCR:
  pip install easyocr opencv-python-headless
"""

from __future__ import annotations
import argparse
import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import chardet

# ---------- опциональные импорты ----------
try:
    import fitz  # PyMuPDF
except Exception as e:
    print("[ERR] Требуется PyMuPDF: pip install pymupdf", file=sys.stderr)
    raise

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    import pytesseract
    from PIL import Image
    TESS_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False

try:
    import easyocr
    EASY_AVAILABLE = True
except Exception:
    EASY_AVAILABLE = False

# OpenCV (для предобработки)
try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except Exception:
    CV_AVAILABLE = False


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
    try:
        needs_colorspace_convert = getattr(pix.colorspace, "n", 3) != 3  # не RGB
    except Exception:
        needs_colorspace_convert = False
    if pix.alpha or needs_colorspace_convert:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


# --- текстовая чистка для OCR ---
_LATIN_TO_CYR = str.maketrans({
    "A":"А","a":"а","B":"В","E":"Е","e":"е","K":"К","k":"к","M":"М","H":"Н","O":"О","o":"о","P":"Р","p":"р","C":"С","c":"с","T":"Т","X":"Х","x":"х","Y":"У","y":"у"
})
def clean_ocr_text(text: str) -> str:
    t = text.replace("\r", "")
    # убрать переносы со знаком дефиса
    t = t.replace("-\n", "")
    # заменить одиночные переводы строки на пробел (сохраним абзацы)
    t = t.replace("\n\n", "<<<PARA>>>").replace("\n", " ").replace("<<<PARA>>>", "\n\n")
    # нормализовать пробелы
    t = " ".join(t.split())
    # частые замены латиницы на кириллицу (не агрессивно)
    t = t.translate(_LATIN_TO_CYR)
    return t.strip()


# ================== OCR Backends ==================

def ocr_page_tesseract(img_pil: "Image.Image", lang: str) -> str:
    if not TESS_AVAILABLE:
        return ""
    cfg = "--oem 1 --psm 6"
    return (pytesseract.image_to_string(img_pil, lang=lang, config=cfg) or "").strip()

def ocr_page_easyocr(img_pil: "Image.Image", reader) -> str:
    if reader is None:
        return ""
    arr = np.array(img_pil.convert("RGB"))
    res = reader.readtext(arr, detail=0, paragraph=True)
    return "\n".join([x.strip() for x in res if x]).strip()

def preprocess_pil(img_pil: "Image.Image") -> "Image.Image":
    """Лёгкая предобработка для OCR (если есть OpenCV)."""
    if not CV_AVAILABLE:
        return img_pil
    img = np.array(img_pil.convert("L"))  # grayscale
    # CLAHE (выравнивание гистограммы)
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img = clahe.apply(img)
    except Exception:
        pass
    # шумоподавление
    img = cv2.fastNlMeansDenoising(img, h=10)
    # бинаризация
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, 15)
    return Image.fromarray(img)


# ================== Извлечение ==================

def ingest_pdf(
    pdf_path: Path,
    *,
    ocr_mode: str,            # "auto"|"always"|"never"
    ocr_backend: str,         # "tesseract"|"easyocr"
    ocr_lang: str,
    dpi: int,
    min_chars: int,
    verbose: bool,
    easy_reader=None
) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[WARN] Не удалось открыть PDF {pdf_path.name}: {e}", file=sys.stderr)
        return pages

    for i, page in enumerate(doc, start=1):
        txt = (page.get_text("text") or "").strip()

        do_ocr = False
        if ocr_mode == "always":
            do_ocr = True
        elif ocr_mode == "auto":
            do_ocr = (len(txt) < min_chars)
        else:  # never
            do_ocr = False

        if do_ocr:
            # Рендер в PIL (dpi влияет лишь на Tesseract; EasyOCR терпим к 2x масштабированию)
            if ocr_backend == "tesseract":
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_pil = pixmap_to_pil(pix)
            else:  # easyocr
                # 2x масштаб для устойчивости к мелкому шрифту
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                img_pil = pixmap_to_pil(pix)

            # Предобработка (OpenCV)
            img_pil = preprocess_pil(img_pil)

            if ocr_backend == "easyocr":
                txt_ocr = ocr_page_easyocr(img_pil, easy_reader)
            else:
                txt_ocr = ocr_page_tesseract(img_pil, ocr_lang)

            if len(txt_ocr) > len(txt):
                txt = txt_ocr
                if verbose:
                    print(f"[OCR-{ocr_backend}] {pdf_path.name} p.{i}: len={len(txt)}")
            elif verbose and ocr_mode != "never":
                print(f"[OCR-{ocr_backend}] {pdf_path.name} p.{i}: OCR не улучшил текст (len={len(txt)})")

        if txt:
            txt = clean_ocr_text(txt)

        pages.append({"page": i, "text": txt})

    return pages


def ingest_docx(docx_path: Path) -> List[Dict[str, Any]]:
    if not docx:
        raise RuntimeError("python-docx не установлен (pip install python-docx)")
    d = docx.Document(str(docx_path))
    buf = [p.text for p in d.paragraphs]
    return [{"page": 1, "text": clean_ocr_text("\n".join(buf).strip())}]


def ingest_txt(txt_path: Path) -> List[Dict[str, Any]]:
    text = detect_text_file(txt_path).strip()
    return [{"page": 1, "text": clean_ocr_text(text)}]


# ================== Основной процесс ==================

def process_one_file(
    f: Path,
    out_dir: Path,
    *,
    force: bool,
    min_chars: int,
    ocr_mode: str,
    ocr_backend: str,
    ocr_lang: str,
    dpi: int,
    verbose: bool
) -> Dict[str, Any]:
    """Процессинг одного файла (без доступа к manifest). Возвращает entry + путь pages."""
    sha = file_sha1(f)
    stem = f.stem
    doc_id = stem  # уникализацию по doc_id решаем на верхнем уровне, если потребуется
    out_pages = out_dir / f"{doc_id}.pages.jsonl"

    # OCR init (easyocr локально в процессе)
    easy_reader = None
    if ocr_mode != "never" and ocr_backend == "easyocr" and EASY_AVAILABLE:
        use_gpu = False
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            pass
        easy_reader = easyocr.Reader(['ru','en'], gpu=use_gpu)

    # Парсинг
    if f.suffix.lower() == ".pdf":
        pages = ingest_pdf(
            f, ocr_mode=ocr_mode, ocr_backend=ocr_backend, ocr_lang=ocr_lang,
            dpi=dpi, min_chars=min_chars, verbose=verbose, easy_reader=easy_reader
        )
    elif f.suffix.lower() == ".docx":
        pages = ingest_docx(f)
    else:
        pages = ingest_txt(f)

    # Сохраняем
    with out_pages.open("w", encoding="utf-8") as w:
        for p in pages:
            rec = {"doc_id": doc_id, "page": p.get("page", 1), "text": p.get("text", "")}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    empty_pages = sum(1 for p in pages if not (p.get("text") or "").strip())

    return {
        "doc_id": doc_id,
        "source_path": str(f),
        "pages": len(pages),
        "lang": "ru",
        "sha1": sha,
        "ocr_backend": ocr_backend,
        "ocr_mode": ocr_mode,
        "ocr_lang": ocr_lang,
        "dpi": dpi,
        "min_chars": min_chars,
        "empty_pages": empty_pages,
        "out_pages": str(out_pages)
    }


def main():
    ap = argparse.ArgumentParser("RAW -> data/*.pages.jsonl (+manifest) с качественным OCR")
    ap.add_argument("--input-dir", default="raw_docs", help="Папка с PDF/DOCX/TXT")
    ap.add_argument("--out-dir", default="data", help="Куда сохранять JSONL и manifest.json")
    ap.add_argument("--force", action="store_true", help="Перепарсить даже без изменений")

    # OCR-настройки
    ap.add_argument("--ocr-mode", choices=["auto","always","never"], default="auto",
                    help="auto: только если мало текста; always: OCR на всех страницах; never: без OCR")
    ap.add_argument("--ocr-backend", choices=["tesseract","easyocr"], default="easyocr",
                    help="Движок OCR")
    ap.add_argument("--ocr-lang", default="rus+eng", help="Языки для tesseract (ignored для easyocr)")
    ap.add_argument("--min-chars", type=int, default=60, help="Порог символов для запуска OCR в режиме auto")
    ap.add_argument("--dpi", type=int, default=300, help="DPI рендера для tesseract")

    # Прочее
    ap.add_argument("--workers", type=int, default=0, help="Параллелизм по файлам (0=CPU count)")
    ap.add_argument("--verbose", action="store_true", help="Подробный лог")

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
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or "docs" not in manifest:
                manifest = {"docs": []}
        except Exception:
            manifest = {"docs": []}

    docs = manifest.get("docs", [])
    by_source: Dict[str, Dict[str, Any]] = {d.get("source_path"): d for d in docs if isinstance(d, dict)}
    existing_ids = {d.get("doc_id") for d in docs if isinstance(d, dict)}

    # План работ: перепарсить только изменённые или force
    plan: List[Path] = []
    for f in files:
        sha = file_sha1(f)
        prev = by_source.get(str(f))
        if args.force or not prev or prev.get("sha1") != sha:
            plan.append(f)
        elif args.verbose:
            print(f"→ Без изменений: {f.name}")

    if not plan:
        print("Нет изменений — ничего делать не нужно.")
        return 0

    # Параллелизм по файлам
    workers = args.workers or max(1, __import__("os").cpu_count() or 1)
    results: List[Dict[str, Any]] = []

    if workers <= 1 or len(plan) == 1:
        for f in plan:
            r = process_one_file(
                f, out_dir, force=args.force, min_chars=args.min_chars,
                ocr_mode=args.ocr_mode, ocr_backend=args.ocr_backend, ocr_lang=args.ocr_lang,
                dpi=args.dpi, verbose=args.verbose
            )
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    process_one_file, f, out_dir,
                    force=args.force, min_chars=args.min_chars,
                    ocr_mode=args.ocr_mode, ocr_backend=args.ocr_backend, ocr_lang=args.ocr_lang,
                    dpi=args.dpi, verbose=args.verbose
                ): f for f in plan
            }
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"[ERR] {futs[fut].name}: {e}", file=sys.stderr)

    # Обновляем manifest
    for r in results:
        src = r["source_path"]
        doc_id = r["doc_id"]
        # уникализируем doc_id при коллизии
        if doc_id in existing_ids:
            base = doc_id
            suf = 2
            while f"{base}_{suf}" in existing_ids:
                suf += 1
            new_id = f"{base}_{suf}"
            # переименуем файл jsonl
            op = Path(r["out_pages"])
            op_renamed = op.with_name(f"{new_id}.pages.jsonl")
            op.rename(op_renamed)
            r["doc_id"] = new_id
            r["out_pages"] = str(op_renamed)
            doc_id = new_id
        existing_ids.add(doc_id)

        prev = by_source.get(src)
        entry = {
            "doc_id": doc_id,
            "source_path": src,
            "pages": r["pages"],
            "lang": "ru",
            "sha1": r["sha1"],
            "ocr": (args.ocr_mode != "never"),
            "ocr_mode": args.ocr_mode,
            "ocr_backend": args.ocr_backend,
            "ocr_lang": args.ocr_lang,
            "dpi": args.dpi,
            "min_chars": args.min_chars,
            "empty_pages": r["empty_pages"]
        }
        if prev:
            prev.update(entry)
        else:
            manifest["docs"].append(entry)

        if args.verbose and r["empty_pages"]:
            print(f"[WARN] {Path(src).name}: пустых страниц {r['empty_pages']}")

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nГотово ✅: обработано файлов = {len(results)}. Обновлён {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
