#!/usr/bin/env python3
"""
RAW (PDF/DOCX/TXT) -> data/*.pages.jsonl + data/manifest.json

Особенности:
- Рекурсивный обход --input-dir (по умолчанию raw_docs/).
- PDF: извлечение текста, при необходимости OCR (tesseract или easyocr).
- DOCX/TXT: нарезка на «псевдо-страницы» фиксированной длины (по умолчанию 1800 символов),
  чтобы down-stream (BM25/Qdrant/цитаты) работал одинаково с PDF.
- Инкрементальность: обрабатываем только новые/изменённые файлы (по SHA1), если НЕ указан --force.
- EasyOCR: «тёплый старт» для скачивания моделей один раз; в воркерах скачивание отключено.
- GPU-настройки для EasyOCR: можно принудительно задать устройство (--ocr-gpu cuda|cpu|auto),
  путь к кэшу моделей и возможность скачивания (--easyocr-allow-downloads).
- ✂️ АНТИ-ЛИТЕРАТУРА: автоматически вырезаем разделы «Список литературы / References».

Зависимости (в контейнере app):
  pip install chardet pymupdf pillow python-docx
  # для OCR (опционально):
  apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus
  pip install easyocr opencv-python-headless
"""

from __future__ import annotations
import argparse
import json
import sys
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from time import perf_counter

import numpy as np
import chardet

# ---------- опциональные импорты ----------
try:
    import fitz  # PyMuPDF
except Exception as e:
    print("[ERR] Требуется PyMuPDF: pip install pymupdf", file=sys.stderr, flush=True)
    raise

# python-docx==1.1.2
try:
    from docx import Document
except Exception:
    Document = None  # обработаем ниже

try:
    import pytesseract
    from PIL import Image
    TESS_AVAILABLE = True
except Exception:
    from PIL import Image  # Pillow всё равно требуется
    TESS_AVAILABLE = False

try:
    import easyocr
    EASY_AVAILABLE = True
except Exception:
    EASY_AVAILABLE = False

# OpenCV (предобработка для OCR)
try:
    import cv2
    CV_AVAILABLE = True
except Exception:
    CV_AVAILABLE = False

# Torch для GPU-инфо (опционально; не падаем, если отсутствует)
try:
    import torch
    _TORCH_OK = True
except Exception:
    torch = None
    _TORCH_OK = False


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


# --- текстовая чистка для OCR/парсеров ---
_LATIN_TO_CYR = str.maketrans({
    "A":"А","a":"а","B":"В","E":"Е","e":"е","K":"К","k":"к","M":"М","H":"Н","O":"О","o":"о",
    "P":"Р","p":"р","C":"С","c":"с","T":"Т","X":"Х","x":"х","Y":"У","y":"у"
})

def clean_text(text: str) -> str:
    t = (text or "").replace("\r", "")
    t = re.sub(r"-\n", "", t)                                # убрать переносы со знаком дефиса
    t = t.replace("\n\n", "<<<PARA>>>").replace("\n", " ").replace("<<<PARA>>>", "\n\n")
    t = re.sub(r"[ \t]+", " ", t)                            # нормализовать пробелы
    t = t.translate(_LATIN_TO_CYR)                           # частые замены латиницы на кириллицу
    return t.strip()

# --- детектор «Списка литературы» / References ---
REFS_HDR_RE = re.compile(
    r'^\s*(?:'
    r'спис[оo]к\s+литератур[аы]|литератур[аы]|источники|'
    r'использованн[а-яё]+\s+литератур[аы]|'
    r'references?|bibliograph\w*'
    r')\s*[:\-–—]?\s*$',
    re.IGNORECASE | re.MULTILINE
)

def strip_tail_references(text: str) -> tuple[str, bool]:
    """
    Найти заголовок библиографии и обрезать всё, что идёт после него.
    Возвращает (новый_текст, было_ли_обрезание).
    """
    if not text:
        return text, False
    m = REFS_HDR_RE.search(text)
    if not m:
        return text, False
    return text[:m.start()].rstrip(), True


def split_text_to_pages(full_text: str, page_size_chars: int = 1800) -> List[Dict[str, Any]]:
    """Нарезаем длинный текст на «псевдо-страницы» по символам, стараясь уважать абзацы."""
    text = clean_text(full_text)
    # попытка обрезать литературу и для текстов/доксов
    text, _ = strip_tail_references(text)

    if not text:
        return [{"page": 1, "text": ""}]

    parts: List[str] = []
    buf = []
    cur_len = 0
    # грубо по абзацам/предложениям
    tokens = re.split(r"(\n\n|[.!?]\s+)", text)
    for t in tokens:
        if t is None:
            continue
        if cur_len + len(t) > page_size_chars and buf:
            parts.append("".join(buf).strip())
            buf, cur_len = [t], len(t)
        else:
            buf.append(t)
            cur_len += len(t)
    if buf:
        parts.append("".join(buf).strip())

    pages: List[Dict[str, Any]] = []
    for i, chunk in enumerate(parts, start=1):
        pages.append({"page": i, "text": chunk})

    return pages or [{"page": 1, "text": text[:page_size_chars]}]


# ================== DOCX ==================

def extract_docx_text(path: Path) -> str:
    """Извлечь текст из DOCX (абзацы + таблицы)."""
    if Document is None:
        print("[ERR] python-docx не установлен. Добавь `python-docx` в зависимости.", file=sys.stderr, flush=True)
        return ""
    try:
        doc = Document(str(path))
    except Exception as e:
        print(f"[ERR] Не удалось открыть DOCX {path.name}: {e}", file=sys.stderr, flush=True)
        return ""

    parts: List[str] = []
    # абзацы
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    # таблицы
    for table in doc.tables:
        for row in table.rows:
            cells = [ (c.text or "").strip() for c in row.cells ]
            line = " | ".join([c for c in cells if c])
            if line:
                parts.append(line)

    return "\n".join(parts).strip()


def ingest_docx(path: Path, page_size_chars: int = 1800) -> List[Dict[str, Any]]:
    """DOCX -> список {page, text} (как у PDF/TXT)."""
    raw = extract_docx_text(path)
    if not raw:
        return [{"page": 1, "text": ""}]
    # ✂️ обрезаем литературу до нарезки
    raw, _ = strip_tail_references(raw)
    return split_text_to_pages(raw, page_size_chars=page_size_chars)


# ================== OCR Backends ==================

def ocr_page_tesseract(img_pil: "Image.Image", lang: str) -> str:
    if not TESS_AVAILABLE:
        return ""
    # NB: ограничиваем треды, чтобы не душить систему
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
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
        print(f"[WARN] Не удалось открыть PDF {pdf_path.name}: {e}", file=sys.stderr, flush=True)
        return pages

    # как только встретили заголовок «Литература/References» — обрежем и остановимся
    for i, page in enumerate(doc, start=1):
        txt = (page.get_text("text") or "").strip()

        # решаем, делать ли OCR
        if ocr_mode == "always":
            do_ocr = True
        elif ocr_mode == "auto":
            do_ocr = (len(txt) < min_chars)
        else:  # "never"
            do_ocr = False

        # сам OCR (ВНЕ ветки "never")
        if do_ocr:
            if ocr_backend == "tesseract":
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_pil = pixmap_to_pil(pix)

                img_pil = preprocess_pil(img_pil)
                txt_ocr = ocr_page_tesseract(img_pil, ocr_lang)

            elif ocr_backend == "easyocr":
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                img_pil = pixmap_to_pil(pix)

                img_pil = preprocess_pil(img_pil)
                txt_ocr = ocr_page_easyocr(img_pil, easy_reader)

            else:
                txt_ocr = ""

            if len(txt_ocr) > len(txt):
                txt = txt_ocr
                if verbose:
                    print(f"[OCR-{ocr_backend}] {pdf_path.name} p.{i}: len={len(txt)}", flush=True)
            elif verbose and ocr_mode != "never" and txt_ocr:
                print(f"[OCR-{ocr_backend}] {pdf_path.name} p.{i}: OCR не улучшил текст (len={len(txt)})", flush=True)

        # ✂️ сначала пробуем обрезать до чистки
        if txt:
            txt_cut, cut0 = strip_tail_references(txt)
            if cut0:
                txt = txt_cut

        if txt:
            txt = clean_text(txt)

        # ✂️ ещё раз пробуем после clean_text (на случай OCR-«каши» в заголовке)
        if txt:
            txt_cut2, cut1 = strip_tail_references(txt)
            if cut1:
                txt = txt_cut2

        pages.append({"page": i, "text": txt})

        # если был срез «литературы» на этой странице — прекращаем разбор следующих страниц
        if ('cut0' in locals() and cut0) or ('cut1' in locals() and cut1):
            if verbose:
                print(f"[CUT] {pdf_path.name}: обнаружен раздел литературы на странице {i}, дальнейшие страницы пропущены", flush=True)
            break

    return pages


def ingest_txt(txt_path: Path, page_size_chars: int = 1800) -> List[Dict[str, Any]]:
    text = detect_text_file(txt_path).strip()
    # ✂️ отрезаем литературу
    text, _ = strip_tail_references(text)
    return split_text_to_pages(text, page_size_chars=page_size_chars)


# ================== Основной процесс ==================

def choose_ocr_backend(requested: str) -> str:
    """
    Возвращает эффективный backend OCR с учётом доступности библиотек.
    priority: requested → fallback tesseract → 'none'
    """
    req = (requested or "").lower()
    if req == "easyocr" and EASY_AVAILABLE:
        return "easyocr"
    if req == "tesseract" and TESS_AVAILABLE:
        return "tesseract"
    if EASY_AVAILABLE:
        return "easyocr"
    if TESS_AVAILABLE:
        return "tesseract"
    return "none"  # OCR недоступен


def _easyocr_models_ready(model_dir: Path) -> bool:
    try:
        mdir = model_dir / "model"
        return mdir.exists() and any(mdir.iterdir())
    except Exception:
        return False


def process_one_file(
    f: Path,
    out_dir: Path,
    *,
    min_chars: int,
    ocr_mode: str,
    ocr_backend_eff: str,
    ocr_lang: str,
    dpi: int,
    verbose: bool,
    page_size_chars: int,
    easyocr_dir: Path,
    easyocr_use_gpu: bool,
    easyocr_allow_downloads: bool,
) -> Dict[str, Any]:
    """Процессинг одного файла (без доступа к manifest). Возвращает entry + путь pages."""
    sha = file_sha1(f)
    stem = f.stem
    doc_id = stem  # уникализацию по doc_id решаем на верхнем уровне, если потребуется
    out_pages = out_dir / f"{doc_id}.pages.jsonl"

    # OCR init (EasyOCR в этом процессе; скачивания по флагу)
    easy_reader = None
    if ocr_mode != "never" and ocr_backend_eff == "easyocr":
        easy_reader = easyocr.Reader(
            ['ru', 'en'],
            gpu=easyocr_use_gpu,
            model_storage_directory=str(easyocr_dir),
            download_enabled=bool(easyocr_allow_downloads),
            verbose=False,
        )

    # Парсинг
    ext = f.suffix.lower()
    if ext == ".pdf":
        pages = ingest_pdf(
            f, ocr_mode=ocr_mode, ocr_backend=ocr_backend_eff, ocr_lang=ocr_lang,
            dpi=dpi, min_chars=min_chars, verbose=verbose, easy_reader=easy_reader
        )
    elif ext == ".docx":
        pages = ingest_docx(f, page_size_chars=page_size_chars)
    else:
        pages = ingest_txt(f, page_size_chars=page_size_chars)

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
        "ocr_backend": ocr_backend_eff,
        "ocr_mode": ocr_mode,
        "ocr_lang": ocr_lang,
        "dpi": dpi,
        "min_chars": min_chars,
        "empty_pages": empty_pages,
        "out_pages": str(out_pages)
    }


def _decide_easyocr_gpu(ocr_gpu_arg: str) -> bool:
    """Возвращает True если следует использовать GPU для EasyOCR."""
    ocr_gpu_arg = (ocr_gpu_arg or "auto").lower()
    if ocr_gpu_arg == "cuda":
        return _TORCH_OK and torch.cuda.is_available()
    if ocr_gpu_arg == "cpu":
        return False
    # auto
    return _TORCH_OK and torch.cuda.is_available()


def main():
    ap = argparse.ArgumentParser("RAW -> data/*.pages.jsonl (+manifest) с OCR и поддержкой DOCX/TXT")
    ap.add_argument("--input-dir", default="raw_docs", help="Папка с PDF/DOCX/TXT (рекурсивно)")
    ap.add_argument("--out-dir", default="data", help="Куда сохранять JSONL и manifest.json")
    ap.add_argument("--force", action="store_true", help="Перепарсить даже без изменений")

    # OCR-настройки (можно задать через env)
    ap.add_argument("--ocr-mode", choices=["auto","always","never"], default=os.getenv("OCR_MODE", "auto"),
                    help="auto: только если мало текста; always: OCR на всех страницах; never: без OCR")
    ap.add_argument("--ocr-backend", choices=["tesseract","easyocr"], default=os.getenv("OCR_BACKEND", "easyocr"),
                    help="Желаемый движок OCR (будет авто-фоллбек)")
    ap.add_argument("--ocr-lang", default=os.getenv("TESS_LANG", "rus+eng"), help="Языки для tesseract")
    ap.add_argument("--min-chars", type=int, default=int(os.getenv("MIN_CHARS", "60")),
                    help="Порог символов для запуска OCR в режиме auto")
    ap.add_argument("--dpi", type=int, default=int(os.getenv("OCR_DPI", "300")), help="DPI рендера для tesseract")

    # EasyOCR/GPU
    ap.add_argument("--ocr-gpu", choices=["auto","cpu","cuda"], default=os.getenv("OCR_GPU", "auto"),
                    help="Устройство для EasyOCR (auto/cpu/cuda)")
    ap.add_argument("--easyocr-dir", default=os.getenv("EASYOCR_DIR", str(Path.home() / ".EasyOCR")),
                    help="Каталог для моделей EasyOCR")
    ap.add_argument("--easyocr-allow-downloads", action="store_true",
                    default=os.getenv("EASYOCR_ALLOW_DOWNLOADS", "0").lower() in ("1","true","yes"),
                    help="Разрешить скачивание моделей EasyOCR в этом запуске")

    # Прочее
    ap.add_argument("--workers", type=int, default=0, help="Параллелизм по файлам (0=CPU count)")
    ap.add_argument("--page-size-chars", type=int, default=1800, help="Размер «псевдо-страницы» для DOCX/TXT")
    ap.add_argument("--verbose", action="store_true", help="Подробный лог")

    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Рекурсивно собираем файлы
    allowed = {".pdf", ".docx", ".txt"}
    files: List[Path] = []
    for p in in_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in allowed:
            continue
        name = p.name
        if name.startswith("~$"):  # временные файлы MS Office
            continue
        files.append(p)
    files = sorted(files)

    if not files:
        print(f"В {in_dir} нет pdf/docx/txt", file=sys.stderr, flush=True)
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

    # План работ: перепарсить только новые/изменённые или --force
    plan: List[Path] = []
    for f in files:
        sha = file_sha1(f)
        prev = by_source.get(str(f))
        if args.force or not prev or prev.get("sha1") != sha:
            plan.append(f)
        elif args.verbose:
            try:
                rel = f.relative_to(in_dir)
            except Exception:
                rel = f
            print(f"→ Без изменений: {rel}", flush=True)

    if not plan:
        print("Нет изменений — ничего делать не нужно.", flush=True)
        return 0

    # Эффективный OCR backend с учётом установленного
    ocr_backend_eff = choose_ocr_backend(args.ocr_backend)
    if ocr_backend_eff == "none":
        args.ocr_mode = "never"

    easyocr_dir = Path(args.easyocr_dir).expanduser()
    ensure_dir(easyocr_dir / "model")

    # EasyOCR warmup: если нужно и моделей ещё нет
    need_easy_warmup = (
        args.ocr_mode != "never"
        and ocr_backend_eff == "easyocr"
        and not _easyocr_models_ready(easyocr_dir)
    )
    if need_easy_warmup:
        print("⏳ EasyOCR warmup: загрузка моделей (один раз)...", flush=True)
        use_gpu = _decide_easyocr_gpu(args.ocr_gpu)
        # Загрузка моделей один раз (по флагу)
        easyocr.Reader(['ru','en'], gpu=use_gpu,
                       model_storage_directory=str(easyocr_dir),
                       download_enabled=bool(args.easyocr_allow_downloads),
                       verbose=False)
        # первый прогон — без параллелизма (исключаем гонки)
        args.workers = 1

    # Параллелизм по файлам
    workers = args.workers or max(1, os.cpu_count() or 1)

    # Если EasyOCR — предпочитаем одиночный процесс, чтобы дать CUDA работать стабильно
    easyocr_use_gpu = False
    if args.ocr_mode != "never" and ocr_backend_eff == "easyocr":
        if workers > 1:
            print("⚠️ EasyOCR: переключаюсь на workers=1 для стабильности и GPU.", flush=True)
            workers = 1
        easyocr_use_gpu = _decide_easyocr_gpu(args.ocr_gpu)
        print(f"EasyOCR init планируется с GPU={easyocr_use_gpu}", flush=True)

    total = len(plan)
    print(f"📦 Ingest started: {total} files (workers={workers}, ocr={args.ocr_mode}/{ocr_backend_eff})", flush=True)

    results: List[Dict[str, Any]] = []
    t_start = perf_counter()

    if workers <= 1 or len(plan) == 1:
        for i, f in enumerate(plan, 1):
            t0 = perf_counter()
            try:
                r = process_one_file(
                    f, out_dir,
                    min_chars=args.min_chars,
                    ocr_mode=args.ocr_mode,
                    ocr_backend_eff=ocr_backend_eff,
                    ocr_lang=args.ocr_lang,
                    dpi=args.dpi,
                    verbose=args.verbose,
                    page_size_chars=args.page_size_chars,
                    easyocr_dir=easyocr_dir,
                    easyocr_use_gpu=easyocr_use_gpu,
                    easyocr_allow_downloads=False,  # в рабочих процессах по умолчанию не качаем
                )
                results.append(r)
                dt = perf_counter() - t0
                print(f"[{i}/{total}] {f.name}: {r['pages']} pages, empty={r['empty_pages']}, ocr={r['ocr_backend']}/{r['ocr_mode']} ({dt:.2f}s)", flush=True)
            except Exception as e:
                print(f"[ERR] {f.name}: {e}", file=sys.stderr, flush=True)
    else:
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
            futs = {
                ex.submit(
                    process_one_file, f, out_dir,
                    min_chars=args.min_chars,
                    ocr_mode=args.ocr_mode,
                    ocr_backend_eff=ocr_backend_eff,
                    ocr_lang=args.ocr_lang,
                    dpi=args.dpi,
                    verbose=args.verbose,
                    page_size_chars=args.page_size_chars,
                    easyocr_dir=easyocr_dir,
                    easyocr_use_gpu=False,  # в мультипроцессе — без CUDA
                    easyocr_allow_downloads=False,
                ): f for f in plan
            }
            done = 0
            for fut in as_completed(futs):
                f = futs[fut]
                try:
                    r = fut.result()
                    results.append(r)
                    done += 1
                    print(f"[{done}/{total}] {f.name}: {r['pages']} pages, empty={r['empty_pages']}, ocr={r['ocr_backend']}/{r['ocr_mode']}", flush=True)
                except Exception as e:
                    print(f"[ERR] {f.name}: {e}", file=sys.stderr, flush=True)

    # Обновляем manifest и уникализируем doc_id при коллизии
    for r in results:
        src = r["source_path"]
        doc_id = r["doc_id"]

        if doc_id in existing_ids:
            base = doc_id
            suf = 2
            while f"{base}_{suf}" in existing_ids:
                suf += 1
            new_id = f"{base}_{suf}"
            # переименуем файл jsonl
            op = Path(r["out_pages"])
            op_renamed = op.with_name(f"{new_id}.pages.jsonl")
            try:
                op.rename(op_renamed)
            except FileNotFoundError:
                pass
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
            "ocr": (args.ocr_mode != "never" and ocr_backend_eff != "none"),
            "ocr_mode": r["ocr_mode"],
            "ocr_backend": r["ocr_backend"],
            "ocr_lang": r["ocr_lang"],
            "dpi": r["dpi"],
            "min_chars": r["min_chars"],
            "empty_pages": r["empty_pages"],
        }
        if prev:
            prev.update(entry)
        else:
            manifest["docs"].append(entry)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    total_dt = perf_counter() - t_start
    print(f"\nГотово ✅: обработано файлов = {len(results)} за {total_dt:.2f}s. Обновлён {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
