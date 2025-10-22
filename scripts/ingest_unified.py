#!/usr/bin/env python3
"""
Единый ingest из популярных форматов с автодетектом и OCR для сканов.
Поддержка: PDF, DOCX, HTML/HTM, TXT.

Что делает:
- Для каждого файла создаёт
  data/<doc_id>.txt           — объединённый текст
  data/<doc_id>.pages.jsonl   — постранично/покусково (1 JSON на строку)
- Ведёт data/manifest.jsonl
- Для PDF без текста пытается прогнать OCR через внешнюю утилиту `ocrmypdf` (если установлена).

Установка зависимостей (минимум):
  pip install pymupdf python-docx beautifulsoup4 lxml
Для OCR (Linux/Ubuntu):
  sudo apt update && sudo apt install -y ocrmypdf tesseract-ocr tesseract-ocr-rus

Запуск:
  python ingest_unified.py --input-dir raw_docs --out-dir data

Программное использование (для «автомата» в бекэнде):
  from ingest_unified import ingest_all
  ingest_all(Path('raw_docs'), Path('data'))

Примечание: это этап извлечения. Чанкинг parent/child и загрузка в Qdrant — chunk_and_index
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Попробуем опциональные модули
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from bs4 import BeautifulSoup  # bs4
except Exception:
    BeautifulSoup = None


# ========= ВСПОМОГАТЕЛЬНЫЕ ========= #

def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    out, prev_blank = [], False
    for ln in lines:
        if not ln:
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(ln)
            prev_blank = False
    return "\n".join(out).strip()


def ensure_ocr_pdf(pdf_path: Path, tmp_dir: Path) -> Path:
    """Если PDF вероятно скановый (мало текста), пробуем OCR через ocrmypdf."""
    if fitz is None:
        return pdf_path
    try:
        doc = fitz.open(pdf_path)
        total_chars = 0
        for i in range(min(len(doc), 5)):  # оценим по первым 5 страницам
            page = doc.load_page(i)
            total_chars += len(page.get_text("text") or "")
        doc.close()
        if total_chars >= 50:
            return pdf_path  # текст уже есть
    except Exception:
        return pdf_path

    # OCR
    ocrmypdf = shutil.which("ocrmypdf")
    if not ocrmypdf:
        print(f"[NOTE] {pdf_path.name}: похоже, скан. OCR пропущен (не найден ocrmypdf). Установи: sudo apt install ocrmypdf tesseract-ocr tesseract-ocr-rus")
        return pdf_path

    ocred = tmp_dir / (pdf_path.stem + ".ocr.pdf")
    ocred.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([ocrmypdf, "--force-ocr", "--skip-text", str(pdf_path), str(ocred)], check=True)
        return ocred
    except subprocess.CalledProcessError as e:
        print(f"[WARN] OCR не удалось для {pdf_path.name}: {e}")
        return pdf_path


# ========= ИЗВЛЕЧЕНИЕ ========= #

def extract_pdf(pdf_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF не установлен: pip install pymupdf")
    tmp = pdf_path.parent / ".tmp_ocr"
    src = ensure_ocr_pdf(pdf_path, tmp)
    doc = fitz.open(src)
    pages, total = [], 0
    for i in range(len(doc)):
        txt = normalize_text(doc.load_page(i).get_text("text") or "")
        pages.append({"page": i + 1, "text": txt})
        total += len(txt)
    doc.close()
    joined = "\n\n".join([f"##### [PAGE {p['page']}]\n{p['text']}" for p in pages])
    return joined, pages


def extract_docx(docx_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    if docx is None:
        raise RuntimeError("python-docx не установлен: pip install python-docx")
    d = docx.Document(str(docx_path))
    paras = [normalize_text(p.text) for p in d.paragraphs]
    text = "\n".join([p for p in paras if p])
    # псевдо-страницы каждые ~2000 символов
    pages, chunk, n, page_no = [], [], 0, 1
    for ln in text.split("\n"):
        chunk.append(ln)
        n += len(ln) + 1
        if n >= 2000:
            pages.append({"page": page_no, "text": "\n".join(chunk)})
            chunk, n, page_no = [], 0, page_no + 1
    if chunk:
        pages.append({"page": page_no, "text": "\n".join(chunk)})
    return text, pages


def extract_html(html_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    if BeautifulSoup is None:
        raise RuntimeError("bs4/lxml не установлены: pip install beautifulsoup4 lxml")
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    # сохраняем заголовки как маркеры
    for h in soup.find_all(["h1", "h2", "h3"]):
        h.insert_before("\n\n### [HEADING] " + h.get_text(strip=True) + "\n")
    raw = soup.get_text("\n")
    text = normalize_text(raw)
    # псевдо-страницы
    pages, chunk, n, page_no = [], [], 0, 1
    for ln in text.split("\n"):
        chunk.append(ln)
        n += len(ln) + 1
        if n >= 2000:
            pages.append({"page": page_no, "text": "\n".join(chunk)})
            chunk, n, page_no = [], 0, page_no + 1
    if chunk:
        pages.append({"page": page_no, "text": "\n".join(chunk)})
    return text, pages


def extract_txt(txt_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    text = normalize_text(txt_path.read_text(encoding="utf-8", errors="ignore"))
    pages, chunk, n, page_no = [], [], 0, 1
    for ln in text.split("\n"):
        chunk.append(ln)
        n += len(ln) + 1
        if n >= 2000:
            pages.append({"page": page_no, "text": "\n".join(chunk)})
            chunk, n, page_no = [], 0, page_no + 1
    if chunk:
        pages.append({"page": page_no, "text": "\n".join(chunk)})
    return text, pages


# ========= ДРАЙВЕР ========= #

def ingest_file(path: Path, out_dir: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    doc_id = path.stem

    if suffix == ".pdf":
        text, pages = extract_pdf(path)
    elif suffix == ".docx":
        text, pages = extract_docx(path)
    elif suffix in {".html", ".htm"}:
        text, pages = extract_html(path)
    elif suffix == ".txt":
        text, pages = extract_txt(path)
    else:
        raise ValueError(f"Неподдерживаемый формат: {suffix}")

    # запись
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    with (out_dir / f"{doc_id}.pages.jsonl").open("w", encoding="utf-8") as f:
        for p in pages:
            rec = {"doc_id": doc_id, "page": p["page"], "text": p["text"], "source_path": str(path.resolve())}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"doc_id": doc_id, "file": str(path.resolve()), "pages": len(pages), "chars": len(text)}


def ingest_all(input_dir: Path, out_dir: Path) -> int:
    files = sorted([p for p in input_dir.glob("**/*") if p.suffix.lower() in {".pdf", ".docx", ".html", ".htm", ".txt"}])
    if not files:
        print(f"[WARN] В {input_dir} нет поддерживаемых файлов")
        return 0
    manifest = out_dir / "manifest.jsonl"
    count = 0
    with manifest.open("a", encoding="utf-8") as man:
        for fp in files:
            print(f"→ {fp.name}")
            try:
                info = ingest_file(fp, out_dir)
            except Exception as e:
                print(f"[ERR] {fp.name}: {e}")
                continue
            man.write(json.dumps(info, ensure_ascii=False) + "\n")
            count += 1
    print(f"Готово: обработано {count} файлов. Вывод: {out_dir}")
    return count


def cli():
    ap = argparse.ArgumentParser(description="Единый ingest: PDF/DOCX/HTML/TXT (+OCR для сканов)")
    ap.add_argument("--input-dir", required=True, type=Path, help="Папка с документами")
    ap.add_argument("--out-dir", required=True, type=Path, help="Папка вывода (например, data)")
    args = ap.parse_args()
    ingest_all(args.input_dir, args.out_dir)


if __name__ == "__main__":
    cli()

