#!/usr/bin/env python3
"""
Инкрементальная сборка BM25 (Lucene/Anserini/Pyserini).

Берём data/*.pages.jsonl (по одному файлу на документ) и генерим:
  - index/bm25_json/<doc_id>/*.json   — per-doc/per-page JSON для JsonCollection
  - index/bm25_idx/                   — Lucene индекс
  - index/bm25_manifest.json          — манифест с SHA1 по pages.jsonl

Режимы:
  --recreate  : полная пересборка индекса (снос index/bm25_idx, генерация JSON для всех, индексирование всего)
  --append    : (по умолчанию) инкрементально — создать JSON и дозалить в индекс только новые/изменённые документы

Требования:
  - openjdk установлен (есть в Dockerfile)
  - pyserini в requirements
  - Папка с json'ами и индексом доступны на запись
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib
import glob
import subprocess

# ------------------------ utils ------------------------

def read_jsonl_pages(p: Path) -> List[Dict]:
    out = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                out.append(obj)
            except Exception:
                continue
    return out

def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def rm_tree(p: Path):
    if p.exists():
        shutil.rmtree(p)

def hardlink_or_copy(src: Path, dst: Path):
    ensure_dir(dst.parent)
    try:
        os.link(src, dst)  # экономим место и время
    except OSError:
        shutil.copy2(src, dst)

# --------------------- json writer ---------------------

def write_doc_pages_json(
    pages: List[Dict],
    out_root: Path,
    doc_id: str
) -> List[Path]:
    """
    Для документа doc_id создаём набор JSON-файлов формата JsonCollection:
      { "id": "<doc_id>#p<page>", "contents": "<text>", "raw": "{\"doc_id\":...,\"page\":...}" }
    Возвращаем список файлов, которые записали.
    """
    dst_dir = out_root / doc_id
    if dst_dir.exists():
        # документ обновился — удаляем старые страницы, чтобы не осталось мусора
        shutil.rmtree(dst_dir)
    ensure_dir(dst_dir)

    written: List[Path] = []
    for rec in pages:
        page = int(rec.get("page", 1) or 1)
        text = (rec.get("text") or "").strip()
        # даже пустые страницы создадим (потом можно фильтровать при поиске)
        obj = {
            "id": f"{doc_id}#p{page}",
            "contents": text,
            "raw": json.dumps({"doc_id": doc_id, "page": page}, ensure_ascii=False),
        }
        out_file = dst_dir / f"{doc_id}_p{page}.json"
        out_file.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        written.append(out_file)
    return written

# --------------------- manifest ------------------------

def load_manifest(p: Path) -> Dict:
    if not p.exists():
        return {"docs": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "docs" not in data:
            return {"docs": {}}
        if not isinstance(data["docs"], dict):
            data["docs"] = {}
        return data
    except Exception:
        return {"docs": {}}

def save_manifest(p: Path, data: Dict):
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# --------------------- indexer -------------------------

def run_pyserini_indexer(
    input_dir: Path,
    index_dir: Path,
    threads: int = 8,
    language: str = "ru",
    append: bool = False
):
    """
    Вызываем Pyserini/Anserini индексатор.
    Важное: для append используем отдельную input_dir, где лежат ТОЛЬКО новые/изменённые документы.
    """
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(input_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storePositions", "--storeDocvectors", "--storeRaw",
        "--language", language,
    ]
    if append:
        cmd.append("--append")

    print("▶️  INDEX:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ---------------------- main --------------------------

def main() -> int:
    ap = argparse.ArgumentParser("BM25 builder (incremental)")
    ap.add_argument("--pages-glob", default="data/*.pages.jsonl",
                    help="Глоб по файлам страниц (по одному jsonl на документ)")
    ap.add_argument("--out-json", default="index/bm25_json",
                    help="Куда класть per-doc JSON для JsonCollection")
    ap.add_argument("--index-dir", default="index/bm25_idx",
                    help="Папка Lucene индекса")
    ap.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--language", default="ru", help="Язык для анализатора Lucene (напр., ru, en)")
    ap.add_argument("--recreate", action="store_true", help="Полная пересборка")
    ap.add_argument("--append", dest="append", action="store_true", help="Инкрементальная индексация (по умолчанию)")
    ap.add_argument("--only-new", dest="append", action="store_true", help="Синоним append")
    ap.set_defaults(append=True)

    args = ap.parse_args()

    pages_files = sorted(glob.glob(args.pages_glob))
    if not pages_files:
        print(f"❌ Не найдено файлов по шаблону: {args.pages_glob}", file=sys.stderr)
        return 1

    out_json_root = Path(args.out_json)
    index_dir = Path(args.index_dir)
    manifest_path = out_json_root / "bm25_manifest.json"
    stage_dir = out_json_root.parent / "_bm25_stage"  # временная папка для новых/изменённых

    ensure_dir(out_json_root)

    # Загружаем текущий манифест
    manifest = load_manifest(manifest_path)
    docs_state: Dict[str, Dict] = manifest.get("docs", {})

    # Сбор списка документов (doc_id -> (pages_path, sha1))
    docs_found: Dict[str, Tuple[Path, str]] = {}
    for pth in pages_files:
        p = Path(pth)
        # doc_id берём из самого файла (строки jsonl содержат doc_id)
        try:
            first_line = next(iter(read_jsonl_pages(p)), None)
            if not first_line:
                continue
            doc_id = str(first_line.get("doc_id") or p.stem)
        except StopIteration:
            continue
        sha = sha1_file(p)
        docs_found[doc_id] = (p, sha)

    if args.recreate:
        print("♻️  Полная пересборка: очищаем индекс и json…")
        rm_tree(index_dir)
        # JSON перегенерим для всех документов
        to_generate = list(docs_found.items())
        # чистим старые per-doc подкаталоги (кроме служебных)
        for child in out_json_root.iterdir():
            if child.is_dir() and child.name not in (".", ".."):
                shutil.rmtree(child)
        docs_state = {}
    else:
        # инкрементально: берём только новые/изменённые
        to_generate = []
        for doc_id, (pages_path, sha) in docs_found.items():
            prev = docs_state.get(doc_id)
            if not prev or prev.get("pages_sha1") != sha:
                to_generate.append((doc_id, (pages_path, sha)))

    print(f"📄 Всего документов: {len(docs_found)}; к генерации: {len(to_generate)}")

    # Генерим per-doc JSON
    generated_any = False
    for i, (doc_id, (pages_path, sha)) in enumerate(to_generate, 1):
        pages = read_jsonl_pages(pages_path)
        # safety: сортируем по page
        pages.sort(key=lambda r: int(r.get("page", 1) or 1))
        written = write_doc_pages_json(pages, out_json_root, doc_id)
        docs_state[doc_id] = {
            "pages_sha1": sha,
            "json_count": len(written),
            "json_dir": str((out_json_root / doc_id).resolve()),
            "pages_file": str(pages_path.resolve()),
        }
        generated_any = True
        if i % 20 == 0 or i == len(to_generate):
            print(f"  └─ [{i}/{len(to_generate)}] {doc_id}: страниц={len(written)}")

    # Сохраняем манифест
    manifest["docs"] = docs_state
    save_manifest(manifest_path, manifest)

    # Индексация
    if args.recreate:
        # индексируем весь out_json_root
        run_pyserini_indexer(out_json_root, index_dir, threads=args.threads,
                             language=args.language, append=False)
        print("✅ Полная пересборка BM25 завершена.")
        return 0

    if not generated_any:
        print("⏭️  Нет новых/изменённых документов — пересборка индекса не требуется.")
        return 0

    # Создаём stage с ТОЛЬКО новыми/изменёнными документами
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    ensure_dir(stage_dir)
    for doc_id, (_pages_path, _sha) in to_generate:
        src_dir = out_json_root / doc_id
        dst_dir = stage_dir / doc_id
        ensure_dir(dst_dir)
        for jf in src_dir.glob("*.json"):
            hardlink_or_copy(jf, dst_dir / jf.name)

    # append индексирование из stage
    run_pyserini_indexer(stage_dir, index_dir, threads=args.threads,
                         language=args.language, append=True)

    # убираем stage
    shutil.rmtree(stage_dir, ignore_errors=True)

    print("✅ Инкрементальное обновление BM25 завершено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
