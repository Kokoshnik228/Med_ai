#!/usr/bin/env python3
"""
Строит BM25-индекс (Pyserini/Lucene) из data/*.pages.jsonl.
Режет страницы на child-чанки и индексирует их.

Зависимости:
  pip install pyserini tqdm
  sudo apt install -y openjdk-17-jre-headless  # для Lucene

Запуск:
  python build_bm25.py --pages-glob "data/*.pages.jsonl" \
    --out-json index/bm25_json --index-dir index/bm25_idx
"""
from __future__ import annotations

import os
import argparse
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm


def words(text: str) -> List[str]:
    return text.split()


def chunk_words(tokens: List[str], max_len: int, overlap: int) -> List[List[str]]:
    chunks: List[List[str]] = []
    i, n = 0, len(tokens)
    while i < n:
        j = min(i + max_len, n)
        chunks.append(tokens[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


def is_noise_text(text: str) -> bool:
    """
    Определяет, является ли текст "мусорным" (оглавление, таблица, слишком короткий и т.п.)
    """
    if not text or len(text.strip()) < 50:
        return True

    # много точек и цифр — вероятно, оглавление
    punct_density = text.count('.') / max(len(text), 1)
    digit_density = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if punct_density > 0.25 and digit_density > 0.15:
        return True

    # слишком много повторов пунктов "1.1", "2.3" и т.п.
    if len(re.findall(r'\d+\.\d+', text)) > 10:
        return True

    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Build BM25 index from pages.jsonl")
    ap.add_argument("--language", default="ru", help="Analyzer language for Lucene (e.g., ru, en, ...)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--pages-glob", required=True)
    ap.add_argument("--out-json", default="index/bm25_json")
    ap.add_argument("--index-dir", default="index/bm25_idx")
    ap.add_argument("--child-w", type=int, default=200)
    ap.add_argument("--child-overlap", type=int, default=40)
    args = ap.parse_args()

    pages_files = sorted(Path().glob(args.pages_glob))
    if not pages_files:
        raise SystemExit(f"Не найдено файлов по маске: {args.pages_glob}")

    out_json_dir = Path(args.out_json)
    out_json_dir.mkdir(parents=True, exist_ok=True)

    # ---- Основной цикл по документам ----
    for fp in tqdm(pages_files, desc="Processing"):
        doc_id = fp.stem.replace(".pages", "")
        out_path = out_json_dir / f"{doc_id}.json"

        # читаем страницы
        pages: List[Dict[str, Any]] = []
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pages.append({"page": rec.get("page", 0), "text": rec.get("text", "")})

        # если в документе вообще нет текста — пропускаем
        if not any(p.get("text") for p in pages):
            print(f"⚠️ Пропуск пустого документа: {fp.name}")
            continue

        kept = 0
        skipped = 0
        with out_path.open("w", encoding="utf-8") as fout:
            for p in pages:
                toks = words(p["text"]) if p["text"] else []
                if not toks:
                    continue
                chunks = chunk_words(toks, args.child_w, args.child_overlap)
                for c_i, chunk in enumerate(chunks, start=1):
                    text = " ".join(chunk)
                    if is_noise_text(text):
                        skipped += 1
                        continue  # пропускаем мусорные чанки
                    kept += 1
                    chunk_id = f"{doc_id}_p{p['page']}_c{c_i}"
                    obj = {
                        "id": chunk_id,
                        "contents": text,
                        "raw": json.dumps({
                            "doc_id": doc_id,
                            "page": p["page"],
                            "child_idx": c_i
                        }, ensure_ascii=False)
                    }
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(f"✅ {doc_id} — записано {kept} чанков (пропущено {skipped} мусорных)")

    # ---- Запуск Lucene индексатора ----
    index_dir = Path(args.index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(out_json_dir.resolve()),
        "--index", str(index_dir.resolve()),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(args.threads),
        "--language", args.language,                 # важно для русского
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]

    print("→ Запуск индексатора:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✅ Готово: BM25 индекс создан в", index_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
