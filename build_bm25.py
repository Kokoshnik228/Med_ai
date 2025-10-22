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
import argparse, json, subprocess
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

def main() -> int:
    ap = argparse.ArgumentParser(description="Build BM25 index from pages.jsonl")
    ap.add_argument("--pages-glob", required=True)
    ap.add_argument("--out-json", default="index/bm25_json")
    ap.add_argument("--index-dir", default="index/bm25_idx")
    ap.add_argument("--child-w", type=int, default=200)
    ap.add_argument("--child-overlap", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    pages_files = sorted(Path().glob(args.pages_glob))
    if not pages_files:
        raise SystemExit(f"Не найдено файлов по маске: {args.pages_glob}")

    out_json_dir = Path(args.out_json)
    out_json_dir.mkdir(parents=True, exist_ok=True)

    # JsonCollection: по файлу на документ; внутри — по строке JSON на каждый child-чанк
    for fp in pages_files:
        doc_id = fp.stem.replace(".pages", "")
        out_path = out_json_dir / f"{doc_id}.json"
        with fp.open("r", encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as fout:
            pages: List[Dict[str, Any]] = []
            for line in f:
                rec = json.loads(line)
                pages.append({"page": rec.get("page", 0), "text": rec.get("text", "")})
            child_idx = 0
            for p in pages:
                toks = words(p["text"]) if p["text"] else []
                if not toks:
                    continue
                chunks = chunk_words(toks, args.child_w, args.child_overlap)
                for c_i, chunk in enumerate(chunks, start=1):
                    child_idx += 1
                    chunk_id = f"{doc_id}_p{p['page']}_c{c_i}"
                    text = " ".join(chunk)
                    obj = {
                        "id": chunk_id,
                        "contents": text,
                        "raw": json.dumps({"doc_id": doc_id, "page": p["page"], "child_idx": c_i}, ensure_ascii=False),
                    }
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    index_dir = Path(args.index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(out_json_dir.resolve()),
        "--index", str(index_dir.resolve()),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(args.threads),
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]
    print("→ Запуск индексатора:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Готово: BM25 индекс в", index_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
