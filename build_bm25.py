#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Инкрементальная сборка BM25-индекса (Pyserini/Lucene) из data/*.pages.jsonl.

Что улучшено по сравнению с исходной версией:
  • Строго такая же схема чанкинга, как в dense-пайплайне: по словам с окнами
    (child_w=180, overlap=40 по умолчанию). Это снижает рассинхрон между BM25 и dense.
  • Робастное чтение .pages.jsonl (грязные строки пропускаются, не валим сборку).
  • Чуть аккуратнее отбрасываем мусор (огрызки, «табличный шум», реферативные хвосты).
  • Обновляем И изменившиеся документы по умолчанию (uniqueDocid обеспечит замену),
    при желании можно пропустить изменившиеся через --skip-changed.
  • Чёткая «дельта»: в Lucene идёт только то, что реально нужно проиндексировать.

Зависимости:
  pip install pyserini tqdm
  sudo apt install -y openjdk-17-jre-headless  # для Lucene

Примеры:
  python build_bm25.py \
    --pages-glob "data/*.pages.jsonl" \
    --out-json index/bm25_json --index-dir index/bm25_idx

  # Полная перестройка (снос индекса и состояния):
  python build_bm25.py --pages-glob "data/*.pages.jsonl" \
    --out-json index/bm25_json --index-dir index/bm25_idx --recreate
"""
from __future__ import annotations

import os
import argparse
import json
import subprocess
import re
import shutil
import hashlib
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm


# ------------------------- утилиты -------------------------

def _clean_jsonl_line(s: str) -> str:
    """Мягкая чистка строки JSONL (убираем NUL/контр.символы, нормализуем NFKC)."""
    if not s:
        return ""
    s = s.replace("\x00", "")
    s = "".join(ch for ch in s if ch.isprintable() or ch in "\t\r\n")
    return unicodedata.normalize("NFKC", s)


def _read_pages_robust(p: Path) -> Tuple[List[Dict[str, Any]], int]:
    """
    Читает .pages.jsonl построчно. Возвращает (pages, skipped_count).
    Плохие строки пропускаем с предупреждением (не валим весь процесс).
    Ожидаемые поля: page(int), text(str).
    """
    pages: List[Dict[str, Any]] = []
    skipped = 0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            line = _clean_jsonl_line(line).strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    pages.append({
                        "page": int(rec.get("page", 0) or 0),
                        "text": rec.get("text", "") or "",
                    })
            except Exception as e:
                skipped += 1
                print(f"⚠️  {p.name}: битая JSONL-строка #{i}: {e}")
    return pages, skipped


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def words(text: str) -> List[str]:
    return (text or "").split()


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


# Признаки «мусора» для BM25 (очень простые и дешёвые)
_CITATION_LINE_RE = re.compile(
    r"(?xi)(?:^\s*\d{1,3}[\).\]]\s+|et\s*al\.?|doi[:\s/]|10\.\d{3,9}/\S+|\b\d{4}\b\s*;\s*\d+)",
)
def _drop_citation_like_lines(text: str) -> str:
    if not text:
        return text
    parts = re.split(r"(?:\n+|(?<=\.)\s+)", text)
    kept = [ln for ln in parts if ln and not _CITATION_LINE_RE.search(ln)]
    return "\n".join(kept).strip()


def is_noise_text(text: str, min_chars: int = 90) -> bool:
    """
    Грубый фильтр мусорных кусков (оглавления, таблицы-огрызки, чистые списки ссылок).
    Лучше недофильтровать, чем переборщить.
    """
    if not text:
        return True
    t = text.strip()
    if len(t) < min_chars:
        return True

    # Явные «табличные»/сверхцифровые куски
    digits = sum(ch.isdigit() for ch in t)
    if digits / max(len(t), 1) > 0.30:
        return True

    # Слишком много пунктуации → список/огрызок
    punct = sum(ch in ".,;:·•▪-|_/\\[]()" for ch in t)
    if punct / max(len(t), 1) > 0.28:
        return True

    # Кусок почти целиком состоит из коротких «строк»-элементов (типа списка литературы)
    lines = [ln.strip() for ln in re.split(r"(?:\n+|(?<=\.)\s+)", t) if ln.strip()]
    if lines and sum(1 for ln in lines if len(ln) < 35) / len(lines) > 0.7:
        return True

    return False


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Возвращает мапу doc_id -> {sha1, source_path, ...}.
    Если файла нет — вернёт пустую мапу.
    """
    if not manifest_path.exists():
        return {}
    try:
        j = json.loads(manifest_path.read_text(encoding="utf-8"))
        docs = j.get("docs") or []
        out: Dict[str, Dict[str, Any]] = {}
        for d in docs:
            if not isinstance(d, dict):
                continue
            did = str(d.get("doc_id") or "").strip()
            if not did:
                continue
            out[did] = d
        return out
    except Exception:
        return {}


def load_state(state_path: Path) -> Dict[str, str]:
    """
    Состояние: doc_id -> sha1, что уже проиндексировано.
    """
    if not state_path.exists():
        return {}
    try:
        j = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(j, dict):
            return {str(k): str(v) for k, v in j.items()}
    except Exception:
        pass
    return {}


def save_state(state_path: Path, state: Dict[str, str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def decide_doc_sha1(doc_id: str, manifest_map: Dict[str, Dict[str, Any]], pages_file: Path) -> str:
    """
    Определяем «контрольную сумму» документа:
    - если в manifest есть sha1 по doc_id, берём его;
    - иначе хешируем сам .pages.jsonl (похуже, но годится).
    """
    md = manifest_map.get(doc_id)
    if md and md.get("sha1"):
        return str(md["sha1"])
    # fallback: хеш файла страниц
    return sha1_file(pages_file)


# ------------------------- основной процесс -------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Incremental BM25 build (Pyserini/Lucene)")
    ap.add_argument("--language", default="ru", help="Analyzer language for Lucene (e.g., ru, en, ...)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--pages-glob", required=True, help='Например: "data/*.pages.jsonl"')
    ap.add_argument("--out-json", default="index/bm25_json", help="Куда писать per-doc JSON для Pyserini")
    ap.add_argument("--index-dir", default="index/bm25_idx", help="Каталог Lucene индекса")

    # Важно: синхрон с dense-пайплайном (узкие окна повышают точность)
    ap.add_argument("--child-w", type=int, default=180, help="Размер child-окна (слов)")
    ap.add_argument("--child-overlap", type=int, default=40, help="Перекрытие child-окон (слов)")

    ap.add_argument("--manifest", default="data/manifest.json", help="Путь к manifest.json из ingest")
    ap.add_argument("--state-path", default="index/.bm25_state.json", help="Где хранить состояние (doc_id -> sha1)")
    ap.add_argument("--recreate", action="store_true", help="Снести индекс и состояние, собрать заново")

    # Политика для изменившихся документов
    ap.add_argument("--skip-changed", action="store_true",
                    help="Если указан, изменившиеся документы будут ПРОПУЩЕНЫ (по умолчанию — обновляем).")

    # Минимальный размер чанка для индексации
    ap.add_argument("--min-chunk-chars", type=int, default=110, help="Минимальный размер чанка в символах")

    args = ap.parse_args()

    pages_files = sorted(Path().glob(args.pages_glob))
    if not pages_files:
        raise SystemExit(f"Не найдено файлов по маске: {args.pages_glob}")

    out_json_dir = Path(args.out_json)
    out_json_dir.mkdir(parents=True, exist_ok=True)

    index_dir = Path(args.index_dir)
    index_dir.parent.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state_path)
    manifest_map = load_manifest(Path(args.manifest))

    # Пересоздание индекса и состояния по запросу
    if args.recreate:
        if index_dir.exists():
            print(f"🧨 Удаляю индекс {index_dir} (recreate)...")
            shutil.rmtree(index_dir, ignore_errors=True)
        if state_path.exists():
            print(f"🧨 Удаляю state {state_path} (recreate)...")
            try:
                state_path.unlink()
            except Exception:
                pass

    state = load_state(state_path)

    # --- Определяем дельту: какие doc_id новые/изменившиеся ---
    to_process: List[Tuple[str, Path]] = []
    new_docs: List[str] = []
    changed_docs: List[str] = []

    for fp in pages_files:
        doc_id = fp.stem.replace(".pages", "")
        sha = decide_doc_sha1(doc_id, manifest_map, fp)
        prev = state.get(doc_id)

        if prev is None:
            new_docs.append(doc_id)
            to_process.append((doc_id, fp))
        elif prev != sha:
            changed_docs.append(doc_id)
            if not args.skip_changed:
                to_process.append((doc_id, fp))
        # если prev == sha → уже индексирован; пропускаем

    if changed_docs:
        if args.skip_changed:
            print("ℹ️  Изменившиеся документы обнаружены, но будут пропущены (--skip-changed).")
        else:
            print("♻️  Изменившиеся документы будут обновлены (uniqueDocid обеспечит замену):")
        print("   doc_id:", ", ".join(changed_docs[:25]) + (" …" if len(changed_docs) > 25 else ""))

    if not to_process:
        print("✅ Нет новых/обновлённых документов — индексация не требуется.")
        return 0

    # --- Готовим delta-директорию: только то, что реально индексируем ---
    delta_dir = index_dir.parent / "bm25_json_delta"
    if delta_dir.exists():
        shutil.rmtree(delta_dir, ignore_errors=True)
    delta_dir.mkdir(parents=True, exist_ok=True)

    # --- Генерация JSON для Pyserini (только для нужных документов) ---
    total_kept, total_skipped = 0, 0
    written_json_files: List[Path] = []

    for doc_id, fp in tqdm(to_process, desc="JSON build (delta)"):
        pages, bad = _read_pages_robust(fp)
        if bad:
            print(f"⚠️  {doc_id}: пропущено битых строк JSONL: {bad}")

        if not any((p.get("text") or "").strip() for p in pages):
            print(f"⚠️ Пропуск пустого документа: {fp.name}")
            continue

        kept = 0
        skipped = 0

        # Основной JSON (per-doc) в out_json_dir
        full_json_path = out_json_dir / f"{doc_id}.json"
        with full_json_path.open("w", encoding="utf-8") as fout:
            for p in pages:
                txt = p.get("text") or ""
                if not txt.strip():
                    continue

                # Лёгкая зачистка хвостов-«литературы», если вдруг просочились
                txt = _drop_citation_like_lines(txt)

                toks = words(txt)
                if not toks:
                    continue

                chunks = chunk_words(toks, args.child_w, args.child_overlap)
                for c_i, chunk in enumerate(chunks, start=1):
                    text = " ".join(chunk).strip()
                    if is_noise_text(text, min_chars=args.min_chunk_chars):
                        skipped += 1
                        continue

                    kept += 1
                    chunk_id = f"{doc_id}_p{int(p['page'])}_c{c_i}"
                    obj = {
                        "id": chunk_id,
                        "contents": text,
                        "raw": json.dumps({
                            "doc_id": doc_id,
                            "page": int(p["page"]),
                            "child_idx": c_i
                        }, ensure_ascii=False)
                    }
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

        if kept == 0:
            print(f"⚠️ {doc_id}: после фильтрации нечего индексировать (возможно, мусор/слишком коротко).")
            # не создаём дельту, но основной JSON уже лежит (0 строк)
            continue

        # Для индексации — копию в delta_dir
        delta_json_path = delta_dir / f"{doc_id}.json"
        try:
            shutil.copy2(full_json_path, delta_json_path)
        except Exception:
            shutil.copy(full_json_path, delta_json_path)

        written_json_files.append(delta_json_path)
        total_kept += kept
        total_skipped += skipped
        print(f"  ✅ {doc_id}: чанков={kept}, пропущено(мусор)={skipped}")

    if not written_json_files:
        print("⚠️ Нет валидных JSON для индексации (все документы пустые/отфильтрованы?)")
        return 0

    # --- Запуск Pyserini индексатора по delta-директории ---
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(delta_dir.resolve()),
        "--index", str(index_dir.resolve()),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(args.threads),
        "--language", args.language,
        "--storePositions", "--storeDocvectors", "--storeRaw",
        "--uniqueDocid"  # важно: тот же id → обновление, без дублей
    ]

    print("\n→ Запуск индексатора только по Δ (новые/обновлённые документы):", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✅ Индексация дельты завершена.")

    # --- Обновляем state по успешно проиндексированным документам ---
    state = load_state(state_path)  # перечитать, если кто-то писал параллельно
    for doc_id, fp in to_process:
        sha = decide_doc_sha1(doc_id, manifest_map, fp)
        state[doc_id] = sha
    save_state(state_path, state)

    print("\n📊 Итог:")
    print(f"  Новых документов: {len(new_docs)}")
    print(f"  Обновлённых документов: {0 if args.skip_changed else len(changed_docs)}")
    print(f"  Добавлено чанков: {total_kept}")
    print(f"  Пропущено мусорных чанков: {total_skipped}")
    print(f"  Индекс: {index_dir}")
    print("✅ Готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
