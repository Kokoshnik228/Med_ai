#!/usr/bin/env python3
"""
Быстрый smoke‑test для Qdrant на локальном порту (по умолчанию http://qdrant:6333).
Создаёт коллекцию `med_kb_v3` (1024‑мерные векторы под bge-m3),
добавляет 3 точки‑заглушки и делает поиск.

Запуск:
  python qdrant_smoketest.py --url http://qdrant:6333 --reset
"""
from __future__ import annotations
import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://qdrant:6333", help="URL Qdrant")
    ap.add_argument("--name", default="med_kb_v3", help="Имя коллекции")
    ap.add_argument("--reset", action="store_true", help="Пересоздать коллекцию")
    args = ap.parse_args()

    client = QdrantClient(url=args.url)

    if args.reset:
        print(f"Recreate collection {args.name}…")
        client.recreate_collection(
            collection_name=args.name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    else:
        print(f"Create collection {args.name} if not exists…")
        try:
            client.get_collection(args.name)
        except Exception:
            client.recreate_collection(
                collection_name=args.name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )

    # Три простые точки с одинаковым вектором (1024 элементов)
    vec = [0.001] * 1024
    points = [
        PointStruct(id=1, vector=vec, payload={"doc_id": "doc1", "section": "test"}),
        PointStruct(id=2, vector=vec, payload={"doc_id": "doc2", "section": "test"}),
        PointStruct(id=3, vector=vec, payload={"doc_id": "doc3", "section": "test"}),
    ]
    client.upsert(collection_name=args.name, points=points)
    print("Upserted 3 points.")

    res = client.search(collection_name=args.name, query_vector=vec, limit=2)
    print("Top2:", [(p.id, round(p.score, 4)) for p in res])
    print("OK — связь с Qdrant работает.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

