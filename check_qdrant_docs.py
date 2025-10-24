from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:7777")

collections = client.get_collections().collections
print("📚 Коллекции в Qdrant:")
for c in collections:
    print(" -", c.name)

print("\nПримеры документов из med_kb:")
points, _ = client.scroll(collection_name="med_kb", limit=10, with_payload=True)
for p in points:
    pl = p.payload
    print(f"  {pl.get('doc_id')}  p.{pl.get('page_start')}-{pl.get('page_end')}")
