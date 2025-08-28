import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMB_MODEL = "intfloat/multilingual-e5-base"
CHUNKS_PATH = "curso/data/chunks.jsonl"
INDEX_PATH = "curso/index/faiss.index"
META_PATH = "curso/index/meta.json"

model = SentenceTransformer(EMB_MODEL)

meta = []
textos = []
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        meta.append(r)
        textos.append(r["text"])

vecs = model.encode(textos, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
vecs = np.array(vecs, dtype="float32")

index = faiss.IndexFlatIP(vecs.shape[1])  # coseno si normalizas
index.add(vecs)

faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False)

print(f"Index listo: {len(meta)} chunks")