import redis
import numpy as np
from tqdm import tqdm
from redis.commands.search.field import TextField, VectorField

r = redis.Redis(host='localhost', port=6380)

INDEX_NAME = "glove_index"
VECTOR_DIM = 100
VECTOR_FIELD_NAME = "vector"
GLOVE_FILE = "datasets/glove/glove.6B.100d.txt"

try:
    r.ft(INDEX_NAME).dropindex(delete_documents=True)
    print(f"🗑️ Dropped existing index '{INDEX_NAME}'")
except Exception:
    print(f"ℹ️ No existing index named '{INDEX_NAME}' found.")

schema = (
    TextField("word"),
    VectorField(
        VECTOR_FIELD_NAME,
        "HNSW", {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": "COSINE",
            "INITIAL_CAP": 400000,
            "M": 16,
            "EF_CONSTRUCTION": 200,
        }
    ),
)

r.ft(INDEX_NAME).create_index(schema)
print(f"✅ Created RediSearch index '{INDEX_NAME}'.")

print(f"📄 Loading vectors from {GLOVE_FILE} into Redis...")

with open(GLOVE_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(tqdm(f, total=400000)):
        parts = line.strip().split()
        word = parts[0]
        vector = np.array(parts[1:], dtype=np.float32)

        key = f"glove:{i}"
        r.hset(key, mapping={
            "word": word,
            VECTOR_FIELD_NAME: vector.tobytes()
        })

print("✅ Done loading GloVe vectors into Redis.")
