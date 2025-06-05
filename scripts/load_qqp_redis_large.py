import redis
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

with open("datasets/large/qqp_with_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

df["embedding"] = df.apply(lambda row: (
    (np.array(row["q1_vector"]) + np.array(row["q2_vector"])) / 2
), axis=1)
df["embedding"] = df["embedding"].apply(lambda vec: vec / np.linalg.norm(vec))

r = redis.Redis(host="localhost", port=6380)

INDEX_NAME = "qqp_idx"
VECTOR_DIM = len(df.iloc[0]["embedding"])
VECTOR_FIELD_NAME = "embedding"

try:
    r.ft(INDEX_NAME).dropindex(delete_documents=True)
    print(f"🗑️ Dropped existing index '{INDEX_NAME}'")
except Exception:
    print(f"ℹ️ No existing index named '{INDEX_NAME}'")

schema = (
    TextField("question1"),
    TextField("question2"),
    VectorField(VECTOR_FIELD_NAME, "HNSW", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": "COSINE",
        "INITIAL_CAP": len(df),
        "M": 16,
        "EF_CONSTRUCTION": 200
    }),
)

r.flushdb()
r.ft(INDEX_NAME).create_index(schema, definition=IndexDefinition(prefix=["qqp:"], index_type=IndexType.HASH))
print(f"✅ Created index '{INDEX_NAME}' for QQP dataset.")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    key = f"qqp:{idx}"
    r.hset(key, mapping={
        "question1": row["question1"],
        "question2": row["question2"],
        "embedding": row["embedding"].astype(np.float32).tobytes()
    })

print("✅ Finished loading QQP dataset into Redis with normalized vectors.")
