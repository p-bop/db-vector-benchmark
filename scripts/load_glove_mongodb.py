import os
import sys
import pymongo
from pymongo import MongoClient
from tqdm import tqdm

GLOVE_DIM = 100  
GLOVE_FILE = f"datasets/glove/glove.6B.{GLOVE_DIM}d.txt"
DB_NAME = "benchmark"
COLLECTION_NAME = "glove_vectors"

try:
    client = MongoClient("localhost", 27017)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print(f"Connected to MongoDB. Using database '{DB_NAME}', collection '{COLLECTION_NAME}'.")
except Exception as e:
    print("Failed to connect to MongoDB:", e)
    sys.exit(1)

if not os.path.exists(GLOVE_FILE):
    print(f"❌ GloVe file not found: {GLOVE_FILE}")
    sys.exit(1)

print(f"📄 Loading vectors from {GLOVE_FILE}...")

with open(GLOVE_FILE, 'r', encoding='utf-8') as f:
    batch = []
    batch_size = 1000  
    for line in tqdm(f, desc="Inserting vectors into MongoDB"):
        parts = line.strip().split()
        if len(parts) != GLOVE_DIM + 1:
            continue  
        word = parts[0]
        vector = list(map(float, parts[1:]))
        doc = {
            "word": word,
            "vector": vector
        }
        batch.append(doc)

        if len(batch) >= batch_size:
            collection.insert_many(batch)
            batch = []

    if batch:
        collection.insert_many(batch)

print("✅ Done loading GloVe vectors into MongoDB.")
