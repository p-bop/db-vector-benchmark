import numpy as np
import time
from pymongo import MongoClient
from tqdm import tqdm

def load_queries(path, num=100):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num:
                break
            parts = line.strip().split()
            word = parts[0]
            vec = list(map(float, parts[1:]))
            queries.append((word, vec))
    return queries

print("📄 Loading 100 queries from GloVe...")
queries = load_queries("datasets/glove/glove.6B.100d.txt")
print(f"✅ Loaded {len(queries)} queries.")

client = MongoClient("mongodb+srv://dbuser:dbpassword@vector-m10.wuxxb.mongodb.net/?retryWrites=true&w=majority&appName=vector-m10")
collection = client.glove.vectors

print("🔍 Running native vector search queries on MongoDB Atlas...")
timings = []
correct = 0

for word, vec in tqdm(queries, desc="🔍 Querying"):
    start = time.time()
    result = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": vec,
                "path": "vector",
                "numCandidates": 100,
                "limit": 1,
                "index": "vector_index"
            }
        },
        {
            "$project": {
                "_id": 0,
                "word": 1
            }
        }
    ])
    best = next(result, None)
    if best and best.get("word", "").lower() == word.lower():
        correct += 1
    timings.append(time.time() - start)

avg_time = sum(timings) / len(timings)
throughput = len(timings) / sum(timings)
recall_at_1 = correct / len(timings)

print("\n📊 MongoDB Native Vector Results (Small Dataset):")
print(f"Average Query Time: {avg_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall_at_1 * 100:.2f}%")
