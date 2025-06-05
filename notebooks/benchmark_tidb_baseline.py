import pymysql
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics.pairwise import cosine_similarity

connection = pymysql.connect(
    host="localhost",
    port=4000,
    user="root",
    password="",
    database="glove"
)

def load_queries(path, num=100):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num:
                break
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            queries.append((word, vec))
    return queries

print("📄 Loading 100 queries...")
queries = load_queries("datasets/glove/glove.6B.100d.txt", num=100)
print(f"✅ Loaded {len(queries)} queries.")

print("🔍 Loading vectors from TiDB...")
with connection.cursor() as cursor:
    cursor.execute("SELECT word, vector FROM glove_vectors")
    all_vectors = [
        (word, np.frombuffer(vec, dtype=np.float32))
        for word, vec in cursor.fetchall()
    ]
print(f"✅ Retrieved {len(all_vectors)} vectors.")

print("🔍 Running queries...")
timings = []
correct = 0

for query_word, query_vec in tqdm(queries):
    start = time.time()

    similarities = [
        (word, cosine_similarity([query_vec], [vec])[0][0])
        for word, vec in all_vectors
    ]

    best_word, _ = max(similarities, key=lambda x: x[1])
    timings.append(time.time() - start)

    if best_word == query_word:
        correct += 1

avg_time = sum(timings) / len(timings)
throughput = len(timings) / sum(timings)
recall = correct / len(timings)

print("\n📊 TiDB Baseline (Brute-Force) Results:")
print(f"Average Query Time: {avg_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall * 100:.2f}%")

connection.close()
