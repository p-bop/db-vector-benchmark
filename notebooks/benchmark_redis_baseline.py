import redis
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

r = redis.Redis(host='localhost', port=6379, db=0)

def load_queries(path, num=100):
    queries = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num:
                break
            parts = line.strip().split()
            word = parts[0]
            vec = list(map(float, parts[1:]))
            queries.append((word, np.array(vec)))
    return queries

print("📄 Loading 100 queries...")
queries = load_queries("datasets/glove/glove.6B.100d.txt", num=100)
print(f"✅ Loaded {len(queries)} queries.")

def load_all_vectors():
    keys = list(r.scan_iter("glove:*"))
    vectors = {}
    for key in tqdm(keys, desc="🔍 Loading vectors from Redis"):
        vec_bytes = r.hget(key, "vector")
        if vec_bytes is None:
            continue
        try:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors[key.decode("utf-8")] = vec
        except Exception as e:
            print(f"⚠️ Could not decode {key}: {e}")
    return vectors

all_vectors = load_all_vectors()

timings = []
top1_matches = 0

print("🔍 Running queries...")
for query_word, query_vec in tqdm(queries, desc="🔍 Running queries"):
    start = time.time()
    similarities = {}
    for word, vec in all_vectors.items():
        sim = cosine_similarity([query_vec], [vec])[0][0]
        similarities[word] = sim
    best_match = max(similarities, key=similarities.get)
    timings.append(time.time() - start)
    if best_match == query_word:
        top1_matches += 1

total_time = sum(timings)
avg_query_time = total_time / len(queries)
throughput = len(queries) / total_time
recall_at_1 = top1_matches / len(queries)

print("\n📊 Redis Baseline (Brute-Force) Results:")
print(f"Average Query Time: {avg_query_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall_at_1*100:.2f}%")
