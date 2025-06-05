import redis
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import time

r = redis.Redis(host="localhost", port=6380, decode_responses=False)

keys = r.keys("qqp:*")
print(f"📦 Retrieved {len(keys)} records from Redis.")

data = []
for key in keys:
    fields = r.hgetall(key)
    vec = pickle.loads(fields[b'embedding'])
    data.append((key, vec))

print("🔍 Running brute-force benchmark (cosine similarity)...")
timings = []
recall = 0

for i in tqdm(range(100)):
    query_key, query_vec = data[i]
    query_vec = np.array(query_vec).reshape(1, -1)

    sims = []
    for candidate_key, candidate_vec in data:
        sim = cosine_similarity(query_vec, [candidate_vec])[0][0]
        sims.append((candidate_key, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    top_match = sims[0][0]

    if top_match == query_key:
        recall += 1

    timings.append(time.time())

avg_time = (timings[-1] - timings[0]) / len(timings)
throughput = len(timings) / (timings[-1] - timings[0])
recall_at_1 = recall / len(timings)

print("\n📊 Redis Baseline Results (Large Dataset):")
print(f"Average Query Time: {avg_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall_at_1*100:.2f}%")
