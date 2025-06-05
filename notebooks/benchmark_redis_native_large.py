import redis
import pickle
import numpy as np
from tqdm import tqdm
import time
from redis.commands.search.query import Query

with open("datasets/large/qqp_with_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

df["embedding"] = df.apply(lambda row: (
    (np.array(row["q1_vector"]) + np.array(row["q2_vector"])) / 2
), axis=1)
df["embedding"] = df["embedding"].apply(lambda vec: vec / np.linalg.norm(vec))

r = redis.Redis(host="localhost", port=6380)

print("🔍 Running Redis native vector benchmark...")
timings = []
recall = 0

for i, row in tqdm(df.iloc[:100].iterrows(), total=100):
    query_vec = row["embedding"].astype(np.float32).tobytes()
    start = time.time()

    q = Query("*=>[KNN 1 @embedding $vec_param]") \
        .sort_by("__embedding_score") \
        .paging(0, 1) \
        .dialect(2)

    res = r.ft("qqp_idx").search(q, query_params={"vec_param": query_vec})
    duration = time.time() - start
    timings.append(duration)

    top_doc_id = res.docs[0].id if res.docs else None
    expected_doc_id = f"qqp:{i}"
    if top_doc_id == expected_doc_id:
        recall += 1

avg_time = sum(timings) / len(timings)
throughput = len(timings) / sum(timings)
recall_at_1 = recall / len(timings)

print("\n📊 Redis Native Vector Results (Large Dataset):")
print(f"Average Query Time: {avg_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall_at_1*100:.2f}%")
