import pymongo
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from sklearn.metrics.pairwise import cosine_similarity

with open("datasets/large/qqp_with_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

df["embedding"] = df.apply(lambda row: (np.array(row["q1_vector"]) + np.array(row["q2_vector"])) / 2, axis=1)
data = df.to_dict(orient="records")

print(f"📄 Loaded {len(data)} records with averaged embeddings.")
print(f"Sample embedding keys: {list(data[0].keys())}")

client = pymongo.MongoClient("mongodb://localhost:27017/")
collection = client.benchmark.qqp_vectors_large

print("🔍 Running cosine similarity baseline benchmark...")
timings = []
recall = 0

for i, query in tqdm(enumerate(data[:100]), total=100):
    query_vec = np.array(query["embedding"])
    similarities = [
        (doc["id"] if "id" in doc else idx, cosine_similarity([query_vec], [np.array(doc["embedding"])])[0][0])
        for idx, doc in enumerate(data)
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_id = similarities[0][0]
    if top_id == (query["id"] if "id" in query else i):
        recall += 1
    timings.append(time.time())

total_time = timings[-1] - timings[0]
avg_query_time = total_time / len(timings)
throughput = len(timings) / total_time
recall_at_1 = recall / len(timings)

print("\n📊 MongoDB Baseline Results (Large Dataset):")
print(f"Average Query Time: {avg_query_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall_at_1*100:.2f}%")
