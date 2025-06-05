import os
import pickle
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm
import time
from dotenv import load_dotenv

load_dotenv()
username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
cluster_url = os.getenv("MONGO_CLUSTER_URL")

uri = f"mongodb+srv://{username}:{password}@{cluster_url}/?retryWrites=true&w=majority&appName=vector-m10"
client = MongoClient(uri)
collection = client.qqp.vectors  

print("📄 Loading 100 queries from QQP dataset...")
with open("datasets/large/qqp_with_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

df["embedding"] = df.apply(lambda row: (
    (np.array(row["q1_vector"]) + np.array(row["q2_vector"])) / 2
), axis=1)
df["embedding"] = df["embedding"].apply(lambda vec: vec / np.linalg.norm(vec))
queries = df["embedding"][:100].tolist()
expected_ids = [f"qqp:{i}" for i in range(100)]

print(f"✅ Loaded {len(queries)} queries.")
print("🔍 Running native vector search queries on MongoDB Atlas...")

timings = []
recall = 0

for i, vec in tqdm(enumerate(queries), total=len(queries), desc="🔍 Querying"):
    start = time.time()
    result = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": vec.tolist(),  
                "path": "embedding",
                "numCandidates": 100,
                "limit": 1,
                "index": "vector_index"
            }
        },
        {
            "$project": {
                "_id": 1
            }
        }
    ])
    top = next(result, None)
    top_id = top["_id"] if top else None
    if top_id == expected_ids[i]:
        recall += 1
    timings.append(time.time() - start)

avg_time = sum(timings) / len(timings)
throughput = len(timings) / sum(timings)
recall_at_1 = recall / len(timings)

print("\n📊 MongoDB Native Vector Results (Large Dataset):")
print(f"Average Query Time: {avg_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall_at_1*100:.2f}%")
