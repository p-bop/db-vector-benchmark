from neo4j import GraphDatabase
import numpy as np
import pickle
from tqdm import tqdm
import time
from sklearn.metrics.pairwise import cosine_similarity

uri = "bolt://localhost:7687"
user = "neo4j"
password = "test1234"
driver = GraphDatabase.driver(uri, auth=(user, password))

with open("datasets/large/qqp_with_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

df["embedding"] = df.apply(lambda row: (
    (np.array(row["q1_vector"]) + np.array(row["q2_vector"])) / 2
), axis=1)
df["embedding"] = df["embedding"].apply(lambda vec: vec / np.linalg.norm(vec))

print(f"📄 Loaded {len(df)} query records.")
print("🔍 Running Neo4j brute-force benchmark...")

recall = 0
timings = []

with driver.session() as session:
    for i, row in tqdm(df.iloc[:100].iterrows(), total=100):
        query_vec = row["embedding"]

        start = time.time()
        result = session.run("MATCH (q:Question) RETURN q.id AS id, q.embedding AS emb")
        scores = []
        for record in result:
            doc_id = record["id"]
            emb = record["emb"]
            if emb is None: continue  
            emb = np.array(emb)
            sim = cosine_similarity([query_vec], [emb])[0][0]
            scores.append((doc_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_id = scores[0][0]
        expected_id = f"qqp:{i}"
        if top_id == expected_id:
            recall += 1
        timings.append(time.time() - start)

avg_time = sum(timings) / len(timings)
throughput = len(timings) / sum(timings)
recall_at_1 = recall / len(timings)

print("\n📊 Neo4j Baseline Results (Large Dataset):")
print(f"Average Query Time: {avg_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall_at_1*100:.2f}%")
