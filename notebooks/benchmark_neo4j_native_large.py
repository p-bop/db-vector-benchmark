from neo4j import GraphDatabase
import pickle
import numpy as np
from tqdm import tqdm
import time

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

print("🔍 Running Neo4j native vector benchmark...")

timings = []
recall = 0

with driver.session() as session:
    for i, row in tqdm(df.iloc[:100].iterrows(), total=100):
        query_vec = row["embedding"].tolist()
        start = time.time()

        result = session.run(
            """
            CALL db.index.vector.queryNodes('qqp_vector_index', 1, $vec)
            YIELD node, score
            RETURN node.id AS id
            """,
            vec=query_vec
        )

        record = result.single()
        top_id = record["id"] if record else None
        expected_id = f"qqp:{i}"
        if top_id == expected_id:
            recall += 1

        timings.append(time.time() - start)

avg_time = sum(timings) / len(timings)
throughput = len(timings) / sum(timings)
recall_at_1 = recall / len(timings)

print("\n📊 Neo4j Native Vector Results (Large Dataset):")
print(f"Average Query Time: {avg_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall_at_1 * 100:.2f}%")
