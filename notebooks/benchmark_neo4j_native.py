import numpy as np
from neo4j import GraphDatabase
from tqdm import tqdm
import time
from sklearn.metrics.pairwise import cosine_similarity

uri = "bolt://localhost:7687"
username = "neo4j"
password = "benchmarking123"

TOP_K = 1
GLOVE_FILE = "datasets/glove/glove.6B.100d.txt"

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
queries = load_queries(GLOVE_FILE, num=100)
print(f"✅ Loaded {len(queries)} queries.")

driver = GraphDatabase.driver(uri, auth=(username, password))

def run_vector_search(tx, query_vec):
    result = tx.run(
        """
        CALL db.index.vector.queryNodes('vector_index', $k, $vector)
        YIELD node, score
        RETURN node.text AS word
        """,
        vector=query_vec.tolist(),
        k=TOP_K
    )
    return result.single()["word"]

print("🔍 Running queries...")
timings = []
correct = 0

with driver.session() as session:
    for query_word, query_vec in tqdm(queries):
        start = time.time()
        try:
            best_word = session.execute_read(run_vector_search, query_vec)
            if best_word == query_word:
                correct += 1
        except Exception as e:
            print(f"Error on query {query_word}: {e}")
        timings.append(time.time() - start)

driver.close()

avg_time = sum(timings) / len(timings)
throughput = len(timings) / sum(timings)
recall = correct / len(timings)

print("\n📊 Neo4j Native Vector Search Results:")
print(f"Average Query Time: {avg_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall * 100:.2f}%")
