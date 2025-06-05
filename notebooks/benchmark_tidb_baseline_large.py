import pandas as pd
import numpy as np
import pymysql
import time
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

connection = pymysql.connect(
    host="127.0.0.1",
    port=4000,
    user="root",
    database="glove",
    ssl={'ssl': {}},
    cursorclass=pymysql.cursors.DictCursor
)

df = pd.read_pickle("datasets/large/qqp_with_embeddings.pkl")
df["embedding"] = df.apply(lambda row: (
    (np.array(row["q1_vector"]) + np.array(row["q2_vector"])) / 2
), axis=1)
df["embedding"] = df["embedding"].apply(lambda vec: vec / np.linalg.norm(vec))

queries = df.iloc[:100]

with connection.cursor() as cursor:
    cursor.execute("SELECT id, embedding FROM qqp_vectors")
    rows = cursor.fetchall()

corpus = [(row["id"], np.frombuffer(row["embedding"], dtype=np.float32)) for row in rows]

timings = []
recall = 0

for i, (_, row) in enumerate(tqdm(queries.iterrows(), total=100)):
    query_vec = row["embedding"]
    start = time.time()

    scores = [(doc_id, cosine_similarity([query_vec], [vec])[0][0]) for doc_id, vec in corpus]
    scores.sort(key=lambda x: x[1], reverse=True)

    top_id = scores[0][0]
    expected_id = f"qqp:{i}"
    if top_id == expected_id:
        recall += 1

    timings.append(time.time() - start)

avg_time = sum(timings) / len(timings)
throughput = len(timings) / sum(timings)
recall_at_1 = recall / len(timings)

print("\n📊 TiDB Baseline Results (Large Dataset):")
print(f"Average Query Time: {avg_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall_at_1*100:.2f}%")
