import pymysql
import numpy as np
import time
import json
import os
import pickle
from tqdm import tqdm

DB_CONFIG = {
    "host": "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    "port": 4000,
    "user": "2PqTALbrMMjaf9T.root",
    "password": "Cyh5LaqnYqcne19X",
    "database": "qqp",
    "ssl_verify_cert": True,
    "ssl_verify_identity": True,
    "ssl": {'ca': 'ca.pem'}
}
VEC_PATH = "datasets/large/qqp_embeddings.npy"
CHECKPOINT = "results/qqp_tidb_ann_half_checkpoint.pkl"
INDEX_NAME = "idx_vec_l2"

print("📦 Loading first half of QQP vectors...")
all_vectors = np.load(VEC_PATH)[:404283]  
NUM_QUERIES = len(all_vectors)

if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT, "rb") as f:
        start_idx, query_times, correct = pickle.load(f)
    print(f"🔄 Resuming from query {start_idx}...")
else:
    start_idx = 0
    query_times = []
    correct = 0

def connect():
    return pymysql.connect(**DB_CONFIG)

conn = connect()
cursor = conn.cursor()

print(f"🔍 Running {NUM_QUERIES} vector similarity queries using TiDB HNSW index...")
for idx in tqdm(range(start_idx, NUM_QUERIES), initial=start_idx, total=NUM_QUERIES):
    vec = all_vectors[idx].tolist()
    json_vector = json.dumps(vec)

    sql = f"""
        SELECT id
        FROM qqp_vectors_native USE INDEX ({INDEX_NAME})
        ORDER BY VEC_L2_DISTANCE(embedding_vec, CAST('{json_vector}' AS VECTOR(384)))
        LIMIT 1;
    """

    try:
        start = time.time()
        cursor.execute(sql)
        result = cursor.fetchone()
        end = time.time()
    except Exception as e:
        print(f"\n⚠️ Query failed at {idx}: {e}\n🔁 Reconnecting...")
        time.sleep(5)
        conn.close()
        conn = connect()
        cursor = conn.cursor()
        continue

    matched_id = result[0] if result else ""
    if matched_id in (f"{idx}_q1", f"{idx}_q2"):
        correct += 1
    query_times.append(end - start)

    if idx % 500 == 0 and idx > 0:
        with open(CHECKPOINT, "wb") as f:
            pickle.dump((idx + 1, query_times, correct), f)
        print(f"💾 Saved checkpoint at {idx}")

recall_at_1 = correct / NUM_QUERIES
avg_time = sum(query_times) / NUM_QUERIES
throughput = NUM_QUERIES / sum(query_times)

print("\n📊 Final Results (Half Dataset, ANN HNSW):")
print(f"Recall@1        : {recall_at_1:.4f}")
print(f"Avg. Query Time : {avg_time:.4f} sec")
print(f"Throughput      : {throughput:.2f} queries/sec")

cursor.close()
conn.close()
