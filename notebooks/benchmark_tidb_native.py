import pymysql
import numpy as np
import time
import random
import ast

conn = pymysql.connect(
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port=4000,
    user="2PqTALbrMMjaf9T.root",
    password="Cyh5LaqnYqcne19X",
    database="glove",
    ssl_verify_cert=True,
    ssl_verify_identity=True,
    ssl={'ca': 'ca.pem'}
)
cursor = conn.cursor()

cursor.execute("SELECT id FROM glove_vectors_native")
all_words = [row[0] for row in cursor.fetchall()]
query_words = random.sample(all_words, 100)

total_time = 0
recall_hits = 0

print("🔍 Running 100 native vector queries using VEC_L2_DISTANCE...")

for word in query_words:
    cursor.execute("SELECT embedding_vec FROM glove_vectors_native WHERE id = %s", (word,))
    result = cursor.fetchone()
    if not result:
        continue
    vec_str = result[0]  

    start = time.time()

    sql = f"""
        SELECT id FROM glove_vectors_native
        ORDER BY VEC_L2_DISTANCE(embedding_vec, VEC_FROM_TEXT('{vec_str}'))
        LIMIT 1
    """
    cursor.execute(sql)
    top_result = cursor.fetchone()
    end = time.time()

    if top_result and top_result[0] == word:
        recall_hits += 1

    total_time += (end - start)

avg_time = total_time / 100
throughput = 100 / total_time
recall = recall_hits / 100

print("\n📊 TiDB Native Benchmark Results (400k vectors):")
print(f"Average Query Time: {avg_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall:.2%}%")

cursor.close()
conn.close()
