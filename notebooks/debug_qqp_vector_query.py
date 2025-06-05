import pymysql
import numpy as np
import random

embeddings = np.load("datasets/large/qqp_embeddings.npy")
print(f"✅ Embeddings loaded: {embeddings.shape}")

conn = pymysql.connect(
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port=4000,
    user="2PqTALbrMMjaf9T.root",
    password="Cyh5LaqnYqcne19X",
    database="qqp",
    ssl_verify_cert=True,
    ssl_verify_identity=True,
    ssl={'ca': 'ca.pem'}
)
cursor = conn.cursor()

i = random.randint(0, len(embeddings) - 1)
query_vec = embeddings[i]
vec_str = "[" + ",".join(map(str, query_vec.tolist())) + "]"

sql = f"""
    SELECT id, VEC_L2_DISTANCE(embedding_vec, VEC_FROM_TEXT('{vec_str}')) as dist
    FROM qqp_vectors_native
    ORDER BY dist
    LIMIT 1
"""
cursor.execute(sql)
result = cursor.fetchone()

print("\n🔎 Debugging Native Vector Search for QQP")
print(f"• Query index: {i}")
print(f"• Expected ID: {i}")
if result:
    print(f"• Returned ID: {result[0]}")
    print(f"• Distance: {result[1]:.4f}")
    try:
        if int(result[0]) == i:
            print("✅ Top result is CORRECT (Recall@1 = 100%)")
        else:
            print("❌ Top result is INCORRECT (Recall@1 = 0%)")
    except Exception as e:
        print(f"⚠️ Could not compare IDs: {e}")
else:
    print("❌ No result returned from query.")

cursor.close()
conn.close()
