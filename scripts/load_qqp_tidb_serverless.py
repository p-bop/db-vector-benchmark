import pandas as pd
import numpy as np
import pymysql
from tqdm import tqdm

CSV_PATH = "datasets/large/qqp_cleaned.csv"
EMBEDDING_PATH = "datasets/large/qqp_embeddings.npy"
BATCH_SIZE = 500  

qqp_df = pd.read_csv(CSV_PATH)
embeddings = np.load(EMBEDDING_PATH)
assert embeddings.shape[0] == len(qqp_df) * 2, "Mismatch in number of embeddings"

print(f"📄 CSV rows: {len(qqp_df)}")
print(f"🔢 Embeddings: {embeddings.shape[0]}")

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

print("🚀 Uploading QQP vectors into TiDB (safe parameterized INSERT)...")

try:
    for start in tqdm(range(0, len(qqp_df), BATCH_SIZE), desc="Uploading"):
        batch = []
        for i in range(start, min(start + BATCH_SIZE, len(qqp_df))):
            q1 = qqp_df.iloc[i]["question1"].replace("'", "''")
            q2 = qqp_df.iloc[i]["question2"].replace("'", "''")
            vec1 = "[" + ",".join(map(str, embeddings[i])) + "]"
            vec2 = "[" + ",".join(map(str, embeddings[i + len(qqp_df)])) + "]"
            batch.append((f"{i}_q1", q1, vec1))
            batch.append((f"{i}_q2", q2, vec2))

        sql = "INSERT INTO qqp_vectors_native (id, sentence, embedding_vec) VALUES " + ",".join(
            ["(%s, %s, VEC_FROM_TEXT(%s))"] * len(batch)
        )
        flat_params = [val for triple in batch for val in triple]

        try:
            cursor.execute(sql, flat_params)
            conn.commit()
        except Exception as batch_err:
            print(f"❌ Batch starting at row {start} failed:", batch_err)

    print("✅ Upload complete.")
except Exception as e:
    print(f"🚨 Fatal error during upload: {e}")
finally:
    cursor.close()
    conn.close()
