import pymysql
import numpy as np
from tqdm import tqdm

glove_path = "datasets/glove/glove.6B.100d.txt"

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

insert_query = """
INSERT INTO glove_vectors_native (id, word, embedding, embedding_vec)
VALUES (%s, %s, %s, %s)
"""

batch = []
batch_size = 500

with open(glove_path, "r", encoding="utf8") as f:
    for line in tqdm(f, desc="Uploading vectors"):
        parts = line.strip().split()
        word = parts[0]
        vec = list(map(float, parts[1:]))

        vec_blob = np.array(vec, dtype=np.float32).tobytes()
        vec_str = "[" + ", ".join(f"{x:.6f}" for x in vec) + "]"

        batch.append((word, word, vec_blob, vec_str))

        if len(batch) >= batch_size:
            cursor.executemany(insert_query, batch)
            conn.commit()
            batch.clear()

if batch:
    cursor.executemany(insert_query, batch)
    conn.commit()

cursor.close()
conn.close()

print("✅ GloVe vectors successfully loaded into TiDB Serverless with native VECTOR column.")
