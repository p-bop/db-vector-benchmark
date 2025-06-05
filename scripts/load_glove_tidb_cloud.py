import pymysql
import numpy as np
from tqdm import tqdm

connection = pymysql.connect(
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port=4000,
    user="Qk3rjqd7MQpjBmc.root",
    password="8mV79PC5h7ZqvxNc",
    database="glove",
    ssl={"ca": "/etc/ssl/cert.pem"}
)

cursor = connection.cursor()

cursor.execute("DELETE FROM glove_vectors")
print("🧹 Cleared existing documents.")

glove_path = "datasets/glove/glove.6B.100d.txt"
print(f"📄 Loading first 100,000 vectors from {glove_path}...")

batch_size = 1000
batch = []

with open(glove_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f, total=100000)):
        if i >= 100000:
            break

        parts = line.strip().split()
        word = parts[0]
        vector = np.array(list(map(float, parts[1:])), dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = (vector / norm).tolist()
        else:
            vector = vector.tolist()
        vector_str = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"

        batch.append((word, vector_str))

        if len(batch) == batch_size:
            cursor.executemany(
                "INSERT INTO glove_vectors (word, vector) VALUES (%s, CAST(%s AS JSON))",
                batch
            )
            connection.commit()
            batch = []

if batch:
    cursor.executemany(
        "INSERT INTO glove_vectors (word, vector) VALUES (%s, CAST(%s AS JSON))",
        batch
    )
    connection.commit()

cursor.close()
connection.close()
print("✅ Loaded 100,000 normalized vectors into TiDB Cloud.")
