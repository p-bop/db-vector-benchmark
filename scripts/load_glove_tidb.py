import pymysql
import numpy as np
from tqdm import tqdm

connection = pymysql.connect(
    host="localhost",
    port=4000,
    user="root",
    password="",
    database="glove"
)

glove_path = "datasets/glove/glove.6B.100d.txt"

print(f"📄 Loading GloVe vectors from {glove_path} into TiDB...")

with connection.cursor() as cursor, open(glove_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=400000):
        parts = line.strip().split()
        word = parts[0]
        vector = np.array(parts[1:], dtype=np.float32).tobytes()
        cursor.execute("INSERT INTO glove_vectors (word, vector) VALUES (%s, %s)", (word, vector))

    connection.commit()

print("✅ Done loading GloVe vectors into TiDB.")
connection.close()