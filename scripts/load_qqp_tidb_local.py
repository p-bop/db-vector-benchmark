import pymysql
import pickle
import numpy as np
from tqdm import tqdm

with open("datasets/large/qqp_with_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

df["embedding"] = df.apply(lambda row: (
    (np.array(row["q1_vector"]) + np.array(row["q2_vector"])) / 2
), axis=1)
df["embedding"] = df["embedding"].apply(lambda vec: vec / np.linalg.norm(vec))

connection = pymysql.connect(
    host="127.0.0.1",
    port=4000,
    user="root",
    database="glove",
    ssl={'ssl': {}},  
    cursorclass=pymysql.cursors.DictCursor
)

connection.autocommit(True)

with connection:
    with connection.cursor() as cursor:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                cursor.execute("""
                    INSERT INTO qqp_vectors (id, question1, question2, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE question1=VALUES(question1), question2=VALUES(question2), embedding=VALUES(embedding)
                """, (
                    f"qqp:{idx}",
                    row["question1"],
                    row["question2"],
                    row["embedding"].astype(np.float32).tobytes()
                ))
            except Exception as e:
                print(f"⚠️ Skipping row {idx} due to error: {e}")

print("✅ Finished loading QQP dataset into local TiDB.")
