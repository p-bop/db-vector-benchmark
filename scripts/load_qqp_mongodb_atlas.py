import pickle
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
cluster_url = os.getenv("MONGO_CLUSTER_URL")

uri = cluster_url.replace("<db_username>", username).replace("<db_password>", password)
client = MongoClient(uri)
collection = client.qqp.vectors

collection.delete_many({})
print("🧹 Cleared 'qqp.vectors' collection.")

with open("datasets/large/qqp_with_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

df["embedding"] = df.apply(lambda row: (
    (np.array(row["q1_vector"]) + np.array(row["q2_vector"])) / 2
), axis=1)
df["embedding"] = df["embedding"].apply(lambda vec: vec / np.linalg.norm(vec))

for idx, row in tqdm(df.iterrows(), total=len(df)):
    doc = {
        "_id": f"qqp:{idx}",
        "question1": row["question1"],
        "question2": row["question2"],
        "embedding": row["embedding"].tolist()
    }
    collection.insert_one(doc)

print("✅ Finished loading QQP vectors into MongoDB Atlas.")
