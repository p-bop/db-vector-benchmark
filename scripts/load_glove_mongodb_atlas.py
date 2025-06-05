import os
import numpy as np
from tqdm import tqdm
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
cluster_url = os.getenv("MONGO_CLUSTER_URL")

uri = f"mongodb+srv://{username}:{password}@{cluster_url}/?retryWrites=true&w=majority&appName=vector-m10"

client = MongoClient(uri)
db = client["glove"]
collection = db["vectors"]

collection.delete_many({})
print("🧹 Cleared existing data in 'glove.vectors'")

glove_file = "datasets/glove/glove.6B.100d.txt"
print(f"📄 Loading GloVe vectors from {glove_file}...")

with open(glove_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f, total=400000)):
        parts = line.strip().split()
        word = parts[0]
        vec = np.array(parts[1:], dtype=np.float32)
        vec /= np.linalg.norm(vec)  

        doc = {
            "_id": f"glove:{i}",
            "word": word,
            "vector": vec.tolist()
        }
        collection.insert_one(doc)

print("✅ Finished loading GloVe vectors into MongoDB Atlas.")
