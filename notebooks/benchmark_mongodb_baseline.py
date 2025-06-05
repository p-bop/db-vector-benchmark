import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time

def load_queries(path, num=100):
    queries = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num:
                break
            parts = line.strip().split()
            word = parts[0]
            vec = list(map(float, parts[1:]))
            queries.append((word, np.array(vec)))
    return queries

queries = load_queries("datasets/glove/glove.6B.100d.txt")
print(f"📄 Loaded {len(queries)} queries.")

client = MongoClient("mongodb://localhost:27017/")
collection = client.benchmark.glove_vectors
cursor = collection.find()
all_vectors = [(doc["word"], np.array(doc["vector"])) for doc in cursor]
words, vectors = zip(*all_vectors)
vectors = np.stack(vectors)

timings = []
correct = 0

for word, qvec in tqdm(queries, desc="🔍 Running queries"):
    start = time.time()
    sims = cosine_similarity([qvec], vectors)[0]
    best_idx = np.argmax(sims)
    best_word = words[best_idx]
    if best_word == word:
        correct += 1
    timings.append(time.time() - start)

avg_time = sum(timings) / len(timings)
throughput = len(timings) / sum(timings)
recall_at_1 = correct / len(timings)

print("\n📊 MongoDB Brute-Force Results:")
print(f"Average Query Time: {avg_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall_at_1*100:.2f}%")
