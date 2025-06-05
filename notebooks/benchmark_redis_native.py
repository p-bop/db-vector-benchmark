import redis
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from redis.commands.search.query import Query

r = redis.Redis(host='localhost', port=6380)

INDEX_NAME = "glove_index"
VECTOR_DIM = 100
VECTOR_FIELD_NAME = "vector"
TOP_K = 1

def load_queries(path, num=100):
    queries = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num:
                break
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            queries.append((word, vec))
    return queries

print("📄 Loading 100 queries...")
queries = load_queries("datasets/glove/glove.6B.100d.txt")
print("✅ Loaded 100 queries.")

print("🔍 Running queries...")
recall_count = 0
timings = []

for word, vec in tqdm(queries):
    query_str = f'*=>[KNN {TOP_K} @{VECTOR_FIELD_NAME} $vec_param AS score]'
    q = Query(query_str) \
        .return_fields("word", "score") \
        .dialect(2)

    params_dict = {"vec_param": vec.tobytes()}

    start = time.time()
    result = r.ft(INDEX_NAME).search(q, query_params=params_dict)
    end = time.time()
    timings.append(end - start)

    if result.docs:
        retrieved_word = result.docs[0].word
        if retrieved_word == word:
            recall_count += 1

avg_time = sum(timings) / len(timings)
throughput = len(timings) / sum(timings)
recall = recall_count / len(queries)

print("\n📊 Redis Native (RediSearch) Results:")
print(f"Average Query Time: {avg_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} queries/sec")
print(f"Recall@1: {recall:.2%}")
