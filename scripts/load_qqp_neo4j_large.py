from neo4j import GraphDatabase
import pickle
import numpy as np
from tqdm import tqdm

uri = "bolt://localhost:7687"
user = "neo4j"
password = "test1234"
driver = GraphDatabase.driver(uri, auth=(user, password))

with open("datasets/large/qqp_with_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

df["embedding"] = df.apply(lambda row: (
    (np.array(row["q1_vector"]) + np.array(row["q2_vector"])) / 2
), axis=1)
df["embedding"] = df["embedding"].apply(lambda vec: (vec / np.linalg.norm(vec)).tolist())

cypher = """
MERGE (q:Question {id: $id})
SET q.question1 = $q1,
    q.question2 = $q2,
    q.embedding = $embedding
"""

with driver.session() as session:
    print("🧹 Clearing previous data...")
    session.run("MATCH (n:Question) DETACH DELETE n")
    print("⬆️ Inserting new data...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        session.run(cypher, {
            "id": f"qqp:{idx}",
            "q1": row["question1"],
            "q2": row["question2"],
            "embedding": row["embedding"]  
        })

print("✅ Finished loading QQP dataset into Neo4j.")
