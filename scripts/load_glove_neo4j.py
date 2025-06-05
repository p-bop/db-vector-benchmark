from neo4j import GraphDatabase
from tqdm import tqdm

uri = "bolt://localhost:7687"
username = "neo4j"
password = "benchmarking123"  

driver = GraphDatabase.driver(uri, auth=(username, password))

def load_vectors(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = list(map(float, parts[1:]))
            yield word, vector

def insert_vector(tx, word, vector):
    tx.run(
        "MERGE (w:Word {text: $word}) "
        "SET w.vector = $vector",
        word=word,
        vector=vector
    )

def main():
    input_file = "datasets/glove/glove.6B.100d.txt"
    print(f"📄 Loading vectors from {input_file} into Neo4j...")

    with driver.session() as session:
        for word, vector in tqdm(load_vectors(input_file), total=400000):
            session.execute_write(insert_vector, word, vector)

    print("✅ Done loading GloVe vectors into Neo4j.")
    driver.close()

if __name__ == "__main__":
    main()
