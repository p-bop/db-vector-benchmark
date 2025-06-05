from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
username = "neo4j"
password = "benchmarking123"

driver = GraphDatabase.driver(uri, auth=(username, password))

def create_vector_index(tx):
    tx.run("""
    CREATE VECTOR INDEX vector_index
    FOR (w:Word)
    ON (w.vector)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 100,
        `vector.similarity_function`: 'cosine'
      }
    }
    """)

with driver.session() as session:
    print("🔧 Creating vector index in Neo4j...")
    session.execute_write(create_vector_index)
    print("✅ Vector index created successfully.")

driver.close()
