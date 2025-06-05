import pymysql

connection = pymysql.connect(
    host="localhost",
    port=4000,
    user="root",
    password=""
)

try:
    with connection.cursor() as cursor:
        cursor.execute("CREATE DATABASE IF NOT EXISTS glove;")
        cursor.execute("USE glove;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS glove_vectors_indexed (
                id INT AUTO_INCREMENT PRIMARY KEY,
                word VARCHAR(100) NOT NULL,
                vector VECTOR(100) NOT NULL
            );
        """)
        cursor.execute("""
            CREATE INDEX vec_index ON glove_vectors_indexed(vector) USING HNSW;
        """)
        connection.commit()
        print("✅ Indexed vector table and index created.")
finally:
    connection.close()
