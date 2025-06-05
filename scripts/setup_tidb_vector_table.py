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
            CREATE TABLE IF NOT EXISTS glove_vectors (
                id INT AUTO_INCREMENT PRIMARY KEY,
                word VARCHAR(100) NOT NULL,
                vector BLOB NOT NULL
            );
        """)
        connection.commit()
        print("✅ TiDB database and table created successfully.")
finally:
    connection.close()