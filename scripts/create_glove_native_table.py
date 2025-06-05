import pymysql

conn = pymysql.connect(
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port=4000,
    user="2PqTALbrMMjaf9T.root",
    password="Cyh5LaqnYqcne19X",
    database="test",
    ssl_verify_cert=True,
    ssl_verify_identity=True,
    ssl={'ca': 'ca.pem'}
)

cursor = conn.cursor()

cursor.execute("CREATE DATABASE IF NOT EXISTS glove;")
cursor.execute("USE glove;")

cursor.execute("""
CREATE TABLE IF NOT EXISTS glove_vectors_native (
    id VARCHAR(255) PRIMARY KEY,
    word VARCHAR(255),
    embedding BLOB,
    embedding_vec VECTOR FLOAT32(100) NOT NULL COMMENT 'vector column'
);
""")

print("✅ Table 'glove_vectors_native' created successfully with native VECTOR column.")

cursor.close()
conn.close()
