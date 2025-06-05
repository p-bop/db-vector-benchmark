import pymysql

conn = pymysql.connect(
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port=4000,
    user="2PqTALbrMMjaf9T.root",
    password="Cyh5LaqnYqcne19X",
    database="qqp",
    ssl_verify_cert=True,
    ssl_verify_identity=True,
    ssl={'ca': 'ca.pem'}
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS qqp_vectors_native (
    id INT PRIMARY KEY,
    sentence TEXT,
    embedding_vec VECTOR(384) NOT NULL COMMENT 'Vector embedding of the sentence'
);
""")

print("✅ Table 'qqp_vectors_native' created successfully with 384-dim vectors.")
cursor.close()
conn.close()
