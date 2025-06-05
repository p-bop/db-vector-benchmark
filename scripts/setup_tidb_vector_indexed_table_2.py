import pymysql

connection = pymysql.connect(
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port=4000,
    user="Qk3rjqd7MQpjBmc.root",
    password="8mV79PC5h7ZqvxNc",
    database="glove",
    ssl={"ca": "/etc/ssl/cert.pem"}
)

with connection.cursor() as cursor:
    print("🔧 Creating vector index on `glove_vectors_indexed.vector` using VSS...")
    cursor.execute("CREATE INDEX vector_index ON glove_vectors_indexed (vector) USING VSS;")
    connection.commit()
    print("✅ Vector index created successfully.")

connection.close()
