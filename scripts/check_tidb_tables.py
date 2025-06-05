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
    cursor.execute("SHOW TABLES;")
    tables = cursor.fetchall()
    print("📋 Tables in TiDB Cloud:")
    for table in tables:
        print(f" - {table[0]}")

connection.close()
