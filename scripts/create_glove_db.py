import pymysql

conn = pymysql.connect(
    host="tidb.kecoxx1rk23p.clusters.tidb-cloud.com",
    port=4000,
    user="root",
    password="rootpass",
    ssl={'ca': 'ca.pem'}
)

cursor = conn.cursor()
cursor.execute("CREATE DATABASE IF NOT EXISTS glove;")
print("Database 'glove' created or already exists.")

cursor.close()
conn.close()
