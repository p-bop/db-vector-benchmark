import pymysql

conn = pymysql.connect(
    host="tidb.kecoxx1rk23p.clusters.tidb-cloud.com",
    port=4000,
    user="root",
    password="rootpass",
    db="test",
    ssl={'ca': 'ca.pem'}
)

cursor = conn.cursor()
cursor.execute("SELECT NOW();")
print("Connection successful:", cursor.fetchone())

cursor.close()
conn.close()
