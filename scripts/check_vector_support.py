import pymysql

conn = pymysql.connect(
    host="tidb.kecoxx1rk23p.clusters.tidb-cloud.com",
    port=4000,
    user="root",
    password="rootpass",
    ssl={'ca': 'ca.pem'}
)

cursor = conn.cursor()
cursor.execute("SHOW VARIABLES LIKE '%vector%';")
for row in cursor.fetchall():
    print(row)

cursor.close()
conn.close()
