import pymysql

conn = pymysql.connect(
    host="127.0.0.1",
    port=4000,
    user="root",
    password="",  
    autocommit=True
)

with conn.cursor() as cursor:
    cursor.execute("CREATE DATABASE IF NOT EXISTS benchmark;")
    print("✅ Created 'benchmark' database.")