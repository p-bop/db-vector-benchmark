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
cursor.execute("CREATE DATABASE IF NOT EXISTS qqp;")
print("✅ Database 'qqp' created or already exists.")

cursor.close()
conn.close()
