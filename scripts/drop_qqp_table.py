import pymysql

conn = pymysql.connect(
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port=4000,
    user="2PqTALbrMMjaf9T.root",
    password="Cyh5LaqnYqcne19X",
    database="qqp",
    ssl_verify_cert=True,
    ssl_verify_identity=True,
    ssl={"ca": "ca.pem"}
)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS qqp_vectors_native")
conn.commit()
cursor.close()
conn.close()
print("✅ Table 'qqp_vectors_native' dropped successfully.")
