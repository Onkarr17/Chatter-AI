import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="postgres",   # use postgres for now
    user="postgres",
    password="postgres"
)

print("Postgres connected successfully")
conn.close()