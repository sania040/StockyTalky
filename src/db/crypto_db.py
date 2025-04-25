import psycopg2

conn = psycopg2.connect(
    dbname="cryptodb",
    user="postgres",
    password="admin",
    host="localhost",
    port="5432"
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS crypto_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    price_usd NUMERIC,
    market_cap_usd NUMERIC,
    volume_24h_usd NUMERIC,
    percent_change_24h NUMERIC,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

conn.commit()
cursor.close()
conn.close()

print("Table created successfully.")
