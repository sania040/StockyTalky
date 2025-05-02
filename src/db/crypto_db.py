import psycopg2
from src.db.get_connection import get_db_connection



def createTable():
    conn = get_db_connection()
    if conn is None:
        print("Failed to connect to the database.")
        return
    else :
        print("Connected to the database.")
        cursor = conn.cursor()
    
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS crypto_prices (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10),
            price_usd NUMERIC,
            market_cap_usd NUMERIC,
            volume_24h_usd NUMERIC,  -- This is correct
            percent_change_24h NUMERIC,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
    
        conn.commit()
        cursor.close()
        conn.close()
    
        print("Table created successfully.")
