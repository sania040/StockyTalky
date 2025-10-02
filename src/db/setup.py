from src.db.get_connection import get_db_connection

def create_tables():
    """Create all necessary database tables"""
    conn = get_db_connection()
    
    if conn is None:
        print("Failed to connect to the database.")
        return False
    
    cursor = conn.cursor()
    
    # Create crypto_prices table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS crypto_prices (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL,
        price_usd NUMERIC NOT NULL,
        market_cap_usd NUMERIC NOT NULL,
        volume_24h_usd NUMERIC NOT NULL,
        percent_change_24h NUMERIC,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_symbol_timestamp (symbol, timestamp)
    );
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print("✅ Database tables created successfully.")
    return True

def reset_database():
    """Drop and recreate all tables (use with caution!)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS crypto_prices CASCADE;")
    conn.commit()
    cursor.close()
    conn.close()
    
    create_tables()