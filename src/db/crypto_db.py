import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect("crypto_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_price(symbol, price):
    conn = sqlite3.connect("crypto_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prices (symbol, price, timestamp)
        VALUES (?, ?, ?)
    """, (symbol, price, datetime.now().isoformat()))
    conn.commit()
    conn.close()
