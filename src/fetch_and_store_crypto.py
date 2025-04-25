import os
import requests
import psycopg2
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("COINMARKETCAP_API_KEY")

# CoinMarketCap endpoint
url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
symbols = ["BTC", "ETH"]

# DB connection
conn = psycopg2.connect(
    dbname="cryptodb",
    user="postgres",
    password="admin",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Fetch and insert data
for symbol in symbols:
    params = {
        'symbol': symbol,
        'convert': 'USD'
    }
    headers = {
        'X-CMC_PRO_API_KEY': API_KEY
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    try:
        quote = data['data'][symbol]['quote']['USD']
        price = quote['price']
        market_cap = quote['market_cap']
        volume_24h = quote['volume_24h']
        percent_change_24h = quote['percent_change_24h']

        # Insert into DB
        cursor.execute("""
            INSERT INTO crypto_prices (symbol, price_usd, market_cap_usd, volume_24h_usd, percent_change_24h)
            VALUES (%s, %s, %s, %s, %s)
        """, (symbol, price, market_cap, volume_24h, percent_change_24h))
        print(f"{symbol} data inserted.")
    except Exception as e:
        print(f"Error inserting data for {symbol}: {e}")

conn.commit()
cursor.close()
conn.close()
print("All done!")
