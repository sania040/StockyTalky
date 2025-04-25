import os
import requests
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

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
        percent_change_24h = quote.get('percent_change_24h', None)
        percent_change_1h = quote.get('percent_change_1h', None)
        percent_change_7d = quote.get('percent_change_7d', None)
        percent_change_30d = quote.get('percent_change_30d', None)
        percent_change_60d = quote.get('percent_change_60d', None)
        market_cap_dominance = quote.get('market_cap_dominance', None)
        fully_diluted_market_cap = quote.get('fully_diluted_market_cap', None)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"Fetched Data for {symbol}: {data}")
        
        # Insert data into the database
        try:
            cursor.execute("""
                INSERT INTO crypto_prices (
                    symbol, price_usd, volume_24h, volume_change_24h,
                    percent_change_1h, percent_change_24h, percent_change_7d,
                    percent_change_30d, percent_change_60d, market_cap,
                    market_cap_dominance, fully_diluted_market_cap, timestamp, percent_change_90d
                ) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                symbol, price, volume_24h, None, percent_change_1h, 
                percent_change_24h, percent_change_7d, percent_change_30d, 
                percent_change_60d, market_cap, market_cap_dominance, 
                fully_diluted_market_cap, timestamp, None
            ))
        except psycopg2.Error as e:
            print(f"Error inserting data for {symbol}: {e}")
            conn.rollback()  # Rollback to continue with the rest of the data

    except KeyError as e:
        print(f"Error parsing data for {symbol}: {e}")
    except Exception as e:
        print(f"Error inserting data for {symbol}: {e}")

# Commit and close
conn.commit()
cursor.close()
conn.close()

print("Data insertion complete!")
