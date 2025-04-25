import os
import requests
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
from src.db.get_connection import get_db_connection

# PydanticAI imports
from pydantic_ai import Agent, tool_plain

load_dotenv()
API_KEY = os.getenv("COINMARKETCAP_API_KEY")


class CryptoDataFetcher:
    def __init__(self, api_key, api_url="https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"):
        self.api_key = api_key
        self.api_url = api_url
        if not self.api_key:
            raise ValueError("API key not found. Please set COINMARKETCAP_API_KEY in your .env file.")

    def fetch_data_for_symbol(self, symbol):
        params = {'symbol': symbol, 'convert': 'USD'}
        headers = {'X-CMC_PRO_API_KEY': self.api_key}
        resp = requests.get(self.api_url, headers=headers, arams=params)
        resp.raise_for_status()
        return resp.json()

    def _prepare_insert_data(self, symbol, data):
        quote = data['data'][symbol]['quote']['USD']
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            symbol,
            quote['price'],
            quote['volume_24h'],
            quote.get('percent_change_24h'),
            quote['market_cap'],
            ts,
        )

    def store_data(self, conn, api_data, symbol):
        insert = self._prepare_insert_data(symbol, api_data)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO crypto_prices
              (symbol, price_usd, volume_24h_usd, percent_change_24h, market_cap_usd, timestamp)
            VALUES (%s,%s,%s,%s,%s,%s)
            """, insert
        )
        conn.commit()
        cur.close()
        return True

# ——— PydanticAI Agent ———
agent = Agent(
    name="CryptoDataAgent",
    system_prompt=(
        "You have two tools:\n"
        "  • fetch(symbol) → returns raw CoinMarketCap JSON\n"
        "  • store(api_data, symbol) → writes into postgres\n"
    )
)

@tool_plain(agent, name="fetch", description="Fetch USD quote JSON for a symbol")
def fetch(symbol: str) -> dict:
    """Fetch the raw JSON payload for a given symbol."""
    fetcher = CryptoDataFetcher(API_KEY)
    data = fetcher.fetch_data_for_symbol(symbol)
    if 'data' not in data or symbol not in data['data']:
        raise ValueError(f"Bad response for {symbol}")
    return data

@tool_plain(agent, name="store", description="Store fetched JSON into crypto_prices")
def store(api_data: dict, symbol: str) -> str:
    """Store the given payload into Postgres and return status."""
    conn = get_db_connection()
    ok = CryptoDataFetcher(API_KEY).store_data(conn, api_data, symbol)
    conn.close()
    return "stored" if ok else "failed"

# Expose
__all__ = ["agent", "fetch", "store", "CryptoDataFetcher"]