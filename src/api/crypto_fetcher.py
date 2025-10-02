import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class CryptoDataFetcher:
    """Handles fetching crypto data from CoinMarketCap API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("COINMARKETCAP_API_KEY")
        self.api_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        
        if not self.api_key:
            raise ValueError("API key not found. Set COINMARKETCAP_API_KEY in .env")
    
    def fetch_data_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Fetch raw data from CoinMarketCap API"""
        params = {'symbol': symbol, 'convert': 'USD'}
        headers = {'X-CMC_PRO_API_KEY': self.api_key}
        
        response = requests.get(self.api_url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        if 'data' not in data or symbol not in data['data']:
            raise ValueError(f"Invalid response for {symbol}")
        
        return data
    
    def prepare_insert_data(self, symbol: str, api_data: Dict[str, Any]) -> tuple:
        """Convert API data to database insert format"""
        quote = api_data['data'][symbol]['quote']['USD']
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return (
            symbol,
            quote['price'],
            quote['volume_24h'],
            quote.get('percent_change_24h'),
            quote['market_cap'],
            timestamp,
        )
    
    def store_data(self, conn, api_data: Dict[str, Any], symbol: str) -> bool:
        """Store data in database"""
        insert_data = self.prepare_insert_data(symbol, api_data)
        
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO crypto_prices
              (symbol, price_usd, volume_24h_usd, percent_change_24h, market_cap_usd, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            insert_data
        )
        conn.commit()
        cursor.close()
        
        return True