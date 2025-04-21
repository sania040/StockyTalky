# Functions to fetch data from CoinMarketCap API

import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('COINMARKETCAP_API_KEY')

headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': API_KEY,
}

def get_crypto_data(symbol='BTC'):
    url = f'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    params = {'symbol': symbol}
    response = requests.get(url, headers=headers, params=params)
    return response.json()