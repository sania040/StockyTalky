from dotenv import load_dotenv
import os
import requests

def get_crypto_price(symbol: str) -> float:
    # load .env
    load_dotenv()
    api_key = os.getenv("COINMARKETCAP_API_KEY")
    if not api_key:
        raise RuntimeError("Set COINMARKETCAP_API_KEY in .env")

    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    headers = {"X-CMC_PRO_API_KEY": api_key}
    params  = {"symbol": symbol, "convert": "USD"}

    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data["data"][symbol]["quote"]["USD"]["price"]

if __name__ == "__main__":
    for coin in ["BTC", "ETH"]:
        price = get_crypto_price(coin)
        print(f"{coin}: ${price:,.2f}")
