import os
import requests
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
from src.db.get_connection import get_db_connection
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from src.db.get_connection import get_db_connection
# PydanticAI imports
from pydantic_ai import Agent

load_dotenv()
API_KEY = os.getenv("COINMARKETCAP_API_KEY")

# ——— Your existing fetcher ———
class CryptoDataFetcher:
    def __init__(self, api_key, api_url="https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"):
        self.api_key = api_key
        self.api_url = api_url
        if not self.api_key:
            raise ValueError("API key not found. Please set COINMARKETCAP_API_KEY in your .env file.")

    def fetch_data_for_symbol(self, symbol):
        params = {'symbol': symbol, 'convert': 'USD'}
        headers = {'X-CMC_PRO_API_KEY': self.api_key}
        resp = requests.get(self.api_url, headers=headers, params=params)
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

# there will be no Description for the tools
@agent.tool_plain(name="fetch") 
def fetch(symbol: str) -> dict:
    """Fetch the raw JSON payload for a given symbol."""
    fetcher = CryptoDataFetcher(API_KEY)
    data = fetcher.fetch_data_for_symbol(symbol)
    if 'data' not in data or symbol not in data['data']:
        raise ValueError(f"Bad response for {symbol}")
    return data

@agent.tool_plain(name="store")
def store(api_data: dict, symbol: str) -> str:
    """Store the given payload into Postgres and return status."""
    conn = get_db_connection()
    ok = CryptoDataFetcher(API_KEY).store_data(conn, api_data, symbol)
    conn.close()
    return "stored" if ok else "failed"



def show_visualizations():
    st.title("Crypto Investment Visualizations")
    
    # Fetch data
    conn = get_db_connection()
    query = """
    SELECT symbol, price_usd, volume_24h_usd, percent_change_24h, 
           market_cap_usd, timestamp 
    FROM crypto_prices 
    ORDER BY timestamp DESC
    """
    df = pd.read_sql(query, conn, parse_dates=['timestamp'])
    conn.close()
    
    if df.empty:
        st.info("No data available. Please collect some data first.")
        return
    
    # Get available symbols
    symbols = df['symbol'].unique().tolist()
    selected_symbols = st.multiselect("Select cryptocurrencies", symbols, default=symbols[:3])
    
    if not selected_symbols:
        st.warning("Please select at least one cryptocurrency")
        return
    
    # Filter data
    filtered_df = df[df['symbol'].isin(selected_symbols)]
    
    # Visualization 1: Price History
    st.subheader("Price History")
    price_fig = px.line(
        filtered_df, 
        x='timestamp', 
        y='price_usd', 
        color='symbol',
        title="Price History",
        labels={"price_usd": "Price (USD)", "timestamp": "Date", "symbol": "Crypto"}
    )
    st.plotly_chart(price_fig, use_container_width=True)
    
    # Visualization 2: Volume Analysis
    st.subheader("Trading Volume")
    volume_fig = px.bar(
        filtered_df, 
        x='timestamp', 
        y='volume_24h_usd', 
        color='symbol',
        title="24h Trading Volume",
        labels={"volume_24h_usd": "Volume (USD)", "timestamp": "Date", "symbol": "Crypto"}
    )
    st.plotly_chart(volume_fig, use_container_width=True)
    
    # Visualization 3: Price Change
    st.subheader("24h Price Change %")
    latest_data = df.sort_values('timestamp').drop_duplicates(subset=['symbol'], keep='last')
    latest_filtered = latest_data[latest_data['symbol'].isin(selected_symbols)]
    
    change_fig = px.bar(
        latest_filtered,
        x='symbol',
        y='percent_change_24h',
        color='percent_change_24h',
        color_continuous_scale=['red', 'green'],
        title="24h Price Change (%)",
        labels={"percent_change_24h": "Change %", "symbol": "Crypto"}
    )
    st.plotly_chart(change_fig, use_container_width=True)
    
    # Visualization 4: Market Cap Comparison
    st.subheader("Market Cap Comparison")
    cap_fig = px.pie(
        latest_filtered,
        values='market_cap_usd',
        names='symbol',
        title="Market Cap Distribution"
    )
    st.plotly_chart(cap_fig, use_container_width=True)
    
    # Visualization 5: Correlation Heatmap
    st.subheader("Price Correlation")
    
    # Pivot data for correlation
    pivot_df = filtered_df.pivot_table(
        index='timestamp', 
        columns='symbol', 
        values='price_usd'
    )
    
    # Calculate correlation
    corr = pivot_df.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Investment Recommendation Section
    st.subheader("Investment Recommendations")
    
    for symbol in selected_symbols:
        symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
        if len(symbol_data) < 2:
            continue
            
        # Simple price trend analysis
        latest_price = symbol_data['price_usd'].iloc[-1]
        price_change = symbol_data['percent_change_24h'].iloc[-1]
        
        # Simple recommendation logic
        if price_change > 5:
            recommendation = "Consider Taking Profit"
            color = "sell"
            reason = f"Price increased by {price_change:.2f}% in 24h"
        elif price_change < -5:
            recommendation = "Consider Buying"
            color = "buy"
            reason = f"Price dropped by {abs(price_change):.2f}% in 24h"
        else:
            recommendation = "Hold"
            color = "hold"
            reason = "Price relatively stable"
        
        # Display recommendation card
        st.markdown(f"""
        <div class="card {color}">
            <h3>{symbol}</h3>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <p class="metric-label">Price</p>
                    <p class="metric-value">${latest_price:.2f}</p>
                </div>
                <div>
                    <p class="metric-label">24h Change</p>
                    <p class="metric-value {'positive' if price_change > 0 else 'negative'}">
                        {'+' if price_change > 0 else ''}{price_change:.2f}%
                    </p>
                </div>
                <div>
                    <p class="metric-label">Recommendation</p>
                    <p class="metric-value">{recommendation}</p>
                </div>
            </div>
            <p>{reason}</p>
        </div>
        """, unsafe_allow_html=True)

# Add this to your main app.py
# In the sidebar navigation or main layout
if __name__ == "__main__":
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Collection", "Visualizations", "Dashboard"])
    
    load_css()  # Your existing CSS function
    
    if page == "Data Collection":
        show_data_collection()  # Your existing function
    elif page == "Visualizations":
        show_visualizations()  # Our new function
    elif page == "Dashboard":
        show_dashboard()  # Your existing function

# Expose
__all__ = ["agent", "fetch", "store", "CryptoDataFetcher"]
__all__ = ["agent", "fetch", "store", "CryptoDataFetcher"]
