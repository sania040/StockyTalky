import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.db.get_connection import get_db_connection
from src.db.query_utils import execute_query
from datetime import datetime, timedelta

# --- Data Loading Functions (Cached for Performance) ---

@st.cache_data(ttl=600)
def load_latest_data():
    """Fetches the single latest data point for each crypto."""
    print("--- Fetching latest data from database ---")
    query = """
    WITH latest_prices AS (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY timestamp DESC) as rn
        FROM crypto_prices
    )
    SELECT symbol, price_usd, volume_24h_usd, percent_change_24h, market_cap_usd, timestamp
    FROM latest_prices
    WHERE rn = 1;
    """
    conn = get_db_connection()
    df = execute_query(conn, query)
    return df

@st.cache_data(ttl=600)
def load_historical_data(symbol):
    """Fetches the full price history for a single, selected cryptocurrency."""
    print(f"--- Fetching historical data for {symbol} ---")
    query = "SELECT timestamp, price_usd, volume_24h_usd FROM crypto_prices WHERE symbol = %s ORDER BY timestamp ASC;"
    conn = get_db_connection()
    df = execute_query(conn, query, params=(symbol,))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# --- Main Dashboard Page ---

def show():
    """Renders the Investment Dashboard page."""
    st.title(" Investment Dashboard")
    latest_df = load_latest_data()

    if latest_df.empty:
        st.warning("No data available. Please ensure the data collector is running.")
        return

    # --- KPI Section ---
    st.subheader("Market Overview")
    total_market_cap = latest_df['market_cap_usd'].sum()
    total_volume = latest_df['volume_24h_usd'].sum()
    weighted_avg_change = (latest_df['percent_change_24h'] * latest_df['market_cap_usd']).sum() / total_market_cap
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Market Cap", f"${total_market_cap/1e12:.2f}T")
    col2.metric("24h Volume", f"${total_volume/1e9:.2f}B")
    col3.metric("Market Avg 24h Change", f"{weighted_avg_change:+.2f}%")
    st.markdown("---")

    # --- NEW: Top & Worst Performers Section ---
    st.subheader(" Daily Market Movers")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("####  Top 5 Gainers")
        top_gainers = latest_df.nlargest(5, 'percent_change_24h')
        for _, row in top_gainers.iterrows():
            st.metric(label=row['symbol'], value=f"${row['price_usd']:.4f}", delta=f"{row['percent_change_24h']:.2f}%")
    with col2:
        st.markdown("#### Top 5 Losers")
        top_losers = latest_df.nsmallest(5, 'percent_change_24h')
        for _, row in top_losers.iterrows():
            st.metric(label=row['symbol'], value=f"${row['price_usd']:.4f}", delta=f"{row['percent_change_24h']:.2f}%")
    st.markdown("---")

    # --- Main Charting Section (Candlestick & Pie Chart) ---
    col1, col2 = st.columns([2, 1]) # Give more space to the candlestick chart
    with col1:
        st.subheader("Deep Dive Analysis")
        selected_symbol = st.selectbox("Select a Cryptocurrency", latest_df['symbol'].unique())
        hist_df = load_historical_data(selected_symbol)

        if hist_df.empty:
            st.info("No historical data available for this symbol.")
        else:
            # Create Candlestick Chart
            daily_df = hist_df.set_index('timestamp')['price_usd'].resample('D').ohlc()
            fig = go.Figure(data=[go.Candlestick(x=daily_df.index, open=daily_df['open'], high=daily_df['high'], low=daily_df['low'], close=daily_df['close'])])
            fig.update_layout(title_text=f"{selected_symbol}/USD Price", xaxis_rangeslider_visible=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Market Dominance")
        # --- IMPROVED: Smart Pie Chart ---
        # Sort by market cap and take top 10
        top_10 = latest_df.nlargest(10, 'market_cap_usd').copy()
        # Calculate sum of the rest
        others_market_cap = latest_df[~latest_df['symbol'].isin(top_10['symbol'])]['market_cap_usd'].sum()
        # Add "Others" to the DataFrame
        others_row = pd.DataFrame([{'symbol': 'Others', 'market_cap_usd': others_market_cap}])
        pie_df = pd.concat([top_10, others_row], ignore_index=True)

        fig_pie = px.pie(pie_df, values='market_cap_usd', names='symbol', title='Market Cap Distribution', hole=0.4)
        fig_pie.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    # --- Data Table Section ---
    st.subheader(" Cryptocurrency Screener")
    st.dataframe(latest_df.style.format({
        'price_usd': '${:,.4f}',
        'percent_change_24h': '{:+.2f}%',
        'market_cap_usd': '${:,.0f}',
        'volume_24h_usd': '${:,.0f}'
    }), use_container_width=True)