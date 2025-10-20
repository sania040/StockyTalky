# src/pages/advanced_analytics.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.db.get_connection import get_db_connection
from src.db.query_utils import execute_query

# --- Data Loading Functions (Efficient and Focused) ---

@st.cache_data(ttl=600)
def load_all_symbols():
    """Loads only the unique crypto symbols for the selectbox."""
    print("--- Fetching symbol list ---")
    query = "SELECT DISTINCT symbol FROM crypto_prices ORDER BY symbol;"
    conn = get_db_connection()
    df = execute_query(conn, query)
    return df['symbol'].tolist()

@st.cache_data(ttl=600)
def load_historical_data(symbol):
    """Fetches the full price history for ONLY the selected cryptocurrency."""
    print(f"--- Fetching historical data for {symbol} ---")
    query = "SELECT timestamp, price_usd, volume_24h_usd FROM crypto_prices WHERE symbol = %s ORDER BY timestamp ASC;"
    conn = get_db_connection()
    df = execute_query(conn, query, params=(symbol,))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.set_index('timestamp')

# --- Calculation Helper Function ---

def calculate_indicators(df):
    """Calculates all technical indicators based on daily OHLC data."""
    if df.empty:
        return pd.DataFrame()

    # Resample to daily OHLC (Open, High, Low, Close) data. This is the correct way to build a candlestick.
    ohlc_df = df['price_usd'].resample('D').ohlc()
    close_price = ohlc_df['close']

    # Simple Moving Averages
    ohlc_df['SMA_20'] = close_price.rolling(window=20).mean()
    ohlc_df['SMA_50'] = close_price.rolling(window=50).mean()

    # Bollinger Bands
    ohlc_df['BB_std'] = close_price.rolling(window=20).std()
    ohlc_df['BB_upper'] = ohlc_df['SMA_20'] + (ohlc_df['BB_std'] * 2)
    ohlc_df['BB_lower'] = ohlc_df['SMA_20'] - (ohlc_df['BB_std'] * 2)

    # RSI
    delta = close_price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    ohlc_df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = close_price.ewm(span=12, adjust=False).mean()
    ema_26 = close_price.ewm(span=26, adjust=False).mean()
    ohlc_df['MACD'] = ema_12 - ema_26
    ohlc_df['MACD_signal'] = ohlc_df['MACD'].ewm(span=9, adjust=False).mean()
    ohlc_df['MACD_hist'] = ohlc_df['MACD'] - ohlc_df['MACD_signal']
    
    # Resample volume data and add it to the daily DataFrame
    ohlc_df['volume'] = df['volume_24h_usd'].resample('D').sum()
    
    return ohlc_df.dropna() # Drop initial rows that have NaN values for indicators

# --- Plotting Helper Function ---

def plot_charts(df, symbol):
    """Creates the main technical analysis chart with all indicators."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )

    # Plot 1: Candlestick, Bollinger Bands, SMAs, and Volume
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), secondary_y=False, row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1.5)), secondary_y=False, row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='purple', width=1.5)), secondary_y=False, row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='Upper Band', line=dict(color='gray', width=1, dash='dash')), secondary_y=False, row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='Lower Band', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), secondary_y=False, row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='rgba(0,100,255,0.3)'), secondary_y=True, row=1, col=1)

    # Plot 2: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#FF6B6B', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="gray", row=2, col=1)

    # Plot 3: MACD
    colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Histogram', marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#4CAF50', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal Line', line=dict(color='#FF9800', width=2)), row=3, col=1)

    # Update Layout
    fig.update_layout(title_text=f"Technical Analysis for {symbol}", height=800, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Volume", secondary_y=True, showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

# --- Main Page Function ---

def show():
    """Renders the Advanced Analytics page."""
    st.title("🔬 Advanced Analytics & Technical Indicators")
    st.info("This page provides a deep-dive technical analysis for a single cryptocurrency, including key momentum and trend indicators.")

    symbols = load_all_symbols()
    if not symbols:
        st.warning("No symbols found in the database. Please ensure the data collector is running.")
        return
    
    selected_symbol = st.selectbox("Select a Cryptocurrency to Analyze", symbols)

    data_df = load_historical_data(selected_symbol)
    if data_df.empty or len(data_df) < 50: # Need enough data for calculations
        st.warning(f"Not enough historical data for {selected_symbol} to generate a full analysis. Please wait for more data to be collected.")
        return

    data_with_indicators = calculate_indicators(data_df)
    plot_charts(data_with_indicators, selected_symbol)

    # --- Explanations Section ---
    st.markdown("---")
    st.header("How to Read These Indicators")
    with st.expander("SMA (Simple Moving Average)"):
        st.write("The average price over a period (20 or 50 days). Used to identify the direction of the trend.")
    with st.expander("Bollinger Bands"):
        st.write("Measures market volatility. Bands widen during high volatility and narrow during low volatility.")
    with st.expander("RSI (Relative Strength Index)"):
        st.write("A momentum indicator on a scale of 0-100. Above 70 is considered overbought, and below 30 is considered oversold.")
    with st.expander("MACD (Moving Average Convergence Divergence)"):
        st.write("A trend-following momentum indicator. Bullish and bearish signals are generated when the MACD line crosses the Signal line.")