# src/pages/advanced_analytics.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.db.get_connection import get_db_connection
from src.db.query_utils import execute_query

# --- Data Loading Functions ---

@st.cache_data(ttl=600)
def load_all_symbols():
    """Loads unique crypto symbols."""
    print("--- Fetching symbol list ---")
    query = "SELECT DISTINCT symbol FROM crypto_prices ORDER BY symbol;"
    conn = get_db_connection()
    df = execute_query(conn, query)
    return df['symbol'].tolist()

@st.cache_data(ttl=600)
def load_historical_data(symbol):
    """Fetches full price history for the selected cryptocurrency."""
    print(f"--- Fetching historical data for {symbol} ---")
    query = "SELECT timestamp, price_usd, volume_24h_usd FROM crypto_prices WHERE symbol = %s ORDER BY timestamp ASC;"
    conn = get_db_connection()
    df = execute_query(conn, query, params=(symbol,))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.set_index('timestamp')

# --- Calculation Helper Function (Using Shorter Windows) ---

def calculate_indicators(df):
    """Calculates technical indicators based on daily OHLC data using shorter windows."""
    if df.empty:
        return pd.DataFrame()

    ohlc_df = df['price_usd'].resample('D').ohlc()
    close_price = ohlc_df['close']

    # --- Use shorter windows ---
    ohlc_df['SMA_7'] = close_price.rolling(window=7).mean()   # Changed from 20
    ohlc_df['SMA_14'] = close_price.rolling(window=14).mean() # Changed from 50

    # --- Base Bollinger Bands on SMA 14 ---
    ohlc_df['BB_std'] = close_price.rolling(window=14).std() # Changed window to 14
    ohlc_df['BB_MA'] = ohlc_df['SMA_14']                     # Use SMA 14 as the middle band
    ohlc_df['BB_upper'] = ohlc_df['BB_MA'] + (ohlc_df['BB_std'] * 2)
    ohlc_df['BB_lower'] = ohlc_df['BB_MA'] - (ohlc_df['BB_std'] * 2)

    # RSI (already uses 14)
    delta = close_price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9) # Avoid division by zero
    ohlc_df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (already uses 12/26)
    ema_12 = close_price.ewm(span=12, adjust=False).mean()
    ema_26 = close_price.ewm(span=26, adjust=False).mean()
    ohlc_df['MACD'] = ema_12 - ema_26
    ohlc_df['MACD_signal'] = ohlc_df['MACD'].ewm(span=9, adjust=False).mean()
    ohlc_df['MACD_hist'] = ohlc_df['MACD'] - ohlc_df['MACD_signal']
    
    ohlc_df['volume'] = df['volume_24h_usd'].resample('D').sum()
    
    # Drop rows with NaN - will now only remove the first ~14 days
    return ohlc_df.dropna() 

# --- Plotting Helper Function (Updated for Shorter SMAs) ---

def plot_charts(df, symbol):
    """Creates the main technical analysis chart with shorter-term indicators."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )

    # Plot 1: Candlestick, Bollinger Bands, SMAs, Volume
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), secondary_y=False, row=1, col=1)
    # --- Plot SMA 7 and SMA 14 ---
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_7'], name='SMA 7', line=dict(color='orange', width=1.5)), secondary_y=False, row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_14'], name='SMA 14', line=dict(color='purple', width=1.5)), secondary_y=False, row=1, col=1)
    # --- Plot Bollinger Bands based on SMA 14 ---
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='Upper Band', line=dict(color='gray', width=1, dash='dash')), secondary_y=False, row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='Lower Band', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), secondary_y=False, row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_MA'], name='BB Middle (SMA 14)', line=dict(color='gray', width=1, dash='dot')), secondary_y=False, row=1, col=1) 
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

# --- Main Page Function (Updated Data Check and Explainers) ---

def show():
    """Renders the Advanced Analytics page."""
    st.title("🔬 Advanced Analytics & Technical Indicators")
    st.info("This page provides a deep-dive technical analysis using short-term indicators (7/14 day SMAs).") # Updated info

    symbols = load_all_symbols()
    if not symbols:
        st.warning("No symbols found...")
        return
    
    selected_symbol = st.selectbox("Select a Cryptocurrency to Analyze", symbols)

    data_df = load_historical_data(selected_symbol)
    
    # --- FIX: Check for at least 14 data points now ---
    if data_df.empty or len(data_df) < 14: 
        st.warning(f"Not enough historical data for {selected_symbol} to generate short-term indicators (need ~14 days). Please wait for more data.")
        return

    # Calculate indicators
    data_with_indicators = calculate_indicators(data_df.copy()) 

    if data_with_indicators.empty:
        st.error(
            "Calculation resulted in empty data after `dropna()`. "
            "This likely means there aren't enough *consecutive daily data points* "
            f"(need at least 14 for SMA_14/RSI) even with {data_df.shape[0]} total rows. "
            "Check the data quality or let the collector run longer."
        )
        return

    # --- Plotting Section ---
    st.markdown("---")
    st.header("Technical Analysis Chart")
    plot_charts(data_with_indicators, selected_symbol)

    # --- Explanations Section (Updated for Shorter SMAs) ---
    st.markdown("---")
    st.header("How to Read These Indicators")
    with st.expander("SMA (Simple Moving Average)"):
        # --- FIX: Updated text ---
        st.write("The average price over a short period (7 or 14 days). Used to identify short-term trends.")
    with st.expander("Bollinger Bands"):
        # --- FIX: Updated text ---
        st.write("Measures market volatility based on the 14-day SMA. Bands widen during high volatility.")
    with st.expander("RSI (Relative Strength Index)"):
        st.write("A momentum indicator (0-100). Above 70 suggests overbought, below 30 suggests oversold.")
    with st.expander("MACD (Moving Average Convergence Divergence)"):
        st.write("A trend-following momentum indicator. Crossovers between the MACD and Signal lines suggest potential buy/sell signals.")