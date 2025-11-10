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

def calculate_indicators(df, min_days: int = 7):
    """Calculates technical indicators based on daily OHLC data using shorter windows.

    This function will reindex the series to a full daily range and forward-fill
    so that short histories (e.g. 3 days) can still produce rolling indicators.
    min_days controls the minimum number of total days required before the page
    considers indicators meaningful (used by callers for checks).
    """
    if df.empty:
        return pd.DataFrame()

    # Resample to daily OHLC and reindex to a full daily range
    ohlc_df = df['price_usd'].resample('D').ohlc()
    if ohlc_df.empty:
        return pd.DataFrame()

    full_idx = pd.date_range(start=ohlc_df.index.min(), end=ohlc_df.index.max(), freq='D')
    ohlc_df = ohlc_df.reindex(full_idx)

    # Forward-fill to create consecutive daily values where possible
    ohlc_df = ohlc_df.ffill()
    close_price = ohlc_df['close']

    # If after forward-fill we still don't have enough days, return empty
    if len(close_price.dropna()) < min_days:
        return pd.DataFrame()

    # Adaptive window sizes (don't exceed available data length)
    win_7 = min(7, max(1, len(close_price)))
    win_14 = min(14, max(1, len(close_price)))

    ohlc_df['SMA_7'] = close_price.rolling(window=win_7).mean()
    ohlc_df['SMA_14'] = close_price.rolling(window=win_14).mean()

    # Bollinger Bands based on SMA_14 (adapted)
    ohlc_df['BB_std'] = close_price.rolling(window=win_14).std()
    ohlc_df['BB_MA'] = ohlc_df['SMA_14']
    ohlc_df['BB_upper'] = ohlc_df['BB_MA'] + (ohlc_df['BB_std'] * 2)
    ohlc_df['BB_lower'] = ohlc_df['BB_MA'] - (ohlc_df['BB_std'] * 2)

    # RSI (adaptive window)
    delta = close_price.diff()
    rsi_win = min(14, max(1, len(close_price)))
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_win).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_win).mean()
    rs = gain / loss.replace(0, 1e-9)
    ohlc_df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (standard parameters)
    ema_12 = close_price.ewm(span=12, adjust=False).mean()
    ema_26 = close_price.ewm(span=26, adjust=False).mean()
    ohlc_df['MACD'] = ema_12 - ema_26
    ohlc_df['MACD_signal'] = ohlc_df['MACD'].ewm(span=9, adjust=False).mean()
    ohlc_df['MACD_hist'] = ohlc_df['MACD'] - ohlc_df['MACD_signal']

    ohlc_df['volume'] = df['volume_24h_usd'].resample('D').sum().reindex(full_idx).ffill()

    # Drop rows where core indicators are still NaN
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

    # Let the user choose how aggressive the indicators should be (minimum consecutive days)
    # Allow custom values up to a year so users can request longer minimum windows
    min_days = st.number_input(
        "Minimum consecutive days for indicators:",
        min_value=3,
        max_value=365,
        value=7,
        step=1,
        help="Choose lower value to allow indicators with shorter histories (less reliable). You can set up to 365 days."
    )

    # --- Allow analysis with min_days; indicators adapt to available history ---
    if data_df.empty or len(data_df) < min_days:
        st.warning(f"Not enough historical data for {selected_symbol} to generate short-term indicators (need ~{min_days} days). Please wait for more data.")
        return

    # Calculate indicators (pass min_days so calculation can require it)
    data_with_indicators = calculate_indicators(data_df.copy(), min_days=min_days)

    if data_with_indicators.empty:
        st.error(
            "Calculation resulted in empty data after `dropna()`. "
            "This likely means there aren't enough *consecutive daily data points* "
            f"(need at least {min_days} for basic short-term indicators) even with {data_df.shape[0]} total rows. "
            "Check the data quality or let the collector run longer."
        )
        return

    # --- Plotting Section ---
    st.markdown("---")
    st.header("Technical Analysis Chart")
    plot_charts(data_with_indicators, selected_symbol)
# --- Explanations Section (Improved) ---
    st.markdown("---")
    st.header("How to Read These Indicators")

    with st.expander("SMA (Simple Moving Average)"):
        st.write("""
            **What it is:** The average closing price over a specific period (here, 7 or 14 days). It smooths out price fluctuations to make the underlying trend clearer.
            **How to read it:**
            * **Trend Direction:** When the price is consistently **above** the SMAs, it suggests an **uptrend**. When below, it suggests a **downtrend**.
            * **Crossovers:** When a shorter-term SMA (like SMA 7) crosses **above** a longer-term SMA (like SMA 14), it can signal potential **upward momentum** (a "Golden Cross" concept). The opposite (crossing below) can signal **downward momentum** (a "Death Cross" concept).
        """)

    with st.expander("Bollinger Bands®"):
        st.write("""
            **What they are:** Bands plotted two standard deviations above and below a central moving average (here, the 14-day SMA, shown as BB Middle). They measure price **volatility** relative to the recent trend.
            **How to read them:**
            * **Volatility:** Bands **widen** when volatility increases and **narrow** ('squeeze') when volatility decreases. A squeeze often precedes a significant price move.
            * **Potential Reversals:** Prices touching the **upper band** might suggest the asset is becoming overbought short-term (potential pullback). Prices touching the **lower band** might suggest it's becoming oversold (potential bounce).
            * **Breakouts:** A strong price move closing outside the bands can signal the start of a new trend.
        """)
        st.caption("Bollinger Bands® is a registered trademark of John Bollinger.")


    with st.expander("RSI (Relative Strength Index)"):
        st.write("""
            **What it is:** A momentum oscillator measuring the **speed and change** of price movements on a scale of 0 to 100. It helps identify overbought or oversold conditions.
            **How to read it:**
            * **Overbought:** RSI reading **above 70** often suggests the asset price has risen quickly and might be due for a correction (potential sell signal).
            * **Oversold:** RSI reading **below 30** often suggests the asset price has fallen quickly and might be due for a rebound (potential buy signal).
            * **Momentum:** The direction of the RSI line indicates the momentum. Rising RSI suggests increasing bullish momentum; falling RSI suggests increasing bearish momentum.
        """)
        

    with st.expander("MACD (Moving Average Convergence Divergence)"):
        st.write("""
            **What it is:** A trend-following momentum indicator showing the relationship between two exponential moving averages (EMAs) of the price.
            **How to read it:**
            * **MACD Line (Blue):** The difference between the 12-period EMA and the 26-period EMA.
            * **Signal Line (Orange):** A 9-period EMA of the MACD line itself.
            * **Histogram (Green/Red Bars):** The difference between the MACD line and the Signal line. It visually shows the momentum.
            * **Bullish Crossover:** When the **MACD line crosses above** the Signal line, it can indicate increasing upward momentum (potential buy signal). The Histogram turns positive (green).
            * **Bearish Crossover:** When the **MACD line crosses below** the Signal line, it can indicate increasing downward momentum (potential sell signal). The Histogram turns negative (red).
        """)