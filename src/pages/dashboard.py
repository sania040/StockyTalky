import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.db.query_utils import execute_query

def show():
    """Display investment dashboard page"""
    st.title("📈 Investment Dashboard")
    
    # Fetch data
    df = execute_query("""
        SELECT symbol, price_usd, volume_24h_usd, percent_change_24h, 
               market_cap_usd, timestamp 
        FROM crypto_prices 
        ORDER BY timestamp DESC
        LIMIT 1000
    """)
    
    if df.empty:
        st.info("No data available. Please collect some data first.")
        return
    
    # Get latest data for each symbol
    latest_df = df.sort_values('timestamp').groupby('symbol').last().reset_index()
    
    # Overview metrics
    st.subheader("Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_market_cap = latest_df['market_cap_usd'].sum()
        st.metric(
            "Total Market Cap",
            f"${total_market_cap/1e9:.2f}B"
        )
    
    with col2:
        total_volume = latest_df['volume_24h_usd'].sum()
        st.metric(
            "24h Volume",
            f"${total_volume/1e9:.2f}B"
        )
    
    with col3:
        avg_change = latest_df['percent_change_24h'].mean()
        st.metric(
            "Avg 24h Change",
            f"{avg_change:+.2f}%",
            delta=f"{avg_change:.2f}%"
        )
    
    with col4:
        num_coins = len(latest_df)
        st.metric(
            "Tracked Coins",
            num_coins
        )
    
    # Top performers
    st.subheader("🏆 Top Performers (24h)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Biggest Gainers")
        top_gainers = latest_df.nlargest(5, 'percent_change_24h')[['symbol', 'price_usd', 'percent_change_24h']]
        
        for _, row in top_gainers.iterrows():
            st.markdown(f"""
            <div class="card buy">
                <strong>{row['symbol']}</strong>: ${row['price_usd']:.2f} 
                <span class="positive">(+{row['percent_change_24h']:.2f}%)</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📉 Biggest Losers")
        top_losers = latest_df.nsmallest(5, 'percent_change_24h')[['symbol', 'price_usd', 'percent_change_24h']]
        
        for _, row in top_losers.iterrows():
            st.markdown(f"""
            <div class="card sell">
                <strong>{row['symbol']}</strong>: ${row['price_usd']:.2f} 
                <span class="negative">({row['percent_change_24h']:.2f}%)</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Market cap distribution
    st.subheader("Market Cap Distribution")
    
    fig = px.pie(
        latest_df,
        values='market_cap_usd',
        names='symbol',
        title='Market Capitalization by Cryptocurrency',
        hole=0.4
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Price comparison chart
    st.subheader("Price Comparison")
    
    selected_coins = st.multiselect(
        "Select cryptocurrencies to compare:",
        latest_df['symbol'].tolist(),
        default=latest_df['symbol'].tolist()[:3]
    )
    
    if selected_coins:
        filtered_df = df[df['symbol'].isin(selected_coins)]
        
        fig = px.line(
            filtered_df,
            x='timestamp',
            y='price_usd',
            color='symbol',
            title='Price History',
            labels={'price_usd': 'Price (USD)', 'timestamp': 'Date'}
        )
        fig.update_layout(hovermode='x unified', height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("📊 All Cryptocurrencies")
    
    display_df = latest_df[['symbol', 'price_usd', 'percent_change_24h', 'market_cap_usd', 'volume_24h_usd']].copy()
    display_df.columns = ['Symbol', 'Price (USD)', '24h Change (%)', 'Market Cap (USD)', '24h Volume (USD)']
    
    st.dataframe(
        display_df.style.format({
            'Price (USD)': '${:,.2f}',
            '24h Change (%)': '{:+.2f}%',
            'Market Cap (USD)': '${:,.0f}',
            '24h Volume (USD)': '${:,.0f}'
        }),
        use_container_width=True
    )