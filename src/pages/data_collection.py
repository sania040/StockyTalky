import streamlit as st
import pandas as pd
from src.api.crypto_fetcher import CryptoDataFetcher
from src.db.get_connection import get_db_connection
from src.db.query_utils import execute_query

def show():
    """Display data collection page"""
    st.title("📊 Data Collection")
    
    # Get available symbols
    symbols_df = execute_query("SELECT DISTINCT symbol FROM crypto_prices")
    symbols = symbols_df['symbol'].tolist() if not symbols_df.empty else []
    
    # Allow adding new symbols
    with st.expander("➕ Add New Cryptocurrency"):
        new_symbol = st.text_input("Enter symbol (e.g., BTC, ETH):").upper()
        if st.button("Add Symbol") and new_symbol:
            if new_symbol not in symbols:
                symbols.append(new_symbol)
                st.success(f"✅ {new_symbol} added to list")
            else:
                st.warning(f"⚠️ {new_symbol} already exists")
    
    if not symbols:
        st.info("No cryptocurrencies found. Add some symbols above to get started.")
        return
    
    # Symbol selection
    selected_symbol = st.selectbox("Select Cryptocurrency:", symbols)
    
    # Fetch and store buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"🔄 Fetch {selected_symbol} Data"):
            fetch_crypto_data(selected_symbol)
    
    with col2:
        if st.session_state.get("fetched_symbol") == selected_symbol:
            if st.button(f"💾 Save to Database"):
                save_crypto_data(selected_symbol)
    
    # Display recent data
    display_recent_data()

def fetch_crypto_data(symbol: str):
    """Fetch data for a cryptocurrency"""
    with st.spinner(f"Fetching {symbol} data..."):
        try:
            fetcher = CryptoDataFetcher()
            data = fetcher.fetch_data_for_symbol(symbol)
            
            # Store in session state
            st.session_state["fetched_data"] = data
            st.session_state["fetched_symbol"] = symbol
            
            # Display fetched data
            crypto_data = data["data"][symbol]
            quote = crypto_data["quote"]["USD"]
            
            st.success(f"✅ {symbol} data fetched successfully")
            
            # Display data card
            st.markdown(f"""
            <div class="card">
                <h3>{symbol} - {crypto_data['name']}</h3>
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <p class="metric-label">Price</p>
                        <p class="metric-value">${quote['price']:.2f}</p>
                    </div>
                    <div>
                        <p class="metric-label">24h Change</p>
                        <p class="metric-value {'positive' if quote['percent_change_24h'] > 0 else 'negative'}">
                            {'+' if quote['percent_change_24h'] > 0 else ''}{quote['percent_change_24h']:.2f}%
                        </p>
                    </div>
                    <div>
                        <p class="metric-label">Market Cap</p>
                        <p class="metric-value">${quote['market_cap']/1e9:.2f}B</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Fetch failed: {str(e)}")

def save_crypto_data(symbol: str):
    """Save fetched data to database"""
    with st.spinner(f"Saving {symbol} data..."):
        try:
            fetcher = CryptoDataFetcher()
            fetched_data = st.session_state.get("fetched_data")
            
            if not fetched_data:
                st.error("No data to save. Please fetch data first.")
                return
            
            conn = get_db_connection()
            fetcher.store_data(conn, fetched_data, symbol)
            conn.close()
            
            st.success(f"✅ {symbol} data saved to database")
            
            # Clear session state
            if "fetched_data" in st.session_state:
                del st.session_state["fetched_data"]
            if "fetched_symbol" in st.session_state:
                del st.session_state["fetched_symbol"]
                
        except Exception as e:
            st.error(f"❌ Save failed: {str(e)}")

def display_recent_data():
    """Display recently collected data"""
    with st.expander("📋 Recent Data (Last 10 entries)"):
        try:
            df = execute_query("""
                SELECT symbol, price_usd, percent_change_24h, 
                       market_cap_usd, volume_24h_usd, timestamp
                FROM crypto_prices
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            
            if df.empty:
                st.info("No data available yet.")
            else:
                st.dataframe(df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading recent data: {str(e)}")