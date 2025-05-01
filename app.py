import streamlit as st
import pandas as pd
from db.crypto_db import createTable
from db.get_connection import get_db_connection
from fetch_and_store_crypto import fetch, store
from eda.crypto_eda import CryptoEDA

# Setup page
st.set_page_config(page_title="StockyTalky", page_icon="📊", layout="wide")

# CSS for styling
def load_css():
    st.markdown("""
    <style>
    .card {
        border-radius: 5px;
        padding: 1rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {color: #333; font-size: 1.5rem; font-weight: bold;}
    .metric-label {color: #666; font-size: 0.9rem;}
    .positive {color: #4CAF50;}
    .negative {color: #F44336;}
    .buy {background-color: rgba(76, 175, 80, 0.1); border-left: 4px solid #4CAF50; padding-left: 1rem;}
    .sell {background-color: rgba(244, 67, 54, 0.1); border-left: 4px solid #F44336; padding-left: 1rem;}
    .hold {background-color: rgba(117, 117, 117, 0.1); border-left: 4px solid #757575; padding-left: 1rem;}
    </style>
    """, unsafe_allow_html=True)

# Data collection page
def show_data_collection():
    st.title("Data Collection")
    
    symbols = ["BTC", "ETH", "SOL", "XRP", "DOGE", "LTC"]
    sel = st.selectbox("Select Symbol:", symbols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"🔄 Fetch {sel} Data", key="fetch_btn"):
            with st.spinner("Fetching..."):
                try:
                    fetched = fetch(sel)
                    st.session_state["fetched_data"] = fetched
                    st.session_state["fetched_symbol"] = sel
                    
                    crypto_data = fetched["data"][sel]
                    price = crypto_data["quote"]["USD"]["price"]
                    change_24h = crypto_data["quote"]["USD"]["percent_change_24h"]
                    market_cap = crypto_data["quote"]["USD"]["market_cap"]
                    
                    st.success(f"✅ {sel} data fetched")
                    
                    st.markdown(f"""
                    <div class="card">
                        <h3>{sel} - {crypto_data['name']}</h3>
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <p class="metric-label">Price</p>
                                <p class="metric-value">${price:.2f}</p>
                            </div>
                            <div>
                                <p class="metric-label">24h Change</p>
                                <p class="metric-value {'positive' if change_24h > 0 else 'negative'}">
                                    {'+' if change_24h > 0 else ''}{change_24h:.2f}%
                                </p>
                            </div>
                            <div>
                                <p class="metric-label">Market Cap</p>
                                <p class="metric-value">${market_cap/1e9:.2f}B</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Fetch failed: {e}")
    
    with col2:
        if st.session_state.get("fetched_symbol") == sel:
            if st.button(f"💾 Save to DB", key="store_btn"):
                with st.spinner("Saving..."):
                    try:
                        fetched = st.session_state["fetched_data"]
                        status = store(api_data=fetched, symbol=sel)
                        if status == "stored":
                            st.success(f"✅ Saved to database")
                            del st.session_state["fetched_data"]
                            del st.session_state["fetched_symbol"]
                        else:
                            st.error("❌ Save failed")
                    except Exception as e:
                        st.error(f"❌ Save failed: {e}")
    
    with st.expander("📋 Recent Data"):
        try:
            conn = get_db_connection()
            df = pd.read_sql("SELECT * FROM crypto_prices ORDER BY timestamp DESC LIMIT 10;", conn)
            conn.close()
            if df.empty:
                st.info("No data yet")
            else:
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Load failed: {e}")

# Dashboard page
def show_dashboard():
    eda = st.session_state['eda']
    available_symbols = eda.get_available_symbols()
    
    if not available_symbols:
        st.title("Welcome to StockyTalky")
        st.info("Go to Data Collection page to start")
        return
    
    recommendations = eda.get_investment_recommendations()
    market_data = eda.get_all_symbols_data()
    
    st.title("Investment Dashboard")
    
    # Market overview section
    if not market_data.empty:
        st.subheader("Market Overview")
        total_cap = market_data['market_cap_usd'].sum() / 1e9
        avg_change = market_data['percent_change_24h'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="card">
                <p class="metric-label">Total Market Cap</p>
                <p class="metric-value">${total_cap:.2f}B</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="card">
                <p class="metric-label">Avg 24h Change</p>
                <p class="metric-value {'positive' if avg_change > 0 else 'negative'}">
                    {'+' if avg_change > 0 else ''}{avg_change:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="card">
                <p class="metric-label">Assets</p>
                <p class="metric-value">{len(market_data)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations section
    st.subheader("Investment Recommendations")
    
    if not recommendations.empty:
        for _, row in recommendations.iterrows():
            symbol = row['symbol']
            price = row['price_usd']
            change = row['change_24h']
            rec = row['recommendation']
            reason = row['reason']
            
            css_class = "buy" if "Buy" in rec else "sell" if "Sell" in rec else "hold"
            
            st.markdown(f"""
            <div class="card {css_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3>{symbol}</h3>
                        <p><strong>{rec}</strong> - {reason}</p>
                    </div>
                    <div style="text-align: right;">
                        <p class="metric-value">${price:.2f}</p>
                        <p class="{'positive' if change > 0 else 'negative'}">
                            {'+' if change > 0 else ''}{change:.2f}%
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Add more data for recommendations")
    
    # Price chart section
    st.subheader("Price Charts")
    selected = st.selectbox("Select asset:", available_symbols)
    
    if selected:
        price_fig, tech_fig = eda.create_dashboard_charts(selected)
        
        if price_fig:
            st.plotly_chart(price_fig, use_container_width=True)
            
            with st.expander("Technical Analysis"):
                st.plotly_chart(tech_fig, use_container_width=True)
        else:
            st.info(f"Not enough data for {selected}")

# Main app
def main():
    # Load CSS
    load_css()
    
    # Initialize database
    try:
        createTable()
    except Exception as e:
        st.error(f"DB setup failed: {e}")
        st.stop()
    
    # Initialize EDA
    if 'eda' not in st.session_state:
        st.session_state['eda'] = CryptoEDA()
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Investment Dashboard", "Data Collection"])
    
    # Show selected page
    if page == "Data Collection":
        show_data_collection()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()