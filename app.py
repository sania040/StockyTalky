import streamlit as st
import pandas as pd
import plotly.express as px  # Add this import
from src.db.crypto_db import createTable
from src.db.get_connection import get_db_connection
from src.fetch_and_store_crypto import fetch, store
from src.eda.crypto_eda import CryptoEDA
from src.visualizations import show_visualizations
from src.chatbot import show_chatbot

# This must be the first Streamlit command
st.set_page_config(
    page_title="StockyTalky - Crypto Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    # Your existing data collection code
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
                                    {'+' if change_24h > 0 else ''}{change_24h:.2f}%                                </p>
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
                        result = store(fetched, sel)
                        st.success(f"✅ {sel} data saved to database")
                    except Exception as e:
                        st.error(f"❌ Save failed: {e}")
    
    with st.expander("📋 Recent Data"):
        try:
            conn = get_db_connection()
            df = pd.read_sql("SELECT * FROM crypto_prices ORDER BY timestamp DESC LIMIT 10;", conn)
            conn.close()
            if df.empty:
                st.info("No data available yet. Fetch and save some data first.")
            else:
                st.dataframe(df)
        except Exception as e:
            st.error(f"Error loading recent data: {e}")

# Dashboard page
def show_dashboard():
    st.title("Investment Dashboard")
    
    # Get connection to database
    conn = get_db_connection()
    
    # Check if there's any data in the database
    check_query = "SELECT COUNT(*) as count FROM crypto_prices"
    count_df = pd.read_sql(check_query, conn)
    
    if count_df['count'].iloc[0] == 0:
        st.info("No data available. Please go to Data Collection page to fetch and store data first.")
        return
    
    # Get available symbols
    symbols_query = "SELECT DISTINCT symbol FROM crypto_prices"
    symbols_df = pd.read_sql(symbols_query, conn)
    available_symbols = symbols_df['symbol'].tolist()
    
    # User selection for portfolio
    with st.sidebar:
        st.subheader("Portfolio Settings")
        selected_symbols = st.multiselect(
            "Select cryptocurrencies for your portfolio:",
            available_symbols,
            default=available_symbols[:min(3, len(available_symbols))]
        )
        
        time_period = st.selectbox(
            "Select time period:",
            ["1 Week", "1 Month", "3 Months", "All Time"],
            index=1
        )
        
        # Calculate days for SQL query
        if time_period == "1 Week":
            days = 7
        elif time_period == "1 Month":
            days = 30
        elif time_period == "3 Months":
            days = 90
        else:
            days = 9999  # All time
    
    if not selected_symbols:
        st.warning("Please select at least one cryptocurrency for your portfolio.")
        return
        
    # Main dashboard content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Portfolio Performance")
        
        # Get historical data for selected symbols - PostgreSQL compatible
        placeholders = ', '.join([f"'{symbol}'" for symbol in selected_symbols])
        historical_query = f"""
        SELECT symbol, price_usd, timestamp
        FROM crypto_prices
        WHERE symbol IN ({placeholders})
        AND timestamp >= CURRENT_DATE - INTERVAL '{days} days'
        ORDER BY timestamp
        """
        historical_df = pd.read_sql(historical_query, conn, parse_dates=['timestamp'])
        
        # Create portfolio performance chart
        if not historical_df.empty:
            # Create pivot table for each symbol
            pivot_df = historical_df.pivot_table(
                index='timestamp', 
                columns='symbol', 
                values='price_usd'
            )
            
            # Calculate percentage change from first available data point
            perf_df = pivot_df.pct_change()
            perf_df.iloc[0] = 0  # Set first row to 0% change
            perf_df = (1 + perf_df).cumprod() - 1  # Calculate cumulative return
            
            # Create performance chart
            fig = px.line(
                perf_df,
                x=perf_df.index,
                y=perf_df.columns,
                title=f"Price Performance ({time_period})",
                labels={
                    "timestamp": "Date",
                    "value": "Return (%)"
                }
            )
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No historical data available for the selected time period ({time_period}).")
    
    with col2:
        st.subheader("Current Holdings")
        
        # Get latest prices - PostgreSQL compatible
        placeholders = ', '.join([f"'{symbol}'" for symbol in selected_symbols])
        latest_query = f"""
        WITH LatestData AS (
            SELECT symbol, MAX(timestamp) as max_ts
            FROM crypto_prices
            WHERE symbol IN ({placeholders})
            GROUP BY symbol
        )
        SELECT cp.symbol, cp.price_usd, cp.percent_change_24h
        FROM crypto_prices cp
        JOIN LatestData ld ON cp.symbol = ld.symbol AND cp.timestamp = ld.max_ts
        """
        latest_df = pd.read_sql(latest_query, conn)
        
        if not latest_df.empty:
            # Hypothetical portfolio - equal investment in each coin
            investment_per_coin = 1000  # $1000 per coin
            
            for _, row in latest_df.iterrows():
                symbol = row['symbol']
                price = row['price_usd']
                change_24h = row['percent_change_24h']
                
                # Calculate holdings
                coins = investment_per_coin / price
                current_value = coins * price
                daily_change = current_value * change_24h / 100
                
                # Create holding card
                color_class = "positive" if change_24h >= 0 else "negative"
                
                st.markdown(f"""
                <div class="card">
                    <h3>{symbol}</h3>
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <p class="metric-label">Quantity</p>
                            <p class="metric-value">{coins:.4f}</p>
                        </div>
                        <div>
                            <p class="metric-label">Value</p>
                            <p class="metric-value">${current_value:.2f}</p>
                        </div>
                        <div>
                            <p class="metric-label">24h Change</p>
                            <p class="metric-value {color_class}">
                                {'+' if change_24h >= 0 else ''}{change_24h:.2f}%                                <br>
                                ${daily_change:.2f}
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No current price data available for the selected cryptocurrencies.")
    
    # Portfolio metrics section
    st.subheader("Portfolio Metrics")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    if not latest_df.empty:
        # Calculate total portfolio value
        total_investment = investment_per_coin * len(selected_symbols)
        total_portfolio_value = sum(investment_per_coin * (1 + row['percent_change_24h']/100) for _, row in latest_df.iterrows())
        portfolio_change = ((total_portfolio_value / total_investment) - 1) * 100
        
        # Average 24h change
        avg_change = latest_df['percent_change_24h'].mean()
        
        # Best and worst performers
        best_performer = latest_df.loc[latest_df['percent_change_24h'].idxmax()]
        worst_performer = latest_df.loc[latest_df['percent_change_24h'].idxmin()]
        
        # Display metrics
        with metrics_col1:
            st.metric(
                "Total Portfolio Value", 
                f"${total_portfolio_value:.2f}", 
                f"{portfolio_change:+.2f}%"
            )
        
        with metrics_col2:
            st.metric(
                "Best Performer", 
                f"{best_performer['symbol']}", 
                f"{best_performer['percent_change_24h']:+.2f}%"
            )
            
        with metrics_col3:
            st.metric(
                "Worst Performer", 
                f"{worst_performer['symbol']}", 
                f"{worst_performer['percent_change_24h']:+.2f}%"
            )
    
    # Correlation analysis
    st.subheader("Price Correlation")
    
    if not historical_df.empty and len(selected_symbols) > 1:
        # Create pivot table for price data
        pivot_df = historical_df.pivot_table(
            index='timestamp', 
            columns='symbol', 
            values='price_usd'
        )
        
        # Calculate correlation matrix
        corr_matrix = pivot_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Price Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
    elif len(selected_symbols) <= 1:
        st.info("Select at least two cryptocurrencies to see correlation analysis.")
    
    # Close database connection
    conn.close()

# Main app
def main():
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state['page'] = "Data Collection"
    
    # Sidebar navigation
    st.sidebar.title("StockyTalky")
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/investment.png", width=100)
    
    selected = st.sidebar.radio(
        "Navigation", 
        ["Data Collection", "Investment Dashboard", "ML Recommendations", "Advanced Analytics", "Crypto Assistant"]
    )
    st.session_state['page'] = selected
    
    # Load CSS
    load_css()
    
    # Display selected page
    if st.session_state['page'] == "Data Collection":
        show_data_collection()
    elif st.session_state['page'] == "Investment Dashboard":
        show_dashboard()
    elif st.session_state['page'] == "ML Recommendations":
        show_visualizations()
    elif st.session_state['page'] == "Advanced Analytics":
        show_advanced_analytics()
    elif st.session_state['page'] == "Crypto Assistant":
        show_chatbot()

def show_advanced_analytics():
    """Displays the advanced analytics page"""
    from src.db.advanceVisualization import show_advanced_visualization
    show_advanced_visualization()

if __name__ == "__main__":
    main()