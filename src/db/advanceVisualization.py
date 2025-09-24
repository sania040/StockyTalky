import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

def show_advanced_visualization():
    """
    Function to be called from main app.py to display the advanced visualization dashboard
    """
    # Skip page config when called from app.py to avoid conflicts
    if __name__ != "__main__":
        # Apply custom CSS for better styling
        st.markdown("""
        <style>
            .main { background-color: #f5f7f9; }
            .stApp { max-width: 1200px; margin: 0 auto; }
            .stTabs [data-baseweb="tab-list"] { gap: 24px; }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: #ffffff;
                border-radius: 4px 4px 0px 0px;
                gap: 1px;
                pAddeding-top: 10px;
                pAddeding-bottom: 10px;
            }
            .stTabs [aria-selected="true"] {
                background-color: #4CAF50;
                color: black;
            }
            div.block-container { pAddeding-top: 2rem; }
            h1, h2, h3 { color: #1E3A8A; }
            .metric-card {
                background-color: white;
                color:black;
                border-radius: 10px;
                pAddeding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .crypto-header {
                background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
                pAddeding: 20px;
                border-radius: 10px;
                color: black;
                margin-bottom: 20px;
                text-align: center;
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Only set page config when running this file directly
        st.set_page_config(
            page_title="Crypto Investment Analysis",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.markdown("""
        <style>
            .main { background-color: #f5f7f9; }
            .stApp { max-width: 1200px; margin: 0 auto; }
            .stTabs [data-baseweb="tab-list"] { gap: 24px; }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: #ffffff;
                border-radius: 4px 4px 0px 0px;
                gap: 1px;
                pAddeding-top: 10px;
                pAddeding-bottom: 10px;
                color: black;
            }
            .stTabs [aria-selected="true"] {
                background-color: #4CAF50;
                color: white;
            }
            div.block-container { pAddeding-top: 2rem; }
            h1, h2, h3 { color: #1E3A8A; }
            .metric-card {
                background-color: white;
                color:black;
                border-radius: 10px;
                pAddeding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .crypto-header {
                background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
                pAddeding: 20px;
                border-radius: 10px;
                color: #FFFFFF;
                margin-bottom: 20px;
                text-align: center;
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Function to load data from database
    @st.cache_data(ttl=3600)  # Cache data for 1 hour
    def load_data():
        from src.db.query_utils import execute_query
        
        try:
            # Get data from database
            df = execute_query("""
                SELECT cp.id, cp.symbol, cp.price_usd, cp.market_cap_usd, 
                    cp.volume_24h_usd as volume24_hours, 
                    cp.percent_change_24h as percentage24hours, 
                    cp.timestamp
                FROM crypto_prices cp
                ORDER BY cp.timestamp DESC
                LIMIT 5000
            """)
            
            if df.empty:
                st.warning("No data found in database. Using sample data for demonstration.")
                return create_sample_data()
                
            # Calculating Addeditional metrics
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(['symbol', 'timestamp'])
            
            # Added moving averages
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                df.loc[df['symbol'] == symbol, 'MA7'] = symbol_data['price_usd'].rolling(window=7).mean()
                df.loc[df['symbol'] == symbol, 'MA30'] = symbol_data['price_usd'].rolling(window=30).mean()
            
            return df
        except Exception as e:
            st.error(f"Error loading data from database: {str(e)}")
            return create_sample_data()
    
    # Function to create sample data (fallback if database fails)
    def create_sample_data():
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        np.random.seed(42)
        data = {
            'id': list(range(1, len(dates) + 1)),
            'symbol': ['BTC'] * len(dates),
            'price_usd': np.random.normal(30000, 5000, len(dates)).cumsum(),
            'market_cap_usd': np.random.normal(500000000000, 50000000000, len(dates)).cumsum(),
            'volume24_hours': np.abs(np.random.normal(20000000000, 5000000000, len(dates))),
            'percentage24hours': np.random.normal(0, 3, len(dates)),
            'timestamp': dates
        }
        
        # Create data for different coins
        coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
        all_data = []
        
        for coin in coins:
            coin_data = data.copy()
            if coin == 'ETH':
                coin_data['price_usd'] = np.random.normal(2000, 500, len(dates)).cumsum()
                coin_data['market_cap_usd'] = np.random.normal(200000000000, 20000000000, len(dates)).cumsum()
            elif coin == 'SOL':
                coin_data['price_usd'] = np.random.normal(100, 30, len(dates)).cumsum()
                coin_data['market_cap_usd'] = np.random.normal(20000000000, 5000000000, len(dates)).cumsum()
            elif coin == 'BNB':
                coin_data['price_usd'] = np.random.normal(300, 50, len(dates)).cumsum()
                coin_data['market_cap_usd'] = np.random.normal(50000000000, 10000000000, len(dates)).cumsum()
            elif coin == 'XRP':
                coin_data['price_usd'] = np.random.normal(0.5, 0.1, len(dates)).cumsum()
                coin_data['market_cap_usd'] = np.random.normal(20000000000, 5000000000, len(dates)).cumsum()
            
            coin_data['symbol'] = [coin] * len(dates)
            all_data.append(pd.DataFrame(coin_data))
        
        df = pd.concat(all_data)
        df['price_usd'] = df['price_usd'].abs() + 100  # Ensure prices are positive
        df['market_cap_usd'] = df['market_cap_usd'].abs() + 1000000000  # Ensure market caps are positive
        
        # Calculating Addeditional metrics
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['symbol', 'timestamp'])
        
        # Added moving averages
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            df.loc[df['symbol'] == symbol, 'MA7'] = symbol_data['price_usd'].rolling(window=7).mean()
            df.loc[df['symbol'] == symbol, 'MA30'] = symbol_data['price_usd'].rolling(window=30).mean()
        
        return df
    
    # Load the data
    df = load_data()
    
    # Define time periods for filtering
    time_periods = {
        "Last Week": timedelta(days=7),
        "Last Month": timedelta(days=30),
        "Last 3 Months": timedelta(days=90),
        "Last 6 Months": timedelta(days=180),
        "Last Year": timedelta(days=365),
        "All Time": timedelta(days=3650)
    }
    
    # Header section
    st.markdown("<div class='crypto-header'><h1 style='color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>🚀 Crypto Investment Analysis Dashboard</h1></div>", unsafe_allow_html=True)
    
    # Sidebar for filters
    st.sidebar.header("Filters")
    
    # Select cryptocurrency
    selected_coin = st.sidebar.selectbox("Select Cryptocurrency", df['symbol'].unique())
    
    # Time period selection
    selected_period = st.sidebar.radio("Select Time Period", list(time_periods.keys()))
    end_date = df['timestamp'].max()
    start_date = end_date - time_periods[selected_period]
    
    # Advanced filters
    st.sidebar.subheader("Advanced Filters")
    show_moving_averages = st.sidebar.checkbox("Show Moving Averages", value=True)
    timeframe = st.sidebar.selectbox("Timeframe", ["1D", "12H", "4H", "1H"], index=0)
    chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=0)
    use_log = st.sidebar.checkbox("Log scale (price)", value=False)
    show_signals = st.sidebar.checkbox("Show MA Cross Signals", value=True)
    
    # Comparison settings
    st.sidebar.subheader("Comparison")
    compare_enabled = st.sidebar.checkbox("Compare with other coins", value=False)
    if compare_enabled:
        comparison_coins = st.sidebar.multiselect(
            "Select coins to compare", 
            [coin for coin in df['symbol'].unique() if coin != selected_coin],
            default=[coin for coin in df['symbol'].unique() if coin != selected_coin][0:1]
        )
    
    # Filter the data based on selections
    filtered_df = df[(df['symbol'] == selected_coin) & (df['timestamp'] >= start_date)]
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💰 Price Analysis", "📈 Market Performance", "🔍 Technical Analysis"])
    
    with tab1:
        st.header(f"{selected_coin} Overview")
        
        # Get the most recent data for the selected coin
        if filtered_df.empty:
            st.info(f"No data available for {selected_coin}.")
            return
        # ensure chronological order then pick the latest row
        filtered_df = filtered_df.sort_values('timestamp')
        latest_data = filtered_df.iloc[-1]
        
        # Calculating metrics
        price_change = filtered_df['percentage24hours'].iloc[-1]
        price_color = "green" if price_change >= 0 else "red"
        price_icon = "↗" if price_change >= 0 else "↘"
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(
                label="Current Price (USD)",
                value=f"${latest_data['price_usd']:,.2f}",
                delta=f"{price_change:.2f}% (24h)"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(
                label="Market Cap (USD)",
                value=f"${latest_data['market_cap_usd']/1_000_000_000:.2f}B"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(
                label="24h Trading Volume",
                value=f"${latest_data['volume24_hours']/1_000_000_000:.2f}B"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.subheader("Price vs Volume")
        
        # Create a figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Added price line
        fig.add_trace(
            go.Scatter(
                x=filtered_df['timestamp'],
                y=filtered_df['price_usd'],
                name="Price (USD)",
                line=dict(color="#1E3A8A", width=2)
            ),
            secondary_y=False
        )
        
        # Added volume bars
        fig.add_trace(
            go.Bar(
                x=filtered_df['timestamp'],
                y=filtered_df['volume24_hours'],
                name="Volume (USD)",
                marker_color="rgba(75, 192, 192, 0.5)"
            ),
            secondary_y=True
        )
        
        # Added titles and layout
        fig.update_layout(
            title_text=f"{selected_coin} Price and Volume Trends",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Volume (USD)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Added market dominance chart if comparing
        if compare_enabled and comparison_coins:
            st.subheader("Market Comparison")
            compare_coins = [selected_coin] + comparison_coins
            market_cap_df = (
                df[df['symbol'].isin(compare_coins)]
                .sort_values(['symbol', 'timestamp'])
                .groupby('symbol', as_index=False)
                .last()
            )
            
            fig = px.pie(
                market_cap_df,
                values='market_cap_usd',
                names='symbol',
                title='Market Cap Distribution',
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Price Analysis")
        
        # Build a trading-style chart
        work = filtered_df.sort_values("timestamp").copy()
        work.set_index("timestamp", inplace=True)
        rule_map = {"1D": "1D", "12H": "12H", "4H": "4H", "1H": "1H"}
        rule = rule_map.get(timeframe, "1D")
        ohlc = work["price_usd"].resample(rule).ohlc().dropna()
        vol = work["volume24_hours"].resample(rule).mean().reindex(ohlc.index).fillna(method="ffill")
        close = ohlc["close"]
        
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rsi = 100 - (100 / (1 + (gain.rolling(14).mean() / loss.rolling(14).mean())))
        
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        if chart_type == "Candlestick":
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                row_heights=[0.6, 0.2, 0.2], specs=[[{"secondary_y": True}], [{"type": "xy"}], [{"type": "xy"}]]
            )
            # Build custom hover text for compatibility
            hover_text = [
                f"<b>{ts:%Y-%m-%d %H:%M}</b><br>O {o:.2f}<br>H {h:.2f}<br>L {l:.2f}<br>C {c:.2f}"
                for ts, o, h, l, c in zip(ohlc.index, ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'])
            ]
            fig.add_trace(go.Candlestick(
                x=ohlc.index, open=ohlc["open"], high=ohlc["high"], low=ohlc["low"], close=ohlc["close"],
                name="OHLC", increasing_line_color="#16a34a", decreasing_line_color="#ef4444",
                text=hover_text, hoverinfo="text"
             ), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Bar(
                x=ohlc.index, y=vol, name="Volume", marker_color="rgba(59,130,246,0.35)"
            ), row=1, col=1, secondary_y=True)
            if show_moving_averages:
                fig.add_trace(go.Scatter(x=ohlc.index, y=ema20, name="EMA 20", line=dict(color="#10B981", width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=ohlc.index, y=ema50, name="EMA 50", line=dict(color="#F59E0B", width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=ohlc.index, y=bb_upper, name="BB Upper", line=dict(color="#EC4899", width=1, dash="dash")), row=1, col=1)
                fig.add_trace(go.Scatter(x=ohlc.index, y=bb_mid, name="BB Mid", line=dict(color="#6b7280", width=1, dash="dot")), row=1, col=1)
                fig.add_trace(go.Scatter(x=ohlc.index, y=bb_lower, name="BB Lower", line=dict(color="#EC4899", width=1, dash="dash"), fill="tonexty", fillcolor="rgba(236,72,153,0.08)"), row=1, col=1)
            if show_signals:
                cross_up = (ema20.shift(1) < ema50.shift(1)) & (ema20 >= ema50)
                cross_dn = (ema20.shift(1) > ema50.shift(1)) & (ema20 <= ema50)
                fig.add_trace(go.Scatter(
                    x=ohlc.index[cross_up], y=close[cross_up], mode="markers", name="Buy",
                    marker=dict(symbol="triangle-up", size=10, color="#16a34a")
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=ohlc.index[cross_dn], y=close[cross_dn], mode="markers", name="Sell",
                    marker=dict(symbol="triangle-down", size=10, color="#ef4444")
                ), row=1, col=1)
            fig.add_trace(go.Scatter(x=ohlc.index, y=rsi, name="RSI", line=dict(color="#FF6B6B", width=2)), row=2, col=1)
            # Horizontal RSI guides (compatible with older Plotly)
            fig.add_trace(go.Scatter(x=ohlc.index, y=pd.Series(70, index=ohlc.index), showlegend=False,
                                     line=dict(color="red", dash="dash")), row=2, col=1)
            fig.add_trace(go.Scatter(x=ohlc.index, y=pd.Series(30, index=ohlc.index), showlegend=False,
                                     line=dict(color="green", dash="dash")), row=2, col=1)
            fig.add_trace(go.Scatter(x=ohlc.index, y=macd, name="MACD", line=dict(color="#4CAF50", width=2)), row=3, col=1)
            fig.add_trace(go.Scatter(x=ohlc.index, y=macd_signal, name="Signal", line=dict(color="#FF9800", width=2)), row=3, col=1)
            fig.add_trace(go.Bar(x=ohlc.index, y=macd_hist, name="Histogram", marker_color=["#16a34a" if v >= 0 else "#ef4444" for v in macd_hist]), row=3, col=1)
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1, type="log" if use_log else "linear")
            fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_layout(
                 title_text=f"{selected_coin} Trading View ({timeframe})",
                 template="plotly_white", hovermode="x unified",
                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                 height=900, margin=dict(l=10, r=10, t=50, b=10),
                 xaxis=dict(
                     rangeselector=dict(
                         buttons=list([
                             dict(count=7, label="1w", step="day", stepmode="backward"),
                             dict(count=1, label="1m", step="month", stepmode="backward"),
                             dict(count=3, label="3m", step="month", stepmode="backward"),
                             dict(step="all")
                         ])
                     ),
                     rangeslider=dict(visible=True)
                )
             )
            st.plotly_chart(fig, use_container_width=True, config={
                "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
                "displaylogo": False
            })
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_df['timestamp'], y=filtered_df['price_usd'],
                name="Price (USD)", line=dict(color="#3B82F6", width=3)
            ))
            if show_moving_averages:
                fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['MA7'], name="7-Day MA", line=dict(color="#10B981", width=1.5, dash='dot')))
                fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['MA30'], name="30-Day MA", line=dict(color="#F59E0B", width=1.5, dash='dot')))
            if compare_enabled and comparison_coins:
                for coin in comparison_coins:
                    compare_df = df[(df['symbol'] == coin) & (df['timestamp'] >= start_date)].sort_values("timestamp")
                    if compare_df.empty:
                        continue
                    base_sel = filtered_df['price_usd'].iloc[0] if not filtered_df.empty else 1
                    base_cmp = compare_df['price_usd'].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=compare_df['timestamp'],
                        y=compare_df['price_usd'] * (base_sel / base_cmp),
                        name=f"{coin} (Normalized)", line=dict(width=1.5), opacity=0.7
                    ))
            fig.update_layout(
                title_text=f"{selected_coin} Price Evolution",
                template="plotly_white", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=550, xaxis_title="Date", yaxis_title="Price (USD)"
            )
            if use_log:
                fig.update_yaxes(type="log")
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Market Performance")
        
        # Market Cap Trends
        st.subheader("Market Capitalization Trends")
        
        fig = go.Figure()
        
        # Added market cap line
        fig.add_trace(
            go.Scatter(
                x=filtered_df['timestamp'],
                y=filtered_df['market_cap_usd'],
                name=f"{selected_coin} Market Cap",
                line=dict(color="#7C3AED", width=3),
                fill='tozeroy',
                fillcolor='rgba(124, 58, 237, 0.1)'
            )
        )
        
        # Added comparison coins if selected
        if compare_enabled and comparison_coins:
            for coin in comparison_coins:
                compare_df = df[(df['symbol'] == coin) & (df['timestamp'] >= start_date)]
                
                fig.add_trace(
                    go.Scatter(
                        x=compare_df['timestamp'],
                        y=compare_df['market_cap_usd'],
                        name=f"{coin} Market Cap",
                        line=dict(width=2)
                    )
                )
        
        # Update layout
        fig.update_layout(
            title_text="Market Capitalization Over Time",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            xaxis_title="Date",
            yaxis_title="Market Cap (USD)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume Analysis
        st.subheader("Trading Volume Analysis")
        
        # Calculating average daily volume
        avg_volume = filtered_df['volume24_hours'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(
                label="Average Daily Volume",
                value=f"${avg_volume/1_000_000_000:.2f}B"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(
                label="Volume to Market Cap Ratio",
                value=f"{(filtered_df['volume24_hours'].iloc[-1] / filtered_df['market_cap_usd'].iloc[-1] * 100):.2f}%"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Create volume chart
        fig = px.bar(
            filtered_df,
            x='timestamp',
            y='volume24_hours',
            title=f"{selected_coin} Trading Volume Over Time",
            color_discrete_sequence=['#4CAF50']
        )
        
        # Added 7-day moving average for volume
        filtered_df['volume_ma7'] = filtered_df['volume24_hours'].rolling(window=7).mean()
        
        fig.add_trace(
            go.Scatter(
                x=filtered_df['timestamp'],
                y=filtered_df['volume_ma7'],
                name="7-Day Volume MA",
                line=dict(color="#FF4500", width=2)
            )
        )
        
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450,
            xaxis_title="Date",
            yaxis_title="Volume (USD)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Technical Analysis")
        
        # Calculating technical indicators
        df_tech = filtered_df.copy()
        
        # Calculating daily returns
        df_tech['daily_return'] = df_tech.groupby('symbol')['price_usd'].pct_change()
        df_tech['daily_return'] = df_tech['daily_return'].fillna(0)
        
        # Calculating annualized volatility
        volatility = df_tech['daily_return'].std() * (252 ** 0.5)

        # Calculating RSI (Relative Strength Index)
        delta = df_tech['price_usd'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df_tech['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculating MACD (Moving Average Convergence Divergence)
        df_tech['EMA12'] = df_tech['price_usd'].ewm(span=12, adjust=False).mean()
        df_tech['EMA26'] = df_tech['price_usd'].ewm(span=26, adjust=False).mean()
        df_tech['MACD'] = df_tech['EMA12'] - df_tech['EMA26']
        df_tech['Signal'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
        df_tech['MACD_Histogram'] = df_tech['MACD'] - df_tech['Signal']
        
        # Calculating Bollinger Bands
        df_tech['MA20'] = df_tech['price_usd'].rolling(window=20).mean()
        df_tech['std20'] = df_tech['price_usd'].rolling(window=20).std()
        df_tech['Upper_Band'] = df_tech['MA20'] + (df_tech['std20'] * 2)
        df_tech['Lower_Band'] = df_tech['MA20'] - (df_tech['std20'] * 2)
        
        # Display RSI Chart
        st.subheader("Relative Strength Index (RSI)")
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df_tech['timestamp'],
                y=df_tech['RSI'],
                name="RSI",
                line=dict(color="#FF6B6B", width=2)
            )
        )
        
        # Added horizontal lines at 30 and 70
        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", annotation_text="Overbought")
        
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            height=350,
            xaxis_title="Date",
            yaxis_title="RSI"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display MACD Chart
        st.subheader("Moving Average Convergence Divergence (MACD)")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Added MACD and Signal lines
        fig.add_trace(
            go.Scatter(
                x=df_tech['timestamp'],
                y=df_tech['MACD'],
                name="MACD",
                line=dict(color="#4CAF50", width=2)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_tech['timestamp'],
                y=df_tech['Signal'],
                name="Signal Line",
                line=dict(color="#FF9800", width=2)
            )
        )
        
        # Added histogram
        colors = ['green' if val >= 0 else 'red' for val in df_tech['MACD_Histogram']]
        
        fig.add_trace(
            go.Bar(
                x=df_tech['timestamp'],
                y=df_tech['MACD_Histogram'],
                name="Histogram",
                marker_color=colors,
                opacity=0.7
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350,
            xaxis_title="Date"
        )
        
        fig.update_yaxes(title_text="MACD / Signal", secondary_y=False)
        fig.update_yaxes(title_text="Histogram", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display Bollinger Bands
        st.subheader("Bollinger Bands")
        
        fig = go.Figure()
        
        # Added price line
        fig.add_trace(
            go.Scatter(
                x=df_tech['timestamp'],
                y=df_tech['price_usd'],
                name="Price",
                line=dict(color="#3B82F6", width=2)
            )
        )
        
        # Added Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df_tech['timestamp'],
                y=df_tech['Upper_Band'],
                name="Upper Band",
                line=dict(color="#EC4899", width=1, dash='dash')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_tech['timestamp'],
                y=df_tech['MA20'],
                name="20-Day MA",
                line=dict(color="#10B981", width=1.5)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_tech['timestamp'],
                y=df_tech['Lower_Band'],
                name="Lower Band",
                line=dict(color="#EC4899", width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(236, 72, 153, 0.1)'
            )
        )
        
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450,
            xaxis_title="Date",
            yaxis_title="Price (USD)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Addeded footnote
    st.markdown("""
    ---
    ⚠️ **Disclaimer**: This application is for informational purposes only and should not be considered financial advice. 
    Cryptocurrency investments are subject to market risks. Always do your own research before investing.
    """)
    
    # Addeded a "Buy/Sell Signal" feature
    st.sidebar.subheader("Investment Signal")
    
    # Simulate a buy/sell signal based on technical indicators
    rsi = df_tech['RSI'].iloc[-1] if not df_tech['RSI'].isna().all() else 50
    macd = df_tech['MACD'].iloc[-1] if not df_tech['MACD'].isna().all() else 0
    signal = df_tech['Signal'].iloc[-1] if not df_tech['Signal'].isna().all() else 0
    price = df_tech['price_usd'].iloc[-1]
    upper_band = df_tech['Upper_Band'].iloc[-1] if not df_tech['Upper_Band'].isna().all() else price * 1.05
    lower_band = df_tech['Lower_Band'].iloc[-1] if not df_tech['Lower_Band'].isna().all() else price * 0.95
    
    # Calculating a simple signal
    buy_signals = 0
    sell_signals = 0
    
    # RSI signals
    if rsi < 30:
        buy_signals += 1
    elif rsi > 70:
        sell_signals += 1
    
    # MACD signals
    if macd > signal:
        buy_signals += 1
    elif macd < signal:
        sell_signals += 1
    
    # Bollinger Bands signals
    if price < lower_band:
        buy_signals += 1
    elif price > upper_band:
        sell_signals += 1
    
    # Display signal
    if buy_signals > sell_signals:
        signal_strength = min(100, buy_signals * 33)
        st.sidebar.markdown(f"""
        <div style="background-color: rgba(0, 128, 0, 0.2); border-radius: 10px; pAddeding: 10px; text-align: center;">
            <h3 style="color: green;">BUY Signal</h3>
            <p>Strength: {signal_strength}%</p>
            <p>Technical indicators suggest a buying opportunity.</p>
        </div>
        """, unsafe_allow_html=True)
    elif sell_signals > buy_signals:
        signal_strength = min(100, sell_signals * 33)
        st.sidebar.markdown(f"""
        <div style="background-color: rgba(255, 0, 0, 0.2); border-radius: 10px; pAddeding: 10px; text-align: center;">
            <h3 style="color: red;">SELL Signal</h3>
            <p>Strength: {signal_strength}%</p>
            <p>Technical indicators suggest a selling opportunity.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"""
        <div style="background-color: rgba(255,165,0,0.2); border-radius: 10px; pAddeding: 10px; text-align: center;">
            <h3 style="color: orange;">HOLD Signal</h3>
            <p>Technical indicators are neutral.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Addeded simple refresh button
    if st.sidebar.button("Refresh Data"):
        try:
            # Clear cached data to force a refresh
            load_data.clear()
            st.rerun()
        except:
            st.sidebar.warning("Refresh failed. Please try again.")
    
    # Addeded last updated timestamp
    st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Make the script runnable both directly and as an imported module
if __name__ == "__main__":
    show_advanced_visualization()