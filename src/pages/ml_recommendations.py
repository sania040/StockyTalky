# src/pages/ml_recommendations.py

import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from src.db.get_connection import get_db_connection
from src.db.query_utils import execute_query

@st.cache_data(ttl=600)
def load_all_data():
    print("--- Fetching all historical data for ML page ---")
    query = "SELECT timestamp, symbol, price_usd FROM crypto_prices ORDER BY timestamp ASC;"
    conn = get_db_connection()
    df = execute_query(conn, query)
    if df.empty:
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def show():
    st.title(" Market Analysis & ML Insights")
    all_data = load_all_data()

    if all_data.empty or len(all_data['symbol'].unique()) < 2:
        st.warning("Not enough diverse data to generate analysis. Please wait...")
        return

    # --- Feature 1: Multi-Coin Performance Comparison ---
    st.header(" Multi-Coin Performance Comparison")
    st.info("Shows percentage growth from a common starting point.")

    all_symbols = sorted(all_data['symbol'].unique())
    default_symbols = all_symbols[:min(5, len(all_symbols))]
    selected_symbols = st.multiselect("Select cryptocurrencies to compare:", options=all_symbols, default=default_symbols)

    if selected_symbols:
        comparison_df = all_data[all_data['symbol'].isin(selected_symbols)]
        normalized_df = pd.DataFrame()
        common_start_time = comparison_df.groupby('symbol')['timestamp'].min().max()
        comparison_df = comparison_df[comparison_df['timestamp'] >= common_start_time]

        for symbol in selected_symbols:
            symbol_data = comparison_df[comparison_df['symbol'] == symbol].copy().sort_values('timestamp')
            if not symbol_data.empty:
                start_price = symbol_data['price_usd'].iloc[0]
                if start_price != 0:
                    symbol_data['normalized_price'] = symbol_data['price_usd'] / start_price * 100
                    normalized_df = pd.concat([normalized_df, symbol_data])

        if not normalized_df.empty:
            fig_performance = px.line(
                normalized_df, x='timestamp', y='normalized_price', color='symbol',
                title='Normalized Price Growth Comparison',
                labels={'normalized_price': 'Growth (Baseline 100)', 'timestamp': 'Date'}
            )
            fig_performance.update_layout(
                hovermode='x unified',
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_performance, use_container_width=True)
        else:
             st.info("Select coins or wait for more overlapping data to see comparison.")

    st.markdown("---")

    # --- Feature 2: Correlation Heatmap ---
    st.header(" Asset Correlation Heatmap")
    
    if len(all_data['timestamp'].dt.date.unique()) > 1:
        pivot_df = all_data.pivot_table(index='timestamp', columns='symbol', values='price_usd')
        returns_df = pivot_df.pct_change().dropna(how='all', axis=0) 
        if not returns_df.empty and returns_df.shape[1] > 1:
            returns_df_cleaned = returns_df.dropna(axis=1, how='all')
            if returns_df_cleaned.shape[1] > 1:
                correlation_matrix = returns_df_cleaned.corr()
                fig_heatmap = px.imshow(
                    correlation_matrix, text_auto=".2f", aspect="auto",
                    color_continuous_scale='RdYlGn', title='Cryptocurrency Price Correlation Matrix'
                )
                fig_heatmap.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                 st.info("Not enough overlapping data.")
        else:
             st.info("Not enough overlapping data.")
    else:
        st.info("Need more than one day of data.")

    st.markdown("---")

    # --- Feature 3: Price Forecasting with Prophet ---
    st.header(" Simple Price Forecasting")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        forecast_symbol = st.selectbox("Select a cryptocurrency to forecast:", options=all_symbols, key="forecast_symbol_select")
        forecast_days = st.slider("Select forecast period (days):", min_value=7, max_value=90, value=30)

    if st.button(f"Generate Forecast for {forecast_symbol}"):
        with st.spinner(f"Training model and forecasting for {forecast_symbol}..."):
            symbol_data = all_data[all_data['symbol'] == forecast_symbol][['timestamp', 'price_usd']].copy()
            prophet_df = symbol_data.rename(columns={'timestamp': 'ds', 'price_usd': 'y'})
            prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

            if len(prophet_df) < 3:
                 st.error(f"Not enough historical data ({len(prophet_df)} points).")
            else:
                model = Prophet()
                model.fit(prophet_df) 
                future = model.make_future_dataframe(periods=forecast_days) 
                forecast = model.predict(future)

                st.subheader(f"Forecast for {forecast_symbol}")
                fig_forecast = plot_plotly(model, forecast)
                
                # --- FIX: Change the color of the actual data points (dots) ---
                # The actual data points are typically the first trace (index 0) or the one with mode='markers'
                for trace in fig_forecast.data:
                    if trace.mode == 'markers': # Find the dots
                        trace.marker.color = 'rgba(255, 255, 255, 0.7)' # Set to White (0.7 opacity)
                        # OR try a bright color like Cyan: trace.marker.color = '#00FFFF'
                        trace.marker.size = 4 # Optional: make them slightly smaller/larger

                fig_forecast.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

                st.subheader(f"Forecast Components")
                fig_components = plot_components_plotly(model, forecast)
                fig_components.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_components, use_container_width=True)