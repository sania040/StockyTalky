# src/pages/ml_recommendations.py

import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from src.db.get_connection import get_db_connection
from src.db.query_utils import execute_query

# --- Data Loading Function ---

@st.cache_data(ttl=600)
def load_all_data():
    """Fetches all historical price data for analysis."""
    print("--- Fetching all historical data for ML page ---")
    query = "SELECT timestamp, symbol, price_usd FROM crypto_prices ORDER BY timestamp ASC;"
    conn = get_db_connection()
    df = execute_query(conn, query)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# --- Main Page Function ---

def show():
    """Renders the ML Recommendations & Market Analysis page."""
    st.title(" Market Analysis & ML Insights")

    all_data = load_all_data()

    if all_data.empty or len(all_data['symbol'].unique()) < 2:
        st.warning("Not enough diverse data to generate analysis. Please wait for more data to be collected.")
        return

    # --- Feature 1: Multi-Coin Performance Comparison ---
    st.header(" Multi-Coin Performance Comparison")
    st.info("This chart shows the percentage growth of different cryptocurrencies from a common starting point.")

    all_symbols = all_data['symbol'].unique()
    selected_symbols = st.multiselect(
        "Select cryptocurrencies to compare:",
        options=all_symbols,
        default=all_symbols[:5].tolist()
    )

    if selected_symbols:
        comparison_df = all_data[all_data['symbol'].isin(selected_symbols)]
        normalized_df = pd.DataFrame()
        for symbol in selected_symbols:
            symbol_data = comparison_df[comparison_df['symbol'] == symbol].copy()
            if not symbol_data.empty:
                symbol_data['normalized_price'] = symbol_data['price_usd'] / symbol_data['price_usd'].iloc[0] * 100
                normalized_df = pd.concat([normalized_df, symbol_data])

        fig_performance = px.line(
            normalized_df, x='timestamp', y='normalized_price', color='symbol',
            title='Normalized Price Growth Comparison',
            labels={'normalized_price': 'Growth (Baseline 100)', 'timestamp': 'Date'}
        )
        st.plotly_chart(fig_performance, use_container_width=True)

    st.markdown("---")

    # --- Feature 2: Correlation Heatmap ---
    st.header(" Asset Correlation Heatmap")
    st.info("This heatmap shows how closely different cryptocurrencies' prices move together. A high value (close to 1.0) means they move in the same direction.")

    pivot_df = all_data.pivot_table(index='timestamp', columns='symbol', values='price_usd')
    returns_df = pivot_df.pct_change().dropna()
    correlation_matrix = returns_df.corr()

    fig_heatmap = px.imshow(
        correlation_matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale='RdYlGn', title='Cryptocurrency Price Correlation Matrix'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

    # --- NEW FEATURE: Price Forecasting with Prophet ---
    st.header(" Simple Price Forecasting")
    st.info("This tool uses the Prophet time-series model to generate a statistical forecast of future price action. This is not financial advice.")

    col1, col2 = st.columns([1, 2])
    with col1:
        forecast_symbol = st.selectbox("Select a cryptocurrency to forecast:", options=all_symbols)
        forecast_days = st.slider("Select forecast period (days):", min_value=7, max_value=90, value=30)
    
    if st.button(f"Generate Forecast for {forecast_symbol}"):
        with st.spinner(f"Training model and forecasting {forecast_days} days for {forecast_symbol}..."):
            # Prepare data for Prophet
            symbol_data = all_data[all_data['symbol'] == forecast_symbol][['timestamp', 'price_usd']]
            prophet_df = symbol_data.rename(columns={'timestamp': 'ds', 'price_usd': 'y'})
            prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
            if len(prophet_df) < 30:
                st.error("Not enough historical data to generate a reliable forecast. Please wait for more data to be collected.")
            else:
                # Initialize and train the model
                model = Prophet()
                model.fit(prophet_df)

                # Make future dataframe and predict
                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)

                # Plot the forecast
                st.subheader(f"Forecast for {forecast_symbol}")
                fig_forecast = model.plot(forecast)
                st.pyplot(fig_forecast)

                # Plot forecast components
                st.subheader(f"Forecast Components")
                fig_components = model.plot_components(forecast)
                st.pyplot(fig_components)