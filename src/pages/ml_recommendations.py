# src/pages/ml_recommendations.py

import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
# <<< BACKTESTING: Keep the import, but we won't use it yet
# from sklearn.metrics import mean_absolute_percentage_error
# >>> END BACKTESTING
from src.db.get_connection import get_db_connection
from src.db.query_utils import execute_query
from src.ml.forecasting import get_available_models

# --- Data Loading Function ---

@st.cache_data(ttl=600)
def load_all_data():
    """Fetches all historical price data for analysis."""
    print("--- Fetching all historical data for ML page ---")
    query = "SELECT timestamp, symbol, price_usd FROM crypto_prices ORDER BY timestamp ASC;"
    conn = get_db_connection()
    df = execute_query(conn, query)
    if df.empty:
        return pd.DataFrame() # Return empty DataFrame if no data
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# --- Main Page Function ---

def show():
    """Renders the ML Recommendations & Market Analysis page."""
    st.title("🧠 Market Analysis & ML Insights")

    all_data = load_all_data()

    if all_data.empty or len(all_data['symbol'].unique()) < 2:
        st.warning("Not enough diverse data to generate analysis. Please wait...")
        return

    # --- Feature 1: Multi-Coin Performance Comparison ---
    st.header("📈 Multi-Coin Performance Comparison")
    st.info("Shows percentage growth from a common starting point.")

    all_symbols = sorted(all_data['symbol'].unique())
    default_symbols = all_symbols[:min(5, len(all_symbols))]
    selected_symbols = st.multiselect(
        "Select cryptocurrencies to compare:",
        options=all_symbols,
        default=default_symbols
    )

    if selected_symbols:
        comparison_df = all_data[all_data['symbol'].isin(selected_symbols)]
        normalized_df = pd.DataFrame()
        # Find the earliest timestamp present across ALL selected symbols for a fair comparison start
        common_start_time = comparison_df.groupby('symbol')['timestamp'].min().max()
        
        comparison_df = comparison_df[comparison_df['timestamp'] >= common_start_time] # Filter data from common start

        for symbol in selected_symbols:
            symbol_data = comparison_df[comparison_df['symbol'] == symbol].copy().sort_values('timestamp') # Ensure sorted
            if not symbol_data.empty:
                 # Use the price at the common start time for normalization
                start_price = symbol_data['price_usd'].iloc[0]
                if start_price != 0: # Avoid division by zero
                    symbol_data['normalized_price'] = symbol_data['price_usd'] / start_price * 100
                    normalized_df = pd.concat([normalized_df, symbol_data])

        if not normalized_df.empty:
            fig_performance = px.line(
                normalized_df, x='timestamp', y='normalized_price', color='symbol',
                title='Normalized Price Growth Comparison',
                labels={'normalized_price': 'Growth (Baseline 100)', 'timestamp': 'Date'}
            )
            st.plotly_chart(fig_performance, use_container_width=True)
        else:
             st.info("Select coins or wait for more overlapping data to see comparison.")


    st.markdown("---")

    # --- Feature 2: Correlation Heatmap ---
    st.header("🔗 Asset Correlation Heatmap")
    st.info("Shows how closely different cryptocurrencies' prices move together.")

    if len(all_data['timestamp'].dt.date.unique()) > 1:
        pivot_df = all_data.pivot_table(index='timestamp', columns='symbol', values='price_usd')
        returns_df = pivot_df.pct_change().dropna(how='all', axis=0) 
        
        if not returns_df.empty and returns_df.shape[1] > 1:
            # Drop columns with insufficient data for correlation calculation
            returns_df_cleaned = returns_df.dropna(axis=1, how='all')
            # Only calculate if we still have multiple columns
            if returns_df_cleaned.shape[1] > 1:
                correlation_matrix = returns_df_cleaned.corr()
                fig_heatmap = px.imshow(
                    correlation_matrix, text_auto=".2f", aspect="auto",
                    color_continuous_scale='RdYlGn', title='Cryptocurrency Price Correlation Matrix'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                 st.info("Not enough overlapping data yet to calculate correlations.")
        else:
             st.info("Not enough overlapping data yet to calculate correlations.")
    else:
        st.info("Need more than one day of data to calculate correlations.")


    st.markdown("---")

    # --- Feature 3: Price Forecasting with ML Models ---
    st.header("🔮 Price Forecasting")
    st.info("Uses selected ML model (Prophet or XGBoost) to forecast future price action.")

    col1, col2 = st.columns([1, 2])
    with col1:
        forecast_symbol = st.selectbox(
            "Select a cryptocurrency to forecast:",
            options=all_symbols,
            key="forecast_symbol_select"
        )
        available_models = get_available_models()
        model_options = list(available_models.keys())
        selected_model = st.selectbox(
            "Select forecasting model:",
            options=model_options,
            key="model_select"
        )
        forecast_days = st.slider("Select forecast period (days):", min_value=7, max_value=90, value=30)

    if st.button(f"Generate Forecast for {forecast_symbol} with {selected_model}"):
        if selected_model not in available_models:
            st.error(f"Model {selected_model} not available.")
        else:
            model_instance = available_models[selected_model]
            symbol_data = all_data[all_data['symbol'] == forecast_symbol][['timestamp', 'price_usd']].copy()
            symbol_data = symbol_data.sort_values('timestamp').reset_index(drop=True)

            if len(symbol_data) < 7:
                st.error(f"Not enough historical data ({len(symbol_data)} points) to generate a forecast (need at least 7 days).")
            else:
                with st.spinner(f"Training {selected_model} and forecasting for {forecast_symbol}..."):
                    if model_instance.fit(symbol_data):
                        forecast_df = model_instance.predict(forecast_days)

                        if not forecast_df.empty:
                            # Display Forecast
                            st.subheader(f"Forecast for {forecast_symbol} using {selected_model}")

                            # Create a plotly figure for forecast
                            fig = px.line()
                            # Add historical data
                            fig.add_scatter(x=symbol_data['timestamp'], y=symbol_data['price_usd'], mode='lines', name='Historical')
                            # Add forecast
                            fig.add_scatter(x=forecast_df['timestamp'], y=forecast_df['forecast'], mode='lines', name='Forecast', line=dict(dash='dash'))
                            fig.add_scatter(x=forecast_df['timestamp'], y=forecast_df['lower_bound'], mode='lines', name='Lower Bound', line=dict(color='gray', dash='dot'))
                            fig.add_scatter(x=forecast_df['timestamp'], y=forecast_df['upper_bound'], mode='lines', name='Upper Bound', line=dict(color='gray', dash='dot'))
                            fig.update_layout(title=f"{selected_model} Forecast for {forecast_symbol}", xaxis_title="Date", yaxis_title="Price (USD)")
                            st.plotly_chart(fig, use_container_width=True)

                            # For Prophet, add components plot
                            if selected_model == 'Prophet':
                                prophet_df = symbol_data.rename(columns={'timestamp': 'ds', 'price_usd': 'y'})
                                prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
                                model = Prophet()
                                model.fit(prophet_df)
                                future = model.make_future_dataframe(periods=forecast_days)
                                forecast = model.predict(future)

                                st.subheader("Forecast Components")
                                fig_components = plot_components_plotly(model, forecast)
                                st.plotly_chart(fig_components, use_container_width=True)
                        else:
                            st.error("Failed to generate forecast.")
                    else:
                        st.error(f"Failed to fit {selected_model}.")
