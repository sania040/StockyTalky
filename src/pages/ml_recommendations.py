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

    # --- Feature 3: Price Forecasting with Prophet ---
    st.header("🔮 Simple Price Forecasting")
    st.info("Uses the Prophet model to forecast future price action.")

    col1, col2 = st.columns([1, 2])
    with col1:
        forecast_symbol = st.selectbox(
            "Select a cryptocurrency to forecast:", 
            options=all_symbols, 
            key="forecast_symbol_select"
        )
        forecast_days = st.slider("Select forecast period (days):", min_value=7, max_value=90, value=30)
        # <<< BACKTESTING: test_days definition (keep this line, used in commented check below)
        # test_days = 7 
        # >>> END BACKTESTING

    if st.button(f"Generate Forecast for {forecast_symbol}"):
        with st.spinner(f"Training model and forecasting for {forecast_symbol}..."): # Updated spinner text
            symbol_data = all_data[all_data['symbol'] == forecast_symbol][['timestamp', 'price_usd']].copy()
            prophet_df = symbol_data.rename(columns={'timestamp': 'ds', 'price_usd': 'y'})
            prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

            # --- Simplified check (runs forecast even with less data for now) ---
            min_data_points_basic = 3 # Prophet needs >= 2
            if len(prophet_df) < min_data_points_basic:
                 st.error(f"Not enough historical data ({len(prophet_df)} points) to generate a forecast (need at least {min_data_points_basic}).")
            # <<< BACKTESTING: Original check requiring more data (keep commented)
            # min_data_points_backtest = 10 + test_days 
            # if len(prophet_df) < min_data_points_backtest:
            #     st.error(f"Not enough historical data ({len(prophet_df)} points) to perform a reliable {test_days}-day backtest (need at least {min_data_points_backtest}).")
            # >>> END BACKTESTING
            else:
                # --- Train Model (using all available data) ---
                model = Prophet()
                model.fit(prophet_df) 

                # --- Make Future Predictions ---
                future = model.make_future_dataframe(periods=forecast_days) 
                forecast = model.predict(future)

                # --- Display Forecast ---
                st.subheader(f"Forecast for {forecast_symbol}")
                fig_forecast = plot_plotly(model, forecast)
                st.plotly_chart(fig_forecast, use_container_width=True)

                # <<< BACKTESTING: Accuracy Calculation Section (Keep Commented Out) ---
                # st.subheader("Backtest Accuracy Check")
                # # Check if enough data existed for the backtest split
                # min_data_points_backtest = 10 + test_days 
                # if len(prophet_df) >= min_data_points_backtest:
                #     try:
                #         # Split data for backtesting
                #         train_df = prophet_df[:-test_days]
                #         test_df = prophet_df[-test_days:]
                # 
                #         # Train a separate model ONLY on the training data
                #         model_backtest = Prophet()
                #         model_backtest.fit(train_df)
                # 
                #         # Predict ONLY the test period
                #         future_test = model_backtest.make_future_dataframe(periods=test_days) 
                #         forecast_test_all = model_backtest.predict(future_test)
                #         # Extract only the predictions corresponding to the actual test dates
                #         forecast_test = forecast_test_all[-test_days:] 
                # 
                #         # Merge actual vs predicted for the test period
                #         eval_df = pd.merge(test_df, forecast_test[['ds', 'yhat']], on='ds', how='inner')
                # 
                #         if not eval_df.empty:
                #             # Calculate MAPE
                #             mape = mean_absolute_percentage_error(eval_df['y'], eval_df['yhat']) * 100
                #             # Display Accuracy Metric
                #             st.metric(
                #                 label=f"Accuracy (MAPE on last {test_days} days)",
                #                 value=f"{mape:.2f}%",
                #                 help="Mean Absolute Percentage Error vs the last 7 days. Lower is better."
                #             )
                #         else:
                #             st.warning("Could not calculate accuracy (data mismatch during backtesting period).")
                #     except Exception as e:
                #         st.warning(f"Could not perform backtest accuracy check: {e}")
                # else:
                #     st.warning(f"Not enough data ({len(prophet_df)} points) to perform a {test_days}-day backtest (need at least {min_data_points_backtest}). Accuracy check skipped.")
                # >>> END BACKTESTING ---

                st.subheader(f"Forecast Components")
                fig_components = plot_components_plotly(model, forecast)
                st.plotly_chart(fig_components, use_container_width=True)