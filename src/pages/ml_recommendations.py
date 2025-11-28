# src/pages/ml_recommendations.py

import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_percentage_error
from src.db.get_connection import get_db_connection
from src.db.query_utils import execute_query

@st.cache_data(ttl=600)
def load_all_data():
    """Fetches all historical price data for analysis."""
    print("--- Fetching all historical data for ML page ---")
    query = "SELECT timestamp, symbol, price_usd FROM crypto_prices ORDER BY timestamp ASC;"
    conn = get_db_connection()
    df = execute_query(conn, query)
    if df.empty:
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def show():
    """Renders the ML Recommendations & Market Analysis page."""
    st.title(" Market Analysis & ML Insights")

    all_data = load_all_data()

    if all_data.empty or len(all_data['symbol'].unique()) < 2:
        st.warning("Not enough diverse data to generate analysis. Please wait for more data to be collected.")
        return

    # --- Feature 1: Multi-Coin Performance Comparison ---
    st.header(" Multi-Coin Performance Comparison")
    
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
            fig_performance.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_performance, use_container_width=True)

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
                fig_heatmap.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                 st.info("Not enough overlapping data yet.")
        else:
             st.info("Not enough overlapping data yet.")
    else:
        st.info("Need more than one day of data to calculate correlations.")

    st.markdown("---")

    # --- Feature 3: Price Forecasting & Accuracy Check ---
    st.header(" Price Forecasting & Accuracy")
    st.info("Uses Prophet to forecast prices. Includes an automated backtest to verify model accuracy against recent data.")

    col1, col2 = st.columns([1, 2])
    with col1:
        forecast_symbol = st.selectbox("Select coin:", options=all_symbols, key="forecast_symbol_select")
        forecast_days = st.slider("Forecast horizon (days):", min_value=7, max_value=90, value=30)

    if st.button(f"Generate Forecast for {forecast_symbol}"):
        with st.spinner(f"Running analysis for {forecast_symbol}..."):
            symbol_data = all_data[all_data['symbol'] == forecast_symbol][['timestamp', 'price_usd']].copy()
            prophet_df = symbol_data.rename(columns={'timestamp': 'ds', 'price_usd': 'y'})
            prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

            if len(prophet_df) < 5:
                 st.error(f"Not enough historical data ({len(prophet_df)} points) to generate any forecast.")
            else:
                # --- 1. MAIN FORECAST (Future) ---
                model_future = Prophet()
                model_future.fit(prophet_df)
                future_dates = model_future.make_future_dataframe(periods=forecast_days)
                forecast_future = model_future.predict(future_dates)

                # --- 2. SMART BACKTESTING LOGIC ---
                history_days = (prophet_df['ds'].max() - prophet_df['ds'].min()).days
                
                # Dynamically decide test size based on available history
                if history_days >= 20:
                    test_days = 7
                elif history_days >= 7:
                    test_days = 3
                elif history_days >= 3:
                    test_days = 1
                else:
                    test_days = 0

                backtest_metric = None
                backtest_msg = ""

                if test_days > 0:
                    try:
                        # Split data: Train on everything EXCEPT the last 'test_days'
                        train_cutoff = prophet_df['ds'].max() - pd.Timedelta(days=test_days)
                        train_df = prophet_df[prophet_df['ds'] <= train_cutoff]
                        test_df = prophet_df[prophet_df['ds'] > train_cutoff]

                        if len(train_df) > 2 and not test_df.empty:
                            model_backtest = Prophet()
                            model_backtest.fit(train_df)
                            
                            # --- FIX: Predict specifically on the test set timestamps ---
                            # Instead of making a future dataframe (which defaults to days),
                            # we pass the exact timestamps from our test data. This ensures matching.
                            forecast_backtest = model_backtest.predict(test_df[['ds']])
                            
                            # Calculate MAPE
                            y_true = test_df['y'].values
                            y_pred = forecast_backtest['yhat'].values
                            
                            if len(y_true) == len(y_pred):
                                mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                                accuracy_score = max(0, 100 - mape) # Ensure not negative
                                backtest_metric = f"{accuracy_score:.2f}%"
                                backtest_msg = f"Verified against the last {test_days} days of data."
                    except Exception as e:
                        print(f"Backtest failed: {e}")

                # --- Display Results ---
                
                if backtest_metric:
                    st.success(f"Model Accuracy: **{backtest_metric}** ({backtest_msg})")
                else:
                    st.info(f"ℹ Showing Forecast. (Accuracy check requires >3 days history, current: {history_days} days)")

                # Show Main Forecast Chart
                st.subheader(f"Price Forecast: {forecast_symbol}")
                fig_forecast = plot_plotly(model_future, forecast_future)
                
                # Dark mode styling for points
                for trace in fig_forecast.data:
                    if trace.mode == 'markers':
                        trace.marker.color = 'rgba(0, 255, 255, 0.7)' # Cyan dots
                        trace.marker.size = 3

                fig_forecast.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_forecast, use_container_width=True)

                # Components
                st.subheader("Forecast Components")
                fig_components = plot_components_plotly(model_future, forecast_future)
                fig_components.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_components, use_container_width=True)