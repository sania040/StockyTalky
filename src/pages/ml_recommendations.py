import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.db.query_utils import execute_query

def show():
    """Display ML recommendations page"""
    st.title("🤖 ML-Powered Investment Recommendations")
    
    # Fetch data
    df = execute_query("""
        SELECT symbol, price_usd, volume_24h_usd, percent_change_24h, 
               market_cap_usd, timestamp 
        FROM crypto_prices 
        ORDER BY timestamp DESC
        LIMIT 2500
    """)
    
    if df.empty:
        st.info("No data available. Please collect some data first.")
        return
    
    # Model selection
    st.subheader("Select Prediction Model")
    
    model_type = st.radio(
        "Choose analysis approach:",
        ["Simple Trend Analysis", "Ensemble Model", "Technical Indicators"],
        horizontal=True
    )
    
    with st.spinner("Analyzing market data and generating recommendations..."):
        
        if model_type == "Simple Trend Analysis":
            recommendations = analyze_simple_trends(df)
        elif model_type == "Ensemble Model":
            recommendations = analyze_with_ml(df)
        else:
            recommendations = analyze_technical_indicators(df)
        
        if recommendations.empty:
            st.warning("Not enough data to generate recommendations.")
            return
        
        # Display top 5 recommendations
        st.subheader("🌟 Top 5 Investment Recommendations")
        
        top_5 = recommendations.nlargest(5, 'predicted_return')
        
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            display_recommendation_card(i, row, model_type)

def analyze_simple_trends(df):
    """Simple trend-based analysis"""
    prediction_data = []
    
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
        
        if len(symbol_data) < 5:
            continue
        
        recent = symbol_data.tail(10)
        
        # Calculate trends
        price_trend = recent['price_usd'].pct_change().mean() * 100
        volume_trend = recent['volume_24h_usd'].pct_change().mean() * 100
        volatility = recent['price_usd'].pct_change().std() * 100
        
        # Simple scoring
        trend_score = (
            price_trend * 0.5 + 
            volume_trend * 0.3 + 
            (10 - min(volatility, 10)) * 0.2
        )
        
        # Calculate RSI
        delta = recent['price_usd'].diff()
        gain = delta.where(delta > 0, 0).mean()
        loss = -delta.where(delta < 0, 0).mean()
        rsi = 100 - (100 / (1 + (gain / loss))) if loss != 0 else 50
        
        # Adjust score based on RSI
        if rsi > 70:
            trend_score *= 0.7
        elif rsi < 30:
            trend_score *= 1.3
        
        prediction_data.append({
            'symbol': symbol,
            'current_price': recent['price_usd'].iloc[-1],
            'price_trend': price_trend,
            'volume_trend': volume_trend,
            'volatility': volatility,
            'rsi': rsi,
            'predicted_return': trend_score,
            'confidence': 0.7
        })
    
    return pd.DataFrame(prediction_data)

def analyze_with_ml(df):
    """ML-based analysis using Random Forest"""
    # Prepare features
    all_features = []
    future_returns = []
    
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
        
        if len(symbol_data) < 20:
            continue
        
        for i in range(15, len(symbol_data)-5):
            window = symbol_data.iloc[i-15:i]
            future = symbol_data.iloc[i+5]
            
            features = {
                'symbol': symbol,
                'current_price': window['price_usd'].iloc[-1],
                'price_mean_5d': window['price_usd'].tail(5).mean(),
                'price_std_5d': window['price_usd'].tail(5).std(),
                'volume_mean_5d': window['volume_24h_usd'].tail(5).mean(),
                'price_trend': window['price_usd'].pct_change().mean() * 100,
                'market_cap': window['market_cap_usd'].iloc[-1]
            }
            
            future_return = ((future['price_usd'] / window['price_usd'].iloc[-1]) - 1) * 100
            
            if not pd.isna(future_return):
                all_features.append(features)
                future_returns.append(future_return)
    
    if len(all_features) < 30:
        return pd.DataFrame()
    
    feature_df = pd.DataFrame(all_features)
    X_cols = [col for col in feature_df.columns if col != 'symbol']
    X = feature_df[X_cols].fillna(0)
    y = future_returns
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Generate predictions
    prediction_data = []
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
        
        if len(symbol_data) < 15:
            continue
        
        recent = symbol_data.tail(15)
        
        pred_features = {
            'current_price': recent['price_usd'].iloc[-1],
            'price_mean_5d': recent['price_usd'].tail(5).mean(),
            'price_std_5d': recent['price_usd'].tail(5).std(),
            'volume_mean_5d': recent['volume_24h_usd'].tail(5).mean(),
            'price_trend': recent['price_usd'].pct_change().mean() * 100,
            'market_cap': recent['market_cap_usd'].iloc[-1]
        }
        
        X_pred = pd.DataFrame([pred_features])[X_cols].fillna(0)
        predicted_return = model.predict(X_pred)[0]
        
        prediction_data.append({
            'symbol': symbol,
            'current_price': recent['price_usd'].iloc[-1],
            'predicted_return': predicted_return,
            'confidence': 0.75
        })
    
    return pd.DataFrame(prediction_data)

def analyze_technical_indicators(df):
    """Technical indicator-based analysis"""
    prediction_data = []
    
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
        
        if len(symbol_data) < 20:
            continue
        
        prices = symbol_data['price_usd']
        
        # Calculate indicators
        sma_short = prices.rolling(5).mean().iloc[-1]
        sma_long = prices.rolling(20).mean().iloc[-1]
        current_price = prices.iloc[-1]
        
        # Technical score
        tech_score = 0
        if current_price > sma_short:
            tech_score += 2
        if current_price > sma_long:
            tech_score += 2
        if sma_short > sma_long:
            tech_score += 2
        
        # RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = -delta.where(delta < 0, 0).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + (gain / loss))) if loss != 0 else 50
        
        if rsi < 30:
            tech_score += 2
        elif rsi > 70:
            tech_score -= 2
        
        predicted_return = tech_score * 1.5
        confidence = min(0.95, 0.5 + abs(tech_score) * 0.05)
        
        prediction_data.append({
            'symbol': symbol,
            'current_price': current_price,
            'predicted_return': predicted_return,
            'confidence': confidence,
            'rsi': rsi
        })
    
    return pd.DataFrame(prediction_data)

def display_recommendation_card(rank, row, model_type):
    """Display a recommendation card"""
    symbol = row['symbol']
    predicted_return = row['predicted_return']
    current_price = row['current_price']
    confidence = row['confidence']
    
    # Determine recommendation type
    if predicted_return > 5:
        recommendation = "Strong Buy"
        color = "buy"
    elif predicted_return > 2:
        recommendation = "Buy"
        color = "buy"
    elif predicted_return > 0:
        recommendation = "Hold"
        color = "hold"
    else:
        recommendation = "Watch"
        color = "hold"
    
    # Generate insight
    if predicted_return > 5:
        insight = f"{symbol} shows promising momentum with strong growth potential."
    elif predicted_return > 2:
        insight = f"{symbol} displays positive signals with moderate growth potential."
    elif predicted_return > 0:
        insight = f"{symbol} appears stable with balanced risk-reward profile."
    else:
        insight = f"{symbol} exhibits cautionary signals; monitor before investing."
    
    st.markdown(f"""
    <div class="card {color}">
        <h3>#{rank} {symbol}</h3>
        <div style="display: flex; justify-content: space-between;">
            <div>
                <p class="metric-label">Current Price</p>
                <p class="metric-value">${current_price:.2f}</p>
            </div>
            <div>
                <p class="metric-label">Predicted Return</p>
                <p class="metric-value {'positive' if predicted_return > 0 else 'negative'}">{predicted_return:.2f}%</p>
            </div>
            <div>
                <p class="metric-label">Recommendation</p>
                <p class="metric-value">{recommendation}</p>
            </div>
        </div>
        <div style="background-color: rgba(0,0,0,0.03); padding: 8px; border-radius: 4px; margin-top: 8px;">
            <p style="margin: 0; font-style: italic;">"{insight}"</p>
        </div>
        <p><small>Confidence: {confidence:.0%} | Model: {model_type}</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)