import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.db.get_connection import get_db_connection
from src.db.query_utils import execute_query

def show_visualizations():
    st.title("Crypto Investment Visualizations")
    
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
    
    # Get available symbols
    symbols = df['symbol'].unique().tolist()
    selected_symbols = st.multiselect("Select cryptocurrencies", symbols, default=symbols[:min(5, len(symbols))])
    
    if not selected_symbols:
        st.warning("Please select at least one cryptocurrency")
        return
    
    # Filter data
    filtered_df = df[df['symbol'].isin(selected_symbols)]
    
    # Visualization 1: Price History
    st.subheader("Price History")
    price_fig = px.line(
        filtered_df, 
        x='timestamp', 
        y='price_usd', 
        color='symbol',
        title="Price History",
        labels={"price_usd": "Price (USD)", "timestamp": "Date", "symbol": "Crypto"}
    )
    st.plotly_chart(price_fig, use_container_width=True)
    
    # Visualization 2: Trading Volume
    st.subheader("Trading Volume")
    volume_fig = px.bar(
        filtered_df, 
        x='timestamp', 
        y='volume_24h_usd', 
        color='symbol',
        title="24h Trading Volume",
        labels={"volume_24h_usd": "Volume (USD)", "timestamp": "Date", "symbol": "Crypto"}
    )
    st.plotly_chart(volume_fig, use_container_width=True)
    
    # Visualization 3: Market Cap Comparison
    st.subheader("Market Cap Comparison")
    latest_data = df.sort_values('timestamp').drop_duplicates(subset=['symbol'], keep='last')
    latest_filtered = latest_data[latest_data['symbol'].isin(selected_symbols)]
    
    cap_fig = px.pie(
        latest_filtered,
        values='market_cap_usd',
        names='symbol',
        title="Market Cap Distribution"
    )
    st.plotly_chart(cap_fig, use_container_width=True)
    
    # ML-Based Investment Recommendations
    st.subheader("Top 5 Investment Recommendations")

    with st.spinner("Analyzing market data and generating recommendations..."):
        # Let's try a different approach - allow users to choose the model
        model_type = st.radio(
            "Select prediction approach:",
            ["Ensemble Model", "Simple Trend Analysis", "Technical Indicators"],
            horizontal=True
        )
        
        if model_type == "Ensemble Model":
            # Enhanced feature engineering and model ensemble approach
            
            # Prepare data with better features
            all_features = []
            future_returns = []
            
            # Define additional technical indicators
            def calculate_rsi(prices, window=14):
                """Calculate Relative Strength Index"""
                delta = prices.diff()
                gain = delta.where(delta > 0, 0).rolling(window=window).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
                
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            def calculate_macd(prices, fast=12, slow=26):
                """Calculate MACD"""
                if len(prices) < slow:
                    return 0
                
                fast_ema = prices.ewm(span=fast, adjust=False).mean()
                slow_ema = prices.ewm(span=slow, adjust=False).mean()
                macd = fast_ema - slow_ema
                return macd.iloc[-1]
            
            # Normalize numeric columns for better model performance
            def normalize_features(df, columns):
                """Min-max normalize features"""
                result = df.copy()
                for column in columns:
                    if result[column].std() > 0:  # Avoid division by zero
                        result[column] = (result[column] - result[column].min()) / (result[column].max() - result[column].min())
                    else:
                        result[column] = 0
                return result
            
            # For each symbol, create better prediction datasets
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
                
                if len(symbol_data) < 20:  # Need more data for reliable indicators
                    continue
                    
                # Create features with more lookback periods
                for i in range(15, len(symbol_data)-5):  # Use 15 data points, predict 5 days ahead
                    # Different window sizes for more robust features
                    window_short = symbol_data.iloc[i-5:i]
                    window_med = symbol_data.iloc[i-10:i]
                    window_long = symbol_data.iloc[i-15:i]
                    
                    # Target is 5-day future return
                    future = symbol_data.iloc[i+5] if i+5 < len(symbol_data) else symbol_data.iloc[-1]
                    
                    # Calculate enhanced features
                    features = {
                        'symbol': symbol,
                        'current_price': window_short['price_usd'].iloc[-1],
                        
                        # Price features at different timeframes
                        'price_mean_5d': window_short['price_usd'].mean(),
                        'price_mean_10d': window_med['price_usd'].mean(),
                        'price_std_5d': window_short['price_usd'].std(),
                        'price_std_10d': window_med['price_usd'].std(),
                        
                        # Volume features
                        'volume_mean_5d': window_short['volume_24h_usd'].mean(),
                        'volume_mean_10d': window_med['volume_24h_usd'].mean(),
                        'volume_std_5d': window_short['volume_24h_usd'].std(),
                        
                        # Trend features
                        'price_trend_5d': window_short['price_usd'].pct_change().mean() * 100,
                        'price_trend_10d': window_med['price_usd'].pct_change().mean() * 100,
                        'volume_trend_5d': window_short['volume_24h_usd'].pct_change().mean() * 100,
                        
                        # Ratios and technical indicators
                        'price_to_ma_ratio': window_short['price_usd'].iloc[-1] / window_long['price_usd'].mean() if window_long['price_usd'].mean() > 0 else 1,
                        'rsi': calculate_rsi(window_long['price_usd']),
                        'macd': calculate_macd(window_long['price_usd']),
                        
                        # Market cap
                        'market_cap': window_short['market_cap_usd'].iloc[-1],
                        'market_cap_trend': window_short['market_cap_usd'].pct_change().mean() * 100,
                    }
                    
                    # Calculate future return - 5-day return instead of 1-day
                    future_return = ((future['price_usd'] / window_short['price_usd'].iloc[-1]) - 1) * 100
                    
                    # Only add data points with valid values
                    if (not pd.isna(future_return) and 
                        not any(pd.isna(val) for val in features.values() if not isinstance(val, str))):
                        all_features.append(features)
                        future_returns.append(future_return)
            
            if len(all_features) < 30:
                st.warning("Not enough historical data to train model. Collect more data points.")
                return
            
            # Create feature dataset
            feature_df = pd.DataFrame(all_features)
            feature_df['future_return'] = future_returns
            
            # All numeric features
            X_cols = [col for col in feature_df.columns if col not in ['symbol', 'future_return']]
            
            # Fill NaN values and normalize
            feature_df[X_cols] = feature_df[X_cols].fillna(0)
            feature_df = normalize_features(feature_df, X_cols)
            
            # Split data correctly
            X = feature_df[X_cols]
            y = feature_df['future_return']
            
            # Use K-Fold cross-validation for more reliable model evaluation
            from sklearn.model_selection import KFold
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import ElasticNet
            
            # Train multiple models using K-Fold validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Dictionary to hold models and their scores
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
                'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            }
            
            # Train and evaluate models
            model_scores = {}
            best_model_name = None
            best_score = -float('inf')
            
            for name, model in models.items():
                fold_scores = []
                for train_idx, test_idx in kf.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    fold_scores.append(score)
                
                avg_score = np.mean(fold_scores)
                model_scores[name] = avg_score
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model_name = name
            
            # Display model comparison
            score_df = pd.DataFrame({
                'Model': list(model_scores.keys()),
                'R² Score': list(model_scores.values())
            })
            
            st.write("Model Performance Comparison")
            st.dataframe(score_df)
            
            # Select best model
            best_model = models[best_model_name]
            best_model.fit(X, y)  # Fit on all data
            
            # Get feature importances if available
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': X_cols,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Display feature importance
                st.write("Feature Importance")
                importance_fig = px.bar(
                    feature_importance, 
                    x='Feature', 
                    y='Importance',
                    title=f"{best_model_name.replace('_', ' ').title()} Feature Importance"
                )
                st.plotly_chart(importance_fig, use_container_width=True)
            
            # Generate predictions for current state of each coin
            prediction_data = []
            
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
                
                if len(symbol_data) < 15:  # Need enough data points
                    continue
                
                # Get most recent data
                recent_short = symbol_data.iloc[-5:]
                recent_med = symbol_data.iloc[-10:] if len(symbol_data) >= 10 else symbol_data
                recent_long = symbol_data.iloc[-15:] if len(symbol_data) >= 15 else symbol_data
                
                # Create prediction features
                pred_features = {
                    'symbol': symbol,
                    'current_price': recent_short['price_usd'].iloc[-1],
                    
                    # Price features at different timeframes
                    'price_mean_5d': recent_short['price_usd'].mean(),
                    'price_mean_10d': recent_med['price_usd'].mean(),
                    'price_std_5d': recent_short['price_usd'].std(),
                    'price_std_10d': recent_med['price_usd'].std(),
                    
                    # Volume features
                    'volume_mean_5d': recent_short['volume_24h_usd'].mean(),
                    'volume_mean_10d': recent_med['volume_24h_usd'].mean(),
                    'volume_std_5d': recent_short['volume_24h_usd'].std(),
                    
                    # Trend features
                    'price_trend_5d': recent_short['price_usd'].pct_change().mean() * 100,
                    'price_trend_10d': recent_med['price_usd'].pct_change().mean() * 100,
                    'volume_trend_5d': recent_short['volume_24h_usd'].pct_change().mean() * 100,
                    
                    # Ratios and technical indicators
                    'price_to_ma_ratio': recent_short['price_usd'].iloc[-1] / recent_long['price_usd'].mean() if recent_long['price_usd'].mean() > 0 else 1,
                    'rsi': calculate_rsi(recent_long['price_usd']),
                    'macd': calculate_macd(recent_long['price_usd']),
                    
                    # Market cap
                    'market_cap': recent_short['market_cap_usd'].iloc[-1],
                    'market_cap_trend': recent_short['market_cap_usd'].pct_change().mean() * 100,
                }
                
                # Add to prediction dataset
                prediction_data.append(pred_features)
            
            # Create prediction dataframe
            pred_df = pd.DataFrame(prediction_data)
            
            # Prepare features for prediction
            X_pred_cols = [col for col in pred_df.columns if col != 'symbol']
            X_pred = pred_df[X_pred_cols].fillna(0)
            
            # Normalize prediction features same way as training
            for col in X_pred.columns:
                if col in X.columns:
                    # If feature exists in training data, predict on it
                    X_pred[col] = X_pred[col]
            
            # Filter to only have columns in the model
            X_pred = X_pred[X_cols]
            
            # Make predictions
            pred_df['predicted_return'] = best_model.predict(X_pred)
            
            # Calculate confidence based on feature values
            # Confidence is higher when key indicators align
            def calculate_confidence(row):
                """Calculate confidence score based on feature alignment"""
                signals = []
                
                # Price trend signals
                if row['price_trend_5d'] > 0:
                    signals.append(1)
                elif row['price_trend_5d'] < 0:
                    signals.append(-1)
                    
                # Volume trend signals    
                if row['volume_trend_5d'] > 0:
                    signals.append(1)
                elif row['volume_trend_5d'] < 0:
                    signals.append(-1)
                    
                # RSI signals
                if row['rsi'] > 70:
                    signals.append(-1)  # Overbought
                elif row['rsi'] < 30:
                    signals.append(1)   # Oversold
                    
                # MACD signals
                if row['macd'] > 0:
                    signals.append(1)
                elif row['macd'] < 0:
                    signals.append(-1)
                    
                # Calculate alignment (how many signals agree)
                if len(signals) == 0:
                    return 0.5  # Neutral
                    
                positive = sum(1 for s in signals if s > 0)
                negative = sum(1 for s in signals if s < 0)
                
                if positive > negative:
                    # Most signals positive
                    return 0.5 + (positive / len(signals)) * 0.5
                elif negative > positive:
                    # Most signals negative
                    return 0.5 - (negative / len(signals)) * 0.5
                else:
                    # Mixed signals
                    return 0.5
            
            # Apply confidence calculation
            pred_df['confidence'] = pred_df.apply(calculate_confidence, axis=1)
        
        elif model_type == "Simple Trend Analysis":
            # Use simple trend analysis - more reliable than ML for limited data
            prediction_data = []
            
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
                
                if len(symbol_data) < 5:
                    continue
                    
                # Get recent data
                recent = symbol_data.iloc[-10:] if len(symbol_data) >= 10 else symbol_data
                
                # Calculate trends
                price_trend = recent['price_usd'].pct_change().mean() * 100
                volume_trend = recent['volume_24h_usd'].pct_change().mean() * 100
                volatility = recent['price_usd'].pct_change().std() * 100
                
                # Simple scoring
                trend_score = (
                    price_trend * 0.5 + 
                    volume_trend * 0.3 + 
                    (10 - min(volatility, 10)) * 0.2  # Lower volatility is better
                )
                
                # Calculate RSI
                delta = recent['price_usd'].diff()
                gain = delta.where(delta > 0, 0).mean()
                loss = -delta.where(delta < 0, 0).mean()
                
                if loss != 0:
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 50
                
                # Adjust score based on RSI
                if rsi > 70:  # Overbought
                    trend_score *= 0.7
                elif rsi < 30:  # Oversold (potential buy)
                    trend_score *= 1.3
                    
                prediction_data.append({
                    'symbol': symbol,
                    'current_price': recent['price_usd'].iloc[-1],
                    'price_trend': price_trend,
                    'volume_trend': volume_trend,
                    'volatility': volatility,
                    'rsi': rsi,
                    'predicted_return': trend_score,
                    'confidence': 0.7  # Higher fixed confidence for this method
                })
                
            pred_df = pd.DataFrame(prediction_data)
            
        else:  # Technical Indicators
            # Use technical indicator approach
            prediction_data = []
            
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
                
                if len(symbol_data) < 20:
                    continue
                    
                # Get recent data
                prices = symbol_data['price_usd']
                volumes = symbol_data['volume_24h_usd']
                
                # Calculate SMAs
                sma_short = prices.rolling(5).mean().iloc[-1]
                sma_med = prices.rolling(10).mean().iloc[-1]
                sma_long = prices.rolling(20).mean().iloc[-1]
                
                # Current price
                current_price = prices.iloc[-1]
                
                # Calculate technical signals
                # 1. Price vs SMAs
                price_above_sma_short = current_price > sma_short
                price_above_sma_med = current_price > sma_med
                price_above_sma_long = current_price > sma_long
                
                # 2. SMA crossovers
                short_above_med = sma_short > sma_med
                med_above_long = sma_med > sma_long
                
                # 3. RSI
                delta = prices.diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean().iloc[-1]
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean().iloc[-1]
                
                if loss != 0:
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 50
                    
                # 4. MACD
                ema_fast = prices.ewm(span=12, adjust=False).mean()
                ema_slow = prices.ewm(span=26, adjust=False).mean()
                macd = ema_fast.iloc[-1] - ema_slow.iloc[-1]
                macd_signal = (ema_fast - ema_slow).ewm(span=9, adjust=False).mean().iloc[-1]
                macd_hist = macd - macd_signal
                
                # 5. Volume trend
                vol_trend = volumes.pct_change().rolling(5).mean().iloc[-1] * 100
                
                # Technical score
                tech_score = 0
                
                # Uptrend indicators
                if price_above_sma_short: tech_score += 1
                if price_above_sma_med: tech_score += 1
                if price_above_sma_long: tech_score += 1
                if short_above_med: tech_score += 1
                if med_above_long: tech_score += 1
                
                # RSI indicators
                if rsi < 30: tech_score += 2  # Oversold - buying opportunity
                elif rsi < 50: tech_score += 1
                elif rsi > 70: tech_score -= 2  # Overbought - potential drop
                elif rsi > 60: tech_score -= 1
                
                # MACD indicators
                if macd > 0 and macd_hist > 0: tech_score += 2  # Strong buy signal
                elif macd > 0: tech_score += 1
                elif macd < 0 and macd_hist < 0: tech_score -= 2  # Strong sell signal
                elif macd < 0: tech_score -= 1
                
                # Volume confirms trend
                if vol_trend > 0 and tech_score > 0: tech_score += 1
                if vol_trend < 0 and tech_score < 0: tech_score -= 1
                
                # Convert to percentage expected return
                predicted_return = tech_score * 1.5
                
                # Calculate confidence
                # Higher when more indicators align
                confidence = min(0.95, 0.5 + abs(tech_score) * 0.05)
                
                prediction_data.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'sma_ratio': current_price / sma_med if sma_med > 0 else 1,
                    'rsi': rsi,
                    'macd': macd,
                    'volume_trend': vol_trend,
                    'tech_score': tech_score,
                    'predicted_return': predicted_return,
                    'confidence': confidence
                })
                
            pred_df = pd.DataFrame(prediction_data)
        
        # Get top 5 recommendations
        top_recommendations = pred_df.sort_values('predicted_return', ascending=False).head(5)
        
        # Display top 5 coins
        for i, (_, row) in enumerate(top_recommendations.iterrows(), 1):
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
            
            # Generate reasons based on model type
            if model_type == "Ensemble Model" and hasattr(best_model, 'feature_importances_'):
                # Use feature importances for explanation
                top_features = feature_importance.head(3)['Feature'].tolist()
                reasons = []
                
                for feature in top_features:
                    if 'price' in feature and 'trend' in feature and row.get(feature, 0) > 0:
                        reasons.append("Positive price trend")
                    elif 'volume' in feature and 'trend' in feature and row.get(feature, 0) > 0:
                        reasons.append("Increasing volume")
                    elif 'std' in feature and row.get(feature, 0) < pred_df[feature].median():
                        reasons.append("Low price volatility")
                    elif 'market_cap' in feature and row.get(feature, 0) > pred_df[feature].median():
                        reasons.append("Strong market cap")
                    elif 'rsi' in feature and 30 <= row.get(feature, 50) <= 70:
                        reasons.append("Balanced RSI")
                    
                reason_text = ", ".join(reasons) if reasons else "Based on model prediction"
            
            elif model_type == "Simple Trend Analysis":
                # Use trend features for explanation
                reasons = []
                if row.get('price_trend', 0) > 0:
                    reasons.append("Positive price trend")
                if row.get('volume_trend', 0) > 0:
                    reasons.append("Increasing volume")
                if row.get('volatility', 100) < 5:
                    reasons.append("Low price volatility")
                if 30 <= row.get('rsi', 50) <= 70:
                    reasons.append("Balanced RSI")
                    
                reason_text = ", ".join(reasons) if reasons else "Based on trend analysis"
                
            else:  # Technical Indicators
                # Use technical signals for explanation
                reasons = []
                if row.get('sma_ratio', 1) > 1:
                    reasons.append("Price above moving average")
                if 40 <= row.get('rsi', 50) <= 60:
                    reasons.append("Neutral RSI")
                elif row.get('rsi', 50) < 30:
                    reasons.append("Oversold (buying opportunity)")
                if row.get('macd', 0) > 0:
                    reasons.append("Positive MACD")
                if row.get('volume_trend', 0) > 0:
                    reasons.append("Increasing volume")
                    
                reason_text = ", ".join(reasons) if reasons else "Based on technical signals"
            
            # Generate simple AI insight without external agent dependency
            if predicted_return > 5:
                ai_insight = f"{symbol} shows promising momentum with multiple indicators suggesting potential upside."
            elif predicted_return > 2:
                ai_insight = f"{symbol} displays positive signals with moderate growth potential in the near term."
            elif predicted_return > 0:
                ai_insight = f"{symbol} appears stable with balanced risk-reward profile at current price levels."
            else:
                ai_insight = f"{symbol} exhibits cautionary signals; consider monitoring before making investment decisions."
            
            # Display recommendation card
            st.markdown(f"""
            <div class="card {color}">
                <h3>#{i} {symbol}</h3>
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
                <p><strong>Why:</strong> {reason_text}</p>
                <div style="background-color: rgba(0,0,0,0.03); padding: 8px; border-radius: 4px; margin-top: 8px;">
                    <p style="margin: 0; font-style: italic;">"{ai_insight}"</p>
                </div>
                <p><small>Model confidence: {confidence:.2f}</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add small spacing between cards
            st.markdown("<br>", unsafe_allow_html=True)