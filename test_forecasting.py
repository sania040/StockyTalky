# test_forecasting.py
"""
Quick test script to validate all forecasting models work correctly.
Run: python test_forecasting.py
"""

import pandas as pd
import numpy as np
from src.ml.forecasting import get_available_models

# Generate sample crypto price data
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
prices = 100 + np.cumsum(np.random.randn(60) * 2)  # Random walk starting at 100

df = pd.DataFrame({
    'timestamp': dates,
    'price_usd': prices
})

print("=" * 80)
print("FORECASTING MODELS TEST")
print("=" * 80)
print(f"\nTest Data: {len(df)} daily price observations")
print(f"Price Range: ${df['price_usd'].min():.2f} - ${df['price_usd'].max():.2f}")
print(f"Current Price: ${df['price_usd'].iloc[-1]:.2f}")

# Get all models
models = get_available_models()
print(f"\nAvailable Models: {', '.join(models.keys())}\n")

for model_name, model in models.items():
    print("-" * 80)
    print(f"Testing: {model_name}")
    print("-" * 80)
    
    # Try to fit
    print(f"  Fitting {model_name}...")
    fit_ok = model.fit(df)
    
    if not fit_ok:
        print(f"  ❌ Failed to fit {model_name}")
        continue
    
    print(f"  ✅ Fit successful")
    
    # Generate forecast
    print(f"  Generating 14-day forecast...")
    forecast = model.predict(periods=14, df=df)
    
    if forecast.empty:
        print(f"  ❌ Forecast generation failed")
        continue
    
    print(f"  ✅ Forecast generated:")
    print(f"      First forecast: ${forecast['forecast'].iloc[0]:.2f}")
    print(f"      Last forecast: ${forecast['forecast'].iloc[-1]:.2f}")
    print(f"      Avg: ${forecast['forecast'].mean():.2f}")
    
    # Run backtest
    print(f"  Running 7-day backtest...")
    results = model.backtest(df, test_days=7)
    
    if 'data_insufficient' in results:
        print(f"  ⚠️  Insufficient data for backtest")
    elif 'error' in results:
        print(f"  ⚠️  Backtest error: {results['error']}")
    elif results['mape'] is None:
        print(f"  ⚠️  Could not compute metrics")
    else:
        print(f"  ✅ Backtest Results:")
        print(f"      MAPE: {results['mape']:.2f}%")
        print(f"      RMSE: ${results['rmse']:.2f}")
    
    print()

print("=" * 80)
print("✅ All tests completed!")
print("=" * 80)
print("\nNow you can:")
print("1. Run the Streamlit app: streamlit run app.py")
print("2. Navigate to 'Market Analysis & ML Insights' page")
print("3. Select a model from the dropdown and generate forecasts")
print("\nFor more info, see: ML_MODELS_README.md")
