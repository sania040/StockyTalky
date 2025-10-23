# src/ml/forecasting.py
"""
Multi-model forecasting module for cryptocurrency price prediction.
Supports: Prophet and XGBoost models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# --- Prophet Availability Check ---
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    import prophet
    # Optional: check version if needed
    PROPHET_AVAILABLE = True
except ImportError:
    pass

# --- XGBoost ---
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


class ForecastingModel:
    """Base class for forecasting models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
    
    def fit(self, df: pd.DataFrame, target_col: str = 'price_usd') -> bool:
        raise NotImplementedError
    
    def predict(self, periods: int, df: pd.DataFrame = None) -> pd.DataFrame:
        raise NotImplementedError
    
    def backtest(self, df: pd.DataFrame, test_days: int = 7, target_col: str = 'price_usd') -> Dict[str, float]:
        raise NotImplementedError


class ProphetModel(ForecastingModel):
    """Prophet time-series forecasting model."""
    
    def __init__(self):
        super().__init__("Prophet")
    
    def fit(self, df: pd.DataFrame, target_col: str = 'price_usd') -> bool:
        if not PROPHET_AVAILABLE:
            print("Prophet not installed. Install with: pip install prophet")
            return False
        
        try:
            # Prepare data
            prophet_df = df[['timestamp', target_col]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            # Ensure timezone-naive — critical for Prophet
            if prophet_df['ds'].dt.tz is not None:
                prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
            
            # Drop NaNs
            prophet_df = prophet_df.dropna(subset=['y'])
            if len(prophet_df) < 2:
                return False

            # Fit model — no warning suppression here
            self.model = Prophet(
                interval_width=0.95,
                daily_seasonality=False,
                yearly_seasonality=False,
                weekly_seasonality=False  # Optional: reduce overfitting
            )
            self.model.fit(prophet_df)
            return True
        except Exception as e:
            print(f"Error fitting Prophet: {e}")
            return False
    
    def predict(self, periods: int, df: pd.DataFrame = None) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        
        try:
            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
            result.columns = ['timestamp', 'forecast', 'lower_bound', 'upper_bound']
            result['model'] = 'Prophet'
            return result.reset_index(drop=True)
        except Exception as e:
            print(f"Error predicting with Prophet: {e}")
            return pd.DataFrame()
    
    def backtest(self, df: pd.DataFrame, test_days: int = 7, target_col: str = 'price_usd') -> Dict[str, float]:
        try:
            if len(df) < 10 + test_days:
                return {'mape': None, 'rmse': None, 'data_insufficient': True}
            
            prophet_df = df[['timestamp', target_col]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            if prophet_df['ds'].dt.tz is not None:
                prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
            prophet_df = prophet_df.dropna(subset=['y'])
            
            if len(prophet_df) < 10 + test_days:
                return {'mape': None, 'rmse': None, 'data_insufficient': True}
            
            train_df = prophet_df[:-test_days]
            test_df = prophet_df[-test_days:]
            
            model = Prophet(
                yearly_seasonality=False,
                daily_seasonality=False,
                weekly_seasonality=False
            )
            model.fit(train_df)
            
            future = model.make_future_dataframe(periods=test_days)
            forecast = model.predict(future)
            forecast_test = forecast[-test_days:]
            
            eval_df = pd.merge(test_df, forecast_test[['ds', 'yhat']], on='ds', how='inner')
            
            if eval_df.empty or len(eval_df) == 0:
                return {'mape': None, 'rmse': None, 'data_insufficient': True}
            
            mape = mean_absolute_percentage_error(eval_df['y'], eval_df['yhat']) * 100
            rmse = np.sqrt(mean_squared_error(eval_df['y'], eval_df['yhat']))
            
            return {'mape': mape, 'rmse': rmse}
        except Exception as e:
            print(f"Error in Prophet backtest: {e}")
            return {'mape': None, 'rmse': None, 'error': str(e)}


class XGBoostModel(ForecastingModel):
    """XGBoost model with engineered time-series features."""
    
    def __init__(self):
        super().__init__("XGBoost")
        self.feature_cols = None
        self.last_data = None
        self.last_price = None
    
    def _engineer_features(self, df: pd.DataFrame, target_col: str = 'price_usd') -> pd.DataFrame:
        df = df.copy()
        prices = df[target_col].values
        
        for lag in [1, 2, 3, 5, 7]:
            df[f'price_lag_{lag}'] = pd.Series(prices).shift(lag).values
        
        df['returns'] = pd.Series(prices).pct_change() * 100
        df['returns_lag_1'] = df['returns'].shift(1)
        
        df['price_ma_7'] = pd.Series(prices).rolling(7).mean().values
        df['price_vol_7'] = pd.Series(prices).rolling(7).std().values
        df['price_mom_7'] = prices - pd.Series(prices).rolling(7).mean().values
        
        df = df.dropna()
        return df
    
    def fit(self, df: pd.DataFrame, target_col: str = 'price_usd') -> bool:
        if not XGB_AVAILABLE:
            print("XGBoost not installed. Install with: pip install xgboost")
            return False
        
        try:
            df_features = self._engineer_features(df, target_col)
            if len(df_features) < 10:
                return False
            
            self.last_data = df_features.tail(10).copy()
            self.last_price = df[target_col].iloc[-1]
            
            self.feature_cols = [col for col in df_features.columns if col not in ['timestamp', target_col]]
            X = df_features[self.feature_cols]
            y = df_features[target_col]
            
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            self.model.fit(X, y)
            return True
        except Exception as e:
            print(f"Error fitting XGBoost: {e}")
            return False
    
    def predict(self, periods: int, df: pd.DataFrame = None, target_col: str = 'price_usd') -> pd.DataFrame:
        if self.model is None or self.feature_cols is None or self.last_data is None:
            return pd.DataFrame()
        
        try:
            predictions = []
            timestamps = []
            last_date = pd.to_datetime(self.last_data['timestamp'].iloc[-1])
            current_price = self.last_price
            current_data = self.last_data.copy()
            
            for step in range(periods):
                future_date = last_date + pd.Timedelta(days=step + 1)
                timestamps.append(future_date)
                
                df_features = self._engineer_features(current_data, target_col)
                if len(df_features) > 0 and all(col in df_features.columns for col in self.feature_cols):
                    X_pred = df_features[self.feature_cols].iloc[[-1]]
                    pred = float(self.model.predict(X_pred)[0])
                    pred = np.clip(pred, current_price * 0.5, current_price * 1.5)
                    pred = max(pred, 0.01)
                    predictions.append(pred)
                    
                    new_row = current_data.iloc[[-1]].copy()
                    new_row[target_col] = pred
                    new_row['timestamp'] = future_date
                    current_data = pd.concat([current_data, new_row], ignore_index=True).tail(10)
                    current_price = pred
                else:
                    predictions.append(current_price)
            
            result_df = pd.DataFrame({
                'timestamp': timestamps,
                'forecast': predictions,
                'lower_bound': np.array(predictions) * 0.95,
                'upper_bound': np.array(predictions) * 1.05,
                'model': 'XGBoost'
            })
            return result_df
        except Exception as e:
            print(f"Error predicting with XGBoost: {e}")
            return pd.DataFrame()
    
    def backtest(self, df: pd.DataFrame, test_days: int = 7, target_col: str = 'price_usd') -> Dict[str, float]:
        try:
            if len(df) < 20 + test_days:
                return {'mape': None, 'rmse': None, 'data_insufficient': True}
            
            df_features = self._engineer_features(df, target_col)
            if len(df_features) < 20 + test_days:
                return {'mape': None, 'rmse': None, 'data_insufficient': True}
            
            train_data = df_features[:-test_days]
            test_data = df_features[-test_days:]
            
            X_train = train_data[self.feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[self.feature_cols]
            y_true = test_data[target_col].values
            
            model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            return {'mape': mape, 'rmse': rmse}
        except Exception as e:
            print(f"Error in XGBoost backtest: {e}")
            return {'mape': None, 'rmse': None, 'error': str(e)}


def get_available_models() -> Dict[str, ForecastingModel]:
    """Get all available forecasting models."""
    models = {}
    
    # Always add XGBoost if available
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBoostModel()
    
    # Add Prophet if import succeeded (no dummy fit test!)
    if PROPHET_AVAILABLE:
        models['Prophet'] = ProphetModel()
    
    return models