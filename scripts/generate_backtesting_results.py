"""
Compute backtesting metrics (MAPE, RMSE) for available models and top N coins.
Writes results to thesis_assets/tables/table_5_2_backtesting_results.csv
"""

from pathlib import Path
import csv
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / 'thesis_assets' / 'tables' / 'table_5_2_backtesting_results.csv'

import sys
sys.path.insert(0, r"f:\code\StockyTalky")
from src.db.get_connection import get_db_connection
from src.db.query_utils import execute_query
from src.ml.forecasting import get_available_models

TOP_N = 5
TEST_DAYS = 7

# utility: load latest table to pick top symbols
conn = get_db_connection()
query = """
WITH latest_prices AS (
    SELECT *, ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY timestamp DESC) as rn
    FROM crypto_prices
)
SELECT symbol, price_usd, market_cap_usd, timestamp
FROM latest_prices
WHERE rn = 1
ORDER BY market_cap_usd DESC
LIMIT %s;
"""

symbols_df = execute_query(conn, query, params=(TOP_N,))
if symbols_df is None or symbols_df.empty:
    print("No symbols found. Exiting.")
    exit(1)

symbols = list(symbols_df['symbol'].values)
print(f"Top symbols: {symbols}")

# helper to load historical
def load_history(symbol: str) -> pd.DataFrame:
    q = "SELECT timestamp, price_usd FROM crypto_prices WHERE symbol = %s ORDER BY timestamp ASC;"
    df = execute_query(conn, q, params=(symbol,))
    if df is None or df.empty:
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

models = get_available_models()

rows = []
for sym in symbols:
    df = load_history(sym)
    if df.empty or len(df) < 20:
        print(f"Skipping {sym}: insufficient data ({len(df)} rows)")
        continue
    for name, model in models.items():
        try:
            res = model.backtest(df, test_days=TEST_DAYS)
            if not res or res.get('data_insufficient'):
                rows.append({
                    'coin': sym,
                    'model': name,
                    'test_days': TEST_DAYS,
                    'test_start': '',
                    'test_end': '',
                    'train_rows': '',
                    'test_rows': '',
                    'dataset_length': len(df),
                    'mape(%)': res.get('mape'),
                    'rmse(USD)': res.get('rmse'),
                    'notes': 'data_insufficient' if res.get('data_insufficient') else res.get('error', '')
                })
            else:
                # estimate train and test rows
                train_rows = len(df) - TEST_DAYS
                test_rows = TEST_DAYS
                test_start = df['timestamp'].iloc[-TEST_DAYS].strftime('%Y-%m-%d')
                test_end = df['timestamp'].iloc[-1].strftime('%Y-%m-%d')
                rows.append({
                    'coin': sym,
                    'model': name,
                    'test_days': TEST_DAYS,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_rows': train_rows,
                    'test_rows': test_rows,
                    'dataset_length': len(df),
                    'mape(%)': res.get('mape'),
                    'rmse(USD)': res.get('rmse'),
                    'notes': ''
                })
        except Exception as e:
            rows.append({
                'coin': sym,
                'model': name,
                'test_days': TEST_DAYS,
                'test_start': '',
                'test_end': '',
                'train_rows': '',
                'test_rows': '',
                'dataset_length': len(df),
                'mape(%)': None,
                'rmse(USD)': None,
                'notes': str(e)
            })

# write CSV
headers = ['coin','model','test_days','test_start','test_end','train_rows','test_rows','dataset_length','mape(%)','rmse(USD)','notes']
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Wrote backtesting CSV: {OUT_CSV}")
