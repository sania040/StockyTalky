"""
Generate thesis-ready plots for StockyTalky and save them as PNG images.

What this script produces (file names suggest chapter.figure slots):
- fig_5_2_kpis_top_marketcap.png: Top market cap bar (proxy KPI visual)
- fig_5_3_daily_movers.png: Top 5 gainers and top 5 losers (bar)
- fig_5_4_price_line_{SYMBOL}.png: Daily price line for a selected symbol (candlestick alt)
- fig_5_5_dominance_pie.png: Market dominance (Top 10 + Others)
- fig_5_7_normalized_growth.png: Multi-coin normalized growth comparison
- fig_5_8_correlation_heatmap.png: Correlation heatmap of daily returns
- fig_5_9_forecast_overlay_{SYMBOL}.png: Forecast overlay (historical + forecast + bounds)

How to run (set your DB and API env vars first):
- Open a terminal in repo root and run this module with Python. The script uses existing DB utils.
- You can edit DEFAULT_SYMBOL and FORECAST_DAYS below.

Note: If forecasting models are unavailable, it falls back to a simple moving-average based naïve forecast.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Make sure repo root is importable for `src.*`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.get_connection import get_db_connection
from src.db.query_utils import execute_query

# Optional forecasting imports
MODEL_FACTORY_AVAILABLE = False
try:
    from src.ml.forecasting import get_available_models
    MODEL_FACTORY_AVAILABLE = True
except Exception:
    MODEL_FACTORY_AVAILABLE = False

OUTPUT_DIR = ROOT / "thesis_assets" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Config ---
DEFAULT_SYMBOL: str = os.getenv("THESIS_FIG_SYMBOL", "BTC")
FORECAST_DAYS: int = int(os.getenv("THESIS_FORECAST_DAYS", "30"))
TOP_N: int = int(os.getenv("THESIS_TOP_N", "10"))

sns.set_context("talk")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


def _safe_save(fig: plt.Figure, name: str):
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def load_latest_data(conn) -> pd.DataFrame:
    query = """
    WITH latest_prices AS (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY symbol ORDER BY timestamp DESC) as rn
        FROM crypto_prices
    )
    SELECT symbol, price_usd, volume_24h_usd, percent_change_24h, market_cap_usd, timestamp
    FROM latest_prices
    WHERE rn = 1;
    """
    df = execute_query(conn, query)
    return df if df is not None else pd.DataFrame()


def load_historical_data(conn, symbol: str) -> pd.DataFrame:
    query = """
    SELECT timestamp, price_usd
    FROM crypto_prices
    WHERE symbol = %s
    ORDER BY timestamp ASC;
    """
    df = execute_query(conn, query, params=(symbol,))
    if df is None or df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_all_prices(conn) -> pd.DataFrame:
    query = "SELECT timestamp, symbol, price_usd FROM crypto_prices ORDER BY timestamp ASC;"
    df = execute_query(conn, query)
    if df is None or df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# --- Figure generators ---

def fig_top_marketcap_bar(latest_df: pd.DataFrame):
    if latest_df.empty:
        return
    top = latest_df.nlargest(TOP_N, "market_cap_usd").copy()
    top = top.sort_values("market_cap_usd", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top["symbol"], top["market_cap_usd"] / 1e9, color="#2a9d8f")
    ax.set_xlabel("Market Cap (B USD)")
    ax.set_title("Top Market Cap (Proxy KPI)")
    ax.bar_label(bars, fmt="{:.1f}")
    _safe_save(fig, "fig_5_2_kpis_top_marketcap.png")


def fig_daily_movers(latest_df: pd.DataFrame):
    if latest_df.empty or "percent_change_24h" not in latest_df.columns:
        return
    gainers = latest_df.nlargest(5, "percent_change_24h").copy()
    losers = latest_df.nsmallest(5, "percent_change_24h").copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    sns.barplot(ax=axes[0], data=gainers, x="percent_change_24h", y="symbol", palette="Greens_r")
    axes[0].set_title("Top 5 Gainers (24h)")
    axes[0].set_xlabel("% Change (24h)")
    axes[0].set_ylabel("")

    sns.barplot(ax=axes[1], data=losers, x="percent_change_24h", y="symbol", palette="Reds")
    axes[1].set_title("Top 5 Losers (24h)")
    axes[1].set_xlabel("% Change (24h)")
    axes[1].set_ylabel("")

    fig.suptitle("Daily Market Movers", y=1.02)
    _safe_save(fig, "fig_5_3_daily_movers.png")


def fig_price_line(conn, symbol: str):
    df = load_historical_data(conn, symbol)
    if df.empty:
        return
    daily = df.set_index("timestamp")["price_usd"].resample("D").last().dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily.index, daily.values, color="#264653")
    ax.set_title(f"{symbol}/USD Daily Close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    _safe_save(fig, f"fig_5_4_price_line_{symbol}.png")


def fig_dominance_pie(latest_df: pd.DataFrame):
    if latest_df.empty:
        return
    top10 = latest_df.nlargest(10, "market_cap_usd").copy()
    others = latest_df[~latest_df["symbol"].isin(top10["symbol"])]["market_cap_usd"].sum()
    labels = list(top10["symbol"]) + (["Others"] if others > 0 else [])
    sizes = list(top10["market_cap_usd"]) + ([others] if others > 0 else [])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title("Market Dominance (Top 10 + Others)")
    _safe_save(fig, "fig_5_5_dominance_pie.png")


def fig_normalized_growth(all_prices: pd.DataFrame, symbols: List[str]):
    if all_prices.empty:
        return
    subset = all_prices[all_prices["symbol"].isin(symbols)].copy()
    if subset.empty:
        return
    # Common start timestamp across selected symbols
    common_start = subset.groupby("symbol")["timestamp"].min().max()
    subset = subset[subset["timestamp"] >= common_start]

    norm_frames = []
    for sym in symbols:
        s = subset[subset["symbol"] == sym].sort_values("timestamp")
        if s.empty:
            continue
        base = s["price_usd"].iloc[0]
        if base == 0:
            continue
        s = s.assign(normalized=lambda x: x["price_usd"] / base * 100.0)
        norm_frames.append(s)

    if not norm_frames:
        return
    norm = pd.concat(norm_frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for sym, grp in norm.groupby("symbol"):
        ax.plot(grp["timestamp"], grp["normalized"], label=sym)
    ax.set_title("Normalized Growth (Baseline = 100)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth (Baseline 100)")
    ax.legend(ncol=2, fontsize=9)
    _safe_save(fig, "fig_5_7_normalized_growth.png")


def fig_correlation_heatmap(all_prices: pd.DataFrame):
    if all_prices.empty:
        return
    pivot = all_prices.pivot_table(index="timestamp", columns="symbol", values="price_usd")
    returns = pivot.pct_change().dropna(how="all", axis=0)
    returns = returns.dropna(how="all", axis=1)
    if returns.shape[1] < 2:
        return
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
    ax.set_title("Correlation of Daily Returns")
    _safe_save(fig, "fig_5_8_correlation_heatmap.png")


def _naive_forecast(series: pd.Series, periods: int) -> pd.Series:
    # Simple moving average of last 7 points as the next value, iterated forward
    window = min(7, len(series))
    hist = series.copy()
    preds = []
    last_vals = hist.iloc[-window:]
    for _ in range(periods):
        preds.append(float(last_vals.mean()))
        last_vals = pd.concat([last_vals.iloc[1:], pd.Series([preds[-1]])], ignore_index=True)
    return pd.Series(preds)


def fig_forecast_overlay(conn, symbol: str, days: int = 30):
    df = load_historical_data(conn, symbol)
    if df.empty or len(df) < 10:
        return
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Try to use project models if available, else fallback to naïve
    future_dates = pd.date_range(df["timestamp"].iloc[-1] + pd.Timedelta(days=1), periods=days, freq="D")

    forecast_vals = None
    lower = None
    upper = None
    model_name = "Naïve-MA7"

    if MODEL_FACTORY_AVAILABLE:
        try:
            models = get_available_models()
            # Prefer Prophet if available, else XGBoost, else naive
            preferred = ["Prophet", "XGBoost"]
            chosen = None
            for m in preferred:
                if m in models:
                    chosen = models[m]
                    model_name = m
                    break
            if chosen is not None and chosen.fit(df[["timestamp", "price_usd"]].copy()):
                fdf = chosen.predict(days)
                if not fdf.empty:
                    forecast_vals = fdf["forecast"].values
                    lower = fdf.get("lower_bound", pd.Series(np.nan, index=fdf.index)).values
                    upper = fdf.get("upper_bound", pd.Series(np.nan, index=fdf.index)).values
        except Exception as e:
            print(f"Model forecast failed, falling back to naïve: {e}")

    if forecast_vals is None:
        # Naïve fallback
        forecast_series = _naive_forecast(df["price_usd"], days)
        forecast_vals = forecast_series.values
        lower = forecast_series.values * 0.95
        upper = forecast_series.values * 1.05

    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot last 120 days of history for context
    hist = df.tail(120)
    ax.plot(hist["timestamp"], hist["price_usd"], label="Historical", color="#1d3557")
    ax.plot(future_dates, forecast_vals, label=f"Forecast ({model_name})", color="#e76f51", linestyle="--")
    ax.fill_between(future_dates, lower, upper, color="#e76f51", alpha=0.15, label="Bounds")
    ax.set_title(f"Forecast Overlay for {symbol}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    _safe_save(fig, f"fig_5_9_forecast_overlay_{symbol}.png")


def main():
    warnings.filterwarnings("ignore")
    try:
        conn = get_db_connection()
    except Exception as e:
        print(f"Could not connect to DB. Please set env vars. Error: {e}")
        return

    latest = load_latest_data(conn)
    all_prices = load_all_prices(conn)

    # Choose defaults from data if available
    symbol = DEFAULT_SYMBOL
    if latest is not None and not latest.empty and symbol not in latest["symbol"].values:
        symbol = latest.sort_values("market_cap_usd", ascending=False).iloc[0]["symbol"]

    # Generate figures
    fig_top_marketcap_bar(latest)
    fig_daily_movers(latest)
    fig_price_line(conn, symbol)
    fig_dominance_pie(latest)

    symbols_for_norm: List[str] = []
    if latest is not None and not latest.empty:
        symbols_for_norm = list(latest.sort_values("market_cap_usd", ascending=False)["symbol"].head(5).values)
    else:
        # fallback if latest is empty but all_prices exists
        if not all_prices.empty:
            symbols_for_norm = list(all_prices["symbol"].unique())[:5]

    if symbols_for_norm:
        fig_normalized_growth(all_prices, symbols_for_norm)

    fig_correlation_heatmap(all_prices)
    fig_forecast_overlay(conn, symbol, days=FORECAST_DAYS)

    print("All requested figures generated (where data was available).")


if __name__ == "__main__":
    main()
