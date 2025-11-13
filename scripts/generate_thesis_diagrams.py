"""
Generate thesis diagrams (architecture, ER, UML, flows) as PNGs (or DOT fallbacks).

Outputs to thesis_assets/diagrams:
- fig_3_1_system_architecture.png
- fig_3_2_deployment_physical.png
- fig_3_3_er_diagram.png
- fig_3_4_use_case.png
- fig_3_5_class_diagram.png
- fig_3_6_sequence_forecast.png
- fig_3_7_sequence_dashboard.png
- fig_4_1_dev_flow.png
- fig_4_2_ingestion_mapping.png
- fig_4_3_db_access.png
- fig_4_4_dashboard_layout.png
- fig_4_8_prophet_steps.png
- fig_4_9_backtest_timeline.png (matplotlib timeline)
- fig_6_1_summary.png
- fig_6_2_limitations.png
- fig_6_3_roadmap.png

If Graphviz is not installed, DOT files with the same base names will be created instead (e.g., fig_3_1_system_architecture.dot).
"""

from pathlib import Path
import sys
import os

# Prefer PNG via graphviz; fall back to DOT files if graphviz missing
GRAPHVIZ_AVAILABLE = False
try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except Exception:
    GRAPHVIZ_AVAILABLE = False

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "thesis_assets" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _render_or_write(dot: Digraph, filename: str):
    base = OUT_DIR / filename
    stem = base.stem
    if GRAPHVIZ_AVAILABLE:
        # Try to render PNG directly using system Graphviz (dot)
        try:
            dot.format = "png"
            # graphviz appends extension; specify directory without suffix
            dot.render(filename=str(OUT_DIR / stem), cleanup=True)
            print(f"Rendered: {stem}.png")
            return
        except Exception as e:
            # Fallback: write DOT file so user can render later after fixing PATH
            with open(OUT_DIR / f"{stem}.dot", "w", encoding="utf-8") as f:
                f.write(dot.source)
            print(f"Graphviz PNG render failed ({e.__class__.__name__}). Wrote DOT: {stem}.dot")
            return
    # If python-graphviz is not available, always write DOT
    with open(OUT_DIR / f"{stem}.dot", "w", encoding="utf-8") as f:
        f.write(dot.source)
    print(f"Graphviz library not available. Wrote DOT: {stem}.dot")


# --- Chapter 3 diagrams ---

def fig_3_1_system_architecture():
    dot = Digraph("system_architecture", graph_attr={"rankdir": "LR"})
    dot.node("user", "User (Browser)", shape="oval")
    dot.node("ui", "Streamlit App\n(Dashboard + ML)", shape="box")
    dot.node("ml", "Forecasting Module\n(Prophet/XGBoost)", shape="box")
    dot.node("db", "Database\n(crypto_prices)", shape="cylinder")
    dot.node("api", "CoinMarketCap API", shape="box")

    dot.edges([("user", "ui"), ("ui", "user")])
    dot.edge("ui", "db", label="read/write")
    dot.edge("ml", "db", label="read")
    dot.edge("ui", "ml", label="select model / forecast")
    dot.edge("ui", "api", label="fetch (collector)")
    dot.edge("api", "ui", label="JSON quotes")
    _render_or_write(dot, "fig_3_1_system_architecture")


def fig_3_2_deployment_physical():
    dot = Digraph("deployment", graph_attr={"rankdir": "LR"})
    with dot.subgraph(name="cluster_host") as c:
        c.attr(label="Host: Python + Streamlit", style="rounded")
        c.node("app", "App Service")
        c.node("svc_db", "DB Client")

    dot.node("browser", "User Browser", shape="oval")
    dot.node("db", "PostgreSQL / Supabase", shape="cylinder")
    dot.node("internet", "Internet", shape="cloud")
    dot.node("cmc", "CoinMarketCap API", shape="box")

    dot.edge("browser", "app")
    dot.edge("app", "svc_db")
    dot.edge("svc_db", "db")
    dot.edges([("app", "internet"), ("internet", "cmc")])
    _render_or_write(dot, "fig_3_2_deployment_physical")


def fig_3_3_er_diagram():
    dot = Digraph("er", graph_attr={"rankdir": "TB"})
    dot.node("crypto_prices", "crypto_prices|PK: (symbol, timestamp)\nprice_usd: NUM\nvolume_24h_usd: NUM\npercent_change_24h: NUM?\nmarket_cap_usd: NUM\ntimestamp: TIMESTAMP", shape="record")
    _render_or_write(dot, "fig_3_3_er_diagram")


def fig_3_4_use_case():
    dot = Digraph("use_case", graph_attr={"rankdir": "LR"})
    dot.node("actor", "Viewer", shape="oval")
    with dot.subgraph(name="cluster_uc") as c:
        c.attr(label="Use Cases", style="dashed")
        c.node("uc1", "View Market Overview")
        c.node("uc2", "Explore Coin History")
        c.node("uc3", "Compare Coins")
        c.node("uc4", "View Correlations")
        c.node("uc5", "Generate Forecast")
        c.node("uc6", "View Prophet Components")
    for uc in ["uc1","uc2","uc3","uc4","uc5","uc6"]:
        dot.edge("actor", uc)
    _render_or_write(dot, "fig_3_4_use_case")


def fig_3_5_class_diagram():
    dot = Digraph("class", graph_attr={"rankdir": "TB"})
    dot.node("fetcher", "CryptoDataFetcher|+fetch_data_for_symbols()\n+prepare_insert_data()\n+store_data()", shape="record")
    dot.node("dbutil", "DB Utils|+execute_query()\n+execute_and_commit()", shape="record")
    dot.node("base", "ForecastingModel|+fit()\n+predict()\n+backtest()", shape="record")
    dot.node("prophet", "ProphetModel: ForecastingModel", shape="record")
    dot.node("xgb", "XGBoostModel: ForecastingModel", shape="record")
    dot.node("pages", "Pages|dashboard.py\nml_recommendations.py", shape="record")
    dot.edge("prophet", "base", arrowhead="empty")
    dot.edge("xgb", "base", arrowhead="empty")
    dot.edge("pages", "dbutil")
    dot.edge("pages", "prophet")
    dot.edge("pages", "xgb")
    dot.edge("fetcher", "dbutil")
    _render_or_write(dot, "fig_3_5_class_diagram")


def fig_3_6_sequence_forecast():
    dot = Digraph("seq_forecast", graph_attr={"rankdir": "LR"})
    dot.node("u", "User", shape="plaintext")
    dot.node("ui", "UI", shape="box")
    dot.node("db", "DB", shape="cylinder")
    dot.node("m", "Model", shape="box")
    dot.edge("u", "ui", label="Select symbol/model/days")
    dot.edge("ui", "db", label="Load history")
    dot.edge("ui", "m", label="fit()")
    dot.edge("m", "ui", label="model ready")
    dot.edge("ui", "m", label="predict(days)")
    dot.edge("m", "ui", label="forecast + bounds")
    dot.edge("ui", "u", label="Render chart")
    _render_or_write(dot, "fig_3_6_sequence_forecast")


def fig_3_7_sequence_dashboard():
    dot = Digraph("seq_dashboard", graph_attr={"rankdir": "LR"})
    dot.node("u", "User", shape="plaintext")
    dot.node("ui", "UI", shape="box")
    dot.node("db", "DB", shape="cylinder")
    dot.edge("u", "ui", label="Open Dashboard")
    dot.edge("ui", "db", label="Load latest per symbol")
    dot.edge("db", "ui", label="Rows")
    dot.edge("ui", "u", label="KPIs, movers, charts")
    _render_or_write(dot, "fig_3_7_sequence_dashboard")


# --- Chapter 4 diagrams ---

def fig_4_1_dev_flow():
    dot = Digraph("dev_flow", graph_attr={"rankdir": "LR"})
    dot.node("api", "API\n(CoinMarketCap)")
    dot.node("ingest", "Ingestion\n(fetch & map)")
    dot.node("db", "DB\n(crypto_prices)", shape="cylinder")
    dot.node("dash", "Dashboard")
    dot.node("ml", "ML Insights\n+ Forecasts")
    dot.edges([("api","ingest"),("ingest","db"),("db","dash"),("db","ml")])
    _render_or_write(dot, "fig_4_1_dev_flow")


def fig_4_2_ingestion_mapping():
    dot = Digraph("ingestion_map", graph_attr={"rankdir": "LR"})
    dot.node("json", "API JSON|symbol\nquote.USD.price\nquote.USD.volume_24h\nquote.USD.percent_change_24h\nquote.USD.market_cap", shape="record")
    dot.node("table", "crypto_prices|symbol\nprice_usd\nvolume_24h_usd\npercent_change_24h\nmarket_cap_usd\ntimestamp", shape="record")
    dot.edge("json", "table", label="field mapping + now() timestamp")
    _render_or_write(dot, "fig_4_2_ingestion_mapping")


def fig_4_3_db_access():
    dot = Digraph("db_access", graph_attr={"rankdir": "TB"})
    dot.node("conn", "get_db_connection()")
    dot.node("read", "execute_query(conn, sql)")
    dot.node("write", "execute_and_commit(conn, sql)")
    dot.edge("conn", "read")
    dot.edge("conn", "write")
    _render_or_write(dot, "fig_4_3_db_access")


def fig_4_4_dashboard_layout():
    dot = Digraph("dash_layout", graph_attr={"rankdir": "TB"})
    with dot.subgraph(name="cluster_top") as c:
        c.attr(label="Top Row: KPIs")
        c.node("k1","Market Cap")
        c.node("k2","24h Volume")
        c.node("k3","Weighted 24h Change")
    with dot.subgraph(name="cluster_mid") as c2:
        c2.attr(label="Middle: Movers + Charts")
        c2.node("gainers","Top Gainers")
        c2.node("losers","Top Losers")
        c2.node("candle","Candlestick")
        c2.node("pie","Dominance Pie")
    with dot.subgraph(name="cluster_bottom") as c3:
        c3.attr(label="Bottom: Screener Table")
        c3.node("screen","Screener")
    _render_or_write(dot, "fig_4_4_dashboard_layout")


def fig_4_8_prophet_steps():
    dot = Digraph("prophet_steps", graph_attr={"rankdir": "LR"})
    dot.node("prep", "Prep: rename to ds/y\nparse datetime\nremove tz")
    dot.node("fit", "Fit Prophet\n(seasonality off)")
    dot.node("future", "Make future dataframe")
    dot.node("pred", "Predict & select tail")
    dot.edges([("prep","fit"),("fit","future"),("future","pred")])
    _render_or_write(dot, "fig_4_8_prophet_steps")


def fig_4_9_backtest_timeline():
    # Use matplotlib to draw a simple timeline bar
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.broken_barh([(0, 8), (8.5, 2)], (10, 9), facecolors=("#2a9d8f", "#e76f51"))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 30)
    ax.set_yticks([])
    ax.set_xticks([0, 8, 10])
    ax.set_xticklabels(["Train start", "Train end", "Test end"])
    ax.set_title("Backtest Window: Train vs Test")
    for spine in ["top","right","left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    png = OUT_DIR / "fig_4_9_backtest_timeline.png"
    fig.savefig(png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Rendered: {png.name}")


# --- Chapter 6 diagrams ---

def fig_6_1_summary():
    dot = Digraph("summary", graph_attr={"rankdir": "LR"})
    dot.node("collect", "Collect")
    dot.node("store", "Store")
    dot.node("analyze", "Analyze")
    dot.node("forecast", "Forecast")
    dot.node("evaluate", "Evaluate")
    dot.edges([("collect","store"),("store","analyze"),("analyze","forecast"),("forecast","evaluate")])
    _render_or_write(dot, "fig_6_1_summary")


def fig_6_2_limitations():
    dot = Digraph("limits", graph_attr={"rankdir": "TB"})
    dot.node("vol", "Volatility spikes")
    dot.node("src", "Single data source")
    dot.node("schema", "Simple schema")
    dot.node("exo", "No exogenous inputs")
    dot.node("bt", "Basic backtesting")
    _render_or_write(dot, "fig_6_2_limitations")


def fig_6_3_roadmap():
    dot = Digraph("roadmap", graph_attr={"rankdir": "LR"})
    dot.node("v1", "Now\n(Student Project)")
    dot.node("v2", "vNext\nAlerts + Portfolio")
    dot.node("v3", "Analytics+\nExogenous inputs")
    dot.node("v4", "Prod\nMonitoring + CI")
    dot.edges([("v1","v2"),("v2","v3"),("v3","v4")])
    _render_or_write(dot, "fig_6_3_roadmap")


def main():
    fig_3_1_system_architecture()
    fig_3_2_deployment_physical()
    fig_3_3_er_diagram()
    fig_3_4_use_case()
    fig_3_5_class_diagram()
    fig_3_6_sequence_forecast()
    fig_3_7_sequence_dashboard()
    fig_4_1_dev_flow()
    fig_4_2_ingestion_mapping()
    fig_4_3_db_access()
    fig_4_4_dashboard_layout()
    fig_4_8_prophet_steps()
    fig_4_9_backtest_timeline()
    fig_6_1_summary()
    fig_6_2_limitations()
    fig_6_3_roadmap()
    print("All diagrams generated (PNG if Graphviz is installed; DOT otherwise).")


if __name__ == "__main__":
    main()
