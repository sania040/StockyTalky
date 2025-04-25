import streamlit as st
from src.db.crypto_db import createTable
from src.db.get_connection import get_db_connection
from src.fetch_and_store_crypto import agent
import os, dotenv, psycopg2, pandas as pd

dotenv.load_dotenv()
st.title("Crypto Investment Advisor")

# Ensure DB & table exist
try:
    createTable()
except Exception as e:
    st.error(f"DB setup failed: {e}")
    st.stop()

symbols = ["BTC","ETH","SOL","XRP","DOGE","LTC"]
sel = st.selectbox("Select Symbol:", symbols)

# Fetch
if st.button(f"Fetch Data for {sel}"):
    with st.spinner("Fetching…"):
        res = agent.invoke({"fetch": sel})
        fetched = res["fetch"]
        st.session_state["fetched_data"] = fetched
        st.session_state["fetched_symbol"] = sel
        st.success("Fetched ✔")
        st.json(fetched["data"][sel])

# Store
if st.session_state.get("fetched_symbol") == sel:
    if st.button(f"Add {sel} Data to DB"):
        with st.spinner("Storing…"):
            fetched = st.session_state["fetched_data"]
            out = agent.invoke({"store": {"api_data": fetched, "symbol": sel}})
            status = out["store"]
            if status == "stored":
                st.success("Stored ✔")
                del st.session_state["fetched_data"]
                del st.session_state["fetched_symbol"]
            else:
                st.error("Store failed 😢")

# View last 10 rows
if st.button("Show Data from DB"):
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM crypto_prices ORDER BY timestamp DESC LIMIT 10;", conn)
    conn.close()
    if df.empty:
        st.info("No rows yet.")
    else:
        st.dataframe(df)
