import streamlit as st
from src.db.crypto_db import createTable
from src.db.get_connection import get_db_connection
from src.fetch_and_store_crypto import CryptoDataFetcher
import os
import dotenv
import psycopg2
import pandas as pd  # Ensure this is here for DB viewing

dotenv.load_dotenv()

API_KEY = os.getenv("COINMARKETCAP_API_KEY")

if 'dataFetcher' not in st.session_state:
    if not API_KEY:
        st.error("COINMARKETCAP_API_KEY not found in environment variables. Please check your .env file.")
        st.stop()
    st.session_state['dataFetcher'] = CryptoDataFetcher(api_key=API_KEY)

dataFetcher = st.session_state['dataFetcher']

st.title('Crypto Investment Advisor')

# Database setup
try:
    createTable()
except psycopg2.Error as e:
    st.error(f"Database connection or table creation failed: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during DB setup: {e}")
    st.stop()

available_symbols = ["BTC", "ETH", "SOL", "XRP", "DOGE", "LTC"]
selected_symbol = st.selectbox("Select Symbol:", available_symbols)

if st.button(f"Fetch Data for {selected_symbol}"):
    if selected_symbol:
        with st.spinner(f"Fetching data for {selected_symbol}..."):
            try:
                fetched_data = dataFetcher.fetch_data_for_symbol(selected_symbol)

                if fetched_data and 'data' in fetched_data and selected_symbol in fetched_data['data']:
                    st.session_state['fetched_data'] = fetched_data
                    st.session_state['fetched_symbol'] = selected_symbol
                    st.success(f"Data fetched successfully for {selected_symbol}!")
                    st.write("Fetched Data Preview:")
                    st.json(fetched_data['data'][selected_symbol])
                elif fetched_data and 'status' in fetched_data and fetched_data['status']['error_message']:
                    st.error(f"API Error for {selected_symbol}: {fetched_data['status']['error_message']}")
                else:
                    st.error(f"Could not fetch valid data for {selected_symbol}. API response might be empty or malformed.")
                    st.write("Raw API Response:")
                    st.json(fetched_data)

            except Exception as e:
                st.error(f"An error occurred during fetching: {e}")
    else:
        st.warning("Please select a symbol.")

# Store to DB
if 'fetched_data' in st.session_state and 'fetched_symbol' in st.session_state:
    if st.session_state['fetched_symbol'] == selected_symbol:
        st.write(f"Data for **{st.session_state['fetched_symbol']}** is ready.")
        if st.button(f"Add {st.session_state['fetched_symbol']} Data to DB"):
            with st.spinner(f"Adding {st.session_state['fetched_symbol']} data to database..."):
                conn = None
                try:
                    data_to_store = st.session_state['fetched_data']
                    symbol_to_store = st.session_state['fetched_symbol']

                    conn = get_db_connection()
                    if conn:
                        success = dataFetcher.store_data(conn, data_to_store, symbol_to_store)
                        if success:
                            st.success(f"Data for {symbol_to_store} added to the database successfully!")
                            del st.session_state['fetched_data']
                            del st.session_state['fetched_symbol']
                            st.rerun()
                        else:
                            st.error(f"Failed to add data for {symbol_to_store} to the database. Check logs.")
                    else:
                        st.error("Failed to establish database connection.")
                except psycopg2.Error as db_err:
                    st.error(f"A database error occurred: {db_err}")
                except Exception as e:
                    st.error(f"An error occurred while adding data to DB: {e}")
                finally:
                    if conn and not conn.closed:
                        conn.close()
                        print("Database connection closed after storing attempt.")
    else:
        st.info(f"Fetched data for {st.session_state['fetched_symbol']} is available. Select it again or fetch new data to store.")

# View from DB
if st.button("Show Data from DB"):
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            query = "SELECT * FROM crypto_prices ORDER BY timestamp DESC LIMIT 10;"
            df = pd.read_sql(query, conn)
            if df.empty:
                st.info("No data found in the database.")
            else:
                st.dataframe(df)
        else:
            st.error("Failed to connect to DB to show data.")
    except psycopg2.Error as db_err:
        st.error(f"Database error while fetching data: {db_err}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        if conn and not conn.closed:
            conn.close()
