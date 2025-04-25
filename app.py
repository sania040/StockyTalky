import streamlit as st
from src.db.crypto_db import createTable
from src.db.get_connection import get_db_connection
from src.fetch_and_store_crypto import CryptoDataFetcher
import os
import dotenv
import psycopg2 # Import for error handling

dotenv.load_dotenv()

API_KEY = os.getenv("COINMARKETCAP_API_KEY")
# symbols_to_fetch = ["BTC", "ETH"] # Initial fetch can be removed or kept based on preference

if 'dataFetcher' not in st.session_state:
    # Initialize fetcher only once and store in session state
    if not API_KEY:
        st.error("COINMARKETCAP_API_KEY not found in environment variables. Please check your .env file.")
        st.stop()
    st.session_state['dataFetcher'] = CryptoDataFetcher(api_key=API_KEY)

# Use the fetcher from session state
dataFetcher = st.session_state['dataFetcher']

if __name__ == "__main__":
    st.title('Crypto Investment Advisor')

    # Create the database and table if they don't exist
    # This should ideally run only once, but get_db_connection handles errors
    try:
        # createTable function in crypto_db.py doesn't take arguments
        createTable()
    except psycopg2.Error as e:
        st.error(f"Database connection or table creation failed: {e}")
        st.stop() # Stop the app if DB setup fails
    except Exception as e:
        st.error(f"An unexpected error occurred during DB setup: {e}")
        st.stop()

    # Optional: Initial data fetch (consider if needed on every run)
    # dataFetcher.process_symbols(symbols_to_fetch) # Uncomment if you want initial fetch

    # Define available symbols (can be dynamic or predefined)
    available_symbols = ["BTC", "ETH", "SOL", "XRP", "DOGE", "LTC"] # Example list

    # Dropdown for symbol selection
    selected_symbol = st.selectbox("Select Symbol:", available_symbols)

    # Button to fetch data
    if st.button(f"Fetch Data for {selected_symbol}"):
        if selected_symbol:
            with st.spinner(f"Fetching data for {selected_symbol}..."):
                try:
                    # Use the new method from CryptoDataFetcher
                    fetched_data = dataFetcher.fetch_data_for_symbol(selected_symbol)

                    if fetched_data and 'data' in fetched_data and selected_symbol in fetched_data['data']:
                        # Store fetched data and symbol in session state
                        st.session_state['fetched_data'] = fetched_data
                        st.session_state['fetched_symbol'] = selected_symbol
                        st.success(f"Data fetched successfully for {selected_symbol}!")
                        # Display fetched data
                        st.write("Fetched Data Preview:")
                        # Display the relevant part of the data
                        st.json(fetched_data['data'][selected_symbol])
                    elif fetched_data and 'status' in fetched_data and fetched_data['status']['error_message']:
                         st.error(f"API Error for {selected_symbol}: {fetched_data['status']['error_message']}")
                    else:
                        st.error(f"Could not fetch valid data for {selected_symbol}. API response might be empty or malformed.")
                        # Optionally display the raw response for debugging
                        st.write("Raw API Response:")
                        st.json(fetched_data)

                except Exception as e:
                    st.error(f"An error occurred during fetching: {e}")
        else:
            st.warning("Please select a symbol.")

    # Check if data has been fetched and show "Add to DB" button
    if 'fetched_data' in st.session_state and 'fetched_symbol' in st.session_state:
        # Ensure the fetched symbol matches the currently selected one for clarity
        if st.session_state['fetched_symbol'] == selected_symbol:
            st.write(f"Data for **{st.session_state['fetched_symbol']}** is ready.")
            if st.button(f"Add {st.session_state['fetched_symbol']} Data to DB"):
                with st.spinner(f"Adding {st.session_state['fetched_symbol']} data to database..."):
                    conn = None # Initialize connection variable
                    try:
                        # Retrieve data from session state
                        data_to_store = st.session_state['fetched_data']
                        symbol_to_store = st.session_state['fetched_symbol']

                        # Get a fresh database connection for this operation
                        conn = get_db_connection()
                        if conn:
                            # Use the new store_data method
                            success = dataFetcher.store_data(conn, data_to_store, symbol_to_store)

                            if success:
                                st.success(f"Data for {symbol_to_store} added to the database successfully!")
                                # Clear session state after successful storage
                                del st.session_state['fetched_data']
                                del st.session_state['fetched_symbol']
                                st.rerun() # Rerun to update UI state
                            else:
                                # store_data method prints specific errors, show a general one here
                                st.error(f"Failed to add data for {symbol_to_store} to the database. Check logs.")
                        else:
                            st.error("Failed to establish database connection.")

                    except psycopg2.Error as db_err:
                        st.error(f"A database error occurred: {db_err}")
                    except Exception as e:
                        st.error(f"An error occurred while adding data to DB: {e}")
                    finally:
                        # Ensure the connection is closed if it was opened
                        if conn and not conn.closed:
                            conn.close()
                            print("Database connection closed after storing attempt.")
        else:
            # Data for a different symbol is in session state, clear it or inform user
            st.info(f"Fetched data for {st.session_state['fetched_symbol']} is available. Select it again or fetch new data to store.")


    # Optional: Add a way to view data already in the DB
    if st.button("Show Data from DB"):
       conn = None
       try:
           conn = get_db_connection()
           if conn:
               # Add logic to query and display data using pandas and st.dataframe
               # Example:
               # import pandas as pd
               # query = "SELECT * FROM crypto_prices ORDER BY timestamp DESC LIMIT 10;"
               # df = pd.read_sql(query, conn)
               # st.dataframe(df)
               st.write("DB data display logic not yet implemented.") # Placeholder
           else:
               st.error("Failed to connect to DB to show data.")
       except psycopg2.Error as db_err:
           st.error(f"Database error while fetching data: {db_err}")
       except Exception as e:
           st.error(f"An error occurred: {e}")
       finally:
           if conn and not conn.closed:
               conn.close()

