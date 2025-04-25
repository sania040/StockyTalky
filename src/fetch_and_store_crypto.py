import os
import requests
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
from src.db.get_connection import get_db_connection

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("COINMARKETCAP_API_KEY")

# CoinMarketCap endpoint
url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
# symbols = ["BTC", "ETH"] # Removed as symbols are passed dynamically

class CryptoDataFetcher:
    def __init__(self, api_key, api_url="https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"):
        """Initializes the CryptoDataFetcher."""
        self.api_key = api_key
        self.api_url = api_url
        self.conn = None
        self.cursor = None
        if not self.api_key:
            raise ValueError("API key not found. Please set COINMARKETCAP_API_KEY in your .env file.")

    def connect_db(self):
        """Establishes the database connection."""
        if not self.conn or self.conn.closed:
            try:
                # Assuming get_db_connection is available in the scope
                self.conn = get_db_connection()
                # Don't create cursor here, manage it per operation or method
                print("Database connection established.")
            except psycopg2.Error as e:
                print(f"Error connecting to database: {e}")
                self.conn = None
                raise # Re-raise the exception to signal connection failure

    def fetch_data(self, symbol):
        """Fetches data for a specific cryptocurrency symbol from the API."""
        params = {
            'symbol': symbol,
            'convert': 'USD'
        }
        headers = {
            'X-CMC_PRO_API_KEY': self.api_key
        }
        try:
            response = requests.get(self.api_url, headers=headers, params=params)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            print(f"Fetched Data for {symbol}")
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {symbol} from API: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during API fetch for {symbol}: {e}")
            return None

    # New method specifically for the Streamlit app's fetch button
    def fetch_data_for_symbol(self, symbol):
        """Fetches data for a single symbol, intended for direct use."""
        return self.fetch_data(symbol)

    def _prepare_insert_data(self, symbol, data):
        """Parses API data and prepares tuple for DB insertion."""
        if not data or 'data' not in data or symbol not in data['data']:
             print(f"No valid data structure received for {symbol}.")
             return None

        try:
            # Ensure data['data'][symbol] is accessed correctly
            crypto_data = data['data'].get(symbol)
            if not crypto_data:
                print(f"Symbol {symbol} not found in API response data.")
                return None

            quote = crypto_data['quote']['USD']
            price_usd = quote['price']
            market_cap_usd = quote['market_cap']
            volume_24h_usd = quote['volume_24h']
            # Use .get() for optional fields to avoid KeyErrors
            percent_change_24h = quote.get('percent_change_24h')
            # percent_change_1h = quote.get('percent_change_1h')
            # percent_change_7d = quote.get('percent_change_7d')
            # percent_change_30d = quote.get('percent_change_30d')
            # percent_change_60d = quote.get('percent_change_60d')
            # percent_change_90d = quote.get('percent_change_90d') # Check if API provides this
            # market_cap_dominance = quote.get('market_cap_dominance')
            # fully_diluted_market_cap = quote.get('fully_diluted_market_cap')
            volume_change_24h = quote.get('volume_change_24h') # Check if API provides this
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Return tuple matching the INSERT statement order
            # Ensure all columns from the INSERT statement are present
            return (
                symbol, price_usd, volume_24h_usd,
                percent_change_24h, market_cap_usd, timestamp,
            )
        except KeyError as e:
            print(f"Error parsing data structure for {symbol}: Missing key {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during data preparation for {symbol}: {e}")
            return None

    # New method specifically for the Streamlit app's store button
    def store_data(self, conn, api_data, symbol):
        """Prepares and stores data for a single symbol using a provided connection."""
        if not conn or conn.closed:
            print("Database connection is closed or invalid.")
            raise psycopg2.InterfaceError("Database connection is closed or invalid.")

        insert_tuple = self._prepare_insert_data(symbol, api_data)
        if not insert_tuple:
            print(f"Failed to prepare data for {symbol}. Not storing.")
            return False # Indicate failure

        cursor = None
        try:
            cursor = conn.cursor()
            # Correct the column name here
            cursor.execute("""
                INSERT INTO crypto_prices (
                    symbol, price_usd, volume_24h_usd,
                    percent_change_24h,
                    market_cap_usd,
                    timestamp
                )
                VALUES (%s, %s, %s, %s, %s, %s)
            """, insert_tuple)
            conn.commit() # Commit the single insertion
            print(f"Data for {symbol} stored successfully.")
            return True # Indicate success
        except psycopg2.Error as e:
            print(f"Database error storing data for {symbol}: {e}")
            if conn:
                conn.rollback() # Rollback the failed transaction
            return False # Indicate failure
        except Exception as e:
             print(f"Unexpected error during data storage for {symbol}: {e}")
             if conn:
                 conn.rollback()
             return False # Indicate failure
        finally:
            if cursor:
                cursor.close()
            # Do not close the connection here, it's managed by the caller (app.py)

    def process_symbols(self, symbols):
        """Fetches and stores data for a list of symbols."""
        conn = None # Manage connection within this method
        try:
            conn = get_db_connection() # Get a fresh connection
            if not conn:
                print("Cannot process symbols without a database connection.")
                return

            cursor = conn.cursor()
            all_success = True
            for symbol in symbols:
                api_data = self.fetch_data(symbol)
                if not api_data:
                    all_success = False
                    print(f"Failed to fetch data for {symbol}. Skipping storage.")
                    continue # Skip this symbol, try the next

                insert_tuple = self._prepare_insert_data(symbol, api_data)
                if not insert_tuple:
                    all_success = False
                    print(f"Failed to prepare data for {symbol}. Skipping storage.")
                    continue # Skip this symbol, try the next

                try:
                    # Execute the insert statement
                    # Correct the column name here as well
                    cursor.execute("""
                        INSERT INTO crypto_prices (
                            symbol, price_usd, volume_24h_usd, volume_change_24h,
                            percent_change_1h, percent_change_24h, percent_change_7d,
                            percent_change_30d, percent_change_60d, market_cap,
                            market_cap_dominance, fully_diluted_market_cap, timestamp, percent_change_90d
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, insert_tuple)
                    print(f"Data for {symbol} queued for insertion.")
                except psycopg2.Error as e:
                    print(f"Database error executing insert for {symbol}: {e}")
                    all_success = False
                    conn.rollback() # Rollback immediately on error for this symbol
                    # Decide if you want to break or continue with other symbols
                    # break # Option 1: Stop all processing on first DB error
                    continue # Option 2: Continue with next symbol after rollback
                except Exception as e:
                     print(f"Unexpected error during insert execution for {symbol}: {e}")
                     all_success = False
                     conn.rollback()
                     # break # Option 1
                     continue # Option 2

            # Commit only if all operations intended were successful (or handled)
            if all_success: # Or adjust logic based on whether you continue/break on error
                conn.commit()
                print("Data processing finished. Changes committed (if any).")
            else:
                print("Errors occurred during processing. Some data might not have been stored.")
                # Rollback might have already happened individually,
                # but an explicit rollback here ensures consistency if needed.
                # conn.rollback() # Uncomment if you didn't rollback individually

        except psycopg2.Error as e:
             print(f"A database error occurred during process_symbols setup: {e}")
             if conn and not conn.closed:
                 conn.rollback()
        except Exception as e:
             print(f"An unexpected error occurred during symbol processing: {e}")
             if conn and not conn.closed:
                 conn.rollback()
        finally:
            # Ensure connection used by this method is closed
            if cursor:
                cursor.close()
            if conn and not conn.closed:
                conn.close()
                print("Database connection closed for process_symbols.")


    def close_connection(self):
        """Closes the database cursor and connection if managed internally (less relevant now)."""
        # This method might be less needed if connections are managed per-method or externally
        if self.cursor:
            try:
                self.cursor.close()
            except psycopg2.Error as e:
                 print(f"Error closing cursor: {e}")
        if self.conn and not self.conn.closed:
            try:
                self.conn.close()
                print("Database connection closed.")
            except psycopg2.Error as e:
                 print(f"Error closing connection: {e}")
        self.cursor = None
        self.conn = None

# ... (rest of the file, including the __main__ block if needed for testing) ...
# Example __main__ block adjusted
if __name__ == "__main__":
    API_KEY = os.getenv("COINMARKETCAP_API_KEY")
    symbols_to_fetch = ["BTC", "ETH", "LTC"] # Example

    if not API_KEY:
        print("Error: COINMARKETCAP_API_KEY not found in environment variables.")
    else:
        try:
            fetcher = CryptoDataFetcher(api_key=API_KEY)
            # Test fetching single symbol
            # btc_data = fetcher.fetch_data_for_symbol("BTC")
            # if btc_data:
            #     print("\nFetched BTC data:")
            #     print(btc_data)
            #     # Test storing single symbol (requires DB connection)
            #     # conn = get_db_connection()
            #     # if conn:
            #     #     fetcher.store_data(conn, btc_data, "BTC")
            #     #     conn.close()

            # Test processing multiple symbols
            print(f"\nProcessing symbols: {symbols_to_fetch}")
            fetcher.process_symbols(symbols_to_fetch)
            print("\nData processing attempt finished.")

        except ValueError as e:
             print(f"Initialization failed: {e}")
        except Exception as e:
             print(f"A critical error occurred: {e}")

# ... (Remove old commented-out procedural code) ...
