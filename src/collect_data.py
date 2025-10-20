import time
import argparse
import os
from dotenv import load_dotenv
import pandas as pd
from src.api.crypto_fetcher import CryptoDataFetcher
from src.db.get_connection import get_db_connection
from src.db.query_utils import execute_query, execute_and_commit

class DataCollector:
    def __init__(self, symbols):
        self.fetcher = CryptoDataFetcher()
        self.symbols = symbols

    def collect_data(self, max_rows):
        """Fetch and store data for all symbols in a single run."""
        conn = None
        try:
            conn = get_db_connection()
            print("Fetching data for all symbols in a single API call...")
            
            # --- FIX #1: Call the correct function for batching ---
            # Make ONE API call for all symbols BEFORE the loop
            all_api_data = self.fetcher.fetch_data_for_symbols(self.symbols)
            print("Successfully fetched batch data.")

            # Now, loop through each symbol to process its data from the response
            for symbol in self.symbols:
                try:
                    # --- FIX #2: Check if the data for this symbol exists in our batch response ---
                    if symbol not in all_api_data.get('data', {}):
                        print(f"⚠️ Data for {symbol} not found in API response. Skipping.")
                        continue

                    # The rest of the logic uses the 'all_api_data' variable
                    count_df = execute_query(conn, "SELECT COUNT(*) FROM crypto_prices WHERE symbol = %s", (symbol,))
                    if count_df.empty:
                        print(f"⚠️ Could not get row count for {symbol}. Skipping.")
                        continue
                    row_count = count_df.iloc[0, 0]

                    # --- FIX #3: REMOVED the redundant API call inside the loop ---
                    # No new API call is needed here!

                    if row_count >= max_rows:
                        print(f"Row limit reached for {symbol}. Updating oldest record...")
                        oldest_id_df = execute_query(conn, """
                            SELECT id FROM crypto_prices WHERE symbol = %s ORDER BY timestamp ASC LIMIT 1
                        """, (symbol,))
                        if oldest_id_df.empty:
                            print(f"⚠️ Could not find oldest record for {symbol}. Skipping update.")
                            continue
                        
                        oldest_id = oldest_id_df.iloc[0, 0]
                        # Use the data we already fetched
                        quote = all_api_data['data'][symbol]['quote']['USD']
                        ts = time.strftime("%Y-%m-%d %H:%M:%S")

                        execute_and_commit(conn, """
                            UPDATE crypto_prices SET price_usd = %s, volume_2h_usd = %s, percent_change_24h = %s,
                                market_cap_usd = %s, timestamp = %s WHERE id = %s
                        """, (
                            quote['price'], quote['volume_24h'], quote.get('percent_change_24h'),
                            quote['market_cap'], ts, oldest_id
                        ))
                        print(f"Updated oldest record for {symbol}")
                    else:
                        print(f"Storing new data for {symbol}...")
                        # Pass the full dataset; the function will extract the relevant symbol
                        self.fetcher.store_data(conn, all_api_data, symbol)
                        print(f"Successfully stored new data for {symbol}")

                except Exception as e:
                    print(f"Error processing {symbol}: {e}")

        except Exception as e:
            print(f"A critical error occurred during data collection: {e}")
        finally:
            if conn and not conn.closed:
                conn.close()
                print("\nDatabase connection closed.")

def main():
    """Main function to parse arguments and run the data collection process once."""
    load_dotenv() # Load environment variables from .env if present for local testing
    print("--- 🕵️ Debugging Secrets ---")
    db_host = os.getenv("DB_HOST")
    api_key = os.getenv("COINMARKETCAP_API_KEY")
    print(f"DB_HOST variable is: {db_host}")
    print(f"API Key variable is set: {api_key is not None}") # Don't print the key itself!
    if not db_host or not api_key:
        print("🔴 CRITICAL: Secrets are NOT loaded. Exiting.")
        return # Exit the script if secrets are missing
    print("--- ✅ Secrets seem to be loaded. Proceeding. ---")

    parser = argparse.ArgumentParser(description="Collect cryptocurrency data a single time.")
    parser.add_argument("symbols", nargs="+", help="Cryptocurrency symbols to track (e.g., BTC ETH XRP)")
    parser.add_argument("-m", "--max-rows", type=int, default=100,
                        help="Maximum number of rows to keep per symbol (default: 100)")

    args = parser.parse_args()

    print(f"🚀 Starting single data collection run for symbols: {args.symbols}")

    collector = DataCollector(args.symbols)
    collector.collect_data(args.max_rows)

    print("✅ Data collection run finished successfully.")

if __name__ == "__main__":
    main()
