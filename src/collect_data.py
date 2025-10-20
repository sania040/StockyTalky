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

    # CHANGED: Removed the 'interval' parameter as it's no longer needed
    def collect_data(self, max_rows):
        """Fetch and store data for all symbols in a single run."""
        conn = None  # Initialize conn to None
        try:
            conn = get_db_connection()
            for symbol in self.symbols:
                try:
                    # Check current row count for this symbol
                    # Pass the connection to the query function for efficiency
                    count_df = execute_query(conn, "SELECT COUNT(*) FROM crypto_prices WHERE symbol = %s", (symbol,))
                    row_count = count_df.iloc[0, 0]
                    
                    print(f"Fetching data for {symbol}...")
                    api_data = self.fetcher.fetch_data_for_symbol(symbol)
                    
                    if row_count >= max_rows:
                        print(f"Row limit reached for {symbol}. Updating oldest record...")
                        # Get the oldest record's ID
                        oldest_id_df = execute_query(conn, """
                            SELECT id FROM crypto_prices 
                            WHERE symbol = %s 
                            ORDER BY timestamp ASC 
                            LIMIT 1
                        """, (symbol,))
                        oldest_id = oldest_id_df.iloc[0, 0]
                        
                        # Prepare data for update
                        quote = api_data['data'][symbol]['quote']['USD']
                        ts = time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Update the oldest record
                        execute_and_commit(conn, """
                            UPDATE crypto_prices 
                            SET price_usd = %s, volume_24h_usd = %s, percent_change_24h = %s, 
                                market_cap_usd = %s, timestamp = %s
                            WHERE id = %s
                        """, (
                            quote['price'], quote['volume_24h'], quote.get('percent_change_24h'),
                            quote['market_cap'], ts, oldest_id
                        ))
                        print(f"Updated oldest record for {symbol}")
                    else:
                        print(f"Storing new data for {symbol}...")
                        self.fetcher.store_data(conn, api_data, symbol)
                        print(f"Successfully stored new data for {symbol}")
                        
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
        
        except Exception as e:
            print(f"A critical error occurred during data collection: {e}")
        finally:
            # This block ensures the connection is closed no matter what
            if conn and not conn.closed:
                conn.close()
                print("\nDatabase connection closed.")

# ADDED: A clean main function to run the script
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
    # REMOVED: The --interval argument is no longer needed
    parser.add_argument("-m", "--max-rows", type=int, default=100,
                        help="Maximum number of rows to keep per symbol (default: 100)")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting single data collection run for symbols: {args.symbols}")
    
    collector = DataCollector(args.symbols)
    collector.collect_data(args.max_rows)
    
    print("✅ Data collection run finished successfully.")


# REMOVED: The old scheduling logic
if __name__ == "__main__":
    main()
