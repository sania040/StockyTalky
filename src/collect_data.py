import time
import argparse
import os
from dotenv import load_dotenv
from db.get_connection import get_db_connection
from fetch_and_store_crypto import CryptoDataFetcher
from src.db.query_utils import execute_query, execute_and_commit

def collect_crypto_data(symbols, interval=10, max_rows=100):
    """
    Continuously collects and stores cryptocurrency data for specified symbols at given intervals.
    Limits data to max_rows per symbol, updating oldest data when limit is reached.
    
    Args:
        symbols (list): List of cryptocurrency symbols to track
        interval (int): Time in seconds between data collection (default: 10)
        max_rows (int): Maximum number of rows to keep per symbol (default: 100)
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("COINMARKETCAP_API_KEY")
    
    if not api_key:
        raise ValueError("API key not found. Please set COINMARKETCAP_API_KEY in your .env file.")
    
    # Initialize the data fetcher
    fetcher = CryptoDataFetcher(api_key)
    
    print(f"Starting data collection for symbols: {', '.join(symbols)}")
    print(f"Collection interval: {interval} seconds")
    print(f"Maximum rows per symbol: {max_rows}")
    print("Press Ctrl+C to stop the collection process")
    
    try:
        while True:
            # Get a fresh database connection for each batch
            conn = get_db_connection()
            
            for symbol in symbols:
                try:
                    # Check current row count for this symbol
                    result_df = execute_query("SELECT COUNT(*) FROM crypto_prices WHERE symbol = %s", (symbol,))
                    row_count = result_df.iloc[0, 0]
                    
                    print(f"Fetching data for {symbol}...")
                    api_data = fetcher.fetch_data_for_symbol(symbol)
                    
                    if row_count >= max_rows:
                        print(f"Row limit reached for {symbol}. Updating oldest record...")
                        # Get the oldest record's ID
                        result_df = execute_query("""
                            SELECT id FROM crypto_prices 
                            WHERE symbol = %s 
                            ORDER BY timestamp ASC 
                            LIMIT 1
                        """, (symbol,))
                        oldest_id = result_df.iloc[0, 0]
                        
                        # Prepare data for update
                        quote = api_data['data'][symbol]['quote']['USD']
                        ts = time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Update the oldest record
                        execute_and_commit("""
                            UPDATE crypto_prices 
                            SET price_usd = %s, 
                                volume_24h_usd = %s, 
                                percent_change_24h = %s, 
                                market_cap_usd = %s, 
                                timestamp = %s
                            WHERE id = %s
                        """, (
                            quote['price'],
                            quote['volume_24h'],
                            quote.get('percent_change_24h'),
                            quote['market_cap'],
                            ts,
                            oldest_id
                        ))
                        print(f"Updated oldest record for {symbol}")
                    else:
                        print(f"Storing new data for {symbol}...")
                        fetcher.store_data(conn, api_data, symbol)
                        print(f"Successfully stored new data for {symbol}")
                        
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                finally:
                    if 'cur' in locals():
                        conn.close()
            
            # Close the connection after processing all symbols
            conn.close()
            
            print(f"Waiting {interval} seconds until next collection...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nData collection stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure connection is closed if an exception occurs
        try:
            if conn and not conn.closed:
                conn.close()
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect cryptocurrency data at regular intervals")
    parser.add_argument("symbols", nargs="+", help="Cryptocurrency symbols to track (e.g., BTC ETH XRP)")
    parser.add_argument("-i", "--interval", type=int, default=10, 
                        help="Time interval between data collections in seconds (default: 10)")
    parser.add_argument("-m", "--max-rows", type=int, default=100,
                        help="Maximum number of rows to keep per symbol (default: 100)")
    
    args = parser.parse_args()
    collect_crypto_data(args.symbols, args.interval, args.max_rows)