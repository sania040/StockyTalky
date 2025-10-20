import time
import argparse
import os
from dotenv import load_dotenv
import schedule
from src.api.crypto_fetcher import CryptoDataFetcher
from src.db.get_connection import get_db_connection
from src.db.query_utils import execute_query, execute_and_commit

class DataCollector:
    def __init__(self, symbols):
        self.fetcher = CryptoDataFetcher()
        self.symbols = symbols

    def collect_data(self, interval, max_rows):
        """Fetch and store data for all symbols"""
        conn = get_db_connection()
        try:
            for symbol in self.symbols:
                try:
                    # Check current row count for this symbol
                    result_df = execute_query("SELECT COUNT(*) FROM crypto_prices WHERE symbol = %s", (symbol,))
                    row_count = result_df.iloc[0, 0]
                    
                    print(f"Fetching data for {symbol}...")
                    api_data = self.fetcher.fetch_data_for_symbol(symbol)
                    
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
                        self.fetcher.store_data(conn, api_data, symbol)
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

def run_collector(symbols, interval=300, max_rows=100):
    """Main function to run the data collection process"""
    collector = DataCollector(symbols)
    
    # Schedule data collection every 5 minutes
    schedule.every(interval).seconds.do(collector.collect_data, interval, max_rows)
    
    # Run initial collection
    print("🚀 Starting initial data collection...")
    collector.collect_data(interval, max_rows)
    
    # Keep running the scheduler
    print(f"📊 Starting scheduled collection every {interval} seconds...")
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect cryptocurrency data at regular intervals")
    parser.add_argument("symbols", nargs="+", help="Cryptocurrency symbols to track (e.g., BTC ETH XRP)")
    parser.add_argument("-i", "--interval", type=int, default=300, 
                        help="Time interval between data collections in seconds (default: 300)")
    parser.add_argument("-m", "--max-rows", type=int, default=100,
                        help="Maximum number of rows to keep per symbol (default: 100)")
    
    args = parser.parse_args()
    run_collector(args.symbols, args.interval, args.max_rows)