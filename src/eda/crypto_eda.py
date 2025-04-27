import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.db.get_connection import get_db_connection

class CryptoEDA:
    def get_available_symbols(self):
        """Get list of available crypto symbols"""
        conn = get_db_connection()
        try:
            query = "SELECT DISTINCT symbol FROM crypto_prices ORDER BY symbol"
            symbols = pd.read_sql(query, conn)
            return symbols['symbol'].tolist()
        except Exception as e:
            print(f"Error: {e}")
            return []
        finally:
            conn.close()
    
    def get_all_symbols_data(self):
        """Get latest data for all symbols"""
        conn = get_db_connection()
        try:
            # Get latest price for each symbol
            query = """
                WITH latest AS (
                    SELECT symbol, MAX(timestamp) as latest_time
                    FROM crypto_prices
                    GROUP BY symbol
                )
                SELECT cp.*
                FROM crypto_prices cp
                JOIN latest l ON cp.symbol = l.symbol AND cp.timestamp = l.latest_time
                ORDER BY market_cap_usd DESC
            """
            return pd.read_sql(query, conn)
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_data_for_symbol(self, symbol):
        """Get historical data for a specific symbol"""
        conn = get_db_connection()
        try:
            query = f"""
                SELECT * FROM crypto_prices 
                WHERE symbol = '{symbol}'
                ORDER BY timestamp
            """
            df = pd.read_sql(query, conn)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_investment_recommendations(self):
        """Generate simple investment recommendations"""
        recommendations = []
        
        # Get latest data for all symbols
        all_data = self.get_all_symbols_data()
        
        for _, row in all_data.iterrows():
            symbol = row['symbol']
            symbol_data = self.get_data_for_symbol(symbol)
            
            if len(symbol_data) >= 3:
                price = row['price_usd']
                change = row['percent_change_24h'] if not pd.isna(row['percent_change_24h']) else 0
                
                # Simple recommendation logic
                if change > 5:
                    rec = "Strong Buy"
                    reason = f"Strong momentum (+{change:.1f}% 24h)"
                elif change > 2:
                    rec = "Buy"
                    reason = "Positive trend"
                elif change < -5:
                    rec = "Sell"
                    reason = "Significant drop"
                elif change < -2:
                    rec = "Buy the Dip"
                    reason = "Potential recovery opportunity"
                else:
                    rec = "Hold/Watch"
                    reason = "No strong signals"
                
                recommendations.append({
                    'symbol': symbol,
                    'price_usd': price,
                    'change_24h': change,
                    'recommendation': rec,
                    'reason': reason
                })
        
        return pd.DataFrame(recommendations)
    
    def create_dashboard_charts(self, symbol):
        """Create price and volume charts"""
        df = self.get_data_for_symbol(symbol)
        if df.empty or len(df) < 2:
            return None, None
        
        # Create subplots for price and volume
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price", "Volume")
        )
        
        # Try to create daily OHLC data for candlesticks
        try:
            # Group by date
            df['date'] = df['timestamp'].dt.date
            daily = df.groupby('date').agg({
                'price_usd': ['first', 'max', 'min', 'last'],
                'volume_24h_usd': 'sum',
                'timestamp': 'first'
            }).reset_index()
            
            # Flatten columns
            daily.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'timestamp']
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=daily['date'],
                    open=daily['open'], high=daily['high'],
                    low=daily['low'], close=daily['close'],
                    increasing_line_color='green',
                    decreasing_line_color='red',
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=daily['date'],
                    y=daily['volume'],
                    name='Volume',
                    marker_color='navy'
                ),
                row=2, col=1
            )
            
            # Add moving average if enough data
            if len(daily) >= 7:
                daily['ma7'] = daily['close'].rolling(window=7).mean()
                fig.add_trace(
                    go.Scatter(
                        x=daily['date'],
                        y=daily['ma7'],
                        name='7-Day MA',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
        except:
            # Fallback to simple line chart
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['price_usd'],
                    name='Price',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['volume_24h_usd'],
                    name='Volume',
                    marker_color='navy'
                ),
                row=2, col=1
            )
        
        # Clean layout
        fig.update_layout(
            title=f"{symbol} Market Data",
            showlegend=True,
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        # Create a simple technical chart with price change
        tech_fig = go.Figure()
        tech_fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['percent_change_24h'],
                name='24h Change',
                marker_color=df['percent_change_24h'].apply(
                    lambda x: 'green' if x >= 0 else 'red'
                )
            )
        )
        
        tech_fig.update_layout(
            title=f"{symbol} 24h Price Change",
            height=300
        )
        
        return fig, tech_fig