import streamlit as st
import pandas as pd
import numpy as np
from src.db.get_connection import get_db_connection
from src.agents.crypto_agents import CoinAnalysisAgent, InvestmentRationaleAgent, MarketSummaryAgent, enhance_prompt_with_data

def get_coin_summary(symbol):
    """Get a comprehensive summary of a cryptocurrency from the database"""
    try:
        conn = get_db_connection()
        
        # Get latest data
        latest_query = f"""
        SELECT * FROM crypto_prices 
        WHERE symbol = '{symbol}' 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        latest_data = pd.read_sql(latest_query, conn)
        
        # Get historical data for trends
        history_query = f"""
        SELECT * FROM crypto_prices 
        WHERE symbol = '{symbol}' 
        ORDER BY timestamp DESC 
        LIMIT 30
        """
        historical_data = pd.read_sql(history_query, conn)
        conn.close()
        
        if latest_data.empty:
            return f"No data available for {symbol}."
            
        # Calculate trends
        if len(historical_data) > 1:
            price_change_7d = ((latest_data['price_usd'].iloc[0] / historical_data['price_usd'].iloc[min(7, len(historical_data)-1)]) - 1) * 100
            volume_trend = historical_data['volume_24h_usd'].pct_change().mean() * 100
            volatility = historical_data['price_usd'].pct_change().std() * 100
        else:
            price_change_7d = 0
            volume_trend = 0
            volatility = 0
            
        # Format the summary
        summary = f"""
## {symbol} Summary

### Current Metrics
- **Price**: ${latest_data['price_usd'].iloc[0]:,.2f}
- **24h Change**: {latest_data['percent_change_24h'].iloc[0]:+.2f}%
- **Market Cap**: ${latest_data['market_cap_usd'].iloc[0]/1e9:,.2f} billion
- **24h Volume**: ${latest_data['volume_24h_usd'].iloc[0]/1e6:,.2f} million

### Trends
- **7-Day Change**: {price_change_7d:+.2f}%
- **Volume Trend**: {'Increasing' if volume_trend > 0 else 'Decreasing'} ({abs(volume_trend):.2f}%)
- **Volatility**: {volatility:.2f}%

### Last Updated
- {latest_data['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')}
        """
        return summary
    except Exception as e:
        return f"Error retrieving data for {symbol}: {str(e)}"

def show_chatbot():
    st.title("StockyTalky Crypto Assistant")
    
    # Initialize agents if not already in session state
    if 'coin_agent' not in st.session_state:
        st.session_state.coin_agent = CoinAnalysisAgent()
    if 'rationale_agent' not in st.session_state:
        st.session_state.rationale_agent = InvestmentRationaleAgent()
    if 'summary_agent' not in st.session_state:
        st.session_state.summary_agent = MarketSummaryAgent()
    
    # Get available symbols from the database
    conn = get_db_connection()
    query = "SELECT DISTINCT symbol FROM crypto_prices"
    symbols_df = pd.read_sql(query, conn)
    conn.close()
    
    available_symbols = symbols_df['symbol'].tolist() if not symbols_df.empty else []
    
    # Chatbot interface
    st.markdown("""
    <div style="background-color:white;color:black;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h3 style="margin-top:0">Welcome to the Crypto Assistant!</h3>
        <p>I can help you with:</p>
        <ul>
            <li>Coin summaries: "Show me a summary of BTC"</li>
            <li>Price info: "What's the current price of ETH?"</li>
            <li>Investment advice: "Should I invest in SOL?"</li>
            <li>Market trends: "What's happening in the crypto market?"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Symbol selection
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_symbol = st.selectbox("Select a cryptocurrency:", [""] + available_symbols, 
                                      index=0, key="symbol_select")
    with col2:
        if selected_symbol and st.button(f"Show {selected_symbol} Summary"):
            summary = get_coin_summary(selected_symbol)
            
            # Add system message to chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
                
            # Add user message
            st.session_state.messages.append({"role": "user", 
                                             "content": f"Show me a summary of {selected_symbol}"})
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": summary})
            
            # Force a rerun to show the new messages
            st.rerun()
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about cryptocurrencies?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process the query
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question..."):
                # Check for summary requests
                if "summary" in prompt.lower() and any(coin in prompt.upper() for coin in available_symbols):
                    # Extract the coin symbol from the prompt
                    requested_symbol = None
                    for symbol in available_symbols:
                        if symbol.upper() in prompt.upper():
                            requested_symbol = symbol
                            break
                    
                    if requested_symbol:
                        response = get_coin_summary(requested_symbol)
                    else:
                        response = "I couldn't identify which cryptocurrency you want a summary for. Please specify a symbol like BTC, ETH, etc."
                
                # Check for price queries
                elif "price" in prompt.lower() and any(coin in prompt.upper() for coin in available_symbols):
                    requested_symbol = None
                    for symbol in available_symbols:
                        if symbol.upper() in prompt.upper():
                            requested_symbol = symbol
                            break
                    
                    if requested_symbol:
                        conn = get_db_connection()
                        price_query = f"""
                        SELECT price_usd, timestamp FROM crypto_prices
                        WHERE symbol = '{requested_symbol}'
                        ORDER BY timestamp DESC
                        LIMIT 1
                        """
                        price_data = pd.read_sql(price_query, conn)
                        conn.close()
                        
                        if not price_data.empty:
                            price = price_data['price_usd'].iloc[0]
                            timestamp = price_data['timestamp'].iloc[0]
                            response = f"The current price of {requested_symbol} is **${price:,.2f}** (as of {timestamp})."
                        else:
                            response = f"I don't have price data for {requested_symbol}."
                    else:
                        response = "I couldn't identify which cryptocurrency you're asking about. Please specify a symbol like BTC, ETH, etc."
                
                # Investment advice
                elif any(term in prompt.lower() for term in ["invest", "buy", "sell", "hold", "should i"]):
                    # Check if a cryptocurrency is already selected in the dropdown
                    if selected_symbol:
                        # Get crypto data for context
                        conn = get_db_connection()
                        data_query = f"""
                        SELECT * FROM crypto_prices
                        WHERE symbol = '{selected_symbol}'
                        ORDER BY timestamp DESC
                        LIMIT 1
                        """
                        crypto_data = pd.read_sql(data_query, conn)
                        conn.close()
                        
                        if not crypto_data.empty:
                            # Use the investment rationale agent
                            enhanced_prompt = enhance_prompt_with_data(
                                st.session_state.rationale_agent,
                                f"The user is asking: '{prompt}' about {selected_symbol}. Provide investment analysis for {selected_symbol}.",
                                crypto_data.iloc[0].to_dict()
                            )
                            response = st.session_state.rationale_agent.ask(enhanced_prompt)
                        else:
                            response = f"I don't have enough data for {selected_symbol} to give investment advice."
                    else:
                        # No cryptocurrency selected, check if any is mentioned in the prompt
                        mentioned_symbol = None
                        for symbol in available_symbols:
                            if symbol.upper() in prompt.upper():
                                mentioned_symbol = symbol
                                break
                                
                        if mentioned_symbol:
                            # Get data for mentioned coin
                            conn = get_db_connection()
                            data_query = f"""
                            SELECT * FROM crypto_prices
                            WHERE symbol = '{mentioned_symbol}'
                            ORDER BY timestamp DESC
                            LIMIT 1
                            """
                            crypto_data = pd.read_sql(data_query, conn)
                            conn.close()
                            
                            if not crypto_data.empty:
                                enhanced_prompt = enhance_prompt_with_data(
                                    st.session_state.rationale_agent,
                                    f"The user is asking: '{prompt}' about {mentioned_symbol}. Provide investment analysis for {mentioned_symbol}.",
                                    crypto_data.iloc[0].to_dict()
                                )
                                response = st.session_state.rationale_agent.ask(enhanced_prompt)
                            else:
                                response = f"I don't have enough data for {mentioned_symbol} to give investment advice."
                        else:
                            # No specific cryptocurrency identified
                            response = """
                            I'd be happy to provide investment analysis, but I need to know which cryptocurrency you're interested in.
                            
                            Please either:
                            1. Select a cryptocurrency from the dropdown above, or
                            2. Specify which cryptocurrency in your question (e.g., "Should I invest in BTC?")
                            """
                
                # Market summary
                elif any(term in prompt.lower() for term in ["market", "trend", "overview", "happening"]):
                    response = st.session_state.summary_agent.ask(prompt)
                
                # General coin analysis
                elif any(coin in prompt.upper() for coin in available_symbols):
                    requested_symbol = None
                    for symbol in available_symbols:
                        if symbol.upper() in prompt.upper():
                            requested_symbol = symbol
                            break
                    
                    if requested_symbol:
                        # Get data for this coin
                        conn = get_db_connection()
                        data_query = f"""
                        SELECT * FROM crypto_prices
                        WHERE symbol = '{requested_symbol}'
                        ORDER BY timestamp DESC
                        LIMIT 1
                        """
                        crypto_data = pd.read_sql(data_query, conn)
                        conn.close()
                        
                        if not crypto_data.empty:
                            enhanced_prompt = enhance_prompt_with_data(
                                st.session_state.coin_agent,
                                f"Analysis for {requested_symbol}: {prompt}",
                                crypto_data.iloc[0].to_dict()
                            )
                            response = st.session_state.coin_agent.ask(enhanced_prompt)
                        else:
                            response = f"I don't have enough data for {requested_symbol} to provide analysis."
                    else:
                        response = "I couldn't identify which cryptocurrency you're asking about. Please specify a symbol like BTC, ETH, etc."
                
                # Greeting or general query
                elif any(greeting in prompt.lower() for greeting in ["hello", "hi", "hey", "greetings"]):
                    response = """
                    Hello! I'm your Crypto Assistant. I can help you with:
                    
                    - Summaries of cryptocurrencies
                    - Current price information
                    - Investment recommendations
                    - Market trends and analysis
                    
                    Try asking about a specific cryptocurrency like "Tell me about BTC" or "Should I invest in ETH?"
                    """
                
                # Default fallback
                else:
                    response = st.session_state.coin_agent.ask(prompt)
                
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.coin_agent.reset_conversation()
        st.session_state.rationale_agent.reset_conversation()
        st.session_state.summary_agent.reset_conversation()
        st.rerun()