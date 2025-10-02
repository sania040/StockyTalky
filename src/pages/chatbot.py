import streamlit as st
import pandas as pd
from datetime import datetime
from src.db.get_connection import get_db_connection
from src.agents.crypto_agents import (
    CoinAnalysisAgent, 
    InvestmentRationaleAgent, 
    MarketSummaryAgent, 
    enhance_prompt_with_data,
    format_agent_response
)
from src.db.query_utils import execute_query

def show():
    """Display professional chatbot page with enhanced UX"""
    
    # Page header with professional styling
    st.markdown("""
        <div class="crypto-header">
            <h1 style="margin:0; color:white;">💬 StockyTalky AI Assistant</h1>
            <p style="margin:5px 0 0 0; color:rgba(255,255,255,0.9);">
                Your intelligent companion for cryptocurrency data analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize agents
    initialize_agents()
    
    # Get available symbols
    symbols_df = execute_query("SELECT DISTINCT symbol FROM crypto_prices ORDER BY symbol")
    available_symbols = symbols_df['symbol'].tolist() if not symbols_df.empty else []
    
    # Symbol selection with real-time data preview
    selected_symbol = display_symbol_selector(available_symbols)
    
    # Update agent context when symbol changes
    if selected_symbol and selected_symbol != st.session_state.get('last_selected_symbol'):
        update_agent_context_with_history(selected_symbol)
        st.session_state['last_selected_symbol'] = selected_symbol
    
    # Quick action buttons
    display_quick_actions(selected_symbol, available_symbols)
    
    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    handle_chat_input(selected_symbol, available_symbols)
    
    # Sidebar
    display_sidebar_info(selected_symbol, available_symbols)


def initialize_agents():
    """Initialize AI agents with error handling"""
    try:
        if 'coin_agent' not in st.session_state:
            st.session_state.coin_agent = CoinAnalysisAgent()
        if 'rationale_agent' not in st.session_state:
            st.session_state.rationale_agent = InvestmentRationaleAgent()
        if 'summary_agent' not in st.session_state:
            st.session_state.summary_agent = MarketSummaryAgent()
    except Exception as e:
        st.error(f"❌ Failed to initialize AI agents: {str(e)}")
        st.stop()


def display_symbol_selector(available_symbols: list):
    """Display enhanced symbol selector with live data"""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_symbol = st.selectbox(
            "🎯 Select Cryptocurrency for Analysis",
            [""] + available_symbols,
            index=0,
            help="Choose a cryptocurrency to analyze its historical data"
        )
    
    with col2:
        if selected_symbol:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
    
    # Display live data preview for selected symbol
    if selected_symbol:
        display_live_data_card(selected_symbol)
    
    return selected_symbol


def display_live_data_card(symbol: str):
    """Display real-time data card for selected cryptocurrency"""
    try:
        query = f"""
        SELECT price_usd, percent_change_24h, market_cap_usd, 
               volume_24h_usd, timestamp
        FROM crypto_prices
        WHERE symbol = '{symbol}'
        ORDER BY timestamp DESC
        LIMIT 1
        """
        df = execute_query(query)
        
        if not df.empty:
            data = df.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "💰 Price",
                    f"${data['price_usd']:,.2f}",
                    help="Current market price in USD"
                )
            
            with col2:
                change = data['percent_change_24h']
                st.metric(
                    "📊 24h Change",
                    f"{change:+.2f}%",
                    delta=f"{change:+.2f}%",
                    help="Price change in last 24 hours"
                )
            
            with col3:
                market_cap = data['market_cap_usd']
                if market_cap >= 1e9:
                    display_cap = f"${market_cap/1e9:.2f}B"
                else:
                    display_cap = f"${market_cap/1e6:.2f}M"
                st.metric(
                    "🏦 Market Cap",
                    display_cap,
                    help="Total market capitalization"
                )
            
            with col4:
                volume = data['volume_24h_usd']
                if volume >= 1e9:
                    display_vol = f"${volume/1e9:.2f}B"
                else:
                    display_vol = f"${volume/1e6:.2f}M"
                st.metric(
                    "📈 24h Volume",
                    display_vol,
                    help="Trading volume in last 24 hours"
                )
            
            # Store in session state
            st.session_state[f'{symbol}_data'] = data.to_dict()
            
    except Exception as e:
        st.warning(f"Unable to load live data: {str(e)}")


def update_agent_context_with_history(symbol: str):
    """Update all agents with selected cryptocurrency context INCLUDING historical data"""
    try:
        # Get current data
        crypto_data = st.session_state.get(f'{symbol}_data')
        
        # Get historical data
        historical_query = f"""
        SELECT symbol, price_usd, percent_change_24h, market_cap_usd, 
               volume_24h_usd, timestamp
        FROM crypto_prices
        WHERE symbol = '{symbol}'
        ORDER BY timestamp ASC
        """
        historical_df = execute_query(historical_query)
        
        # Update all agents with both current and historical data
        if not historical_df.empty:
            st.session_state.coin_agent.set_context(symbol, crypto_data, historical_df)
            st.session_state.rationale_agent.set_context(symbol, crypto_data, historical_df)
            st.session_state.summary_agent.set_context(symbol, crypto_data, historical_df)
            
            # Store historical data in session state
            st.session_state[f'{symbol}_history'] = historical_df
        
    except Exception as e:
        st.warning(f"Could not load historical data: {str(e)}")


def display_quick_actions(selected_symbol: str, available_symbols: list):
    """Display quick action buttons for common queries"""
    
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Define actions with proper structure
    actions = [
        ("📊 Analyze Data", f"Analyze the historical data for {selected_symbol or 'Bitcoin'}"),
        ("💡 Best Entry", f"When was the best time to buy {selected_symbol or 'Ethereum'} based on our data?"),
        ("📈 Price Trends", f"Show me price trends for {selected_symbol or 'Bitcoin'} in our database"),
        ("🔍 Compare Highs/Lows", f"Compare {selected_symbol or 'Bitcoin'} current price to historical highs and lows")
    ]
    
    columns = [col1, col2, col3, col4]
    
    for idx, (col, (label, query)) in enumerate(zip(columns, actions)):
        with col:
            if st.button(label, use_container_width=True, key=f"quick_action_{idx}"):
                st.session_state.messages.append({
                    "role": "user", 
                    "content": query,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                st.session_state['pending_query'] = query
                st.rerun()


def display_chat_history():
    """Display chat messages with professional formatting"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                timestamp = message.get("timestamp", "")
                if timestamp:
                    st.caption(f"🕒 {timestamp}")
            st.markdown(message["content"])
    
    # Process pending query if exists
    if 'pending_query' in st.session_state:
        query = st.session_state['pending_query']
        del st.session_state['pending_query']
        
        # Get selected symbol
        symbols_df = execute_query("SELECT DISTINCT symbol FROM crypto_prices ORDER BY symbol")
        available_symbols = symbols_df['symbol'].tolist() if not symbols_df.empty else []
        selected_symbol = st.session_state.get('last_selected_symbol', '')
        
        # Process and display response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing data..."):
                response = process_query(query, selected_symbol, available_symbols)
                timestamp = datetime.now().strftime("%I:%M %p")
                st.caption(f"🕒 {timestamp}")
                st.markdown(response)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "timestamp": timestamp
        })


def handle_chat_input(selected_symbol: str, available_symbols: list):
    """Handle chat input with intelligent routing"""
    
    if selected_symbol:
        placeholder = f"Ask about {selected_symbol} data... (e.g., 'When was the best time to buy {selected_symbol}?')"
    else:
        placeholder = "Ask about cryptocurrency data in our database..."
    
    if prompt := st.chat_input(placeholder):
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing data..."):
                response = process_query(prompt, selected_symbol, available_symbols)
                timestamp = datetime.now().strftime("%I:%M %p")
                st.caption(f"🕒 {timestamp}")
                st.markdown(response)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "timestamp": timestamp
        })
        
        st.rerun()


def process_query(prompt: str, selected_symbol: str, available_symbols: list) -> str:
    """Process user query and route to appropriate agent"""
    
    active_symbol = selected_symbol if selected_symbol else None
    
    if not active_symbol:
        for sym in available_symbols:
            if sym.upper() in prompt.upper():
                active_symbol = sym
                break
    
    prompt_lower = prompt.lower()
    
    # Investment/Trading intent
    if any(term in prompt_lower for term in ["invest", "buy", "sell", "hold", "trade", "purchase", "should i", "entry", "best time"]):
        return handle_investment_query(prompt, active_symbol)
    
    # Market overview intent
    elif any(term in prompt_lower for term in ["market", "compare", "overview", "sentiment", "all coins"]):
        return handle_market_query(prompt)
    
    # Analysis intent (default)
    else:
        return handle_analysis_query(prompt, active_symbol)


def handle_investment_query(prompt: str, symbol: str) -> str:
    """Handle investment-related queries"""
    if symbol:
        crypto_data = st.session_state.get(f'{symbol}_data')
        
        if crypto_data:
            enhanced_prompt = enhance_prompt_with_data(
                st.session_state.rationale_agent,
                prompt,
                crypto_data
            )
            response = st.session_state.rationale_agent.ask(enhanced_prompt)
            return format_agent_response(response, "Investment Data Analyst")
        else:
            return f"⚠️ No data available for {symbol}. Please select a cryptocurrency with available data."
    else:
        return "🎯 **Please select a cryptocurrency** from the dropdown above to analyze investment opportunities."


def handle_market_query(prompt: str) -> str:
    """Handle market overview queries"""
    response = st.session_state.summary_agent.ask(prompt)
    return format_agent_response(response, "Market Data Intelligence")


def handle_analysis_query(prompt: str, symbol: str) -> str:
    """Handle general analysis queries"""
    if symbol:
        crypto_data = st.session_state.get(f'{symbol}_data')
        if crypto_data:
            enhanced_prompt = f"{prompt}\n\nFocus on analyzing the historical data for {symbol}."
        else:
            enhanced_prompt = prompt
        
        response = st.session_state.coin_agent.ask(enhanced_prompt)
    else:
        response = st.session_state.coin_agent.ask(prompt)
    
    return format_agent_response(response, "Data-Driven Analyst")


def display_sidebar_info(selected_symbol: str, available_symbols: list):
    """Display clean and organized sidebar with essential information"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 15px 0;">
            <h2 style="margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       background-clip: text; font-size: 1.5rem;">
                🤖 Data Analyst
            </h2>
            <p style="color: #666; font-size: 0.8rem; margin: 5px 0 0 0;">Analyzing YOUR Database</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if selected_symbol:
            # Show data statistics
            history_df = st.session_state.get(f'{selected_symbol}_history')
            if history_df is not None and not history_df.empty:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 12px; border-radius: 8px; color: white; margin-bottom: 15px; text-align: center;">
                    <div style="font-size: 0.75rem; opacity: 0.9;">Analyzing</div>
                    <div style="font-size: 1.2rem; font-weight: bold; margin: 5px 0;">{selected_symbol}</div>
                    <div style="font-size: 0.7rem; opacity: 0.8;">✓ {len(history_df)} Data Points</div>
                </div>
                """, unsafe_allow_html=True)
        
        message_count = len(st.session_state.get('messages', []))
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💬 Messages", message_count)
        with col2:
            st.metric("📊 Assets", len(available_symbols))
        
        st.markdown("---")
        
        with st.expander("🎯 **What I Analyze**"):
            st.markdown("""
            **📊 Historical Data**  
            Price trends, highs, lows
            
            **💡 Best Opportunities**  
            When to buy/sell based on data
            
            **📈 Performance**  
            Gains/losses over time
            
            **🔍 Comparisons**  
            Current vs historical prices
            """)
        
        with st.expander("💭 **Example Questions**"):
            example_questions = [
                "When was the best time to buy?",
                "Show me price trends",
                "Am I buying at a good price?",
            ]
            
            for i, question in enumerate(example_questions, 1):
                if st.button(f"💬 {question}", key=f"example_{i}", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": question,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })
                    st.session_state['pending_query'] = question
                    st.rerun()
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.session_state.coin_agent.reset_conversation()
                st.session_state.rationale_agent.reset_conversation()
                st.session_state.summary_agent.reset_conversation()
                if 'last_selected_symbol' in st.session_state:
                    del st.session_state['last_selected_symbol']
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #d1ecf1; padding: 8px; border-radius: 4px; 
                    border-left: 3px solid #0c5460;">
            <small><strong>💡 Tip</strong><br/>
            I analyze YOUR database data - ask about specific dates and prices!</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding-top: 15px; color: #999; font-size: 0.7rem;">
            StockyTalky © 2025
        </div>
        """, unsafe_allow_html=True)