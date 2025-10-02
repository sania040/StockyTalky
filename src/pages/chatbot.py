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
                Your intelligent companion for cryptocurrency analysis and insights
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize agents with proper error handling
    initialize_agents()
    
    # Get available symbols
    symbols_df = execute_query("SELECT DISTINCT symbol FROM crypto_prices ORDER BY symbol")
    available_symbols = symbols_df['symbol'].tolist() if not symbols_df.empty else []
    
    # Professional welcome section
    display_welcome_section(available_symbols)
    
    # Symbol selection with real-time data preview
    selected_symbol = display_symbol_selector(available_symbols)
    
    # Update agent context when symbol changes
    if selected_symbol and selected_symbol != st.session_state.get('last_selected_symbol'):
        update_agent_context(selected_symbol)
        st.session_state['last_selected_symbol'] = selected_symbol
    
    # Quick action buttons
    display_quick_actions(selected_symbol, available_symbols)
    
    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history with professional formatting
    display_chat_history()
    
    # Chat input with suggestions
    handle_chat_input(selected_symbol, available_symbols)
    
    # Sidebar with helpful information
    display_sidebar_info(selected_symbol)


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


def display_welcome_section(available_symbols: list):
    """Display professional welcome section"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="margin: 0 0 10px 0;">👋 Welcome to Your AI Crypto Advisor</h3>
            <p style="margin: 0; font-size: 0.95rem; line-height: 1.6;">
                Get professional analysis, investment insights, and market intelligence 
                powered by advanced AI. Ask anything about cryptocurrencies!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric(
            "Tracked Assets", 
            len(available_symbols),
            delta="Live Data",
            help="Number of cryptocurrencies with available data"
        )


def display_symbol_selector(available_symbols: list):
    """Display enhanced symbol selector with live data"""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_symbol = st.selectbox(
            "🎯 Select Cryptocurrency for Analysis",
            [""] + available_symbols,
            index=0,
            help="Choose a cryptocurrency to get context-aware insights"
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
            
            # Store in session state for agent context
            st.session_state[f'{symbol}_data'] = data.to_dict()
            
    except Exception as e:
        st.warning(f"Unable to load live data: {str(e)}")


def update_agent_context(symbol: str):
    """Update all agents with selected cryptocurrency context"""
    crypto_data = st.session_state.get(f'{symbol}_data')
    
    if crypto_data:
        st.session_state.coin_agent.set_context(symbol, crypto_data)
        st.session_state.rationale_agent.set_context(symbol, crypto_data)
        st.session_state.summary_agent.set_context(symbol, crypto_data)


def display_quick_actions(selected_symbol: str, available_symbols: list):
    """Display quick action buttons for common queries"""
    
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    quick_actions = {
        "col1": ("📊 Analyze", f"Give me a detailed analysis of {selected_symbol or 'Bitcoin'}"),
        "col2": ("💡 Investment View", f"Should I consider investing in {selected_symbol or 'Ethereum'}?"),
        "col3": ("📈 Market Trend", "What's the current crypto market sentiment?"),
        "col4": ("🔍 Compare", "Compare top cryptocurrencies by market cap")
    }
    
    for col_name, (label, query) in quick_actions.items():
        with locals()[col_name]:
            if st.button(label, use_container_width=True, key=f"quick_{label}"):
                process_quick_action(query, selected_symbol, available_symbols)


def display_chat_history():
    """Display chat messages with professional formatting"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Add timestamp to assistant messages
                timestamp = message.get("timestamp", "")
                if timestamp:
                    st.caption(f"🕒 {timestamp}")
            st.markdown(message["content"])


def handle_chat_input(selected_symbol: str, available_symbols: list):
    """Handle chat input with intelligent routing"""
    
    # Provide smart suggestions based on context
    if selected_symbol:
        placeholder = f"Ask me about {selected_symbol}... (e.g., 'Is {selected_symbol} a good investment?')"
    else:
        placeholder = "Ask me anything about cryptocurrencies..."
    
    if prompt := st.chat_input(placeholder):
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query with professional response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                response = process_query(prompt, selected_symbol, available_symbols)
                
                # Add timestamp
                timestamp = datetime.now().strftime("%I:%M %p")
                st.caption(f"🕒 {timestamp}")
                st.markdown(response)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "timestamp": timestamp
        })
        
        # Auto-scroll to bottom
        st.rerun()


def process_query(prompt: str, selected_symbol: str, available_symbols: list) -> str:
    """Process user query and route to appropriate agent"""
    
    # Determine active symbol
    active_symbol = selected_symbol if selected_symbol else None
    
    # Check if user mentioned a specific symbol in their query
    if not active_symbol:
        for sym in available_symbols:
            if sym.upper() in prompt.upper():
                active_symbol = sym
                break
    
    # Route to appropriate agent based on query intent
    prompt_lower = prompt.lower()
    
    # Investment/Trading intent
    if any(term in prompt_lower for term in ["invest", "buy", "sell", "hold", "trade", "purchase", "should i"]):
        return handle_investment_query(prompt, active_symbol)
    
    # Market overview intent
    elif any(term in prompt_lower for term in ["market", "trend", "overview", "sentiment", "outlook"]):
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
            return format_agent_response(response, "Investment Analyst")
        else:
            return f"⚠️ No current data available for {symbol}. Please select a cryptocurrency with available data."
    else:
        return "🎯 **Please select a cryptocurrency** from the dropdown above to get investment insights, or specify one in your question."


def handle_market_query(prompt: str) -> str:
    """Handle market overview queries"""
    response = st.session_state.summary_agent.ask(prompt)
    return format_agent_response(response, "Market Intelligence")


def handle_analysis_query(prompt: str, symbol: str) -> str:
    """Handle general analysis queries"""
    if symbol:
        # Enhance with context if available
        crypto_data = st.session_state.get(f'{symbol}_data')
        if crypto_data:
            enhanced_prompt = f"{prompt}\n\nFocus on {symbol} specifically."
            st.session_state.coin_agent.set_context(symbol, crypto_data)
        else:
            enhanced_prompt = prompt
        
        response = st.session_state.coin_agent.ask(enhanced_prompt)
    else:
        response = st.session_state.coin_agent.ask(prompt)
    
    return format_agent_response(response, "Cryptocurrency Analyst")


def process_quick_action(query: str, selected_symbol: str, available_symbols: list):
    """Process quick action button clicks"""
    st.session_state.messages.append({
        "role": "user", 
        "content": query,
        "timestamp": datetime.now().strftime("%I:%M %p")
    })
    st.rerun()


def display_sidebar_info(selected_symbol: str):
    """Display helpful information in sidebar"""
    with st.sidebar:
        st.markdown("---")
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - **Select a cryptocurrency** for context-aware analysis
        - **Use quick actions** for instant insights
        - **Ask follow-up questions** to dive deeper
        - **Compare multiple coins** for better decisions
        """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 What I Can Help With")
        st.markdown("""
        - **Technical Analysis**: Charts, indicators, trends
        - **Fundamental Analysis**: Technology, team, roadmap
        - **Investment Insights**: Risk/reward, timing
        - **Market Intelligence**: Trends, sentiment, news
        - **Portfolio Advice**: Diversification, allocation
        """)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.coin_agent.reset_conversation()
            st.session_state.rationale_agent.reset_conversation()
            st.session_state.summary_agent.reset_conversation()
            if 'last_selected_symbol' in st.session_state:
                del st.session_state['last_selected_symbol']
            st.success("Chat history cleared!")
            st.rerun()
        
        st.markdown("---")
        st.caption(f"💬 Messages: {len(st.session_state.get('messages', []))}")
        st.caption(f"🤖 AI Model: openai/gpt-oss-120b")