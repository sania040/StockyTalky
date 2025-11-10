import streamlit as st
import os
import ast
from datetime import datetime
from decimal import Decimal
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory

# --- Database Connection (Cached for performance) ---
@st.cache_resource
def get_engine():
    """
    Get SQLAlchemy engine (cached for performance).
    Separate from SQLDatabase to avoid Streamlit serialization issues.
    """
    try:
        # Try using SUPABASE_DB_URL first
        db_url = os.getenv("SUPABASE_DB_URL")
        if db_url:
            # Ensure the URL uses the correct driver
            if db_url.startswith("postgresql://"):
                db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
            return create_engine(db_url, echo=False)
        else:
            # Fall back to individual credentials
            db_user = os.getenv('DB_USER')
            db_password = os.getenv('DB_PASSWORD')
            db_host = os.getenv('DB_HOST')
            db_port = os.getenv('DB_PORT', '5432')
            db_name = os.getenv('DB_NAME')
            
            if not all([db_user, db_password, db_host, db_name]):
                raise ValueError("Database credentials not properly configured")
                
            db_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            return create_engine(db_uri, echo=False)
    except Exception as e:
        raise ValueError(f"Database connection error: {e}")

def get_db():
    """
    Get SQLDatabase wrapper (not cached to avoid Streamlit serialization issues).
    Uses the cached engine internally.
    """
    try:
        engine = get_engine()
        return SQLDatabase(engine, include_tables=['crypto_prices'])
    except Exception as e:
        raise ValueError(f"Database wrapper error: {e}")

# --- Helper Function to Parse Database Results ---

def parse_db_result(result_str: str) -> list:
    """
    Safely parse database result strings containing Decimal and datetime objects.
    Returns a list of tuples with properly converted values.
    """
    import datetime as dt
    
    # Create a safe namespace for evaluation
    safe_dict = {
        'Decimal': Decimal,
        'datetime': dt,
    }
    
    try:
        # Use eval with limited namespace instead of literal_eval to handle Decimal and datetime
        result = eval(result_str, {"__builtins__": {}}, safe_dict)
        return result if isinstance(result, list) else [result]
    except Exception as e:
        raise ValueError(f"Failed to parse database result: {str(e)}")

# --- Define Tools for Agent ---

@tool
def get_latest_crypto_stats(symbol: str) -> dict:
    """
    Retrieves the most recent price, market cap, volume, and 24h change for a given cryptocurrency symbol.
    Always fetches the single most recent entry using ORDER BY timestamp DESC LIMIT 1.
    Returns a dictionary with keys: symbol, price_usd, market_cap_usd, volume_24h_usd, percent_change_24h, timestamp
    """
    db = get_db()
    # Sanitize symbol to prevent SQL injection
    symbol_upper = symbol.upper().replace("'", "''")
    query = f"""
    SELECT symbol, price_usd, market_cap_usd, volume_24h_usd, percent_change_24h, timestamp
    FROM crypto_prices 
    WHERE symbol = '{symbol_upper}'
    ORDER BY timestamp DESC 
    LIMIT 1;
    """
    
    try:
        result = db.run(query)
        
        if not result or result == "[]":
            return {"error": f"No data found for symbol: {symbol}"}
        
        parsed_result = parse_db_result(result)
        if not parsed_result or len(parsed_result) == 0:
            return {"error": f"No data found for symbol: {symbol}"}
        
        row = parsed_result[0]
        return {
            "symbol": row[0],
            "price_usd": float(row[1]),
            "market_cap_usd": float(row[2]),
            "volume_24h_usd": float(row[3]),
            "percent_change_24h": float(row[4]),
            "timestamp": row[5].isoformat() if hasattr(row[5], 'isoformat') else str(row[5])
        }
    except Exception as e:
        return {"error": f"Failed to retrieve data for {symbol}: {str(e)}"}

@tool
def compare_crypto_stats(symbol1: str, symbol2: str) -> dict:
    """
    Compares the most recent price, market cap, volume, and 24h change for two cryptocurrency symbols.
    Always fetches the single most recent entry for each using ORDER BY timestamp DESC LIMIT 1.
    """
    db = get_db()
    
    # Sanitize symbols to prevent SQL injection
    symbol1_upper = symbol1.upper().replace("'", "''")
    symbol2_upper = symbol2.upper().replace("'", "''")
    
    query1 = f"""
    SELECT symbol, price_usd, market_cap_usd, volume_24h_usd, percent_change_24h, timestamp
    FROM crypto_prices 
    WHERE symbol = '{symbol1_upper}'
    ORDER BY timestamp DESC 
    LIMIT 1;
    """
    query2 = f"""
    SELECT symbol, price_usd, market_cap_usd, volume_24h_usd, percent_change_24h, timestamp
    FROM crypto_prices 
    WHERE symbol = '{symbol2_upper}'
    ORDER BY timestamp DESC 
    LIMIT 1;
    """
    
    data1 = data2 = None
    
    try:
        result1 = db.run(query1)
        if result1 and result1 != "[]":
            parsed = parse_db_result(result1)
            if parsed:
                r = parsed[0]
                data1 = {
                    "symbol": r[0],
                    "price_usd": float(r[1]),
                    "market_cap_usd": float(r[2]),
                    "volume_24h_usd": float(r[3]),
                    "percent_change_24h": float(r[4])
                }
    except Exception as e:
        pass
        
    try:
        result2 = db.run(query2)
        if result2 and result2 != "[]":
            parsed = parse_db_result(result2)
            if parsed:
                r = parsed[0]
                data2 = {
                    "symbol": r[0],
                    "price_usd": float(r[1]),
                    "market_cap_usd": float(r[2]),
                    "volume_24h_usd": float(r[3]),
                    "percent_change_24h": float(r[4])
                }
    except Exception as e:
        pass
    
    if not data1 or not data2:
        return {"error": f"Could not retrieve data for both {symbol1} and {symbol2}."}
    
    return {
        "comparison": {
            symbol1.upper(): data1,
            symbol2.upper(): data2
        }
    }

@tool
def compare_multiple_crypto_stats(symbols: str) -> dict:
    """
    Compares the most recent price, market cap, volume, and 24h change for multiple cryptocurrency symbols.
    Accepts a comma-separated string of symbols (e.g., 'BTC,ETH,SHIB') or space-separated (e.g., 'BTC ETH SHIB').
    Dynamically handles ANY number of cryptocurrencies (2, 3, 5, 10, etc.).
    Returns a dictionary with stats for each symbol, sorted by price descending.
    """
    db = get_db()
    
    # Parse symbols - handle both comma and space separated
    if ',' in symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        symbol_list = [s.strip().upper() for s in symbols.split()]
    
    symbol_list = list(set([s for s in symbol_list if s]))  # Remove duplicates
    
    if len(symbol_list) == 0:
        return {"error": "No symbols provided"}
    
    results = {}
    
    for symbol in symbol_list:
        # Sanitize symbol to prevent SQL injection
        symbol_sanitized = symbol.replace("'", "''")
        query = f"""
        SELECT symbol, price_usd, market_cap_usd, volume_24h_usd, percent_change_24h, timestamp
        FROM crypto_prices 
        WHERE symbol = '{symbol_sanitized}'
        ORDER BY timestamp DESC 
        LIMIT 1;
        """
        
        try:
            result = db.run(query)
            if result and result != "[]":
                parsed = parse_db_result(result)
                if parsed:
                    r = parsed[0]
                    results[symbol] = {
                        "symbol": r[0],
                        "price_usd": float(r[1]),
                        "market_cap_usd": float(r[2]),
                        "volume_24h_usd": float(r[3]),
                        "percent_change_24h": float(r[4]),
                        "timestamp": r[5].isoformat() if hasattr(r[5], 'isoformat') else str(r[5])
                    }
            else:
                results[symbol] = {"error": f"No data found for {symbol}"}
        except Exception as e:
            results[symbol] = {"error": str(e)}
    
    # Sort by price descending for easier comparison
    sorted_results = dict(sorted(
        results.items(),
        key=lambda x: x[1].get("price_usd", 0) if "price_usd" in x[1] else 0,
        reverse=True
    ))
    
    return {"comparison": sorted_results}

# --- Agent Initialization ---

def get_conversational_sql_agent():
    """Initializes a conversational agent with custom tools and memory."""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found! Please set it in your environment or secrets.")
            st.stop()
            
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7, groq_api_key=groq_api_key)

        # Define tools
        tools = [get_latest_crypto_stats, compare_crypto_stats, compare_multiple_crypto_stats]

        # Create the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are StockyTalky, a friendly yet professional cryptocurrency market assistant. Think of yourself as a knowledgeable friend who genuinely cares about helping people understand crypto markets.

**Your Communication Style:**
- Be warm, conversational, and approachable—not robotic or formulaic
- Use natural language and vary your sentence structure
- Show enthusiasm when discussing market movements or interesting data
- Be personable: use phrases like "Great question!", "Interesting!", "Let me pull that data for you"
- Feel free to add brief context or insights when relevant

**Formatting Guidelines:**
- Always format prices with commas and 2 decimal places (e.g., $111,486.14)
- Use natural currency units: use "K" for thousands, "M" for millions, "B" for billions, "T" for trillions
- Include the full crypto name with ticker: "Bitcoin (BTC)", "Ethereum (ETH)"
- Market caps and volumes should also use these units for readability

**How to Present Data:**
When you retrieve crypto stats, weave them naturally into your response. Here are conversational examples:
- "Bitcoin's currently trading at $111,486.14, with a solid 24-hour gain of 2.80%."
- "Ethereum's at $2,840.30 right now. It's up 1.5% over the last 24 hours, which is a nice move!"
- When comparing multiple cryptos, highlight the best and worst performers

**Comparison Feature - Dynamic:**
- When comparing 2+ cryptocurrencies, use the compare_multiple_crypto_stats tool
- You can compare any number of cryptocurrencies dynamically (2, 3, 5, 10+, etc.)
- Example: "compare BTC, ETH and SHIB" → passes "BTC,ETH,SHIB" to the tool
- When presenting comparisons, highlight the top performer and lagging performers
- Point out interesting patterns (e.g., "Bitcoin's leading with a +5% gain while SHIB is down 3%")

**Key Rules:**
- Always use the most recent data from the database
- Be honest: if you don't have data for something, say "I don't have current data for that one"
- For off-topic questions, gently redirect
- Keep responses concise but friendly
- Show personality: be authentic and conversational, not stiff
"""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return agent_executor

    except Exception as e:
        st.error(f"Failed to initialize the AI agent: {e}")
        st.stop()

# --- Main Page Function ---

def show():
    """Renders the Crypto Assistant chatbot page."""
    st.title("💰 Crypto Market Data")

    # Initialize agent in session state (avoids caching issues)
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = get_conversational_sql_agent()
    
    agent_executor = st.session_state.agent_executor

    # Initialize chat history for the UI
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to the Crypto Market Data Center. I provide real-time prices and trends for Bitcoin, Ethereum, and other major cryptocurrencies. What would you like to know?"}
        ]

    # Initialize memory for the agent
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=5)

    # Display prior chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask about crypto prices, market cap, or trends..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving live market data..."):
                try:
                    # Load chat history
                    chat_history = st.session_state.agent_memory.load_memory_variables({})["chat_history"]
                    
                    # Invoke the agent
                    response = agent_executor.invoke({
                        "input": prompt,
                        "chat_history": chat_history
                    })
                    
                    assistant_response = response["output"]
                    st.markdown(assistant_response)
                    
                    # Save context
                    st.session_state.agent_memory.save_context(
                        {"input": prompt}, 
                        {"output": assistant_response}
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": assistant_response}
                    )
                    
                except Exception as e:
                    fallback = "I'm having trouble retrieving live data right now. Please try asking about BTC or ETH prices again."
                    st.error(f"An error occurred while processing your request: {str(e)}")
                    print(f"DEBUG: Chatbot error - {str(e)}")  # For debugging
                    import traceback
                    traceback.print_exc()
                    st.session_state.messages.append({"role": "assistant", "content": fallback})