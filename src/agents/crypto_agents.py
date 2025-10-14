import os
import json
import requests
from dotenv import load_dotenv
<<<<<<< HEAD
from typing import List, Dict, Any
=======
from typing import List, Dict, Any, Optional
>>>>>>> raw

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class CryptoAgent:
<<<<<<< HEAD
    """Base agent class for crypto analysis"""
=======
    """Base agent class for crypto analysis with context awareness"""
>>>>>>> raw
    
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.conversation_history = []
<<<<<<< HEAD
        self.reset_conversation()
    
=======
        self.current_symbol = None
        self.current_data = None
        self.historical_data = None
        self.reset_conversation()
    
    def set_context(self, symbol: str, crypto_data: Optional[Dict[str, Any]] = None, historical_data: Optional[Any] = None):
        """Set the current cryptocurrency context for the agent"""
        self.current_symbol = symbol
        self.current_data = crypto_data
        self.historical_data = historical_data
        
        # Add context to conversation history
        context_parts = []
        
        if crypto_data:
            context_parts.append(self._format_crypto_data(symbol, crypto_data))
        
        if historical_data is not None and not historical_data.empty:
            context_parts.append(self._format_historical_data(symbol, historical_data))
        
        if context_parts:
            full_context = f"Current context: Analyzing {symbol}.\n\n" + "\n\n".join(context_parts)
            self.conversation_history.append({
                "role": "system", 
                "content": full_context
            })
    
    def _format_crypto_data(self, symbol: str, data: Dict[str, Any]) -> str:
        """Format crypto data into a readable context string"""
        try:
            return f"""
**Latest Market Data for {symbol}:**
- Current Price: ${data.get('price_usd', 0):,.2f} USD
- 24h Change: {data.get('percent_change_24h', 0):+.2f}%
- Market Cap: ${data.get('market_cap_usd', 0):,.0f} USD
- 24h Volume: ${data.get('volume_24h_usd', 0):,.0f} USD
- Last Updated: {data.get('timestamp', 'N/A')}
"""
        except Exception:
            return f"Analyzing {symbol} (limited data available)"
    
    def _format_historical_data(self, symbol: str, df) -> str:
        """Format historical data into analysis context"""
        try:
            if df.empty:
                return f"No historical data available for {symbol}."
            
            # Calculate statistics
            total_records = len(df)
            date_range = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
            
            price_high = df['price_usd'].max()
            price_low = df['price_usd'].min()
            price_current = df['price_usd'].iloc[-1]
            price_avg = df['price_usd'].mean()
            
            # Find best and worst times
            highest_date = df.loc[df['price_usd'].idxmax(), 'timestamp']
            lowest_date = df.loc[df['price_usd'].idxmin(), 'timestamp']
            
            # Calculate % from high/low
            pct_from_high = ((price_current - price_high) / price_high) * 100
            pct_from_low = ((price_current - price_low) / price_low) * 100
            
            # Trend analysis
            price_change_total = ((price_current - df['price_usd'].iloc[0]) / df['price_usd'].iloc[0]) * 100
            
            return f"""
**Historical Data Analysis for {symbol}:**

📊 **Data Coverage:**
- Total Records: {total_records}
- Date Range: {date_range}

💰 **Price Statistics:**
- Current Price: ${price_current:,.2f}
- All-Time High (in our data): ${price_high:,.2f} (on {highest_date})
- All-Time Low (in our data): ${price_low:,.2f} (on {lowest_date})
- Average Price: ${price_avg:,.2f}

📈 **Position Analysis:**
- Distance from High: {pct_from_high:+.2f}% ({abs(pct_from_high):.1f}% {'below' if pct_from_high < 0 else 'above'} peak)
- Distance from Low: {pct_from_low:+.2f}% ({pct_from_low:.1f}% above bottom)
- Overall Trend: {price_change_total:+.2f}% change since first record

💡 **Key Insights:**
- Best buying opportunity was at ${price_low:,.2f} on {lowest_date}
- Selling at peak ${price_high:,.2f} on {highest_date} would have been optimal
- Current price is {'above' if price_current > price_avg else 'below'} historical average

**IMPORTANT:** Base your analysis on THIS ACTUAL DATA from our database, not general market knowledge.
"""
        except Exception as e:
            return f"Error analyzing historical data: {str(e)}"
    
>>>>>>> raw
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
<<<<<<< HEAD
    
    def ask(self, query: str) -> str:
        """Send a query to the LLM and get a response"""
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
=======
        self.current_symbol = None
        self.current_data = None
        self.historical_data = None
    
    def ask(self, query: str, include_context: bool = True) -> str:
        """Send a query to the LLM and get a response"""
        # Enhance query with current context if available
        enhanced_query = query
        if include_context and self.current_symbol:
            enhanced_query = f"{query}\n\n[Context: Currently analyzing {self.current_symbol} based on OUR DATABASE data]"
        
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": enhanced_query})
>>>>>>> raw
        
        # Make API request to LLM
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
<<<<<<< HEAD
            "model": "llama3-70b-8192",  # Using Llama 3 70B via Groq
            "messages": self.conversation_history,
            "temperature": 0.7,
            "max_tokens": 1024
=======
            "model": "llama-3.3-70b-versatile",
            "messages": self.conversation_history,
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.9
>>>>>>> raw
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
<<<<<<< HEAD
                json=payload
=======
                json=payload,
                timeout=30
>>>>>>> raw
            )
            response.raise_for_status()
            
            result = response.json()
            assistant_response = result["choices"][0]["message"]["content"]
            
            # Add assistant response to conversation history
<<<<<<< HEAD
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
        
        except Exception as e:
            return f"Error: {str(e)}"


# Specialized agents
class CoinAnalysisAgent(CryptoAgent):
    """Agent specialized in analyzing specific cryptocurrencies"""
    
    def __init__(self):
        system_prompt = """You are a cryptocurrency analysis expert. 
        Your job is to provide detailed analysis about specific cryptocurrencies.
        Focus on fundamentals, technology, team, market position, and potential risks.
        Be objective and balanced in your assessment. 
        Use data-driven insights when available.
        Always provide a balanced view with both potential upsides and risks."""
        
        super().__init__("Coin Analyst", system_prompt)


class InvestmentRationaleAgent(CryptoAgent):
    """Agent specialized in explaining investment rationales"""
    
    def __init__(self):
        system_prompt = """You are an investment rationale expert for cryptocurrencies.
        Your job is to provide a clear investment analysis when users ask questions like "Should I buy/invest in this?"
        
        When answering investment questions:
        1. Start with a disclaimer that you're not providing financial advice
        2. Present a balanced analysis with key pros and cons
        3. Discuss current metrics (price, market cap, volume, etc.)
        4. Explain relevant technical indicators (when available)
        5. Mention broader market context
        6. End with action-oriented considerations, not direct "buy" or "sell" commands
        
        Keep responses concise, factual and directly answer the question the user is asking.
        Do not ask the user which cryptocurrency they're interested in - that information should already be in the prompt.
        Use the provided cryptocurrency data to inform your analysis.
        """
        
        super().__init__("Investment Rationale", system_prompt)


class MarketSummaryAgent(CryptoAgent):
    """Agent specialized in summarizing market conditions"""
    
    def __init__(self):
        system_prompt = """You are a cryptocurrency market summarization expert.
        Your job is to provide clear, concise summaries of market conditions, trends, and sentiment.
        Focus on:
        1. Overall market direction and key indicators
        2. Major sector trends (DeFi, L1s, meme coins, etc.)
        3. Macro factors influencing the crypto market
        4. Notable news events affecting sentiment
        5. be to the point and avoid unnecessary jargon
        6. Use data and statistics to support your summary
        Be concise and factual. Avoid speculation or personal opinion."""
        
        super().__init__("Market Summarizer", system_prompt)


# Add function to inject crypto data into agent prompts
=======
            self.conversation_history.append({
                "role": "assistant", 
                "content": assistant_response
            })
            
            return assistant_response
        
        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            return f"🔌 Connection error: Unable to reach the AI service. Please check your internet connection."
        except Exception as e:
            return f"❌ An unexpected error occurred: {str(e)}"


class CoinAnalysisAgent(CryptoAgent):
    """Agent specialized in analyzing specific cryptocurrencies based on actual database data"""
    
    def __init__(self):
        system_prompt = """You are a DATA-DRIVEN cryptocurrency analyst. Your specialty is analyzing ACTUAL HISTORICAL DATA from the user's database.

**CRITICAL RULES:**
1. ALWAYS base your analysis on the HISTORICAL DATA provided in the context
2. NEVER make up information about coins - use ONLY the data shown
3. Focus on: price trends, highs/lows, best buying/selling opportunities from the DATA
4. Reference specific dates and prices from the historical data
5. Calculate and mention percentage changes, trends, and patterns you observe in the DATA

**Your Analysis Should Include:**
- 📊 Data coverage (how much data we have, date range)
- 💰 Price analysis (current vs historical high/low/average)
- 📈 Trend identification (uptrend/downtrend/sideways from the data)
- 💡 Actionable insights (when to buy/sell based on historical patterns)
- ⚠️ Data-based observations (volatility, support/resistance levels from data)

**Example Good Response:**
"Based on our database, we have 150 records for BTC spanning from Jan 1 to Dec 31. The price peaked at $65,000 on Nov 10 and bottomed at $40,000 on July 15. Currently at $50,000, it's 23% below the high and 25% above the low. The data shows an overall uptrend of +15% during this period. Best buying opportunity was at $40,000 in July."

**What NOT to do:**
- Don't talk about who created the coin or its technical specifications
- Don't discuss general blockchain concepts
- Don't mention information not in the provided data
- Don't give generic crypto advice

Remember: You're analyzing THEIR DATA, not the entire crypto market. Be specific with dates, prices, and percentages from the historical data provided."""
        
        super().__init__("Data-Driven Analyst", system_prompt)


class InvestmentRationaleAgent(CryptoAgent):
    """Agent specialized in explaining investment opportunities based on actual data"""
    
    def __init__(self):
        system_prompt = """You are an investment analyst who ONLY uses ACTUAL DATABASE DATA to provide insights.

**YOUR MISSION:**
Analyze the historical price data to identify investment opportunities, risks, and timing strategies.

**ALWAYS Include:**
1. **Current Position Analysis:**
   - Where is the price now vs historical high/low?
   - Is it near resistance or support (based on data)?

2. **Historical Performance:**
   - Best entry points (lowest prices in data)
   - Worst entry points (highest prices in data)
   - Average performance over the data period

3. **Data-Based Recommendations:**
   - When has buying been profitable (based on historical patterns)?
   - What price levels show support/resistance?
   - Risk/reward based on historical volatility

4. **Timing Insights:**
   - "If you had bought at $X on [date], you'd be up/down Y%"
   - Best and worst case scenarios from the data
   - Current price position (expensive/cheap compared to history)

**Disclaimer First:**
"⚠️ This analysis is based solely on historical data in your database and is for educational purposes only. Past performance does not guarantee future results."

**Focus on DATA, not speculation:**
✅ "In our data, BTC dropped to $40K on July 15 - that was the best entry point"
❌ "Bitcoin was created by Satoshi Nakamoto in 2009"

Be specific, use actual numbers, dates, and percentages from the historical data."""
        
        super().__init__("Investment Data Analyst", system_prompt)


class MarketSummaryAgent(CryptoAgent):
    """Agent specialized in summarizing market conditions from database data"""
    
    def __init__(self):
        system_prompt = """You are a market data analyst who summarizes trends across multiple cryptocurrencies in the database.

**Your Role:**
Provide market summaries based on ACTUAL DATA from the user's database, not general market knowledge.

**When Summarizing:**
1. **Data-Driven Overview:**
   - Which coins have data available?
   - Date ranges and record counts
   - Overall market trend from the data

2. **Performance Comparison:**
   - Best/worst performers based on the data
   - Price movements and volatility
   - Relative strength between coins

3. **Opportunity Identification:**
   - Which coins show buying opportunities (based on historical patterns)?
   - Which are near historical highs/lows?
   - Risk assessment based on data volatility

**Example:**
"Across our database, we're tracking 5 cryptocurrencies with 500+ records each. Over the past 30 days of data, BTC is up 12% while ETH is down 5%. BTC is currently 15% below its data high, presenting a potential opportunity. The data shows..."

**Remember:**
- Only discuss coins that have data in the database
- Reference specific time periods from the data
- Use actual percentages and numbers
- No speculation - only data-based observations"""
        
        super().__init__("Market Data Intelligence", system_prompt)


>>>>>>> raw
def enhance_prompt_with_data(agent: CryptoAgent, query: str, crypto_data: Dict[str, Any]) -> str:
    """
    Enhance the user query with available crypto data before sending to agent
    """
    if not crypto_data:
        return query
<<<<<<< HEAD
        
    # Format the data as a string
    data_context = "Here is the latest data for this cryptocurrency:\n"
    for key, value in crypto_data.items():
        if key != "symbol":  # Skip the symbol as it's likely in the query
            if isinstance(value, float):
                data_context += f"- {key}: {value:.2f}\n"
            else:
                data_context += f"- {key}: {value}\n"
    
    # Combine with original query
    enhanced_query = f"{query}\n\nPlease use the following data in your analysis:\n{data_context}"
    return enhanced_query
=======
    
    symbol = crypto_data.get('symbol', agent.current_symbol or 'Unknown')
    
    data_context = f"\n\n📊 **Current Market Data for {symbol}:**\n"
    
    if 'price_usd' in crypto_data:
        data_context += f"💰 Price: ${crypto_data['price_usd']:,.2f} USD\n"
    
    if 'percent_change_24h' in crypto_data:
        change = crypto_data['percent_change_24h']
        emoji = "📈" if change > 0 else "📉"
        data_context += f"{emoji} 24h Change: {change:+.2f}%\n"
    
    if 'market_cap_usd' in crypto_data:
        market_cap = crypto_data['market_cap_usd']
        if market_cap >= 1e9:
            data_context += f"🏦 Market Cap: ${market_cap/1e9:.2f}B\n"
        elif market_cap >= 1e6:
            data_context += f"🏦 Market Cap: ${market_cap/1e6:.2f}M\n"
        else:
            data_context += f"🏦 Market Cap: ${market_cap:,.0f}\n"
    
    if 'volume_24h_usd' in crypto_data:
        volume = crypto_data['volume_24h_usd']
        if volume >= 1e9:
            data_context += f"📊 24h Volume: ${volume/1e9:.2f}B\n"
        elif volume >= 1e6:
            data_context += f"📊 24h Volume: ${volume/1e6:.2f}M\n"
        else:
            data_context += f"📊 24h Volume: ${volume:,.0f}\n"
    
    if 'timestamp' in crypto_data:
        data_context += f"⏰ Last Updated: {crypto_data['timestamp']}\n"
    
    enhanced_query = f"{query}{data_context}\n\n**IMPORTANT: Analyze based on the historical data provided in your context, not general knowledge.**"
    
    return enhanced_query


def format_agent_response(response: str, agent_name: str) -> str:
    """Format agent response with professional styling"""
    header = f"### 🤖 {agent_name}\n\n"
    footer = "\n\n---\n*💡 Tip: Ask follow-up questions about specific dates or price levels in our data.*"
    return f"{header}{response}{footer}"
>>>>>>> raw
