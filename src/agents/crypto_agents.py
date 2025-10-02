import os
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class CryptoAgent:
    """Base agent class for crypto analysis with context awareness"""
    
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.conversation_history = []
        self.current_symbol = None
        self.current_data = None
        self.reset_conversation()
    
    def set_context(self, symbol: str, crypto_data: Optional[Dict[str, Any]] = None):
        """Set the current cryptocurrency context for the agent"""
        self.current_symbol = symbol
        self.current_data = crypto_data
        
        # Add context to conversation history
        if crypto_data:
            context_message = self._format_crypto_data(symbol, crypto_data)
            self.conversation_history.append({
                "role": "system", 
                "content": f"Current context: Analyzing {symbol}. {context_message}"
            })
    
    def _format_crypto_data(self, symbol: str, data: Dict[str, Any]) -> str:
        """Format crypto data into a readable context string"""
        try:
            return f"""
Latest data for {symbol}:
- Current Price: ${data.get('price_usd', 0):,.2f} USD
- 24h Change: {data.get('percent_change_24h', 0):+.2f}%
- Market Cap: ${data.get('market_cap_usd', 0):,.0f} USD
- 24h Volume: ${data.get('volume_24h_usd', 0):,.0f} USD
- Last Updated: {data.get('timestamp', 'N/A')}
"""
        except Exception:
            return f"Analyzing {symbol} (limited data available)"
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.current_symbol = None
        self.current_data = None
    
    def ask(self, query: str, include_context: bool = True) -> str:
        """Send a query to the LLM and get a response"""
        # Enhance query with current context if available
        enhanced_query = query
        if include_context and self.current_symbol and self.current_data:
            enhanced_query = f"{query}\n\n[Context: Currently analyzing {self.current_symbol}]"
        
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": enhanced_query})
        
        # Make API request to LLM
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "openai/gpt-oss-120b",
            "messages": self.conversation_history,
            "temperature": 0.7,
            "max_tokens": 1500,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            assistant_response = result["choices"][0]["message"]["content"]
            
            # Add assistant response to conversation history
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
    """Agent specialized in analyzing specific cryptocurrencies"""
    
    def __init__(self):
        system_prompt = """You are a professional cryptocurrency analysis expert with deep knowledge of blockchain technology, market dynamics, and digital assets.

Your role is to provide comprehensive, data-driven analysis about cryptocurrencies while maintaining a professional and accessible tone.

When analyzing a cryptocurrency, consider:
1. **Technology & Fundamentals**: Blockchain architecture, consensus mechanism, use case, and innovation
2. **Market Position**: Market cap ranking, adoption rate, competitive advantages
3. **Team & Development**: Core team credentials, development activity, community support
4. **Tokenomics**: Supply dynamics, distribution, inflation/deflation mechanisms
5. **Risk Factors**: Technical risks, regulatory concerns, market volatility, competition
6. **Recent Developments**: Updates, partnerships, ecosystem growth

Guidelines:
- Use clear, professional language suitable for both beginners and experienced investors
- Present balanced views with both opportunities and risks
- Support claims with data when available
- Avoid speculation; focus on verifiable information
- Use analogies to explain complex concepts
- Structure responses with clear headings and bullet points
- Be conversational yet authoritative

Remember: You're advising someone making important financial decisions. Be thorough, objective, and helpful."""
        
        super().__init__("Cryptocurrency Analyst", system_prompt)


class InvestmentRationaleAgent(CryptoAgent):
    """Agent specialized in explaining investment rationales"""
    
    def __init__(self):
        system_prompt = """You are a professional cryptocurrency investment advisor providing educational analysis to help investors make informed decisions.

Your role is to deliver clear, actionable investment perspectives while maintaining regulatory compliance and professional standards.

When providing investment analysis:
1. **Disclaimer First**: Always start with a brief disclaimer about not providing financial advice
2. **Current Metrics**: Analyze price action, volume, market cap, and momentum indicators
3. **Technical Analysis**: Discuss support/resistance levels, trends, and key indicators (RSI, moving averages)
4. **Fundamental Strengths**: Highlight project fundamentals, adoption metrics, and value proposition
5. **Risk Assessment**: Clearly outline potential downsides and risk factors
6. **Market Context**: Consider broader crypto market conditions and macro factors
7. **Actionable Insights**: Provide considerations for timing, position sizing, and risk management

Response Structure:
- **Investment Outlook**: Brief summary (Bullish/Neutral/Bearish with confidence level)
- **Key Strengths**: 2-3 main positive factors
- **Key Risks**: 2-3 main concerns
- **Market Metrics**: Current price action and indicators
- **Considerations**: Practical points for decision-making

Tone:
- Professional yet approachable
- Confident but not overly definitive
- Educational and informative
- Risk-aware and balanced

Never:
- Give direct "buy" or "sell" commands
- Make price predictions
- Guarantee returns
- Ignore risks or downsides"""
        
        super().__init__("Investment Analyst", system_prompt)


class MarketSummaryAgent(CryptoAgent):
    """Agent specialized in summarizing market conditions"""
    
    def __init__(self):
        system_prompt = """You are a professional cryptocurrency market analyst specializing in market intelligence and trend analysis.

Your role is to provide clear, actionable market summaries that help investors understand the broader cryptocurrency landscape.

When summarizing markets, cover:
1. **Market Sentiment**: Overall bullish/bearish sentiment and confidence levels
2. **Major Trends**: Dominant narratives (DeFi, NFTs, L2s, AI tokens, etc.)
3. **Top Performers**: Leading gainers and their drivers
4. **Key Movers**: Significant price movements and why they matter
5. **Sector Analysis**: Performance across different crypto categories
6. **Macro Factors**: Interest rates, regulations, institutional activity
7. **Technical Outlook**: Key support/resistance levels for BTC/ETH

Response Format:
- **Market Overview**: One-sentence summary of current conditions
- **Key Highlights**: 3-5 bullet points of most important developments
- **Sector Performance**: Quick breakdown by category
- **What to Watch**: Upcoming events or developing situations

Style Guidelines:
- Start with the most important information
- Use clear, concise language
- Include specific data points (percentages, volumes)
- Maintain objectivity; avoid hype or fear-mongering
- Structure for quick scanning (bullets, short paragraphs)
- Professional tone suitable for investors and traders

Remember: Readers may be making decisions based on your analysis. Be accurate, timely, and balanced."""
        
        super().__init__("Market Intelligence", system_prompt)


def enhance_prompt_with_data(agent: CryptoAgent, query: str, crypto_data: Dict[str, Any]) -> str:
    """
    Enhance the user query with available crypto data before sending to agent
    
    Args:
        agent: The CryptoAgent instance
        query: User's original query
        crypto_data: Dictionary containing cryptocurrency data
    
    Returns:
        Enhanced query string with formatted data context
    """
    if not crypto_data:
        return query
    
    # Extract symbol from data or use agent's current context
    symbol = crypto_data.get('symbol', agent.current_symbol or 'Unknown')
    
    # Format the data professionally
    data_context = f"\n\n📊 **Current Market Data for {symbol}:**\n"
    
    # Price information
    if 'price_usd' in crypto_data:
        data_context += f"💰 Price: ${crypto_data['price_usd']:,.2f} USD\n"
    
    # 24h change with emoji indicator
    if 'percent_change_24h' in crypto_data:
        change = crypto_data['percent_change_24h']
        emoji = "📈" if change > 0 else "📉"
        data_context += f"{emoji} 24h Change: {change:+.2f}%\n"
    
    # Market cap
    if 'market_cap_usd' in crypto_data:
        market_cap = crypto_data['market_cap_usd']
        if market_cap >= 1e9:
            data_context += f"🏦 Market Cap: ${market_cap/1e9:.2f}B\n"
        elif market_cap >= 1e6:
            data_context += f"🏦 Market Cap: ${market_cap/1e6:.2f}M\n"
        else:
            data_context += f"🏦 Market Cap: ${market_cap:,.0f}\n"
    
    # Volume
    if 'volume_24h_usd' in crypto_data:
        volume = crypto_data['volume_24h_usd']
        if volume >= 1e9:
            data_context += f"📊 24h Volume: ${volume/1e9:.2f}B\n"
        elif volume >= 1e6:
            data_context += f"📊 24h Volume: ${volume/1e6:.2f}M\n"
        else:
            data_context += f"📊 24h Volume: ${volume:,.0f}\n"
    
    # Timestamp
    if 'timestamp' in crypto_data:
        data_context += f"⏰ Last Updated: {crypto_data['timestamp']}\n"
    
    # Combine with original query
    enhanced_query = f"{query}{data_context}\n\nPlease incorporate this current market data into your analysis."
    
    return enhanced_query


def format_agent_response(response: str, agent_name: str) -> str:
    """
    Format agent response with professional styling
    
    Args:
        response: Raw response from agent
        agent_name: Name of the responding agent
    
    Returns:
        Formatted response string
    """
    # Add professional header
    header = f"### 🤖 {agent_name}\n\n"
    
    # Add footer with helpful tips
    footer = "\n\n---\n*💡 Tip: You can ask follow-up questions for more detailed analysis.*"
    
    return f"{header}{response}{footer}"