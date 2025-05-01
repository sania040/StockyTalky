import os
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class CryptoAgent:
    """Base agent class for crypto analysis"""
    
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.conversation_history = []
        self.reset_conversation()
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def ask(self, query: str) -> str:
        """Send a query to the LLM and get a response"""
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Make API request to LLM
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama3-70b-8192",  # Using Llama 3 70B via Groq
            "messages": self.conversation_history,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            assistant_response = result["choices"][0]["message"]["content"]
            
            # Add assistant response to conversation history
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
def enhance_prompt_with_data(agent: CryptoAgent, query: str, crypto_data: Dict[str, Any]) -> str:
    """
    Enhance the user query with available crypto data before sending to agent
    """
    if not crypto_data:
        return query
        
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