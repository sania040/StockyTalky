# src/pages/chatbot.py

import streamlit as st
import os
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

# --- Agent Initialization (Cached for performance) ---

@st.cache_resource
def get_conversational_sql_agent():
    """Initializes a conversational agent with memory that can query the database."""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found! Please set it in your environment or secrets.")
            st.stop()
            
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, groq_api_key=groq_api_key)
        
        db_uri = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME')}"
        db = SQLDatabase.from_uri(db_uri)

        # --- FIX: A much more robust prompt for the agent ---
        AGENT_PROMPT_PREFIX = """
        You are a friendly and helpful crypto data analyst named StockyTalky.
        You have access to a PostgreSQL database with a table called `crypto_prices`.
        You have a conversation history to understand follow-up questions.
        You must work step-by-step.

        **RULES:**
        1.  If the user's question can be answered by querying the database (e.g., asking for a price, volume, trend), you MUST use your tools to query for the data.
        2.  When asked for a price or metric without a specific time, ALWAYS query for the single most recent entry by using `ORDER BY timestamp DESC LIMIT 1`.
        3.  If the user asks a general conversational question that CANNOT be answered from the database (e.g., "hi", "what is market cap?", "how are you?"), you MUST respond with a friendly, conversational answer **without using your tools**. Your response should be your final answer.
        """
        
        # We need to manage memory manually for this agent type
        if "agent_memory" not in st.session_state:
            st.session_state.agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="zero-shot-react-description",
            prefix=AGENT_PROMPT_PREFIX,
            verbose=True,
            handle_parsing_errors=True,
        )
        
        return agent_executor

    except Exception as e:
        st.error(f"Failed to initialize the AI agent: {e}")
        st.stop()

# --- Main Page Function ---

def show():
    """Renders the Crypto Assistant chatbot page."""
    st.title("💬 Crypto Assistant")

    agent_executor = get_conversational_sql_agent()

    # Initialize chat history for the UI
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm StockyTalky. Ask me anything about the crypto data I've collected!"}]

    # Display prior chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get chat history from our session state memory
                    chat_history = st.session_state.agent_memory.load_memory_variables({})["chat_history"]

                    # Invoke the agent, providing both the new input and the chat history
                    response = agent_executor.invoke({
                        "input": prompt,
                        "chat_history": chat_history
                    })
                    
                    st.markdown(response["output"])
                    
                    # Save the conversation turn to memory
                    st.session_state.agent_memory.save_context({"input": prompt}, {"output": response["output"]})
                    # Save the assistant's response for UI display
                    st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                    
                except Exception as e:
                    st.error(f"Sorry, I ran into an error: {e}")