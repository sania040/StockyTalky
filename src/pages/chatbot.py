# src/pages/chatbot.py

import streamlit as st
import os
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import MessagesPlaceholder

# --- Agent Initialization (Cached for performance) ---

@st.cache_resource
def get_conversational_sql_agent():
    """Initializes a conversational agent with memory that queries the crypto_prices table."""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found! Please set it in your environment or secrets.")
            st.stop()
            
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, groq_api_key=groq_api_key)
        
        db_uri = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME')}"
        db = SQLDatabase.from_uri(db_uri, include_tables=['crypto_prices'])

        # --- FIX: Emphasized the LIMIT 1 rule ---
        AGENT_PROMPT_PREFIX = """
        You are StockyTalky, an AI assistant answering questions using the `crypto_prices` table ONLY.
        You have chat history for context.

        **CRITICAL RULE: When asked for the current price, market cap, volume, or any single metric, you MUST retrieve only the single most recent entry. Use `ORDER BY timestamp DESC LIMIT 1` in your SQL query.**

        To find the 'best'/'worst' performer, query the latest entries and order by `percent_change_24h` DESC/ASC `LIMIT 1`.
        If the question is conversational (like "hi"), respond conversationally without querying.
        Do NOT use `sql_db_list_tables`. Assume `crypto_prices` is the ONLY table.
        First, check the schema of `crypto_prices` using `sql_db_schema`. Then, construct and run your query using `sql_db_query`.
        """
        
        if "agent_memory" not in st.session_state:
            st.session_state.agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=5)

        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="zero-shot-react-description",
            prefix=AGENT_PROMPT_PREFIX,
            verbose=True,
            handle_parsing_errors=True,
            agent_executor_kwargs={
                "memory": st.session_state.agent_memory,
                "input_variables": ["input", "agent_scratchpad", "chat_history"],
            }
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
                    # Get chat history
                    chat_history = st.session_state.agent_memory.load_memory_variables({})["chat_history"]

                    # Invoke the agent
                    response = agent_executor.invoke({
                        "input": prompt,
                        "chat_history": chat_history
                    })
                    
                    st.markdown(response["output"])
                    
                    # Save context manually
                    st.session_state.agent_memory.save_context({"input": prompt}, {"output": response["output"]})
                    
                    # Save for UI
                    st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                    
                except Exception as e:
                    st.error(f"Sorry, I ran into an error: {e}")