# src/pages/chatbot.py

import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

# --- Agent Initialization ---

@st.cache_resource
def get_sql_agent_chain():
    """Initializes the LLM, database connection, and the agent chain."""
    try:
        # Check for API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found! Please set it in your environment or secrets.")
            st.stop()
            
        # Initialize the LLM
        llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0,
            groq_api_key=groq_api_key
        )
        
        # Initialize the database connection via LangChain
        db_uri = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME')}"
        db = SQLDatabase.from_uri(db_uri)

        # Create the prompt template
        template = """
        You are a helpful crypto data analyst. You can query a SQL database to answer questions.
        Based on the user's question, first, create a syntactically correct PostgreSQL query to run, then look at the results of the query and return a user-friendly answer.
        
        Do not make up answers. If you don't know the answer, just say that you don't know.
        
        Here is the database schema:
        {schema}
        
        Question: {question}
        SQL Query:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Create the agent chain
        sql_chain = (
            RunnablePassthrough.assign(schema=lambda _: db.get_table_info())
            | prompt
            | llm.bind(stop=["\nSQLResult:"])
            | StrOutputParser()
        )
        
        full_chain = (
            sql_chain
            | (lambda sql_query: {"sql_query": sql_query, "result": db.run(sql_query)})
            | (lambda inputs: f"SQL Query: {inputs['sql_query']}\nSQLResult: {inputs['result']}")
        )
        
        return full_chain

    except Exception as e:
        st.error(f"Failed to initialize the AI agent: {e}")
        st.stop()

# --- Main Page Function ---

def show():
    """Renders the Crypto Assistant chatbot page."""
    st.title("💬 Crypto Assistant")
    st.info("Ask me questions about the cryptocurrency data in my database. For example: 'What is the latest price of BTC?' or 'Show me the top 3 gainers yesterday'.")

    agent_chain = get_sql_agent_chain()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to UI and history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke the agent to get the SQL and the result
                    intermediate_result = agent_chain.invoke({"question": prompt})

                    # Now, create a final prompt for the LLM to give a natural answer
                    final_prompt = f"""
                    Based on the user's question and the following SQL query and its result, please provide a concise, user-friendly answer.
                    
                    Question: {prompt}
                    {intermediate_result}
                    
                    Answer:
                    """
                    
                    final_llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.2, groq_api_key=os.getenv("GROQ_API_KEY"))
                    response = final_llm.invoke(final_prompt).content
                    st.markdown(response)

                    # Add the final response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Sorry, I ran into an error: {e}")