import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Get the Groq API key from environment variables
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize the Groq client
client = Groq(api_key=api_key)

def call_groq_mcp(messages, model="llama3-8b-8192", temperature=0.7, max_tokens=1024):
    """
    Calls the Groq API with the provided messages and parameters.

    Args:
        messages (list): A list of message dictionaries, e.g.,
                         [{'role': 'user', 'content': 'Explain the importance of low-latency LLMs.'}]
        model (str): The Groq model to use.
        temperature (float): Controls randomness. Lower is more deterministic.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        str: The content of the AI's response, or None if an error occurs.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            # top_p=1, # Optional: Controls nucleus sampling
            # stop=None, # Optional: Sequence where the API will stop generating
            # stream=False, # Set to True for streaming responses
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred calling Groq API: {e}")
        return None

# if __name__ == '__main__':
#     # Example usage:
#     # Define roles and content from potentially other files/sources
#     system_prompt = "You are a helpful assistant."
#     user_query = "What is the capital of France?"

#     # Structure the messages for the API call
#     conversation_history = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_query},
#     ]

#     print(f"User: {user_query}")

#     # Call the MCP function
#     response = call_groq_mcp(conversation_history)

#     if response:
#         print(f"AI: {response}")

#     # Example with a different role and content
#     follow_up_question = "And what is its population?"
#     conversation_history.append({"role": "assistant", "content": response or "Paris."}) # Add previous AI response
#     conversation_history.append({"role": "user", "content": follow_up_question})

#     print(f"\nUser: {follow_up_question}")
#     response_2 = call_groq_mcp(conversation_history)

#     if response_2:
#         print(f"AI: {response_2}")