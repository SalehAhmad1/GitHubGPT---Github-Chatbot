import streamlit as st
from dotenv import load_dotenv
from RAG import GitHubGPT  # Assuming this is the class from your notebook
import os

# Load environment variables
load_dotenv()

# Initialize the GitHubGPT class (adjust based on the actual class name and usage)
gpt_bot = GitHubGPT()

# Set up the title and description
st.title("GitHubGPT Chatbot")
st.write("Interact with your codebase through this RAG-based chatbot!")

# Initialize chat history if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input using the new chat_input component
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display chatbot response
    with st.chat_message("assistant"):
        # Replace the following line with the actual call to your chatbot's query method
        response = gpt_bot.query(prompt)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})