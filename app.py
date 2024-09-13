import streamlit as st
from dotenv import load_dotenv
from RAG import GitHubGPT  # Assuming this is the class from your file
import os

# Load environment variables
load_dotenv()

# Initialize the GitHubGPT class
@st.cache_resource
def initialize_gpt():
    bot = GitHubGPT()
    bot.add_repo('https://github.com/SaschaNe/creatify-app')
    # bot.load_repo()
    return bot

gpt_bot = initialize_gpt()

# Create placeholders for thread ID and assistant ID at the top
thread_id_placeholder = st.empty()  # Placeholder for Thread ID (initially empty)
assistant_id_placeholder = st.empty()  # Placeholder for Assistant ID

# Set up the title and description
st.title("GitHubGPT Chatbot")
st.write("Interact with your codebase through this RAG-based chatbot!")

# Display the assistant ID immediately at the top
assistant_id_placeholder.write(f"**Assistant ID:** {gpt_bot.assistant_id}")

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
    
    # Create a placeholder for streaming assistant response
    assistant_message = st.chat_message("assistant")
    message_placeholder = assistant_message.markdown("...")

    # Stream chatbot response
    response_stream = gpt_bot.query(prompt)  # Stream the response as it's generated
    response = ""

    # Concatenate the response as it's streamed
    for chunk in response_stream:
        response += chunk
        message_placeholder.markdown(response)  # Update the displayed message chunk by chunk

    # Add assistant response to chat history once streaming is complete
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Once the thread ID is set (after the first query), display it
    if gpt_bot.thread_id:
        thread_id_placeholder.write(f"**Thread ID:** {gpt_bot.thread_id}")
