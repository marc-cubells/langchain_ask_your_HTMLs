import streamlit as st
import random
from streamlit_chat import message

# from hr_agent_backend_azure import get_response
from hr_agent_backend import get_response

def process_input(user_input):
    response = get_response(user_input)
    return response

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with your HR Bot ", page_icon=":speech_balloon:")
    st.header("Chat with your HR Bot  💬")
    st.info("Type your query in the chat window.")

    # Initialize the Streamlit chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input and display the chatbot's response
    if user_input := st.chat_input("Ask your questions from the HTML files"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
        result = process_input(user_input)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = result
            message_placeholder.markdown(full_response + "|")

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()            