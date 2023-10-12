# Importing necessary libraries
from dotenv import load_dotenv
import os

# Importing Streamlit, a Python framework for building interactive web apps
import streamlit as st

# Importing the classes needed from the langchain modules
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter     import RecursiveCharacterTextSplitter
from langchain.chains            import ConversationalRetrievalChain
from langchain.chat_models       import ChatOpenAI
from langchain.memory            import ConversationBufferMemory
from langchain.vectorstores      import FAISS

# Directory where the HTML files are stored
HTML_FILES_DIR = "./html_files"

# read all the HTML files and concatenate their content into a single string
def get_html_text():
    text = ""
    for filename in os.listdir(HTML_FILES_DIR):
        if filename.endswith(".html"):
            with open(os.path.join(HTML_FILES_DIR, filename), 'r', encoding='utf-8') as f:
                text += f.read()
    return text

# Load environment variables
load_dotenv()

# Main Streamlit app
def main():
    # Set the title and icon for the Streamlit page
    st.set_page_config(page_title="Chat with multiple HTML files", page_icon=":speech_balloon:")
    st.header("Chat with multiple HTML files  💬")

    try:
        # Retrieve the concatenated content from all the HTML files
        text = get_html_text()
        st.info("Type your query in the chat window.")
    except FileNotFoundError:
        st.error(f"Error: HTMLs files not found: {HTML_FILES_DIR}")
        return
    except Exception as e:
        st.error(f"Error occurred while reading the HTML files: {e}")
        return

    # Instantiate a text splitter to divide the HTML content into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size      = 1000,
        chunk_overlap   = 150,
        length_function = len
    )

    # Process the HTML content and create the list of document chunks
    documents = text_splitter.split_text(text=text)

    # Vectorize the documents and create a vectorstore using FAISS
    embeddings  = OpenAIEmbeddings(model = "text-embedding-ada-002")
    vectorstore = FAISS.from_texts(documents, embedding=embeddings)

    # Save the processed data in Streamlit's session state for later use
    st.session_state.processed_data = {
        "document_chunks": documents,
        "vectorstore": vectorstore,
    }

    # Initialize the Langchain chatbot using the OpenAI model
    openai_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    # Set up a buffer for conversation history
    conversation_memory = ConversationBufferMemory(
        memory_key      = "chat_history", 
        return_messages = True, 
        output_key      = "answer"
    )
    
    # Create a retriever for searching the document chunks based on similarity
    retriever = vectorstore.as_retriever(search_type = "similarity")

    # Set up the main ConversationalRetrievalChain for QA with the chatbot
    qa = ConversationalRetrievalChain.from_llm(
        llm                     = openai_llm, 
        retriever               = retriever, 
        memory                  = conversation_memory,
        return_source_documents = False, 
        verbose                 = True,
        output_key              = "answer",
        chain_type              = "stuff",
        max_tokens_limit        = None
    )
    
    # Initialize the Streamlit chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input and display the chatbot's response
    if prompt := st.chat_input("Ask your questions from the HTML files"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = result["answer"]
            message_placeholder.markdown(full_response + "|")

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()