import streamlit as st
from typing import List, Dict
from pathlib import Path
from Driver_streamlit import Driver
from TextPreprocessor import TextPreprocessor
import datetime
import tempfile
import json

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'driver' not in st.session_state:
        st.session_state.driver = Driver(
            llm_type='ollama',
            llm_model='granite3.1-dense:8b-instruct-q4_0'
        )
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'context_files' not in st.session_state:
        st.session_state.context_files = []

def process_uploaded_files(uploaded_files) -> List[str]:
    """Process uploaded files and store them temporarily."""
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        temp_path = Path(temp_dir) / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(str(temp_path))
    
    return saved_paths

def download_conversation(messages: List[Dict], format: str = 'json'):
    """Create downloadable conversation file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == 'json':
        content = json.dumps(messages, indent=2)
        mime = "application/json"
        filename = f"conversation_{timestamp}.json"
    else:  # format == 'txt'
        content = "\n\n".join([
            f"{msg['role'].upper()}:\n{msg['content']}"
            for msg in messages
        ])
        mime = "text/plain"
        filename = f"conversation_{timestamp}.txt"
    
    return content, mime, filename

def main():
    st.set_page_config(
        page_title="Chat with Granite",
        page_icon="💭",
        layout="wide"
    )


    # Add custom CSS styling
    st.markdown(
        """
        <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 100px);
        }

        .stChatMessageContent {
            max-width: 90%;
            margin-left: calc(var(--sidebar-width, 17rem) + 1rem);
        }

        [data-testid="stChatInput"] {
            position: fixed;
            bottom: 0;
            left: calc(var(--sidebar-width, 17rem) + 1rem);
            right: 1rem;
            padding: 1rem;
            z-index: 1000;
            transition: left 0.3s ease;
        }

        .main .block-container {
            padding-bottom: 5rem;
            margin-left: var(--sidebar-width, 17rem);
            transition: margin-left 0.3s ease;
        }

        .stChatMessage {
            margin-left: calc(var(--sidebar-width, 17rem) + 1rem);
            transition: margin-left 0.3s ease;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("💭 Chat with Granite")
    initialize_session_state()

    # Sidebar configuration
    with st.sidebar:
        st.header("Settings")
        
        # Model settings
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more creative"
        )

        st.divider()
        st.header("Conversation Management")
        
        # Export options
        if st.session_state.messages:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save JSON"):
                    content, mime, filename = download_conversation(
                        st.session_state.messages, 'json'
                    )
                    st.download_button(
                        "Download JSON",
                        content,
                        filename,
                        mime
                    )
            with col2:
                if st.button("📄 Save Text"):
                    content, mime, filename = download_conversation(
                        st.session_state.messages, 'txt'
                    )
                    st.download_button(
                        "Download Text",
                        content,
                        filename,
                        mime
                    )

        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
            st.session_state.driver.clear_conversation_history()
            st.rerun()

        # Context file upload
        st.divider()
        st.header("Context Files")
        uploaded_files = st.file_uploader(
            "Upload context files",
            accept_multiple_files=True,
            type=['txt'],
            help="Upload text files for additional context"
        )
        
        if uploaded_files:
            st.session_state.context_files = process_uploaded_files(uploaded_files)
            st.success(f"Uploaded {len(uploaded_files)} files")

    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        with chat_container:
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Add context if available
                    context = ""
                    if st.session_state.context_files:
                        for file_path in st.session_state.context_files:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                context += f"\nContext from {Path(file_path).name}:\n{f.read()}"
                    
                    full_prompt = f"{context}\n\nQuery: {prompt}" if context else prompt
                    response = st.session_state.driver.process_query(full_prompt)
                    
                    text_processor = TextPreprocessor()
                    formatted_response = text_processor.format_text(response)
                    st.markdown(formatted_response)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": formatted_response
                    })

if __name__ == "__main__":
    main()
