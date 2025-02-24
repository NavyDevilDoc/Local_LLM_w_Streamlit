import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os
from typing import List
from pathlib import Path
from Setup import load_environment_variables
from Driver import Driver
import json
import glob
import tempfile
import tkinter as tk
from tkinter import filedialog
from fpdf import FPDF
import datetime

# streamlit run streamlit_app.py


def initialize_driver(llm_type: str, llm_model: str, temperature: float = 0.2):
    """Initialize the Driver with selected model settings."""
    file_path = Path(r"C:\Users\docsp\Desktop\LLM_Only\file_variables.env")
    load_environment_variables(file_path)
    
    return Driver(
        env_path=os.getenv("ENV_PATH"),
        json_path=os.getenv("JSON_PATH"),
        llm_type=llm_type,
        llm_model=llm_model
    )

#=======================================================================================================#

def load_conversation_files():
    """Load available conversation history files."""
    json_path = os.getenv("JSON_PATH")
    if json_path:
        # Include both current and backup conversation files
        history_files = glob.glob(os.path.join(json_path, "conversation_history_*.json"))
        backup_files = glob.glob(os.path.join(json_path, "conversation_backup_*.json"))
        return history_files + backup_files
    return []

#=======================================================================================================#

def select_file_dialog(title: str, filetypes: list) -> str:
    """Open a file dialog and return selected file path."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    return file_path

#=======================================================================================================#

def save_conversation_dialog(messages: list, json_path: str) -> str:
    """Open save dialog and save conversation to selected location."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"conversation_history_{timestamp}.json"
    
    save_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        initialfile=default_filename,
        initialdir=json_path,
        title="Save Conversation As",
        filetypes=[("JSON files", "*.json")]
    )
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2)
    return save_path

#=======================================================================================================#

def process_uploaded_files(uploaded_files: List[UploadedFile]) -> List[str]:
    """Process uploaded files and return their paths."""
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        # Create a temporary file path
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save uploaded file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(temp_path)
        
    return saved_paths

#=======================================================================================================#

def create_conversation_pdf(messages: list, output_path: str):
    """Create a PDF from conversation history with Unicode support."""
    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=15)
            # Use system fonts instead of downloading
            self.default_font = 'Arial'
            self.code_font = 'Courier'

    def output_text(self, text: str, is_code: bool = False):
        """Handle text output with proper encoding and font selection."""
        if is_code:
            self.set_font(self.code_font, size=10)
        else:
            self.set_font(self.default_font, size=12)
        
        # Use UTF-8 encoding
        text = text.encode('utf-8', 'replace').decode('utf-8')
        self.multi_cell(0, 10, txt=text)

    def sanitize_text(text: str) -> str:
        """Sanitize text for PDF output."""
        # Common replacements
        replacements = {
            '\u2014': '-',    # em dash
            '\u2013': '-',    # en dash
            '\u2018': "'",    # single quote
            '\u2019': "'",    # single quote
            '\u201c': '"',    # double quote
            '\u201d': '"',    # double quote
            '\u03c1': 'p',    # rho
            '\u03b1': 'a',    # alpha
            '\u03b2': 'b',    # beta
            '\u03b3': 'y',    # gamma
            '\u03b4': 'd',    # delta
            '\u03bc': 'u',    # mu
            '\u03c0': 'pi',   # pi
            '\u03c3': 's',    # sigma
            '\u03c4': 't',    # tau
            '\u03c6': 'f',    # phi
            '\u03c8': 'ps',   # psi
            '\u03c9': 'w',    # omega

            # Mathematical symbols
            '\u222b': '∫',    # integral
            '\u2211': 'Σ',    # summation
            '\u220f': 'Π',    # product
            '\u221e': '∞',    # infinity
            '\u2248': '≈',    # approximately equal
            '\u2260': '≠',    # not equal
            '\u2264': '≤',    # less than or equal
            '\u2265': '≥',    # greater than or equal
            '\u221a': '√',    # square root
            '\u2202': '∂',    # partial derivative
            '\u2207': '∇',    # nabla/del
            '\u2208': '∈',    # element of
            '\u2209': '∉',    # not element of
            '\u2229': '∩',    # intersection
            '\u222a': '∪',    # union
            
            # Arrows
            '\u2190': '<-',   # left arrow
            '\u2192': '->',   # right arrow
            '\u2194': '<->',  # left-right arrow
            '\u21d2': '=>',   # rightwards double arrow
            '\u21d4': '<=>',  # left-right double arrow
        }

        # Replace known special characters
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Replace any remaining non-Latin1 characters with their closest ASCII equivalent
        return text.encode('ascii', 'replace').decode('ascii')

    try:
        pdf = PDF()
        pdf.add_page()
        font = pdf.default_font
        pdf.set_font(font, size=12)
        
        # Add title
        pdf.set_font(font, 'B', 16)
        pdf.cell(200, 10, txt="Conversation History", ln=True, align='C')
        pdf.set_font(font, size=12)
        
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(200, 10, txt=f"Generated: {timestamp}", ln=True, align='L')
        pdf.ln(10)
        
        # Add messages
        for msg in messages:
            role = msg["role"].upper()
            pdf.set_font(font, 'B', 12)
            pdf.cell(200, 10, txt=f"{role}:", ln=True)
            pdf.set_font(font, size=12)
            
            # Handle multi-line content
            content = sanitize_text(msg["content"].strip())
            
            pdf.multi_cell(0, 10, txt=content)
            pdf.ln(5)
        
        pdf.output(output_path)
        return True
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return False

#=======================================================================================================#

def main():
    st.set_page_config(page_title="LLM Chat Interface", layout="wide")
    st.title("💬 LLM Conversation Interface")

    
    # Initialize session state first
    if 'driver' not in st.session_state:
        st.session_state.driver = initialize_driver('ollama', 'granite3.1-dense:8b-instruct-q4_0')
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7

#=======================================================================================================#

    # Sidebar controls
    with st.sidebar:
        st.header("Model Settings")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Higher values make output more creative, lower more deterministic"
        )

        llm_type = st.selectbox(
            "Select LLM Type",
            ["ollama", "openai"],
            help="Choose between local Ollama models or OpenAI"
        )
        

        # Model selection based on type
        if llm_type == "ollama":
            model_options = [
                "granite3.1-dense:8b-instruct-q4_0",
                "granite3.1-dense:8b-instruct-q8_0",
                "granite-code:20b",
                "llama3.2:3b-instruct-fp16"
            ]
        else:
            model_options = [
                "gpt-4o", 
                "gpt-4o-mini"
            ]
            
        llm_model = st.selectbox("Select Model", model_options)

        # If model changed, reinitialize
        if ('current_model' not in st.session_state or 
            st.session_state.current_model != llm_model):
            st.session_state.driver = initialize_driver(llm_type, llm_model, temperature)
            st.session_state.current_model = llm_model
            st.success(f"Switched to model: {llm_model}"
            )


        
#=======================================================================================================#

        st.header("Conversation Management")
        # Create two rows of columns for better organization
        save_col1, save_col2 = st.columns(2)
        load_col1, load_col2 = st.columns(2)

        # First row: Save and New
        with save_col1:
            if st.button("💾 Save Conversation"):
                if st.session_state.messages:
                    save_path = save_conversation_dialog(
                        st.session_state.messages,
                        os.getenv("JSON_PATH")
                    )
                    if save_path:
                        st.success(f"Saved as: {os.path.basename(save_path)}")
                else:
                    st.warning("No conversation to save!")

        with save_col2:
            if 'show_new_chat_dialog' not in st.session_state:
                st.session_state.show_new_chat_dialog = False

            if st.button("🗑️ New Conversation"):
                st.session_state.show_new_chat_dialog = True

            # Show dialog if needed
            if st.session_state.show_new_chat_dialog and st.session_state.messages:
                st.warning("Save current conversation?")
                # Use radio buttons instead of columns
                save_choice = st.radio(
                    "Choose an option:",
                    ["Save & Start New", "Start New Without Saving", "Cancel"],
                    key="new_chat_choice",
                    help="Select how to handle the current conversation"
                )
                
                if save_choice == "Save & Start New":
                    save_path = save_conversation_dialog(
                        st.session_state.messages,
                        os.getenv("JSON_PATH")
                    )
                    if save_path:
                        st.success(f"Previous conversation saved as: {os.path.basename(save_path)}")
                    st.session_state.messages = []
                    st.session_state.driver = initialize_driver(
                        st.session_state.driver.llm_type,
                        st.session_state.driver.llm_model
                    )
                    st.session_state.show_new_chat_dialog = False
                    st.rerun()
                
                elif save_choice == "Start New Without Saving":
                    st.session_state.messages = []
                    st.session_state.driver = initialize_driver(
                        st.session_state.driver.llm_type,
                        st.session_state.driver.llm_model
                    )
                    st.session_state.show_new_chat_dialog = False
                    st.rerun()
                
                elif save_choice == "Cancel":
                    st.session_state.show_new_chat_dialog = False
                    st.rerun()
            
            # If no messages, just start new conversation
            elif st.session_state.show_new_chat_dialog:
                st.session_state.messages = []
                st.session_state.driver = initialize_driver(
                    st.session_state.driver.llm_type,
                    st.session_state.driver.llm_model
                )
                st.session_state.show_new_chat_dialog = False
                st.success("Started new conversation!")

        # Second row: Load and Export
        with load_col1:
            if st.button("📂 Load Conversation"):
                file_path = filedialog.askopenfilename(
                    title="Select Conversation File",
                    initialdir=os.getenv("JSON_PATH"),
                    filetypes=[("JSON files", "*.json")],
                    defaultextension=".json"
                )
                if file_path:
                    try:
                        st.session_state.driver.load_existing_conversation(file_path)
                        history = st.session_state.driver.get_conversation_history()
                        if history:
                            st.success(f"Loaded {len(history)} messages into conversation history")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            conversation = json.load(f)
                            st.session_state.messages = [
                                {"role": msg["role"], "content": msg["content"]}
                                for msg in conversation
                            ]
                        st.success(f"Loaded conversation: {os.path.basename(file_path)}")
                    except Exception as e:
                        st.error(f"Error loading conversation: {str(e)}")
                else:
                    st.info("No file selected")

        with load_col2:
            if st.button("📄 Export to PDF"):
                if st.session_state.messages:
                    with st.spinner("Creating PDF..."):
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        default_filename = f"conversation_{timestamp}.pdf"
                        save_path = filedialog.asksaveasfilename(
                            defaultextension=".pdf",
                            initialfile=default_filename,
                            filetypes=[("PDF files", "*.pdf")]
                        )
                        if save_path:
                            if create_conversation_pdf(st.session_state.messages, save_path):
                                st.balloons()
                                st.success(f"✅ PDF saved as: {os.path.basename(save_path)}")
                            else:
                                st.error("Failed to create PDF. Check the error message above.")
                        else:
                            st.info("PDF export cancelled")
                else:
                    st.warning("No conversation to export!")

        st.header("Context Files")
        uploaded_files = st.file_uploader(
            "Upload context files",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'doc', 'docx'],
            help="Upload files to provide context for the conversation"
        )
        
        if uploaded_files:
            context_files = process_uploaded_files(uploaded_files)
            st.session_state.context_files = context_files
            st.success(f"Uploaded {len(context_files)} files for context")
            
            # Add a preview button for uploaded files
            if st.button("Preview Uploaded Files"):
                for file_path in context_files:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        st.text_area(
                            f"Preview of {os.path.basename(file_path)}", 
                            value=f.read()[:1000] + "...",
                            height=150
                        )

#=======================================================================================================#

    # Initialize session state
    if 'driver' not in st.session_state:
        st.session_state.driver = initialize_driver(llm_type, llm_model, temperature)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Chat messages area
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Fixed chat input
    if prompt := st.chat_input("What would you like to ask?", key="chat_input"):
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    context = ""
                    if 'context_files' in st.session_state:
                        for file_path in st.session_state.context_files:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                context += f"\nContent from {os.path.basename(file_path)}:\n{f.read()}"
                    
                    full_prompt = f"Context:\n{context}\n\nQuery: {prompt}" if context else prompt
                    response = st.session_state.driver.process_query(full_prompt,use_history=True)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()