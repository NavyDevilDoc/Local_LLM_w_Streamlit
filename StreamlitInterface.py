import streamlit as st
import asyncio
from typing import Optional
import fpdf
import json
from datetime import datetime
import io
from PIL import Image
import PyPDF2
from docx import Document

class StreamlitInterface:
    def __init__(self, model_interface):
        self.model_interface = model_interface
        # Initialize session state for chat history if it doesn't exist
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ""

#============================================================================#
#============================================================================#

    def clear_input(self):
        """Callback to clear input after sending"""
        if 'user_input' in st.session_state:
            st.session_state.user_input = ""

#============================================================================#
#============================================================================#

    def process_image_for_dalle(self, image_bytes: bytes) -> bytes:
        """Process image to meet DALL-E requirements (square aspect ratio)"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Get dimensions
            width, height = image.size
            st.sidebar.info(f"Original image dimensions: {width}x{height}")
            
            # Calculate the size for square crop (use smaller dimension)
            size = min(width, height)
            
            # Calculate coordinates for center crop
            left = (width - size) // 2
            top = (height - size) // 2
            right = left + size
            bottom = top + size
            
            # Crop image to square
            square_image = image.crop((left, top, right, bottom))
            st.sidebar.info(f"Cropped to: {size}x{size}")
            
            # Save to bytes
            output = io.BytesIO()
            square_image.save(output, format=image.format)
            return output.getvalue()
            
        except Exception as e:
            st.sidebar.error(f"Error processing image: {str(e)}")
            return None

#============================================================================#
#============================================================================#

    def setup_ui(self):
        st.title("Multi-Model Chat Interface")
    
        # Add task selection at the top
        task_type = st.radio(
            "Task:",
            ("Chat", "Image Generation"),
            horizontal=True
        )

        # Sidebar for file upload and export
        with st.sidebar:
            st.header("Options")
            
            # Clear chat history
            #st.subheader("Clear Chat")
            if st.button("Clear Chat", type="primary"):
                self.clear_chat()

            # Chat history section
            st.subheader("Chat History")
            chat_file = st.file_uploader(
                "Load previous chat",
                type=['json'],
                key="chat_loader"
            )
            
            if chat_file:
                if st.button("Load Chat"):
                    self.load_chat(chat_file)
            
            # Context file section
            st.subheader("Context File")
            if task_type == "Chat":
                context_file = st.file_uploader(
                    "Upload a file for context",
                    type=['txt', 'pdf', 'doc', 'docx'],
                    key="context_loader"
                )
            else:
                context_file = st.file_uploader(
                    "Upload reference image",
                    type=['png', 'jpg', 'jpeg'],
                    key="image_context_loader"
                )
            
            # Export options
            st.subheader("Export Options")
            if st.session_state.messages:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save Chat"):
                        self.save_chat()
                with col2:
                    if st.button("Export PDF"):
                        self.export_chat()
        
        # Model selection based on task
        if task_type == "Chat":
            model_choice = st.radio(
                "Model:",
                ("GPT-4o", "GPT-4o-mini")
            )
        else:
            model_choice = st.radio(
                "Model:",
                ("DALL-E 3"),
                key="image_model"  # Different key to prevent state conflicts
            )
    
        return model_choice, context_file, task_type

#============================================================================#
#============================================================================#

    def create_chat_interface(self):
        # Display chat messages in a scrollable container
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])  # Changed from st.write to st.markdown
        
        # Create a container at the bottom of the page for input
        input_container = st.container()
        with input_container:
            # Add some space to push the input to the bottom
            st.markdown("<br>" * 2, unsafe_allow_html=True)
            
            # Create a row for input and button
            cols = st.columns([8, 1])
            with cols[0]:
                user_input = st.text_input(
                    "Message",
                    key="user_input",
                    label_visibility="collapsed",
                )
            with cols[1]:
                send_button = st.button("Send", use_container_width=True)
        
        return user_input, send_button

#============================================================================#
#============================================================================#

    def create_image_interface(self):
        cols = st.columns([8, 1])
        with cols[0]:
            image_prompt = st.text_input(
                "Image Description",
                key="image_prompt",
                label_visibility="collapsed",
            )
        with cols[1]:
            generate_button = st.button("Generate", use_container_width=True)
        
        # Create two columns for size and quality settings
        setting_cols = st.columns(2)
        
        with setting_cols[0]:
            size_options = {
                "Square (1024x1024)": "1024x1024",
                "Landscape (1792x1024)": "1792x1024",
                "Portrait (1024x1792)": "1024x1792"
            }
            image_size = st.select_slider(
                "Image Size",
                options=list(size_options.keys()),
                value="Square (1024x1024)"
            )
        
        with setting_cols[1]:
            quality = st.radio(
                "Quality",
                options=["standard", "hd"],
                horizontal=True,
                help="HD quality costs 2x credits"
            )

            # Add caption generation option
            generate_caption = st.checkbox(
                "Generate caption",
                help="Use GPT-4 to generate a detailed caption"
            )
        
        return image_prompt, generate_button, size_options[image_size], quality, generate_caption

#============================================================================#
#============================================================================#

    async def handle_response(self, model_choice: str, user_input: str, context: str = "") -> Optional[str]:
        if not user_input:
            return None
            
        # Create conversation history string
        conversation_history = ""
        if st.session_state.messages:
            for msg in st.session_state.messages:
                conversation_history += f"{msg['role'].title()}: {msg['content']}\n\n"
        
        # Construct the full prompt with context and history
        full_prompt = f"""Previous conversation:
    {conversation_history}
    Context:
    {context}

    Current message:
    {user_input}

    Please respond to the current message while considering the previous conversation and context."""
            
        with st.spinner('Getting response...'):
            try:
                if model_choice in ["GPT-4o", "GPT-4o-mini"]:
                    response = await self.model_interface.get_openai_response(full_prompt, model_choice)
                    return response
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return None

#============================================================================#
#============================================================================#

    def export_chat(self):
        try:
            # Get custom filename from user
            default_filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            custom_filename = st.sidebar.text_input(
                "Enter filename (without .pdf):",
                value=default_filename,
                key="pdf_filename"
        )

            # Create PDF
            pdf = fpdf.FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Add title and timestamp
            pdf.cell(200, 10, txt="Chat Export", ln=True, align='C')
            pdf.cell(200, 10, 
                    txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                    ln=True, 
                    align='C')
            pdf.ln(10)
            
            # Add messages
            for message in st.session_state.messages:
                pdf.set_font("Arial", 'B', size=12)
                pdf.cell(200, 10, txt=f"{message['role'].title()}:", ln=True)
                pdf.set_font("Arial", size=12)
                
                content = message['content']
                # Check if content contains image markdown
                if content.startswith("!["):
                    pdf.cell(200, 10, txt="[Image Generated - See URL below]", ln=True)
                    url = content.split("(")[1].rstrip(")")
                    pdf.set_font("Arial", size=10)
                    pdf.cell(200, 10, txt=url, ln=True)
                else:
                    content = str(content).encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(200, 10, txt=content)
                pdf.ln(5)
            
            # Save PDF to bytes
            pdf_data = pdf.output(dest='S').encode('latin-1')
            
            # Create download button
            st.sidebar.download_button(
                label="Download Chat PDF",
                data=pdf_data,
                file_name=f"{custom_filename}.pdf",
                mime="application/pdf"
            )
            
        except Exception as e:
            st.sidebar.error(f"Error creating PDF: {str(e)}")

#============================================================================#
#============================================================================#

    def process_uploaded_file(self, uploaded_file, task_type) -> str:
        if uploaded_file is None:
            return ""
            
        try:
            with st.spinner('Processing file...'):
                # Get file extension
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                # Handle image files for DALL-E context
                if task_type == "Image Generation":
                    if uploaded_file.type in ['image/png', 'image/jpeg', 'image/jpg']:
                        # Process image to square format
                        image_bytes = uploaded_file.getvalue()
                        processed_image = self.process_image_for_dalle(image_bytes)
                        
                        if processed_image:
                            return io.BytesIO(processed_image)
                        else:
                            st.sidebar.error("Failed to process image")
                            return None
                    else:
                        st.sidebar.error("Please upload a valid image file (PNG or JPEG)")
                        return None

                if file_extension == 'txt':
                    # Handle text files
                    return uploaded_file.getvalue().decode('utf-8')
                    
                elif file_extension == 'pdf':
                    # Handle PDF files
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
                    
                elif file_extension in ['doc', 'docx']:
                    # Handle Word documents
                    doc = Document(io.BytesIO(uploaded_file.getvalue()))
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    return text
                    
                else:
                    st.sidebar.error(f"Unsupported file type: {file_extension}")
                    return ""
                    
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")
            return ""

#============================================================================#
#============================================================================#

    def render(self):
        model_choice, uploaded_file, task_type = self.setup_ui()
        if task_type == "Chat":
            # Set up callback for input clearing
            if 'clear_next' not in st.session_state:
                st.session_state.clear_next = False
            
            # Create interface with conditional clearing
            if st.session_state.clear_next:
                st.session_state.user_input = ""
                st.session_state.clear_next = False
            
            user_input, send_button = self.create_chat_interface()
            
            # Process file if uploaded
            context = self.process_uploaded_file(uploaded_file, task_type)
            
            if send_button and user_input:
                # Store the input before clearing
                current_input = user_input
                
                # Format user message with context if available
                display_message = current_input
                full_prompt = f"{context}\n\n{current_input}" if context else current_input
                
                # Add user message to chat
                st.session_state.messages.append({
                    "role": "user",
                    "content": display_message
                })
                
                # Get assistant response
                response = asyncio.run(
                    self.handle_response(model_choice, full_prompt, context)
                )
                
                if response:
                    # Add assistant response to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Flag for clearing on next render
                    st.session_state.clear_next = True
                    
                    # Rerun to update the UI
                    st.rerun()
        else:  # Image Generation
            image_prompt, generate_button, image_size, quality, generate_caption = self.create_image_interface()
            context_image = None
            
            # Process uploaded file if it exists
            if uploaded_file:
                context_image = self.process_uploaded_file(uploaded_file, task_type)
                if context_image:
                    # Display the reference image in the sidebar
                    st.sidebar.image(uploaded_file, caption="Reference Image")
            
            if generate_button and image_prompt:
                with st.spinner('Generating image...'):
                    try:
                        # Enhance prompt if we have a reference image
                        enhanced_prompt = image_prompt
                        if context_image:
                            st.write("Attempting to generate with reference image...")
                            enhanced_prompt = f"Using the provided reference image as inspiration: {image_prompt}"
                        
                        image_url = asyncio.run(
                            self.model_interface.generate_image(
                                prompt=enhanced_prompt,
                                size=image_size,
                                reference_image=context_image if context_image else None
                            )
                        )
                        if image_url:
                            # Display the generated image
                            st.image(image_url, caption=image_prompt)
                            quality_tag = "🖼️ HD" if quality == "hd" else "🖼️"

                            # Generate caption if requested
                            caption = None
                            if generate_caption:
                                with st.spinner('Generating caption...'):
                                    caption = asyncio.run(
                                        self.model_interface.generate_caption(image_url)
                                    )

                            # Add to chat history
                            st.session_state.messages.append({
                                "role": "user",
                                "content": f"{quality_tag} Generated image with prompt: {image_prompt}"
                                + ("\n(Used reference image)" if context_image else "")
                            })
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"![Generated Image]({image_url})"
                            })
                        else:
                            st.error("Failed to generate image")
                            if context_image:
                                st.error("Image generation failed. Please check the following:")
                                st.error("- Image must be square")
                                st.error("- Image must be less than 4MB")
                                st.error("- Image must be PNG or JPEG format")
                    except Exception as e:
                        st.error(f"Error generating image: {str(e)}")
                        st.error("Full error details:", exc_info=True)
#============================================================================#
#============================================================================#

    def save_chat(self):
        """Save chat history to JSON file"""
        try:
            chat_data = {
                "messages": [
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                    } for msg in st.session_state.messages
                ],
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Convert to JSON string
            json_str = json.dumps(chat_data, indent=2)
            
            # Create download button for JSON
            st.sidebar.download_button(
                label="Download Chat History",
                data=json_str,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.sidebar.error(f"Error saving chat: {str(e)}")

#============================================================================#
#============================================================================#

    def load_chat(self, uploaded_file) -> bool:
        """Load chat history from JSON file"""
        try:
            if uploaded_file is None:
                return False
                
            content = uploaded_file.getvalue().decode('utf-8')
            chat_data = json.loads(content)
            
            # Validate JSON structure
            if not isinstance(chat_data, dict) or "messages" not in chat_data:
                st.sidebar.error("Invalid chat history format")
                return False
            
            # Ensure each message has required fields
            for msg in chat_data["messages"]:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    st.sidebar.error("Invalid message format in chat history")
                    return False
            
            # Update session state with loaded messages
            st.session_state.messages = chat_data["messages"]
            
            st.sidebar.success(f"Chat loaded from {uploaded_file.name}")
            return True
            
        except json.JSONDecodeError:
            st.sidebar.error("Invalid JSON file")
            return False
        except Exception as e:
            st.sidebar.error(f"Error loading chat: {str(e)}")
            return False

#============================================================================#
#============================================================================#

    def clear_chat(self):
        """Clear all messages from chat history"""
        st.session_state.messages = []
        st.rerun()