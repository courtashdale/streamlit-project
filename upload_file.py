import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document
import io
import PyPDF2
import mammoth

# Load your OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment.")
    st.stop()

client = OpenAI(api_key=api_key)

st.title("ChatGPT Chatbot 🤖")
st.caption("Ask anything! Powered by OpenAI (v1 SDK).")

# Available models
models_available = ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]

# ---- SESSION STATE INIT ----
if "MAXCHAT_model_chosen" not in st.session_state:
    st.session_state["MAXCHAT_model_chosen"] = "gpt-3.5-turbo"

if "previous_model" not in st.session_state:
    st.session_state["previous_model"] = st.session_state["MAXCHAT_model_chosen"]

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi there! What would you like to know?"}]

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = {}

# ---- HELPER FUNCTIONS ----
def extract_text_from_pdf(file_bytes):
    """Extract text from PDF file bytes"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def extract_text_from_docx(file_bytes):
    """Extract text from DOCX file bytes"""
    try:
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        return text.strip()
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")

def extract_text_from_docx_mammoth(file_bytes):
    """Alternative DOCX extraction using mammoth (better formatting)"""
    try:
        with io.BytesIO(file_bytes) as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            return result.value.strip()
    except Exception as e:
        raise Exception(f"Error reading DOCX with mammoth: {str(e)}")

def get_file_icon(file_type):
    """Return appropriate icon for file type"""
    if file_type == "application/pdf":
        return "📄"
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "📝"
    elif file_type == "text/plain":
        return "📄"
    else:
        return "📁"

def process_uploaded_file(uploaded_file):
    """Process uploaded file and return content"""
    try:
        file_content = ""
        
        if uploaded_file.type == "text/plain":
            file_content = uploaded_file.getvalue().decode("utf-8")
            
        elif uploaded_file.type == "application/pdf":
            file_content = extract_text_from_pdf(uploaded_file.getvalue())
            
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                file_content = extract_text_from_docx(uploaded_file.getvalue())
            except:
                file_content = extract_text_from_docx_mammoth(uploaded_file.getvalue())
        else:
            file_content = uploaded_file.getvalue().decode("utf-8")
        
        return file_content
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")

def display_file_attachment(filename, file_type, content_length, file_key):
    """Display file as an attachment card"""
    icon = get_file_icon(file_type)
    
    # Create a styled container for the file
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 2])
        
        with col1:
            st.markdown(f"### {icon}")
        
        with col2:
            st.markdown(f"**{filename}**")
            st.caption(f"{content_length:,} characters")
        
        with col3:
            if st.button("👁️", key=f"view_{file_key}", help="View content"):
                st.session_state[f"show_content_{file_key}"] = not st.session_state.get(f"show_content_{file_key}", False)
    
    # Show content if toggled
    if st.session_state.get(f"show_content_{file_key}", False):
        with st.expander("📖 File content", expanded=True):
            content = st.session_state["uploaded_files"][file_key]["content"]
            st.text_area(
                "Content:",
                value=content[:2000] + ("..." if len(content) > 2000 else ""),
                height=200,
                disabled=True,
                key=f"content_area_{file_key}"
            )

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    chosen_model = st.selectbox(
        "Choose your model",
        models_available,
        index=models_available.index(st.session_state["MAXCHAT_model_chosen"])
    )

    # Detect model change
    if chosen_model != st.session_state["MAXCHAT_model_chosen"]:
        st.session_state["previous_model"] = st.session_state["MAXCHAT_model_chosen"]
        st.session_state["MAXCHAT_model_chosen"] = chosen_model
        st.session_state["messages"].append({
            "role": "system",
            "content": f"Switched model to **{chosen_model}**"
        })

    if st.button("🗑️ Clear Chat"):
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi there! What would you like to know?"}]
        st.session_state["uploaded_files"] = {}
        st.rerun()

    # File upload section
    st.markdown("### 📎 Upload Files")
    uploaded_file = st.file_uploader(
        "Drop files here or click to upload", 
        type=["txt", "pdf", "docx"], 
        help="Supported formats: TXT, PDF, DOCX"
    )
    
    # Process uploaded file automatically
    if uploaded_file:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Only process if it's a new file
        if file_key not in st.session_state["uploaded_files"]:
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    file_content = process_uploaded_file(uploaded_file)
                    
                    # Store file info
                    st.session_state["uploaded_files"][file_key] = {
                        "name": uploaded_file.name,
                        "type": uploaded_file.type,
                        "content": file_content,
                        "size": len(file_content)
                    }
                    
                    # Automatically add to chat
                    file_message = {
                        "role": "user", 
                        "content": f"📎 Uploaded file: **{uploaded_file.name}**",
                        "file_attachment": {
                            "name": uploaded_file.name,
                            "type": uploaded_file.type,
                            "content": file_content,
                            "key": file_key
                        }
                    }
                    st.session_state["messages"].append(file_message)
                    
                    # Add system message about file processing
                    st.session_state["messages"].append({
                        "role": "system",
                        "content": f"✅ File '{uploaded_file.name}' has been processed and is ready for analysis."
                    })
                    
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    
    # Show uploaded files
    if st.session_state["uploaded_files"]:
        st.markdown("### 📁 Uploaded Files")
        for file_key, file_info in st.session_state["uploaded_files"].items():
            with st.expander(f"{get_file_icon(file_info['type'])} {file_info['name']}"):
                st.caption(f"Size: {file_info['size']:,} characters")
                if st.button("🗑️ Remove", key=f"remove_{file_key}"):
                    del st.session_state["uploaded_files"][file_key]
                    st.rerun()

# ---- DISPLAY MESSAGES ----
for i, msg in enumerate(st.session_state["messages"]):
    if msg["role"] == "system":
        st.caption(msg["content"])
    else:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user" and "file_attachment" in msg:
                # Display file attachment
                file_info = msg["file_attachment"]
                
                # File attachment card
                st.markdown(f"""
                <div style="
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 12px;
                    margin: 8px 0;
                    background-color: #f8f9fa;
                    display: flex;
                    align-items: center;
                    gap: 12px;
                ">
                    <div style="font-size: 24px;">{get_file_icon(file_info['type'])}</div>
                    <div style="flex-grow: 1;">
                        <div style="font-weight: bold; margin-bottom: 4px;">{file_info['name']}</div>
                        <div style="font-size: 12px; color: #666;">{len(file_info['content']):,} characters</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Option to view file content
                if st.button("👁️ View content", key=f"view_msg_{i}"):
                    with st.expander("📖 File content", expanded=True):
                        st.text_area(
                            "Content:",
                            value=file_info['content'][:2000] + ("..." if len(file_info['content']) > 2000 else ""),
                            height=200,
                            disabled=True,
                            key=f"msg_content_{i}"
                        )
            else:
                st.markdown(msg["content"])

# ---- CHAT INPUT ----
if prompt := st.chat_input("Ask something..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    full_response = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            # Prepare messages for API (include file content in context)
            api_messages = []
            for msg in st.session_state["messages"]:
                if msg["role"] in ["user", "assistant"]:
                    if msg["role"] == "user" and "file_attachment" in msg:
                        # Include file content in the message for the API
                        file_info = msg["file_attachment"]
                        content_to_send = f"File: {file_info['name']}\n\nContent:\n{file_info['content'][:8000]}"
                        if len(file_info['content']) > 8000:
                            content_to_send += "\n\n[Content truncated due to length]"
                        api_messages.append({"role": "user", "content": content_to_send})
                    else:
                        api_messages.append({"role": msg["role"], "content": msg["content"]})
            
            stream = client.chat.completions.create(
                model=st.session_state["MAXCHAT_model_chosen"],
                messages=api_messages,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"⚠️ Error: {e}"
            st.error(full_response)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})