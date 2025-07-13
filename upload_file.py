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
        st.rerun()

    # File upload section
    st.markdown("### 📎 File Upload")
    uploaded_file = st.file_uploader(
        "Upload a file", 
        type=["txt", "pdf", "docx"], 
        help="Supported formats: TXT, PDF, DOCX"
    )
    
    if uploaded_file:
        try:
            file_content = ""
            
            if uploaded_file.type == "text/plain":
                file_content = uploaded_file.getvalue().decode("utf-8")
                
            elif uploaded_file.type == "application/pdf":
                file_content = extract_text_from_pdf(uploaded_file.getvalue())
                
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # Try primary method first, fallback to mammoth if needed
                try:
                    file_content = extract_text_from_docx(uploaded_file.getvalue())
                except:
                    file_content = extract_text_from_docx_mammoth(uploaded_file.getvalue())
            else:
                # Fallback for other text files
                file_content = uploaded_file.getvalue().decode("utf-8")
            
            # Display file info
            st.success(f"✅ File '{uploaded_file.name}' processed successfully!")
            st.info(f"📄 Content length: {len(file_content)} characters")
            
            # Preview content
            with st.expander("📖 Preview file content"):
                st.text_area(
                    "File content preview:",
                    value=file_content[:1000] + ("..." if len(file_content) > 1000 else ""),
                    height=150,
                    disabled=True
                )
            
            # Add to chat button
            if st.button("📎 Add file content to chat", key="add_file"):
                # Truncate content if too long (keep reasonable length for API)
                max_length = 8000
                content_to_add = file_content[:max_length] + ("...\n\n[Content truncated due to length]" if len(file_content) > max_length else "")
                
                file_message = f"**File content from {uploaded_file.name}:**\n\n{content_to_add}"
                st.session_state["messages"].append({"role": "user", "content": file_message})
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("💡 Make sure you have the required libraries installed:\n- PyPDF2 for PDF files\n- python-docx for DOCX files\n- mammoth for advanced DOCX processing")

# ---- DISPLAY MESSAGES ----
for msg in st.session_state["messages"]:
    if msg["role"] == "system":
        st.caption(msg["content"])
    else:
        with st.chat_message(msg["role"]):
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
            stream = client.chat.completions.create(
                model=st.session_state["MAXCHAT_model_chosen"],
                messages=[
                    m for m in st.session_state["messages"]
                    if m["role"] in ["user", "assistant"]
                ],
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