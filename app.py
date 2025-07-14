import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document
import io
import PyPDF2
import mammoth
from utils.rag_utils import (
    chunk_text, 
    get_embeddings_openai, 
    create_tfidf_embeddings,
    find_relevant_chunks,
    create_context_from_chunks,
    update_document_index
)
import numpy as np

# Load your OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment.")
    st.stop()

client = OpenAI(api_key=api_key)

st.title("Doc Chatbot 🤖")
st.caption("Ask anything or upload docs to discuss them.")

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

if "document_chunks" not in st.session_state:
    st.session_state["document_chunks"] = []

if "document_metadata" not in st.session_state:
    st.session_state["document_metadata"] = []

if "document_embeddings" not in st.session_state:
    st.session_state["document_embeddings"] = None

if "use_rag" not in st.session_state:
    st.session_state["use_rag"] = True

if "embedding_method" not in st.session_state:
    st.session_state["embedding_method"] = "Auto"

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

def add_document_to_rag_index(file_content, file_metadata):
    """Add document to RAG index with chunking and embeddings"""
    try:
        # Update document index
        content_size = len(file_content)
        if content_size > 1000000:  # 1MB
            st.warning(f"Large file detected ({content_size:,} characters). Processing may take time...")
        
        with st.spinner("Creating document chunks..."):
            try:
                updated_chunks, updated_metadata = update_document_index(
                    file_content=file_content,
                    file_metadata=file_metadata,
                    existing_chunks=st.session_state["document_chunks"],
                    existing_metadata=st.session_state["document_metadata"]
                )
            except Exception as chunk_error:
                st.error(f"Chunking failed: {str(chunk_error)}")
                return False
        
        # Update session state
        st.session_state["document_chunks"] = updated_chunks
        st.session_state["document_metadata"] = updated_metadata
        
        # Generate embeddings for all chunks
        if updated_chunks:
            chunks_count = len(updated_chunks)
            st.info(f"Processing {chunks_count} document chunks...")
            
            # Get embedding method preference
            embedding_method = st.session_state.get("embedding_method", "Auto")
            
            # Decide which embedding method to use
            use_tfidf = (
                embedding_method == "TF-IDF" or 
                (embedding_method == "Auto" and chunks_count > 100)
            )
            
            if use_tfidf or embedding_method == "TF-IDF":
                if chunks_count > 100:
                    st.info("Large document detected - using TF-IDF embeddings for faster processing...")
                try:
                    with st.spinner("Generating TF-IDF embeddings..."):
                        embeddings, vectorizer = create_tfidf_embeddings(updated_chunks)
                        st.session_state["document_embeddings"] = embeddings
                        st.session_state["tfidf_vectorizer"] = vectorizer
                        st.success("✅ TF-IDF embeddings created successfully!")
                except Exception as tfidf_error:
                    st.error(f"TF-IDF embeddings failed: {str(tfidf_error)}")
                    return False
            else:
                # Try OpenAI embeddings
                try:
                    with st.spinner("Generating OpenAI embeddings..."):
                        embeddings = get_embeddings_openai(updated_chunks, client)
                        st.session_state["document_embeddings"] = embeddings
                        if "tfidf_vectorizer" in st.session_state:
                            del st.session_state["tfidf_vectorizer"]
                        st.success("✅ OpenAI embeddings created successfully!")
                except Exception as e:
                    st.warning(f"OpenAI embeddings failed: {str(e)}")
                    
                    # Fallback to TF-IDF only if method is Auto
                    if embedding_method == "Auto":
                        st.info("Falling back to TF-IDF embeddings...")
                        try:
                            with st.spinner("Generating TF-IDF embeddings..."):
                                embeddings, vectorizer = create_tfidf_embeddings(updated_chunks)
                                st.session_state["document_embeddings"] = embeddings
                                st.session_state["tfidf_vectorizer"] = vectorizer
                                st.success("✅ TF-IDF embeddings created successfully!")
                        except Exception as tfidf_error:
                            st.error(f"TF-IDF embeddings also failed: {str(tfidf_error)}")
                            return False
                    else:
                        st.error("OpenAI embeddings failed and no fallback available with manual selection.")
                        return False
        
        return True
    except Exception as e:
        st.error(f"Error adding document to RAG index: {str(e)}")
        return False

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
    st.markdown("# Settings")

    chosen_model = st.selectbox(
        "Select your model",
        models_available,
        index=models_available.index(st.session_state["MAXCHAT_model_chosen"])
    )
    
    # RAG toggle
    st.session_state["use_rag"] = st.checkbox(
        "Use RAG (Retrieval-Augmented Generation)",
        value=st.session_state["use_rag"],
        help="When enabled, uses intelligent document chunking and retrieval instead of simple truncation"
    )
    
    # Embedding method selection
    if st.session_state["use_rag"]:
        embedding_method = st.selectbox(
            "Embedding Method",
            ["Auto", "OpenAI", "TF-IDF"],
            help="Auto: OpenAI for <100 chunks, TF-IDF for larger documents"
        )
        st.session_state["embedding_method"] = embedding_method

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
        st.session_state["document_chunks"] = []
        st.session_state["document_metadata"] = []
        st.session_state["document_embeddings"] = None
        if "tfidf_vectorizer" in st.session_state:
            del st.session_state["tfidf_vectorizer"]
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
                    
                    # Add to RAG index if enabled
                    if st.session_state["use_rag"]:
                        file_metadata = {
                            "filename": uploaded_file.name,
                            "file_type": uploaded_file.type,
                            "file_key": file_key
                        }
                        
                        # Add timeout handling for RAG processing
                        try:
                            rag_success = add_document_to_rag_index(file_content, file_metadata)
                            if not rag_success:
                                st.error("Failed to add document to RAG index. File uploaded but RAG disabled for this session.")
                                st.session_state["use_rag"] = False
                        except Exception as rag_error:
                            st.error(f"RAG processing failed: {str(rag_error)}")
                            st.info("File uploaded successfully but RAG processing disabled for this session.")
                            st.session_state["use_rag"] = False
                    
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
    
    # Show RAG status
    if st.session_state["use_rag"]:
        st.markdown("### 🔍 RAG Status")
        chunks_count = len(st.session_state["document_chunks"])
        
        if chunks_count > 0:
            st.success(f"✅ {chunks_count} document chunks indexed")
            embedding_type = "OpenAI" if "tfidf_vectorizer" not in st.session_state else "TF-IDF"
            st.caption(f"Using {embedding_type} embeddings")
        else:
            st.info("Upload documents to enable RAG")
    
    # Show uploaded files
    if st.session_state["uploaded_files"]:
        st.markdown("### 📁 Uploaded Files")
        for file_key, file_info in st.session_state["uploaded_files"].items():
            with st.expander(f"{get_file_icon(file_info['type'])} {file_info['name']}"):
                st.caption(f"Size: {file_info['size']:,} characters")
                if st.button("🗑️ Remove", key=f"remove_{file_key}"):
                    del st.session_state["uploaded_files"][file_key]
                    
                    # Remove from RAG index - rebuild embeddings
                    if st.session_state["use_rag"]:
                        # Filter out chunks from this file
                        remaining_chunks = []
                        remaining_metadata = []
                        
                        for chunk, metadata in zip(st.session_state["document_chunks"], st.session_state["document_metadata"]):
                            if metadata.get("file_key") != file_key:
                                remaining_chunks.append(chunk)
                                remaining_metadata.append(metadata)
                        
                        st.session_state["document_chunks"] = remaining_chunks
                        st.session_state["document_metadata"] = remaining_metadata
                        
                        # Rebuild embeddings
                        if remaining_chunks:
                            try:
                                embeddings = get_embeddings_openai(remaining_chunks, client)
                                st.session_state["document_embeddings"] = embeddings
                            except:
                                embeddings, vectorizer = create_tfidf_embeddings(remaining_chunks)
                                st.session_state["document_embeddings"] = embeddings
                                st.session_state["tfidf_vectorizer"] = vectorizer
                        else:
                            st.session_state["document_embeddings"] = None
                            if "tfidf_vectorizer" in st.session_state:
                                del st.session_state["tfidf_vectorizer"]
                    
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
            # Prepare messages for API
            api_messages = []
            
            # Add document context using RAG if enabled and available
            chunks_available = len(st.session_state["document_chunks"]) > 0
            embeddings_available = st.session_state["document_embeddings"] is not None
            
            if (st.session_state["use_rag"] and chunks_available and embeddings_available):
                
                # Debug info
                st.info(f"🔍 RAG Search: '{prompt}' in {len(st.session_state['document_chunks'])} chunks")
                
                # Find relevant chunks for the current query
                try:
                    if "tfidf_vectorizer" in st.session_state:
                        # Using TF-IDF
                        relevant_chunks = find_relevant_chunks(
                            query=prompt,
                            chunks=st.session_state["document_chunks"],
                            embeddings=st.session_state["document_embeddings"],
                            top_k=3,
                            vectorizer=st.session_state["tfidf_vectorizer"]
                        )
                    else:
                        # Using OpenAI embeddings
                        relevant_chunks = find_relevant_chunks(
                            query=prompt,
                            chunks=st.session_state["document_chunks"],
                            embeddings=st.session_state["document_embeddings"],
                            top_k=3,
                            client=client
                        )
                    
                    if relevant_chunks:
                        # Create context from relevant chunks
                        context = create_context_from_chunks(relevant_chunks)
                        
                        # Add context as system message
                        api_messages.append({
                            "role": "system",
                            "content": f"Relevant document context:\n\n{context}\n\nUse this context to answer the user's question when relevant."
                        })
                        
                        # Show user that RAG is working
                        st.success(f"🔍 Retrieved {len(relevant_chunks)} relevant document sections")
                    else:
                        st.warning("⚠️ No relevant chunks found for this query")
                        
                except Exception as e:
                    st.warning(f"RAG retrieval failed, falling back to simple approach: {str(e)}")
            else:
                # Debug why RAG is not being used
                if not st.session_state["use_rag"]:
                    st.info("ℹ️ RAG is disabled in settings")
                elif not chunks_available:
                    st.info("ℹ️ No document chunks available")
                elif not embeddings_available:
                    st.info("ℹ️ No embeddings available")
            
            # Add conversation history
            for msg in st.session_state["messages"]:
                if msg["role"] in ["user", "assistant"]:
                    if msg["role"] == "user" and "file_attachment" in msg:
                        # For file attachments, just mention the file was uploaded
                        file_info = msg["file_attachment"]
                        api_messages.append({
                            "role": "user", 
                            "content": f"[Uploaded file: {file_info['name']}]"
                        })
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