import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document
import io
import PyPDF2
import mammoth
import hashlib
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load your OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment.")
    st.stop()

client = OpenAI(api_key=api_key)

st.title("ChatGPT Chatbot 🤖")
st.caption("Ask anything! Powered by OpenAI with Document Intelligence.")

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

if "vectorizer" not in st.session_state:
    st.session_state["vectorizer"] = None

if "chunk_vectors" not in st.session_state:
    st.session_state["chunk_vectors"] = None

if "trigger_ai_response" not in st.session_state:
    st.session_state["trigger_ai_response"] = None

# ---- RAG HELPER FUNCTIONS ----
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def get_embeddings_openai(texts):
    """Get embeddings using OpenAI API"""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"Error getting embeddings: {e}")
        return None

def create_tfidf_embeddings(chunks):
    """Create TF-IDF embeddings as fallback"""
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    vectors = vectorizer.fit_transform(chunks)
    return vectorizer, vectors.toarray()

def find_relevant_chunks(query, top_k=3):
    """Find most relevant chunks for a query using RAG"""
    if not st.session_state["document_chunks"]:
        return []
    
    chunks = st.session_state["document_chunks"]
    
    # For multi-file queries, increase top_k and search across all files
    if detect_multi_file_query(query):
        top_k = min(top_k * 2, len(chunks))  # Get more chunks for multi-file queries
        
        # Ensure we get chunks from different files for comparison
        file_representation = {}
        for chunk in chunks:
            file_key = chunk["file_key"]
            if file_key not in file_representation:
                file_representation[file_key] = []
            file_representation[file_key].append(chunk)
    
    # Try OpenAI embeddings first, fallback to TF-IDF
    try:
        query_embedding = get_embeddings_openai([query])
        if query_embedding and st.session_state.get("openai_embeddings"):
            query_vec = np.array(query_embedding[0])
            chunk_vecs = np.array(st.session_state["openai_embeddings"])
            similarities = cosine_similarity([query_vec], chunk_vecs)[0]
        else:
            raise Exception("Using TF-IDF fallback")
    except:
        # Fallback to TF-IDF
        if st.session_state["vectorizer"] and st.session_state["chunk_vectors"] is not None:
            query_vec = st.session_state["vectorizer"].transform([query])
            similarities = cosine_similarity(query_vec, st.session_state["chunk_vectors"])[0]
        else:
            return chunks[:top_k]  # Return first chunks if no embeddings
    
    # Get top-k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    relevant_chunks = []
    
    # For multi-file queries, ensure representation from each file
    if detect_multi_file_query(query) and len(st.session_state["uploaded_files"]) > 1:
        chunks_per_file = max(1, top_k // len(st.session_state["uploaded_files"]))
        file_chunks_added = {}
        
        # First pass: get best chunks from each file
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Lower threshold for multi-file
                chunk_info = chunks[idx].copy()
                chunk_info["similarity"] = similarities[idx]
                file_key = chunk_info["file_key"]
                
                if file_key not in file_chunks_added:
                    file_chunks_added[file_key] = 0
                
                if file_chunks_added[file_key] < chunks_per_file:
                    relevant_chunks.append(chunk_info)
                    file_chunks_added[file_key] += 1
        
        # Second pass: fill remaining slots with best remaining chunks
        remaining_slots = top_k - len(relevant_chunks)
        for idx in top_indices:
            if remaining_slots <= 0:
                break
            if similarities[idx] > 0.05:
                chunk_info = chunks[idx].copy()
                chunk_info["similarity"] = similarities[idx]
                
                # Check if we already added this chunk
                already_added = any(
                    c["file_key"] == chunk_info["file_key"] and 
                    c["chunk_id"] == chunk_info["chunk_id"] 
                    for c in relevant_chunks
                )
                
                if not already_added:
                    relevant_chunks.append(chunk_info)
                    remaining_slots -= 1
    else:
        # Regular single-file or general query
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Similarity threshold
                chunk_info = chunks[idx].copy()
                chunk_info["similarity"] = similarities[idx]
                relevant_chunks.append(chunk_info)
    
    return relevant_chunks

def update_document_index():
    """Update the document search index when files change"""
    all_chunks = []
    
    for file_key, file_info in st.session_state["uploaded_files"].items():
        content = file_info["content"]
        chunks = chunk_text(content, chunk_size=400, overlap=50)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "file_name": file_info["name"],
                "file_key": file_key,
                "chunk_id": i,
                "file_type": file_info["type"]
            })
    
    st.session_state["document_chunks"] = all_chunks
    
    if all_chunks:
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        
        # Try OpenAI embeddings first
        try:
            embeddings = get_embeddings_openai(chunk_texts)
            if embeddings:
                st.session_state["openai_embeddings"] = embeddings
                st.session_state["vectorizer"] = None
                st.session_state["chunk_vectors"] = None
            else:
                raise Exception("OpenAI embeddings failed")
        except:
            # Fallback to TF-IDF
            vectorizer, vectors = create_tfidf_embeddings(chunk_texts)
            st.session_state["vectorizer"] = vectorizer
            st.session_state["chunk_vectors"] = vectors
            st.session_state["openai_embeddings"] = None

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

def create_typing_animation():
    """Create a typing animation effect"""
    typing_placeholder = st.empty()
    for i in range(3):
        for dots in [".", "..", "..."]:
            typing_placeholder.markdown(f"🤖 Thinking{dots}")
            time.sleep(0.3)
    typing_placeholder.empty()

def stream_response_with_animation(response_text, placeholder):
    """Stream response with better visual feedback"""
    displayed_text = ""
    
    # Simulate typing with character-by-character display
    for i, char in enumerate(response_text):
        displayed_text += char
        
        # Show cursor effect
        if i % 5 == 0:  # Update every 5 characters for performance
            placeholder.markdown(displayed_text + "▌")
        
        # Small delay for typing effect
        time.sleep(0.01)
    
    # Final display without cursor
    placeholder.markdown(displayed_text)

def detect_multi_file_query(query):
    """Detect if query is asking about multiple files"""
    multi_file_keywords = [
        "compare", "summarize all", "all documents", "all files", 
        "across documents", "between files", "common themes",
        "differences between", "similarities", "overview of all"
    ]
    return any(keyword in query.lower() for keyword in multi_file_keywords)

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
        st.session_state["document_chunks"] = []
        st.session_state["vectorizer"] = None
        st.session_state["chunk_vectors"] = None
        st.session_state["trigger_ai_response"] = None
        st.rerun()

    # RAG Settings
    st.markdown("### 🧠 Document Intelligence")
    rag_enabled = st.checkbox("Enable Smart Document Search", value=True, help="Use RAG to find relevant document sections")
    max_chunks = st.slider("Max relevant sections per query", 1, 10, 3)

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
                with st.spinner(f"🔍 Processing {uploaded_file.name}..."):
                    file_content = process_uploaded_file(uploaded_file)
                    
                    # Store file info
                    st.session_state["uploaded_files"][file_key] = {
                        "name": uploaded_file.name,
                        "type": uploaded_file.type,
                        "content": file_content,
                        "size": len(file_content),
                        "upload_time": time.time()
                    }
                    
                    # Update document index
                    with st.spinner("🧠 Building document index..."):
                        update_document_index()
                    
                    # Automatically add to chat
                    file_message = {
                        "role": "user", 
                        "content": f"📎 Uploaded file: **{uploaded_file.name}**",
                        "file_attachment": {
                            "name": uploaded_file.name,
                            "type": uploaded_file.type,
                            "key": file_key
                        }
                    }
                    st.session_state["messages"].append(file_message)
                    
                    st.success(f"✅ {uploaded_file.name} processed with {len(st.session_state['document_chunks'])} searchable sections")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    
    # Show uploaded files with stats
    if st.session_state["uploaded_files"]:
        st.markdown("### 📁 Document Library")
        
        # Show total stats
        total_files = len(st.session_state["uploaded_files"])
        total_chunks = len(st.session_state["document_chunks"])
        st.info(f"📊 {total_files} files • {total_chunks} searchable sections")
        
        for file_key, file_info in st.session_state["uploaded_files"].items():
            with st.expander(f"{get_file_icon(file_info['type'])} {file_info['name']}"):
                st.caption(f"📏 {file_info['size']:,} characters")
                
                # Show file chunks
                file_chunks = [c for c in st.session_state["document_chunks"] if c["file_key"] == file_key]
                st.caption(f"🔍 {len(file_chunks)} searchable sections")
                
                if st.button("🗑️ Remove", key=f"remove_{file_key}"):
                    del st.session_state["uploaded_files"][file_key]
                    update_document_index()
                    st.rerun()
    
    # Quick action buttons
    if st.session_state["uploaded_files"]:
        st.markdown("### 🚀 Quick Actions")
        if st.button("📊 Summarize All Documents"):
            summary_prompt = "Please provide a comprehensive summary of all uploaded documents, highlighting key themes and main points."
            st.session_state["messages"].append({"role": "user", "content": summary_prompt})
            st.session_state["trigger_ai_response"] = summary_prompt
            st.rerun()
        
        if st.button("🔍 Compare Documents"):
            compare_prompt = "Compare the uploaded documents and identify similarities, differences, and unique insights from each."
            st.session_state["messages"].append({"role": "user", "content": compare_prompt})
            st.session_state["trigger_ai_response"] = compare_prompt
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
                        <div style="font-size: 12px; color: #666;">Ready for intelligent search</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

# ---- CHAT INPUT ----
# Handle triggered AI responses from quick action buttons
if st.session_state.get("trigger_ai_response"):
    prompt = st.session_state["trigger_ai_response"]
    st.session_state["trigger_ai_response"] = None  # Clear the trigger
    
    # Process the AI response
    full_response = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Show thinking animation
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🧠 Analyzing documents...")

        try:
            # Build context using RAG
            context_parts = []
            
            if rag_enabled and st.session_state["uploaded_files"]:
                # For multi-file queries, get more chunks
                query_max_chunks = max_chunks * 2 if detect_multi_file_query(prompt) else max_chunks
                
                # Find relevant chunks
                relevant_chunks = find_relevant_chunks(prompt, top_k=query_max_chunks)
                
                if relevant_chunks:
                    context_parts.append("=== RELEVANT DOCUMENT SECTIONS ===\n")
                    
                    # Group chunks by file for better organization
                    chunks_by_file = {}
                    for chunk in relevant_chunks:
                        file_name = chunk['file_name']
                        if file_name not in chunks_by_file:
                            chunks_by_file[file_name] = []
                        chunks_by_file[file_name].append(chunk)
                    
                    for file_name, file_chunks in chunks_by_file.items():
                        context_parts.append(f"\n📄 FROM DOCUMENT: {file_name}")
                        for chunk in file_chunks:
                            context_parts.append(f"Section (Relevance: {chunk.get('similarity', 0):.2f}): {chunk['text']}\n")
                    
                    context_parts.append("=== END DOCUMENT SECTIONS ===\n")
                    context_parts.append(f"User Question: {prompt}")
                    
                    thinking_placeholder.markdown(f"🔍 Found {len(relevant_chunks)} relevant sections from {len(chunks_by_file)} documents")
                else:
                    context_parts.append(f"User Question: {prompt}")
                    thinking_placeholder.markdown("🤔 No specific document sections found, using general knowledge")
            else:
                context_parts.append(f"User Question: {prompt}")
                thinking_placeholder.empty()
            
            # Prepare API message
            final_prompt = "\n".join(context_parts)
            
            # Prepare messages for API
            api_messages = []
            for msg in st.session_state["messages"]:
                if msg["role"] in ["user", "assistant"] and "file_attachment" not in msg:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Replace the last user message with the enhanced prompt
            if api_messages and api_messages[-1]["role"] == "user":
                api_messages[-1]["content"] = final_prompt
            
            thinking_placeholder.empty()
            
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
    st.rerun()

# Regular chat input
if prompt := st.chat_input("Ask something about your documents..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    full_response = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Show thinking animation
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🧠 Analyzing documents...")

        try:
            # Build context using RAG
            context_parts = []
            
            if rag_enabled and st.session_state["uploaded_files"]:
                # For multi-file queries, get more chunks
                query_max_chunks = max_chunks * 2 if detect_multi_file_query(prompt) else max_chunks
                
                # Find relevant chunks
                relevant_chunks = find_relevant_chunks(prompt, top_k=query_max_chunks)
                
                if relevant_chunks:
                    context_parts.append("=== RELEVANT DOCUMENT SECTIONS ===\n")
                    
                    # Group chunks by file for better organization
                    chunks_by_file = {}
                    for chunk in relevant_chunks:
                        file_name = chunk['file_name']
                        if file_name not in chunks_by_file:
                            chunks_by_file[file_name] = []
                        chunks_by_file[file_name].append(chunk)
                    
                    for file_name, file_chunks in chunks_by_file.items():
                        context_parts.append(f"\n📄 FROM DOCUMENT: {file_name}")
                        for chunk in file_chunks:
                            context_parts.append(f"Section (Relevance: {chunk.get('similarity', 0):.2f}): {chunk['text']}\n")
                    
                    context_parts.append("=== END DOCUMENT SECTIONS ===\n")
                    context_parts.append(f"User Question: {prompt}")
                    
                    thinking_placeholder.markdown(f"🔍 Found {len(relevant_chunks)} relevant sections from {len(chunks_by_file)} documents")
                else:
                    context_parts.append(f"User Question: {prompt}")
                    thinking_placeholder.markdown("🤔 No specific document sections found, using general knowledge")
            else:
                context_parts.append(f"User Question: {prompt}")
                thinking_placeholder.empty()
            
            # Prepare API message
            final_prompt = "\n".join(context_parts)
            
            # Prepare messages for API
            api_messages = []
            for msg in st.session_state["messages"]:
                if msg["role"] in ["user", "assistant"] and "file_attachment" not in msg:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Replace the last user message with the enhanced prompt
            if api_messages and api_messages[-1]["role"] == "user":
                api_messages[-1]["content"] = final_prompt
            
            thinking_placeholder.empty()
            
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