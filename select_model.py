import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document
import io

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
        st.experimental_rerun()

    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"], label_visibility="collapsed")
    if uploaded_file:
        try:
            if uploaded_file.type == "text/plain":
                content = uploaded_file.getvalue().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                st.warning("PDF processing requires additional libraries. Displaying raw content.")
                content = str(uploaded_file.getvalue())
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = Document(io.BytesIO(uploaded_file.getvalue()))
                content = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
            else:
                content = uploaded_file.getvalue().decode("utf-8")
            
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            
            if st.button("📎 Add file content to chat"):
                file_message = f"**File content from {uploaded_file.name}:**\n\n{content[:2000]}{'...' if len(content) > 2000 else ''}"
                st.session_state["messages"].append({"role": "user", "content": file_message})
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

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