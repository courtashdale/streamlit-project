import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load your OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment.")
    st.stop()

client = OpenAI(api_key=api_key)

st.title("ChatGPT Chatbot 🤖")
st.caption("Ask anything! Powered by OpenAI (v1 SDK).")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi there! What would you like to know?"}]

# Sidebar: Model selector
models_available = ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]
default_model = st.session_state.get("MAXCHAT_model_chosen", "gpt-3.5-turbo")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    chosen_model = st.selectbox("Choose your model", models_available, index=models_available.index(default_model))
    st.session_state["MAXCHAT_model_chosen"] = chosen_model

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask something..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response using streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model=st.session_state["MAXCHAT_model_chosen"],
                messages=st.session_state.messages,
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

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})