# 👑 Streamlit Project
This is my first Streamlit-based chatbot application integrated with OpenAI's API. It allows conversations and file uploads.

> ⚠️ Streamlit is a Python framework that makes it easier to develop ML/AI apps. A common alternative is [**Gradio**](https://www.gradio.app/), as seen on many [**🤗 Hugging Face**](https://huggingface.co/) apps.

## Features
- 🤖 **Multi-Model Support**: Choose between `GPT-3.5-turbo`, `GPT-4` and `GPT-4-1106-preview`.
- 📄 **Document Upload**: Support for PDF, DOCX, and TXT files.
- 🔍 **Document Analysis**: Upload docs and ask questions about the content.
- 💬 **Real-time Stream Chat**: Streamed responses with chat history.
- 🗃️ **File Management**: View, preview, and remove uploaded files.
- 👋 **Responsive UI**: Clean Streamlit interface with file attachment cards.

## Installation
1. Setup
```bash
git clone https://github.com/courtashdale/streamlit-project
cd streamlit-project
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env      
```
2. Run
```bash
streamlit run app.py
```
