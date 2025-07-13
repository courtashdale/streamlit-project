# utils/ui_components.py
"""
Reusable UI/UX components for Streamlit app
"""

import streamlit as st
import time
from typing import Dict, Optional, Any, List


def get_file_icon(file_type: str) -> str:
    """
    Return appropriate icon for file type.
    
    Args:
        file_type: MIME type of the file
        
    Returns:
        str: Emoji icon for the file type
    """
    icon_map = {
        "application/pdf": "📄",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "📝",
        "text/plain": "📋",
        "text/csv": "📊",
        "application/vnd.ms-excel": "📊",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "📊",
        "image/png": "🖼️",
        "image/jpeg": "🖼️",
        "image/jpg": "🖼️",
        "application/json": "📋",
        "text/html": "🌐",
        "text/markdown": "📝",
        "application/zip": "📦",
        "application/x-rar-compressed": "📦"
    }
    return icon_map.get(file_type, "📁")


def render_file_card(file_info: Dict[str, Any], file_key: str, show_actions: bool = True):
    """
    Render a file attachment card with consistent styling.
    
    Args:
        file_info: Dictionary containing file information
        file_key: Unique key for the file
        show_actions: Whether to show action buttons
    """
    icon = get_file_icon(file_info.get("type", ""))
    
    # Create styled container
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 2])
        
        with col1:
            st.markdown(f"### {icon}")
        
        with col2:
            st.markdown(f"**{file_info['name']}**")
            size = file_info.get('size', len(file_info.get('content', '')))
            st.caption(f"{size:,} characters")
        
        with col3:
            if show_actions:
                if st.button("👁️", key=f"view_{file_key}", help="View content"):
                    toggle_content_view(file_key)
    
    # Show content if toggled
    if st.session_state.get(f"show_content_{file_key}", False):
        with st.expander("📖 File content", expanded=True):
            content = file_info.get('content', '')
            st.text_area(
                "Content:",
                value=content[:2000] + ("..." if len(content) > 2000 else ""),
                height=200,
                disabled=True,
                key=f"content_area_{file_key}"
            )


def render_file_attachment_html(file_info: Dict[str, Any]) -> str:
    """
    Render file attachment as HTML card (for chat messages).
    
    Args:
        file_info: Dictionary containing file information
        
    Returns:
        str: HTML string for file card
    """
    icon = get_file_icon(file_info.get("type", ""))
    
    return f"""
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
        <div style="font-size: 24px;">{icon}</div>
        <div style="flex-grow: 1;">
            <div style="font-weight: bold; margin-bottom: 4px;">{file_info['name']}</div>
            <div style="font-size: 12px; color: #666;">{len(file_info.get('content', '')):,} characters</div>
        </div>
    </div>
    """


def stream_response_with_animation(text: str, placeholder, typing_speed: float = 0.01):
    """
    Display text with typing animation effect.
    
    Args:
        text: Text to display
        placeholder: Streamlit placeholder element
        typing_speed: Delay between characters in seconds
    """
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text + "▌")
        time.sleep(typing_speed)
    placeholder.markdown(displayed_text)


def create_typing_animation(text: str, container=None, typing_speed: float = 0.01):
    """
    Create a typing animation for text display.
    
    Args:
        text: Text to animate
        container: Streamlit container (defaults to main area)
        typing_speed: Delay between characters
        
    Returns:
        Streamlit element with animated text
    """
    if container is None:
        container = st
    
    placeholder = container.empty()
    stream_response_with_animation(text, placeholder, typing_speed)
    return placeholder


def toggle_content_view(file_key: str):
    """
    Toggle the content view state for a file.
    
    Args:
        file_key: Unique identifier for the file
    """
    key = f"show_content_{file_key}"
    st.session_state[key] = not st.session_state.get(key, False)


def render_sidebar_header():
    """Render consistent sidebar header."""
    st.markdown("### ⚙️ Settings")


def render_chat_header():
    """Render consistent chat header."""
    st.title("ChatGPT Chatbot 🤖")
    st.caption("Ask anything! Powered by OpenAI (v1 SDK).")


def render_model_selector(models: list, current_model: str) -> str:
    """
    Render model selection dropdown.
    
    Args:
        models: List of available models
        current_model: Currently selected model
        
    Returns:
        str: Selected model
    """
    return st.selectbox(
        "Choose your model",
        models,
        index=models.index(current_model) if current_model in models else 0
    )


def render_clear_chat_button() -> bool:
    """
    Render clear chat button.
    
    Returns:
        bool: True if button was clicked
    """
    return st.button("🗑️ Clear Chat")


def render_file_upload_section(supported_types: list) -> Any:
    """
    Render file upload section.
    
    Args:
        supported_types: List of supported file extensions
        
    Returns:
        Uploaded file object or None
    """
    st.markdown("### 📎 Upload Files")
    return st.file_uploader(
        "Drop files here or click to upload", 
        type=supported_types, 
        help=f"Supported formats: {', '.join([t.upper() for t in supported_types])}"
    )


def render_uploaded_files_list(uploaded_files: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """
    Render list of uploaded files with management options.
    
    Args:
        uploaded_files: Dictionary of uploaded files
        
    Returns:
        str: Key of file to remove, or None
    """
    if uploaded_files:
        st.markdown("### 📁 Uploaded Files")
        for file_key, file_info in uploaded_files.items():
            with st.expander(f"{get_file_icon(file_info['type'])} {file_info['name']}"):
                st.caption(f"Size: {file_info['size']:,} characters")
                if st.button("🗑️ Remove", key=f"remove_{file_key}"):
                    return file_key  # Return key of file to remove
    return None


def render_system_message(message: str):
    """
    Render a system message with consistent styling.
    
    Args:
        message: System message to display
    """
    st.caption(message)


def render_error_message(error: str):
    """
    Render an error message with consistent styling.
    
    Args:
        error: Error message to display
    """
    st.error(f"❌ {error}")


def render_success_message(message: str):
    """
    Render a success message with consistent styling.
    
    Args:
        message: Success message to display
    """
    st.success(f"✅ {message}")


def render_info_message(message: str):
    """
    Render an info message with consistent styling.
    
    Args:
        message: Info message to display
    """
    st.info(f"ℹ️ {message}")


def render_warning_message(message: str):
    """
    Render a warning message with consistent styling.
    
    Args:
        message: Warning message to display
    """
    st.warning(f"⚠️ {message}")


def render_processing_spinner(message: str = "Processing..."):
    """
    Create a context manager for showing a processing spinner.
    
    Args:
        message: Message to display while processing
        
    Returns:
        Streamlit spinner context manager
    """
    return st.spinner(message)


def create_message_container(role: str):
    """
    Create a chat message container.
    
    Args:
        role: Role of the message sender ("user" or "assistant")
        
    Returns:
        Streamlit chat message container
    """
    return st.chat_message(role)


def render_chart_container(fig, use_container_width: bool = True):
    """
    Render a Plotly chart with consistent settings.
    
    Args:
        fig: Plotly figure object
        use_container_width: Whether to use full container width
    """
    st.plotly_chart(fig, use_container_width=use_container_width)


def create_columns_layout(ratios: list):
    """
    Create a columns layout with specified ratios.
    
    Args:
        ratios: List of column width ratios
        
    Returns:
        List of column objects
    """
    return st.columns(ratios)


def render_metric_card(label: str, value: Any, delta: Optional[Any] = None, delta_color: str = "normal"):
    """
    Render a metric card.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
        delta_color: Color of delta ("normal", "inverse", "off")
    """
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def render_progress_bar(value: float, text: Optional[str] = None):
    """
    Render a progress bar.
    
    Args:
        value: Progress value (0.0 to 1.0)
        text: Optional text to display
    """
    if text:
        st.text(text)
    st.progress(value)


def render_divider():
    """Render a horizontal divider."""
    st.divider()


def render_expander(label: str, expanded: bool = False):
    """
    Create an expander container.
    
    Args:
        label: Expander label
        expanded: Whether to start expanded
        
    Returns:
        Streamlit expander container
    """
    return st.expander(label, expanded=expanded)


def render_tabs(tab_names: List[str]) -> List:
    """
    Create tabs layout.
    
    Args:
        tab_names: List of tab names
        
    Returns:
        List of tab containers
    """
    return st.tabs(tab_names)


def render_empty_state(message: str, icon: str = "📭"):
    """
    Render an empty state message.
    
    Args:
        message: Empty state message
        icon: Icon to display
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"<h2 style='text-align: center;'>{icon}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: #666;'>{message}</p>", unsafe_allow_html=True)


def render_json_data(data: dict, expanded: bool = True):
    """
    Render JSON data in a nice format.
    
    Args:
        data: Dictionary to display
        expanded: Whether to expand by default
    """
    import json
    st.json(json.dumps(data, indent=2), expanded=expanded)


def render_code_block(code: str, language: str = "python"):
    """
    Render a code block with syntax highlighting.
    
    Args:
        code: Code to display
        language: Programming language for syntax highlighting
    """
    st.code(code, language=language)


def render_dataframe(df, use_container_width: bool = True):
    """
    Render a pandas DataFrame.
    
    Args:
        df: DataFrame to display
        use_container_width: Whether to use full container width
    """
    st.dataframe(df, use_container_width=use_container_width)


def render_download_button(
    label: str,
    data: Any,
    file_name: str,
    mime: str = "text/plain",
    key: Optional[str] = None
) -> bool:
    """
    Render a download button.
    
    Args:
        label: Button label
        data: Data to download
        file_name: Name of the downloaded file
        mime: MIME type of the file
        key: Optional unique key
        
    Returns:
        bool: True if button was clicked
    """
    return st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        key=key
    )


def render_text_input(
    label: str,
    value: str = "",
    placeholder: Optional[str] = None,
    key: Optional[str] = None,
    disabled: bool = False
) -> str:
    """
    Render a text input field.
    
    Args:
        label: Input label
        value: Default value
        placeholder: Placeholder text
        key: Optional unique key
        disabled: Whether input is disabled
        
    Returns:
        str: Input value
    """
    return st.text_input(
        label=label,
        value=value,
        placeholder=placeholder,
        key=key,
        disabled=disabled
    )


def render_text_area(
    label: str,
    value: str = "",
    height: int = 150,
    placeholder: Optional[str] = None,
    key: Optional[str] = None,
    disabled: bool = False
) -> str:
    """
    Render a text area input.
    
    Args:
        label: Input label
        value: Default value
        height: Height in pixels
        placeholder: Placeholder text
        key: Optional unique key
        disabled: Whether input is disabled
        
    Returns:
        str: Input value
    """
    return st.text_area(
        label=label,
        value=value,
        height=height,
        placeholder=placeholder,
        key=key,
        disabled=disabled
    )


def render_checkbox(
    label: str,
    value: bool = False,
    key: Optional[str] = None,
    help: Optional[str] = None
) -> bool:
    """
    Render a checkbox.
    
    Args:
        label: Checkbox label
        value: Default value
        key: Optional unique key
        help: Help text
        
    Returns:
        bool: Checkbox state
    """
    return st.checkbox(label=label, value=value, key=key, help=help)


def render_slider(
    label: str,
    min_value: float,
    max_value: float,
    value: float,
    step: float = 1.0,
    key: Optional[str] = None
) -> float:
    """
    Render a slider.
    
    Args:
        label: Slider label
        min_value: Minimum value
        max_value: Maximum value
        value: Default value
        step: Step size
        key: Optional unique key
        
    Returns:
        float: Slider value
    """
    return st.slider(
        label=label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        key=key
    )