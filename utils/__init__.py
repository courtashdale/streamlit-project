# utils/__init__.py
"""
Utils package initialization
"""

from .file_handler import (
    process_uploaded_file,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_with_mammoth,
    get_file_metadata,
    validate_file
)

from .rag_utils import (
    chunk_text,
    get_embeddings_openai,
    create_tfidf_embeddings,
    find_relevant_chunks,
    create_context_from_chunks,
    update_document_index
)

from .charts import (
    detect_chart_request,
    extract_data_from_text,
    create_streamlit_chart,
    auto_detect_chart_type
)

from .ui_components import (
    get_file_icon,
    render_file_card,
    render_file_attachment_html,
    stream_response_with_animation,
    create_typing_animation,
    toggle_content_view,
    render_sidebar_header,
    render_chat_header,
    render_model_selector,
    render_clear_chat_button,
    render_file_upload_section,
    render_uploaded_files_list,
    render_system_message,
    render_error_message,
    render_success_message,
    render_info_message,
    render_warning_message,
    render_processing_spinner,
    create_message_container,
    render_chart_container,
    create_columns_layout,
    render_metric_card,
    render_progress_bar,
    render_divider,
    render_expander,
    render_tabs,
    render_empty_state,
    render_json_data,
    render_code_block,
    render_dataframe,
    render_download_button,
    render_text_input,
    render_text_area,
    render_checkbox,
    render_slider
)

__all__ = [
    # file_handler
    'process_uploaded_file',
    'extract_text_from_pdf',
    'extract_text_from_docx',
    'extract_text_with_mammoth',
    'get_file_metadata',
    'validate_file',
    
    # rag_utils
    'chunk_text',
    'get_embeddings_openai',
    'create_tfidf_embeddings',
    'find_relevant_chunks',
    'create_context_from_chunks',
    'update_document_index',
    
    # charts
    'detect_chart_request',
    'extract_data_from_text',
    'create_streamlit_chart',
    'auto_detect_chart_type',
    
    # ui_components
    'get_file_icon',
    'render_file_card',
    'render_file_attachment_html',
    'stream_response_with_animation',
    'create_typing_animation',
    'toggle_content_view',
    'render_sidebar_header',
    'render_chat_header',
    'render_model_selector',
    'render_clear_chat_button',
    'render_file_upload_section',
    'render_uploaded_files_list',
    'render_system_message',
    'render_error_message',
    'render_success_message',
    'render_info_message',
    'render_warning_message',
    'render_processing_spinner',
    'create_message_container',
    'render_chart_container',
    'create_columns_layout',
    'render_metric_card',
    'render_progress_bar',
    'render_divider',
    'render_expander',
    'render_tabs',
    'render_empty_state',
    'render_json_data',
    'render_code_block',
    'render_dataframe',
    'render_download_button',
    'render_text_input',
    'render_text_area',
    'render_checkbox',
    'render_slider'
    'render_clear_chat_button',
    'render_file_upload_section',
    'render_uploaded_files_list',
    'render_system_message',
    'render_error_message',
    'render_success_message',
    'render_info_message',
    'render_processing_spinner',
    'create_message_container',
    'render_chart_container',
    'create_columns_layout',
    'render_metric_card',
    'render_progress_bar'
]