"""
Chart detection, data extraction, and visualization utilities using Streamlit built-in chart elements.
"""

import re
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import streamlit as st


def detect_chart_request(prompt: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if the user is requesting a chart or visualization.
    Returns a tuple (is_chart_request, chart_type).
    """
    prompt_lower = prompt.lower()

    chart_keywords = {
        "bar": ["bar chart", "bar graph", "bars", "column chart"],
        "line": ["line chart", "line graph", "trend", "time series"],
        "pie": ["pie chart", "pie graph", "proportion", "percentage breakdown"],
        "scatter": ["scatter plot", "scatter chart", "correlation", "x vs y"],
        "histogram": ["histogram", "distribution", "frequency"],
        "heatmap": ["heatmap", "heat map", "correlation matrix"],
        "area": ["area chart", "area graph"],
        "bubble": ["bubble chart", "bubble plot"]
    }
    visualization_triggers = [
        "visualize", "plot", "graph", "chart", "show me",
        "create a", "generate a", "make a", "draw a",
        "analyze", "display", "represent"
    ]

    has_trigger = any(trigger in prompt_lower for trigger in visualization_triggers)
    for chart_type, keywords in chart_keywords.items():
        if any(keyword in prompt_lower for keyword in keywords):
            return True, chart_type

    if has_trigger and any(word in prompt_lower for word in ["data", "numbers", "statistics", "values"]):
        return True, "auto"

    return False, None


def extract_data_from_text(text: str, prompt: str) -> Optional[Dict[str, Any]]:
    """
    Extract data from text to structure it for visualization.
    Returns a configuration dict or None.
    """
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    kv_pattern = r'(\w+):\s*([-+]?\d*\.?\d+)'
    kv_matches = re.findall(kv_pattern, text)

    if kv_matches:
        return {
            "type": "bar",
            "data": {
                "labels": [match[0] for match in kv_matches],
                "values": [float(match[1]) for match in kv_matches]
            },
            "title": "Data Visualization",
            "x_label": "Categories",
            "y_label": "Values"
        }
    return None


def create_streamlit_chart(data_config: Dict[str, Any]):
    """
    Create and display a chart using Streamlit's built-in chart elements based on the configuration.
    """
    chart_type = data_config.get("type", "bar")
    data = data_config.get("data", {})
    title = data_config.get("title", "")
    x_label = data_config.get("x_label", "")
    y_label = data_config.get("y_label", "Values")

    # Prepare DataFrame
    if chart_type in ("bar", "line", "area"):
        if "labels" in data and "values" in data:
            df = pd.DataFrame({y_label: data["values"]}, index=data["labels"])
        else:
            df = pd.DataFrame(data)
    elif chart_type == "scatter":
        df = pd.DataFrame({"x": data.get("x", []), "y": data.get("y", [])})
    elif chart_type == "map":
        df = pd.DataFrame(data.get("points", []))
    else:
        st.write(f"Chart type '{chart_type}' not supported with Streamlit built-ins.")
        return

    # Display title
    if title:
        st.markdown(f"### {title}")

    # Render chart
    if chart_type == "bar":
        st.bar_chart(df)
    elif chart_type == "line":
        st.line_chart(df)
    elif chart_type == "area":
        st.area_chart(df)
    elif chart_type == "scatter":
        st.scatter_chart(df)
    elif chart_type == "map":
        st.map(df)
    else:
        st.write(df)


def auto_detect_chart_type(data: pd.DataFrame) -> str:
    """
    Automatically detect the best chart type based on data characteristics.
    """
    num_numeric = data.select_dtypes(include=['number']).shape[1]
    num_categorical = data.select_dtypes(include=['object']).shape[1]
    num_rows = len(data)

    if num_numeric == 1 and num_categorical == 1 and num_rows < 20:
        return "bar"
    elif num_numeric == 2 and num_categorical == 0:
        return "scatter"
    elif num_numeric == 1 and num_categorical == 1 and num_rows < 10:
        return "pie"
    elif num_numeric >= 1 and data.index.name == "date":
        return "line"
    elif num_numeric == 1 and num_categorical == 0:
        return "histogram"
    return "bar"