"""Streamlit UI for the Operations Data AI Agent."""

import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agents.orchestrator import DataAgent
from src.data.database import get_schema_info, execute_query

# Page config
st.set_page_config(
    page_title="Operations Data AI Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .assistant-message {
        background-color: #F5F5F5;
    }
    .sql-code {
        background-color: #263238;
        color: #AEDDFF;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None


def init_agent():
    """Initialize the AI agent."""
    if st.session_state.agent is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
            st.stop()

        with st.spinner("Initializing AI Agent..."):
            st.session_state.agent = DataAgent()


def render_sidebar():
    """Render the sidebar with info and examples."""
    with st.sidebar:
        st.markdown("### 📊 Operations Data Agent")
        st.markdown("---")

        st.markdown("#### Example Queries")
        example_queries = [
            "Show me total revenue by region",
            "What are the top 10 products by sales?",
            "Show monthly revenue trend",
            "Compare Q3 vs Q4 sales",
            "Forecast revenue for next 3 months",
            "Which customers have the highest order value?",
            "Show sales by channel",
            "What's the average order value by segment?",
        ]

        for query in example_queries:
            if st.button(query, key=f"example_{query[:20]}", use_container_width=True):
                st.session_state.selected_query = query

        st.markdown("---")

        st.markdown("#### Database Schema")
        with st.expander("View Tables"):
            schema = get_schema_info()
            st.code(schema, language="text")

        st.markdown("---")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown(
            "<small>Built with LangGraph + OpenAI</small>",
            unsafe_allow_html=True
        )


def render_message(message: dict):
    """Render a chat message."""
    role = message["role"]
    content = message["content"]

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            # Response text
            if "response" in message:
                st.markdown(message["response"])

            # Data table
            if "data" in message and message["data"] is not None:
                df = message["data"]
                if not df.empty:
                    with st.expander(f"📋 View Data ({len(df)} rows)", expanded=len(df) <= 10):
                        st.dataframe(df, use_container_width=True)

            # Chart
            if "chart" in message and message["chart"] is not None:
                st.plotly_chart(message["chart"], use_container_width=True)


def process_query(query: str):
    """Process a user query and get response."""
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Get response from agent
    with st.spinner("Thinking..."):
        result = st.session_state.agent.run(query)

    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("response", ""),
        "response": result.get("response", ""),
        "data": result.get("data"),
        "chart": result.get("chart"),
        "sql_query": result.get("sql_query"),
    })


def main():
    """Main application."""
    init_session_state()

    # Header
    st.markdown('<p class="main-header">📊 Operations Data AI Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your data in natural language</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    render_sidebar()

    # Initialize agent
    init_agent()

    # Check for example query selection
    if "selected_query" in st.session_state:
        query = st.session_state.selected_query
        del st.session_state.selected_query
        process_query(query)
        st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        render_message(message)

    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        process_query(prompt)
        st.rerun()

    # Welcome message if no messages
    if not st.session_state.messages:
        st.markdown("""
        ### 👋 Welcome!

        I'm your AI-powered data analyst. I can help you:

        - **Query your data** in natural language
        - **Visualize trends** with automatic charts
        - **Forecast future values** using time series analysis
        - **Generate insights** and recommendations

        Try clicking one of the example queries in the sidebar, or type your own question below!

        ---

        **Sample questions to get started:**
        - "Show me total revenue by region"
        - "What are the top 10 products by sales?"
        - "Forecast revenue for the next 3 months"
        """)


if __name__ == "__main__":
    main()
