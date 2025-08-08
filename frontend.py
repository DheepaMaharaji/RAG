import streamlit as st
import requests
import json
from datetime import datetime
import time

# Configure Streamlit page
st.set_page_config(
    page_title="NU Global Services Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend configuration
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #CC0000;
        padding: 1rem 0;
        border-bottom: 2px solid #CC0000;
        margin-bottom: 2rem;
    }

    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #CC0000;
    }

    .user-message {
        background-color: #f0f2f6;
        margin-left: 2rem;
    }

    .bot-message {
        background-color: #ffffff;
        margin-right: 2rem;
        border: 1px solid #e6e6e6;
    }

    .timestamp {
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 0.5rem;
    }

    .status-indicator {
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
    }

    .status-healthy {
        background-color: #d4edda;
        color: #155724;
    }

    .status-unhealthy {
        background-color: #f8d7da;
        color: #721c24;
    }

    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    """Check if the backend is healthy and return status info."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "unhealthy", "error": str(e)}


def get_system_stats():
    """Get system statistics from the backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None


def send_message(message):
    """Send a message to the backend and return the response."""
    try:
        payload = {"message": message}
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Backend error (HTTP {response.status_code})"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg += f": {error_data['error']}"
            except:
                pass
            return {"error": error_msg}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}


def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎓 NU Global Services Assistant</h1>
            <p>Ask questions about scholarships, fellowships, and academic opportunities at Northeastern University</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("System Status")

        # Check backend health
        with st.spinner("Checking backend status..."):
            health_status = check_backend_health()

        if health_status["status"] == "healthy":
            st.markdown(
                '<span class="status-indicator status-healthy">🟢 Backend Online</span>',
                unsafe_allow_html=True
            )

            # Display additional health info
            if health_status.get("rag_initialized"):
                st.success("✅ RAG System Initialized")
            else:
                st.warning("⚠️ RAG System Not Initialized")

            if health_status.get("collection_count", 0) > 0:
                st.info(f"📚 {health_status['collection_count']} documents loaded")
            else:
                st.warning("📚 No documents loaded")

        else:
            st.markdown(
                '<span class="status-indicator status-unhealthy">🔴 Backend Offline</span>',
                unsafe_allow_html=True
            )
            st.error(f"Error: {health_status.get('error', 'Unknown error')}")

        # System Statistics
        st.header("System Info")
        stats = get_system_stats()
        if stats:
            st.markdown(f"""
            <div class="sidebar-section">
                <strong>Collection:</strong> {stats.get('collection_name', 'N/A')}<br>
                <strong>Documents:</strong> {stats.get('document_count', 0)}
            </div>
            """, unsafe_allow_html=True)

        # Sample Questions
        st.header("Sample Questions")
        sample_questions = [
            "What scholarships are available for international students?",
            "Tell me about the Double Husky Scholarship",
            "What are the fellowship opportunities at Northeastern?",
            "How do I apply for graduate scholarships?",
            "What is the Full Circle Scholarship?"
        ]

        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}", use_container_width=True):
                st.session_state.sample_question = question

        # Clear Chat Button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Handle sample question selection
    if hasattr(st.session_state, 'sample_question'):
        st.session_state.user_input = st.session_state.sample_question
        delattr(st.session_state, 'sample_question')

    # Main chat interface
    col1, col2 = st.columns([3, 1])

    with col1:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                timestamp = message.get("timestamp", "")

                if message["role"] == "user":
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <div class="timestamp">You • {timestamp}</div>
                            <div><strong>🧑‍🎓 You:</strong> {message["content"]}</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-message bot-message">
                            <div class="timestamp">Assistant • {timestamp}</div>
                            <div><strong>🤖 Assistant:</strong> {message["content"]}</div>
                        </div>
                    """, unsafe_allow_html=True)

    # Chat input
    st.markdown("---")

    # Create input form
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            user_input = st.text_input(
                "Ask your question:",
                placeholder="Type your question about NU Global Services here...",
                key="user_input_field"
            )

        with col2:
            submit_button = st.form_submit_button("Send 📤", use_container_width=True)

    # Process user input
    if submit_button and user_input:
        # Add user message to chat history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })

        # Show thinking indicator
        with st.spinner("🤔 Thinking..."):
            # Send message to backend
            response = send_message(user_input)

            # Add assistant response to chat history
            timestamp = datetime.now().strftime("%H:%M:%S")
            if "error" in response:
                assistant_message = f"❌ {response['error']}"
            else:
                assistant_message = response.get("response", "Sorry, I couldn't generate a response.")

            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_message,
                "timestamp": timestamp
            })

        # Rerun to show the new messages
        st.rerun()

    # Auto-scroll to bottom (using empty container)
    if st.session_state.messages:
        st.empty()

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            NU Global Services Assistant • Powered by RAG & AI<br>
            For urgent matters, please contact the Office of Global Services directly.
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()