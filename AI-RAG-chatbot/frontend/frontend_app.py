import streamlit as st
import requests
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="🤖 Mistral Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoints
API_BASE = "http://localhost:8000"
CHAT_URL = f"{API_BASE}/chat"
SESSIONS_URL = f"{API_BASE}/sessions"
NEW_SESSION_URL = f"{API_BASE}/sessions/new"

# Initialize session state
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_documents" not in st.session_state:
    st.session_state.search_documents = True

def load_sessions():
    """Fetches the list of chat sessions from the backend."""
    try:
        response = requests.get(SESSIONS_URL)
        if response.status_code == 200:
            st.session_state.sessions = response.json().get("sessions", [])
    except Exception as e:
        st.error(f"Error loading sessions: {e}")

def create_new_session():
    """Creates a new chat session."""
    try:
        response = requests.post(NEW_SESSION_URL, json={"title": "New Chat"})
        if response.status_code == 200:
            data = response.json()
            st.session_state.current_session_id = data["session_id"]
            st.session_state.messages = []
            load_sessions()
            st.rerun()
    except Exception as e:
        st.error(f"Error creating session: {e}")

def load_session(session_id):
    """Loads a specific chat session's history."""
    try:
        response = requests.get(f"{SESSIONS_URL}/{session_id}")
        if response.status_code == 200:
            session = response.json()
            st.session_state.current_session_id = session_id
            st.session_state.messages = session["messages"]
            st.rerun()
    except Exception as e:
        st.error(f"Error loading session: {e}")

def delete_session(session_id):
    """Deletes a chat session."""
    try:
        response = requests.delete(f"{SESSIONS_URL}/{session_id}")
        if response.status_code == 200:
            if st.session_state.current_session_id == session_id:
                st.session_state.current_session_id = None
                st.session_state.messages = []
            load_sessions()
            st.rerun()
    except Exception as e:
        st.error(f"Error deleting session: {e}")

def group_sessions_for_chatgpt_layout(sessions):
    """
    Groups sessions into relative timeframes like "Today", "Yesterday", "Previous 7 Days", etc.
    """
    grouped = {}
    now = datetime.now(pytz.utc)
    today = now.date()
    
    for session in sessions:
        try:
            date_str = session['created_at'].replace('Z', '+00:00')
            session_time = datetime.fromisoformat(date_str)
            session_date = session_time.date()
            delta = today - session_date
            
            group_name = None
            if delta.days == 0:
                group_name = "Today"
            elif delta.days == 1:
                group_name = "Yesterday"
            elif 1 < delta.days <= 7:
                group_name = "Previous 7 Days"
            elif 7 < delta.days <= 30:
                group_name = "Previous 30 Days"
            else:
                group_name = session_time.strftime("%B %Y")
            
            if group_name:
                if group_name not in grouped:
                    grouped[group_name] = []
                grouped[group_name].append(session)
        except (ValueError, KeyError):
            if "Unknown" not in grouped:
                grouped["Unknown"] = []
            grouped["Unknown"].append(session)
    return grouped

# --- Main App Logic ---

if not st.session_state.sessions:
    load_sessions()

# Sidebar for chat history
with st.sidebar:
    st.title("💬 Chat History")
    if st.button("➕ New Chat", use_container_width=True):
        create_new_session()
    st.divider()
    st.checkbox(
        "🔍 Search documents",
        value=st.session_state.search_documents,
        key="search_documents",
        help="When checked, searches your document database. When unchecked, uses general knowledge only."
    )
    st.divider()

    if st.session_state.sessions:
        grouped_sessions = group_sessions_for_chatgpt_layout(st.session_state.sessions)
        group_order = ["Today", "Yesterday", "Previous 7 Days", "Previous 30 Days"]
        month_keys = sorted(
            [key for key in grouped_sessions if key not in group_order and key != "Unknown"],
            key=lambda m: datetime.strptime(m, "%B %Y"),
            reverse=True
        )
        all_groups_in_order = group_order + month_keys
        if "Unknown" in grouped_sessions:
            all_groups_in_order.append("Unknown")

        for group_name in all_groups_in_order:
            if group_name in grouped_sessions:
                st.caption(group_name.upper())
                for session in grouped_sessions[group_name]:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        is_selected = session['session_id'] == st.session_state.current_session_id
                        button_type = "primary" if is_selected else "secondary"
                        if st.button(
                            f"💬 {session['title']}",
                            key=f"btn_{session['session_id']}",
                            use_container_width=True,
                            type=button_type,
                            help=f"Created: {datetime.fromisoformat(session['created_at'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')}"
                        ):
                            load_session(session["session_id"])
                    with col2:
                        if st.button("🗑️", key=f"del_{session['session_id']}", help="Delete chat"):
                            delete_session(session["session_id"])
                st.divider()
    else:
        st.info("No chat sessions yet. Start a new chat!")

# Main chat area
st.title("🤖 Manoj GPT")
st.write("Chat with your documents using Mistral-7B")
search_status = "🔍 **Searching documents**" if st.session_state.search_documents else "💭 **General knowledge only**"
st.caption(search_status)

if st.session_state.current_session_id:
    current_session = next((s for s in st.session_state.sessions if s["session_id"] == st.session_state.current_session_id), None)
    if current_session:
        st.subheader(f"📝 {current_session['title']}")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # --- CHANGE IS HERE ---
            # The st.expander for "Sources Used" and the captions have been removed.
            # The main content from the LLM already includes the "Source: ..." line,
            # so we no longer need to display it separately.
            # --- END OF CHANGE ---

# Handle chat input
if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat(),
        "search_used": st.session_state.search_documents
    })

    try:
        with st.spinner("Thinking..."):
            response = requests.post(CHAT_URL, json={
                "message": user_input,
                "session_id": st.session_state.current_session_id,
                "search_documents": st.session_state.search_documents
            })

            if response.status_code == 200:
                data = response.json()
                st.session_state.current_session_id = data["session_id"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["message"],
                    "sources": data.get("sources", []),
                    "timestamp": datetime.now().isoformat(),
                    "search_used": st.session_state.search_documents
                })
                load_sessions()
            else:
                error_message = f"Error from API: {response.status_code} - {response.text}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant", "content": error_message, "timestamp": datetime.now().isoformat()
                })
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        st.session_state.messages.append({
            "role": "assistant", "content": f"Sorry, I couldn't connect to the server: {e}", "timestamp": datetime.now().isoformat()
        })
    
    st.rerun()

if not st.session_state.current_session_id and not st.session_state.messages:
    st.info("💡 Click 'New Chat' in the sidebar or just start typing to begin a conversation!")