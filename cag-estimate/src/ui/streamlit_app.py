"""
CAG Estimate Chat Interface
Feature 1: Chat Interface with Message History (custom iMessage-style bubbles, st.chat_input)
Feature 2: Real-Time Token-by-Token Streaming from API
Feature 3: Sidebar with Context, Metrics, and System Info
"""

import streamlit as st
from datetime import datetime
import requests
import json
import re
import html as html_lib
import sys
from pathlib import Path

# This file lives at src/ui/streamlit_app.py; add src/ (its parent) to the
# path so `cag_estimate` is importable as a top-level package regardless of
# whether it's pip-installed in the active environment.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cag_estimate.context.examples import ESTIMATION_EXAMPLES

# Page configuration
st.set_page_config(
    page_title="CAG Estimate Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "call_count" not in st.session_state:
        st.session_state.call_count = 0

    if "session_started" not in st.session_state:
        st.session_state.session_started = datetime.now().isoformat()

    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    if "session_cost" not in st.session_state:
        st.session_state.session_cost = 0.0

    if "last_call" not in st.session_state:
        st.session_state.last_call = None


init_session_state()


# API Configuration
API_BASE_URL = "http://localhost:8000"  # FastAPI running in Docker
ESTIMATE_ENDPOINT = f"{API_BASE_URL}/api/v1/estimate"
STREAM_ENDPOINT = f"{API_BASE_URL}/api/v1/estimate/stream"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Default hourly rate
DEFAULT_HOURLY_RATE = 40


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def stream_estimation(transcription: str, hourly_rate: int = DEFAULT_HOURLY_RATE):
    """
    Call the streaming estimation endpoint and yield parsed SSE events
    as they arrive, one token at a time.

    Yields:
        dict: {"type": "token", "content": "..."}
              {"type": "done", "tokens_used": {...}, "cost_breakdown": {...}, ...}
              {"type": "error", "message": "..."}
    """
    payload = {"transcription": transcription, "hourly_rate": hourly_rate}

    with requests.post(STREAM_ENDPOINT, json=payload, stream=True, timeout=120) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data_str = raw_line[len("data: "):]
            try:
                yield json.loads(data_str)
            except json.JSONDecodeError:
                continue


def markdown_to_bubble_html(text: str) -> str:
    """
    Escape user/LLM text for safe HTML embedding, then convert the small
    subset of Markdown we actually produce (bold, bullet lines, newlines)
    into HTML so it renders correctly inside a custom bubble <div> — plain
    st.markdown() would otherwise treat that div as a raw HTML block and
    skip Markdown processing of its contents entirely.
    """
    escaped = html_lib.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?m)^- (.+)$", r"&nbsp;&nbsp;• \1", escaped)
    escaped = escaped.replace("\n", "<br>")
    return escaped


def render_message_html(role: str, text: str, timestamp: str = "", streaming: bool = False) -> str:
    """Render a message as a WhatsApp/iMessage-style chat bubble: user on the
    right in sky blue, assistant on the left in neutral gray."""
    content_html = markdown_to_bubble_html(text)
    cursor = " ▌" if streaming else ""
    timestamp_html = (
        f'<div style="font-size:0.7rem; opacity:0.7; margin-top:6px; '
        f'text-align:{"right" if role == "user" else "left"};">{timestamp}</div>'
        if timestamp else ""
    )

    if role == "user":
        justify, radius, bg, color = "flex-end", "18px 18px 4px 18px", "#87CEEB", "#08303f"
    else:
        justify, radius, bg, color = "flex-start", "18px 18px 18px 4px", "#F0F2F6", "#1a1a1a"

    return f"""
    <div style="display:flex; justify-content:{justify}; margin:10px 0;">
      <div style="background-color:{bg}; color:{color}; padding:10px 16px;
                  border-radius:{radius}; max-width:75%; line-height:1.5;
                  box-shadow:0 1px 2px rgba(0,0,0,0.12); word-wrap:break-word;">
        {content_html}{cursor}
        {timestamp_html}
      </div>
    </div>
    """


THINKING_BUBBLE_HTML = """
<div style="display:flex; justify-content:flex-start; margin:10px 0;">
  <div style="background-color:#F0F2F6; padding:10px 16px; border-radius:18px 18px 18px 4px;
              box-shadow:0 1px 2px rgba(0,0,0,0.12);">
    <div class="thinking-indicator">
      <span>🤖 Thinking</span>
      <span class="thinking-dot"></span>
    </div>
  </div>
</div>
<style>
.thinking-indicator { display: flex; align-items: center; gap: 10px; color: #666; }
.thinking-dot {
  position: relative;
  width: 6px; height: 6px; border-radius: 50%;
  background-color: #9880ff;
  animation: thinking-blink 1s infinite linear alternate;
  animation-delay: .5s;
}
.thinking-dot::before, .thinking-dot::after {
  content: ''; position: absolute; top: 0;
  width: 6px; height: 6px; border-radius: 50%;
  background-color: #9880ff;
}
.thinking-dot::before { left: -12px; animation: thinking-blink 1s infinite linear alternate; animation-delay: 0s; }
.thinking-dot::after { left: 12px; animation: thinking-blink 1s infinite linear alternate; animation-delay: 1s; }
@keyframes thinking-blink {
  0% { background-color: #9880ff; }
  50%, 100% { background-color: rgba(152, 128, 255, 0.2); }
}
</style>
"""


# Header
st.title("💬 Ready to estimate your projects?")
st.markdown("**Your AI assistant, expert in software project estimation**")
st.divider()


# Display chat history
if not st.session_state.messages:
    st.info("👋 Start a conversation by describing your project below!")
else:
    for message in st.session_state.messages:
        st.markdown(
            render_message_html(message["role"], message["content"], message.get("timestamp", "")),
            unsafe_allow_html=True,
        )


# Chat input - st.chat_input auto-clears itself after submission
prompt = st.chat_input(
    placeholder="Describe your project... (you can paste a meeting transcription)"
)

if prompt:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save + show user message immediately (right side, sky-blue bubble)
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp,
    })
    st.markdown(render_message_html("user", prompt, timestamp), unsafe_allow_html=True)

    # Stream the assistant response token-by-token, live as the model generates it
    # (left side, gray bubble). Uses the placeholder + delta pattern so each new
    # chunk can be re-wrapped in the bubble HTML as it arrives.
    message_placeholder = st.empty()
    message_placeholder.markdown(THINKING_BUBBLE_HTML, unsafe_allow_html=True)

    full_text = ""
    final_event = None
    error_message = None

    if not check_api_health():
        error_message = "⚠️ API is not running. Start it with: `docker-compose up`"
        message_placeholder.error(error_message)
    else:
        try:
            for event in stream_estimation(prompt, DEFAULT_HOURLY_RATE):
                event_type = event.get("type")

                if event_type == "token":
                    full_text += event.get("content", "")
                    message_placeholder.markdown(
                        render_message_html("assistant", full_text, streaming=True),
                        unsafe_allow_html=True,
                    )

                elif event_type == "done":
                    final_event = event

                elif event_type == "error":
                    error_message = event.get("message", "Unknown error")

        except requests.exceptions.ConnectionError:
            error_message = (
                "❌ Cannot connect to API. Make sure Docker container is running: "
                "`docker-compose up`"
            )
        except requests.exceptions.Timeout:
            error_message = "❌ API request timed out. Try again."
        except Exception as e:
            error_message = f"❌ Error calling API: {str(e)}"

    if final_event and full_text:
        tokens_used = final_event.get("tokens_used", {})
        cost_breakdown = final_event.get("cost_breakdown", {})

        metrics_line = (
            f"\n\n---\n**API Metrics:** {tokens_used.get('total_tokens', 0):,} tokens · "
            f"${cost_breakdown.get('total_cost_usd', 0):.6f}"
        )
        full_content = full_text + metrics_line
        response_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        message_placeholder.markdown(
            render_message_html("assistant", full_content, response_timestamp),
            unsafe_allow_html=True,
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_content,
            "timestamp": response_timestamp,
            "tokens": tokens_used.get("total_tokens", 0),
            "cost": cost_breakdown.get("total_cost_usd", 0),
        })

        st.session_state.call_count += 1
        st.session_state.total_tokens += tokens_used.get("total_tokens", 0)
        st.session_state.session_cost += cost_breakdown.get("total_cost_usd", 0)
        st.session_state.last_call = {
            "model_name": final_event.get("model", "N/A"),
            "provider": final_event.get("provider", "N/A"),
            "input_tokens": tokens_used.get("input_tokens", 0),
            "output_tokens": tokens_used.get("output_tokens", 0),
            "total_tokens": tokens_used.get("total_tokens", 0),
            "cost_usd": cost_breakdown.get("total_cost_usd", 0),
            "finish_reason": final_event.get("finish_reason", "stop"),
        }

    elif error_message:
        message_placeholder.error(error_message)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ {error_message}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })


# Sidebar with session info
with st.sidebar:
    st.header("⚙️ Controls & Info")

    # 1. New Chat button (first)
    if st.button("🔄 New Chat", use_container_width=True, help="Start a fresh conversation"):
        st.session_state.messages = []
        st.session_state.call_count = 0
        st.session_state.total_tokens = 0
        st.session_state.session_cost = 0.0
        st.session_state.last_call = None
        st.rerun()

    st.divider()

    # 2. Conversation Stats
    user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
    assistant_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])

    st.info(f"""
    **Conversation Stats:**
    - 👤 Your messages: {user_msgs}
    - 🤖 Assistant messages: {assistant_msgs}
    """)

    st.divider()

    # 3. Session Metrics (Calls Made, Tokens, Cost)
    st.subheader("📊 Session Metrics")

    api_healthy = check_api_health()
    status_icon = "🟢" if api_healthy else "🔴"
    st.caption(f"{status_icon} API: {'Connected' if api_healthy else 'Disconnected'}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Calls Made", st.session_state.call_count)
        st.metric("Total Tokens", st.session_state.total_tokens)
    with col2:
        st.metric("Session Cost", f"${st.session_state.session_cost:.6f}")
        st.metric("Avg Tokens/Call",
                  int(st.session_state.total_tokens / max(st.session_state.call_count, 1)))

    if st.session_state.last_call:
        with st.expander("📋 Last Call Info", expanded=False):
            last = st.session_state.last_call
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model", last["model_name"])
                st.metric("Input Tokens", f"{last['input_tokens']:,}")
            with col2:
                st.metric("Output Tokens", f"{last['output_tokens']:,}")
                st.metric("Cost", f"${last['cost_usd']:.6f}")
            st.caption(f"Provider: {last['provider']} · Finish reason: {last['finish_reason']}")

    st.divider()

    # 4. How to Use
    with st.expander("❓ How to Use", expanded=False):
        st.markdown("""
        1. **Describe your project** in the chat box below
        2. **Paste meeting transcriptions** if you have them
        3. **Press Enter** to get an estimation
        4. **Continue the conversation** with follow-up questions
        5. **View history** - all messages persist during your session

        **Tips:**
        - Be specific about requirements
        - Include technologies you want to use
        - Ask about specific components
        """)

    st.divider()

    # 5. Context (last)
    st.subheader("📚 Context")
    st.caption(
        f"{len(ESTIMATION_EXAMPLES)} reference project(s) are injected into the "
        "system prompt on every call (CAG)."
    )

    with st.expander("View reference examples", expanded=False):
        for example in ESTIMATION_EXAMPLES:
            summary = example.get("summary", {})
            st.markdown(f"**{example.get('project_name', 'N/A')}**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Hours", summary.get("total_hours", "N/A"))
                st.metric("Team", summary.get("team_size", "N/A"))
            with col2:
                st.metric("Cost", f"${summary.get('total_cost_usd', 0):,}")
                st.metric("Duration", f"{summary.get('estimated_duration_weeks', 'N/A')} wks")

            st.caption(f"{len(example.get('tasks', []))} tasks · ${summary.get('hourly_rate', 0)}/hour")
            st.divider()


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    <p>CAG Estimate Chat Interface | Real-Time Streaming</p>
    <p>Built with Streamlit & Claude AI</p>
</div>
""", unsafe_allow_html=True)
