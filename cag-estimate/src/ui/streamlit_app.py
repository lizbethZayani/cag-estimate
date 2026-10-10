"""
CAG Estimate Chat Interface

- Chat with message history kept in ``st.session_state`` and re-rendered on every rerun.
- Three modes: conversation with session memory (default), streaming markdown
  (token by token) and structured JSON (task table).
- Friendly guardrail messages; raw server errors are never displayed.

Note: the streaming and structured modes are single-shot per message: their history is
a UI convenience and is NOT sent back as context. Conversation mode uses an API
session (project memory + sliding window) that lives in the API process memory.
"""

import html as html_lib
import re
from datetime import datetime
from typing import Any

import estimate_client
import streamlit as st
import view_models

MODE_CONVERSATION = "Conversation (memory)"
MODE_STREAMING = "Streaming (markdown)"
MODE_STRUCTURED = "Structured (JSON)"
DEFAULT_HOURLY_RATE = 40
MIN_HOURLY_RATE, MAX_HOURLY_RATE = 30, 150
CONVERSATION_KEY = "conversation_messages"
ATTACHMENT_TYPES = ["pdf", "txt", "md", "csv", "json"]
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

st.set_page_config(
    page_title="CAG Estimate Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def session_defaults() -> dict[str, Any]:
    return {
        "messages": [],
        "call_count": 0,
        "total_tokens": 0,
        "session_cost": 0.0,
        "last_call": None,
    }


def apply_state(defaults: dict[str, Any], overwrite: bool = False) -> None:
    """Set state defaults; with ``overwrite`` reset the keys to their defaults."""
    for key, value in defaults.items():
        if overwrite:
            st.session_state[key] = value
        else:
            st.session_state.setdefault(key, value)


def init_session_state() -> None:
    """Initialize session state variables."""
    apply_state(session_defaults())


def now() -> str:
    return datetime.now().astimezone().strftime(TIMESTAMP_FORMAT)


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


def render_structured(result: dict[str, Any], cached: bool, caption: str | None = None) -> None:
    """Render a ProjectEstimation dict: cache badge, summary, task table and totals."""
    with st.chat_message("assistant"):
        if cached:
            st.success("⚡ Served from cache")
        else:
            st.info("🧠 Freshly generated by the model")
        st.subheader(result["project_name"])
        st.write(result["meeting_summary"])
        st.dataframe(view_models.task_rows(result), hide_index=True)
        for column, (label, value) in zip(
            st.columns(5), view_models.totals(result).items(), strict=True
        ):
            column.metric(label, value)
        assumptions = result["summary"].get("assumptions") or []
        if assumptions:
            st.markdown("**Assumptions**\n" + "\n".join(f"- {a}" for a in assumptions))
        if caption:
            st.caption(caption)


def render_message(message: dict[str, Any]) -> None:
    """Render one stored message according to its kind."""
    kind = message.get("kind", "text")
    if kind == "structured":
        render_structured(message["result"], message["cached"], message.get("caption"))
    elif kind == "error":
        st.error(message["content"])
    else:
        st.markdown(
            render_message_html(message["role"], message["content"], message.get("timestamp", "")),
            unsafe_allow_html=True,
        )


def store_message(role: str, target: str = "messages", **fields: Any) -> None:
    """Append a timestamped message to the chat history kept under ``target``."""
    st.session_state[target].append({"role": role, "timestamp": now(), **fields})


def extract_usage(final_event: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Token usage dict and total cost (USD) from a ``done`` event."""
    tokens = final_event.get("tokens_used", {})
    cost = final_event.get("cost_breakdown", {}).get("total_cost_usd", 0)
    return tokens, cost


def record_usage(final_event: dict[str, Any]) -> None:
    tokens, cost = extract_usage(final_event)
    st.session_state.call_count += 1
    st.session_state.total_tokens += tokens.get("total_tokens", 0)
    st.session_state.session_cost += cost
    st.session_state.last_call = {
        "model_name": final_event.get("model", "N/A"),
        "provider": final_event.get("provider", "N/A"),
        "input_tokens": tokens.get("input_tokens", 0),
        "output_tokens": tokens.get("output_tokens", 0),
        "cost_usd": cost,
        "finish_reason": final_event.get("finish_reason", "stop"),
    }


def consume_stream(prompt: str, placeholder: Any) -> tuple[str, dict[str, Any] | None, str | None]:
    """Render tokens live; return (text, done event, safe error message)."""
    full_text, final_event, error = "", None, None
    for event in estimate_client.stream_events(prompt, DEFAULT_HOURLY_RATE):
        if event["type"] == "token":
            full_text += event.get("content", "")
            placeholder.markdown(
                render_message_html("assistant", full_text, streaming=True),
                unsafe_allow_html=True,
            )
        elif event["type"] == "done":
            final_event = event
        elif event["type"] == "error":
            error = event["message"]
    return full_text, final_event, error


def metrics_footer(final_event: dict[str, Any]) -> str:
    tokens, cost = extract_usage(final_event)
    return f"\n\n---\n**API Metrics:** {tokens.get('total_tokens', 0):,} tokens · ${cost:.6f}"


def run_streaming(prompt: str) -> None:
    """Stream the markdown answer live, then store it (or the safe error)."""
    placeholder = st.empty()
    placeholder.markdown(THINKING_BUBBLE_HTML, unsafe_allow_html=True)
    full_text, final_event, error = consume_stream(prompt, placeholder)
    placeholder.empty()
    if error or not (final_event and full_text):
        error = error or view_models.GENERIC_ERROR
        store_message("assistant", kind="error", content=error)
        st.error(error)
        return
    store_message("assistant", kind="text", content=full_text + metrics_footer(final_event))
    render_message(st.session_state.messages[-1])
    record_usage(final_event)


def run_structured(prompt: str) -> None:
    """Call the structured endpoint and store the table-ready answer (or the safe error)."""
    with st.spinner("Estimating..."):
        answer = estimate_client.fetch_structured(prompt, DEFAULT_HOURLY_RATE)
    if answer.result is None:
        store_message("assistant", kind="error", content=answer.error)
        st.error(answer.error)
        return
    store_message("assistant", kind="structured", result=answer.result, cached=answer.cached)
    render_message(st.session_state.messages[-1])
    st.session_state.call_count += 1


def handle_prompt(prompt: str, mode: str) -> None:
    store_message("user", kind="text", content=prompt)
    render_message(st.session_state.messages[-1])
    if not estimate_client.check_api_health():
        message = "⚠️ API is not running. Start it with: `docker-compose up`"
        store_message("assistant", kind="error", content=message)
        st.error(message)
    elif mode == MODE_STRUCTURED:
        run_structured(prompt)
    else:
        run_streaming(prompt)


def render_mode_switch() -> str:
    with st.sidebar:
        st.header("⚙️ Controls & Info")
        return st.radio("Response mode", [MODE_CONVERSATION, MODE_STREAMING, MODE_STRUCTURED])


def reset_conversation() -> None:
    apply_state(session_defaults(), overwrite=True)


def render_last_call(last: dict[str, Any]) -> None:
    with st.expander("📋 Last Call Info", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", last["model_name"])
            st.metric("Input Tokens", f"{last['input_tokens']:,}")
        with col2:
            st.metric("Output Tokens", f"{last['output_tokens']:,}")
            st.metric("Cost", f"${last['cost_usd']:.6f}")
        st.caption(f"Provider: {last['provider']} · Finish reason: {last['finish_reason']}")


def render_sidebar_info() -> None:
    with st.sidebar:
        if st.button("🔄 New Chat", use_container_width=True, help="Start a fresh conversation"):
            reset_conversation()
            st.rerun()
        st.divider()
        messages = st.session_state.messages
        st.info(
            "**Conversation Stats:**\n"
            f"- 👤 Your messages: {sum(m['role'] == 'user' for m in messages)}\n"
            f"- 🤖 Assistant messages: {sum(m['role'] == 'assistant' for m in messages)}"
        )
        st.subheader("📊 Session Metrics")
        healthy = estimate_client.check_api_health()
        st.caption(f"{'🟢' if healthy else '🔴'} API: {'Connected' if healthy else 'Disconnected'}")
        col1, col2 = st.columns(2)
        col1.metric("Calls Made", st.session_state.call_count)
        col1.metric("Total Tokens", st.session_state.total_tokens)
        col2.metric("Session Cost", f"${st.session_state.session_cost:.6f}")
        if st.session_state.last_call:
            render_last_call(st.session_state.last_call)
        st.divider()
        st.caption(
            "Each message is estimated on its own: the backend is single-shot and does "
            "not use earlier messages as context."
        )


def conversation_defaults() -> dict[str, Any]:
    return {
        CONVERSATION_KEY: [],
        "project_metadata": None,
        "turns": 0,
        "session_error": None,
        "input_nonce": 0,
    }


def open_session() -> None:
    """Create an API session and reset its memory view; keep a safe error if it fails."""
    started = estimate_client.create_session()
    st.session_state.session_id = started.session_id
    st.session_state.session_error = started.error
    st.session_state.project_metadata = None
    st.session_state.turns = 0


def start_new_conversation() -> None:
    apply_state(conversation_defaults(), overwrite=True)
    open_session()
    st.session_state.input_nonce += 1


def ensure_session() -> None:
    """Create the session lazily, once per browser session (retried while it is missing)."""
    apply_state(conversation_defaults())
    if st.session_state.get("session_id") is None:
        open_session()


def apply_session_answer(answer: estimate_client.SessionAnswer) -> bool:
    """Store the outcome of a turn; return True when it succeeded."""
    if answer.result is None:
        store_message("assistant", CONVERSATION_KEY, kind="error", content=answer.error)
        if answer.expired:
            open_session()
        return False
    st.session_state.project_metadata = answer.project_metadata
    st.session_state.turns = answer.turns
    store_message(
        "assistant",
        CONVERSATION_KEY,
        kind="structured",
        result=answer.result,
        cached=False,
        caption=view_models.turn_caption(answer.turns),
    )
    return True


def send_turn(text: str, hourly_rate: int, uploads: list[Any]) -> None:
    files = [(f.name, f.getvalue(), f.type or "application/octet-stream") for f in uploads]
    names = [name for name, _, _ in files]
    store_message(
        "user", CONVERSATION_KEY, kind="text", content=view_models.user_turn_text(text, names)
    )
    with st.spinner("Estimating... this can take up to a minute"):
        answer = estimate_client.send_session_estimate(
            st.session_state.session_id, text, hourly_rate, files
        )
    if apply_session_answer(answer):
        st.session_state.input_nonce += 1
    st.rerun()


def render_conversation_inputs() -> None:
    nonce = st.session_state.input_nonce
    text = st.text_area(
        "Message or meeting transcript",
        key=f"transcript_{nonce}",
        placeholder="Describe your project or the change you want re-estimated...",
    )
    uploads = st.file_uploader(
        "Attachments (optional)",
        type=ATTACHMENT_TYPES,
        accept_multiple_files=True,
        key=f"uploads_{nonce}",
    )
    rate = st.number_input(
        "Hourly rate (USD)", MIN_HOURLY_RATE, MAX_HOURLY_RATE, DEFAULT_HOURLY_RATE
    )
    if not st.button("Send", type="primary"):
        return
    if not text.strip():
        st.warning("Write a message first.")
    elif st.session_state.session_id is None:
        st.error(st.session_state.session_error or view_models.GENERIC_ERROR)
    else:
        send_turn(text, int(rate), uploads)


def render_conversation() -> None:
    ensure_session()
    if st.session_state.session_error:
        st.error(st.session_state.session_error)
    messages = st.session_state[CONVERSATION_KEY]
    if not messages:
        st.info("👋 Describe your project. Follow-up messages build on what was agreed before.")
    for message in messages:
        render_message(message)
    render_conversation_inputs()


def refresh_memory() -> None:
    snapshot = estimate_client.get_session(st.session_state.session_id)
    if snapshot.expired:
        store_message("assistant", CONVERSATION_KEY, kind="error", content=snapshot.error)
        open_session()
    elif snapshot.error is None:
        st.session_state.project_metadata = snapshot.project_metadata
        st.session_state.turns = snapshot.turns


def render_memory_panel() -> None:
    with st.expander("🧠 Project memory (project_metadata)", expanded=True):
        for label, value in view_models.metadata_rows(st.session_state.project_metadata):
            st.markdown(f"**{label}:** {value}")
        session_id = st.session_state.session_id
        st.caption(
            f"Session {view_models.short_session_id(session_id)} · "
            f"{st.session_state.turns} turn(s)"
        )
        st.caption(
            "This memory is kept separately from the sliding-window history (the last "
            "turns) that the model also sees."
        )
        if session_id and st.button("Refresh memory", use_container_width=True):
            refresh_memory()
            st.rerun()


def render_conversation_sidebar() -> None:
    with st.sidebar:
        if st.button("🆕 New conversation", use_container_width=True):
            start_new_conversation()
            st.rerun()
        render_memory_panel()


def main() -> None:
    init_session_state()
    mode = render_mode_switch()
    st.title("💬 Ready to estimate your projects?")
    st.markdown("**Your AI assistant, expert in software project estimation**")
    st.divider()
    if mode == MODE_CONVERSATION:
        render_conversation()
        render_conversation_sidebar()
        return
    if not st.session_state.messages:
        st.info("👋 Start a conversation by describing your project below!")
    for message in st.session_state.messages:
        render_message(message)
    prompt = st.chat_input(
        placeholder="Describe your project... (you can paste a meeting transcription)"
    )
    if prompt:
        handle_prompt(prompt, mode)
    render_sidebar_info()


main()
