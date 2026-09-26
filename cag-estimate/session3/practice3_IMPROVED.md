# CAG Estimate Streamlit Chat UI - Requirements
**Focus:** Chat Interface, Real-Time Streaming, and System Sidebar  
**Version:** 2.0 (Tech Lead + Business Review)

---

## 📋 Executive Summary

Build a clean, interactive Streamlit chat interface that streams project estimations in real-time while providing system transparency through a sidebar dashboard.

**Why it matters:**
- 🚀 Real-time feedback: Users see results instantly (not wait for completion)
- 💡 Transparency: Full visibility into what's being processed and how much it costs
- 🎯 UX: Familiar chat interface (like WhatsApp/ChatGPT)
- 💼 Business: Demonstrates product capability and builds user confidence

---

## 🎯 Product Vision

**Problem:** Current API-only experience is technical and opaque
- No real-time feedback
- Can't see what system prompt is being used
- No cost visibility
- Limited accessibility for non-technical users

**Solution:** Streamlit chat UI with full transparency
- Interactive, conversational interface
- Real-time token streaming for immediate feedback
- Live cost and usage metrics
- System context visible in sidebar

---

## 🎨 Feature 1: Chat Interface

### User Experience Flow
```
User Opens App
    ↓
Sees Clean Chat Window + System Sidebar
    ↓
User Types Project Description (or Pastes)
    ↓
Clicks Send / Presses Enter
    ↓
Message Appears in Chat History (User)
    ↓
AI Response Streams Token-by-Token
    ↓
Conversation Saved in Memory
    ↓
User Can Continue Conversation (Multi-turn)
```

### Technical Requirements

**Chat History Display:**
```python
# Display all messages from this session
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        st.caption(f"⏰ {message['timestamp']}")
```

**Specifications:**
- ✅ Display conversation history chronologically
- ✅ Distinguish user messages vs assistant messages (different colors/icons)
- ✅ Show timestamps for each message
- ✅ Persist messages during session (clear on refresh)
- ✅ Support multi-line text input

**User Input:**
```python
# Accept user input at bottom of chat
user_input = st.chat_input(
    placeholder="Describe your project... (you can paste transcriptions)",
    key="chat_input"
)
```

**Specifications:**
- ✅ Always-visible input box at bottom
- ✅ Accept text AND pasted content (multi-line)
- ✅ Clear input after sending
- ✅ Placeholder text hints at use ("paste transcription")
- ✅ Works on mobile and desktop

**Message Storage (Session State):**
```python
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add messages
st.session_state.messages.append({
    "role": "user",
    "content": user_input,
    "timestamp": datetime.now().isoformat()
})

st.session_state.messages.append({
    "role": "assistant",
    "content": response,
    "timestamp": datetime.now().isoformat(),
    "tokens": token_count,
    "cost": cost_usd
})
```

**Definition of Done:**
- ✅ Type message → appears in chat immediately
- ✅ Send message → clears input field
- ✅ Messages persist during session
- ✅ Timestamps visible for each message
- ✅ Clear distinction between user and assistant
- ✅ Can have multi-turn conversations

---

## 🌊 Feature 2: Real-Time Token Streaming

### Why Streaming Matters
**Without Streaming:** User sees loading spinner for 3-5 seconds, then entire response appears
**With Streaming:** User sees text appearing 1-by-1, feels interactive (like real conversation)

**Impact:**
- 👍 Feels faster (perceived latency lower)
- 👍 User sees work is happening
- 👍 Can stop reading once answer is found

### Technical Implementation

**Streaming Pattern:**
```python
# When user sends message
if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    # Create container for assistant message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Stream tokens one-by-one
        full_response = ""
        for token in stream_estimation_tokens(user_input):
            full_response += token
            message_placeholder.write(full_response)  # Update in real-time
        
        # Save complete response
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
```

**Specifications:**
- ✅ First token appears within 2 seconds (psychological threshold)
- ✅ Tokens stream at natural reading pace (100+ tokens/second)
- ✅ Update happens inside chat message (not replace)
- ✅ Looks smooth, not jerky
- ✅ Can be read in real-time

**Edge Cases:**
1. **Response too long (hits token limit):**
   - Show: "Response cut off at token limit..."
   - Action: Automatically request continuation
   - Display: "Continuing..." then stream rest

2. **Provider fails during streaming:**
   - Show: "Provider switched mid-stream, continuing..."
   - Action: Switch provider, resume from where left off
   - No data loss: User sees full response

3. **User cancels mid-stream:**
   - Show: Partial response with "Stopped by user"
   - Keep: What was generated so far
   - Allow: User to click "Continue" or ask new question

**Definition of Done:**
- ✅ Tokens appear 1-by-1 in real-time
- ✅ First token within 2 seconds
- ✅ Streaming at ≥100 tokens/second
- ✅ Text looks smooth, not flickering
- ✅ Handles incomplete responses gracefully
- ✅ Mobile browsers support streaming

---

## 📊 Feature 3: System Sidebar (Transparency Dashboard)

### Why Sidebar is Important
**Business:** Builds trust - users see what's happening behind the scenes  
**Technical:** Provides debugging info - what prompt is used, what examples are in context  
**Cost:** Shows real impact of AI usage - how many tokens, what it costs  

### Sidebar Layout

**Section 1: Active System Prompt**
```python
with st.sidebar.expander("⚙️ System Prompt", expanded=False):
    st.info("""
    Role: Software Estimation Expert
    
    Context includes 2 reference projects with similar scopes.
    System analyzes your transcription against these examples.
    """)
    
    # Show first 500 chars of prompt
    st.code(system_prompt[:500] + "...", language="markdown")
    st.caption("(Click to expand for full prompt)")
```

**Section 2: CAG Context Examples**
```python
with st.sidebar.expander("📚 Reference Examples in Context", expanded=False):
    for example in context_examples:
        st.subheader(example["project_name"])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hours", example["summary"]["total_hours"])
        with col2:
            st.metric("Cost", f"${example['summary']['total_cost_usd']:,}")
        with col3:
            st.metric("Team", example["summary"]["team_size"])
        st.divider()
```

**Section 3: Real-Time Session Metrics (Always Visible)**
```python
st.sidebar.title("📈 Session Metrics")

metrics_col1, metrics_col2 = st.sidebar.columns(2)
with metrics_col1:
    st.metric(
        "Calls Made",
        st.session_state.call_count,
        help="Number of estimation requests in this session"
    )
    st.metric(
        "Total Tokens",
        st.session_state.total_tokens,
        help="Input + output tokens used"
    )

with metrics_col2:
    st.metric(
        "Session Cost",
        f"${st.session_state.session_cost:.6f}",
        help="Total USD spent this session"
    )
    st.metric(
        "Cache Hits",
        st.session_state.cache_hits,
        help="Repeated requests (no API call)"
    )
```

**Section 4: Last API Call Details**
```python
with st.sidebar.expander("📋 Last Call Info", expanded=False):
    if st.session_state.last_call:
        last = st.session_state.last_call
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Model", last["model_name"])
            st.metric("Input Tokens", last["input_tokens"])
            st.metric("Duration", f"{last['response_time_ms']}ms")
        
        with col2:
            st.metric("Output Tokens", last["output_tokens"])
            st.metric("Cost", f"${last['cost_usd']:.6f}")
            st.metric("Provider", last["provider"])
        
        if last.get("cached"):
            st.success("✅ Result from cache (saved tokens!)")
        
        # Show finish reason
        if last["finish_reason"] == "length":
            st.warning("⚠️ Response cut off at token limit (auto-continued)")
```

**Specifications:**
- ✅ System prompt visible (read-only, for understanding)
- ✅ Context examples show what LLM is comparing against
- ✅ Session metrics update in real-time after each call
- ✅ Last call details show transparency
- ✅ Cost shown in USD for clear business impact
- ✅ Cache hits highlighted (user feels savings)

**Definition of Done:**
- ✅ Sidebar shows all 4 sections
- ✅ Metrics update in real-time
- ✅ System prompt readable
- ✅ Cost displays accurately
- ✅ Context examples visible and clear
- ✅ Mobile: Sidebar collapses (hamburger menu)

---

## 📐 UI Layout Specification

### Desktop Layout (1920x1080)
```
┌────────────────────────────────────────────────────┬──────────────────┐
│ CAG ESTIMATE CHAT                           [Menu] │  📊 System Info  │
├────────────────────────────────────────────────────┤  📈 Metrics      │
│                                                    │  📚 Context      │
│ User: "Estimate iOS app for..."                   │  📋 Last Call    │
│                                                    │                  │
│ Assistant: "Project: iOS Chat App               │  Real-time        │
│ Total Hours: 276..."  [streaming...]             │  Updates          │
│                                                    │                  │
│ User: "Can you break down the chat feature?"     │                  │
│                                                    │                  │
│ Assistant: "Chat Feature breakdown..." [streaming]│                  │
│                                                    │                  │
│ [Chat Input Field ...........................]     │                  │
│ [Send Button]                                     │                  │
└────────────────────────────────────────────────────┴──────────────────┘
```

### Mobile Layout (375x667)
```
┌──────────────────────────┐
│ ☰  CAG CHAT         ⚙️   │
├──────────────────────────┤
│ User: "Estimate iOS..."  │
│                          │
│ Assistant: "Project..."  │
│ [streaming text...]      │
│                          │
│ User: "Break down chat.."│
│                          │
│ Assistant: [streaming]   │
│                          │
├──────────────────────────┤
│ [Input field...]         │
│ [Send]                   │
└──────────────────────────┘

Sidebar (hamburger):
┌──────────────────────────┐
│ ✕ System Info            │
│ ⚙️ Prompt                  │
│ 📚 Context                │
│ 📈 Metrics                │
│   Calls: 2                │
│   Cost: $0.00012          │
└──────────────────────────┘
```

---

## 🛠️ Implementation Tasks

### Task 1: Chat Interface Setup
**Goal:** Build working chat UI with message history

**Deliverables:**
- ✅ `streamlit_app.py` created in project root
- ✅ Chat history displays correctly
- ✅ User can type and send messages
- ✅ Messages persist in session
- ✅ Basic styling applied

**Acceptance:**
```bash
$ streamlit run streamlit_app.py
# Opens UI, user can type messages and see them appear
```

---

### Task 2: Streaming Integration
**Goal:** Stream estimation responses token-by-token

**Deliverables:**
- ✅ Connect to `/api/v1/estimate` endpoint
- ✅ Stream tokens using `st.write()`
- ✅ First token within 2 seconds
- ✅ Handle token limit (continuation)
- ✅ Handle provider failures

**Acceptance:**
```bash
# Type in chat: "Estimate mobile app with chat feature"
# Response appears token-by-token in real-time
# Takes ~3-5 seconds to complete
```

---

### Task 3: Sidebar Dashboard
**Goal:** Show system transparency and real-time metrics

**Deliverables:**
- ✅ Sidebar expanders for each section
- ✅ System prompt shown (collapsed by default)
- ✅ Context examples visible
- ✅ Real-time metrics (calls, tokens, cost)
- ✅ Last call details logged

**Acceptance:**
- ✅ Send message → metrics update immediately
- ✅ Cost shown accurately (matches API response)
- ✅ Context examples match those in examples.py
- ✅ All info readable and clear

---

## ✅ Verification Tests

### Test 1: Basic Chat Flow
```
1. Open app: streamlit run streamlit_app.py
2. Type: "Estimate a web app"
3. Expected: Message appears in chat
4. Type: "include authentication"
5. Expected: New message in history, estimation appears
6. Result: ✅ PASS if both messages visible
```

### Test 2: Real-Time Streaming
```
1. Send: "Estimate iOS app with chat"
2. Observe: Text appears token-by-token
3. Measure: First token within 2 seconds
4. Result: ✅ PASS if streaming smooth and fast
```

### Test 3: Sidebar Metrics
```
1. Send estimation request
2. Check sidebar: Metrics updated?
3. Verify: Cost = (input_tokens/1M * 3) + (output_tokens/1M * 15)
4. Result: ✅ PASS if all metrics match API response
```

### Test 4: Multi-Turn Conversation
```
1. Send: "Estimate iOS app"
2. Send: "How long for testing?"
3. Send: "What's the team size?"
4. Expected: 3 messages visible in history
5. Result: ✅ PASS if conversation flows naturally
```

### Test 5: Error Handling
```
1. Send very long transcription
2. Expected: Streaming continues beyond token limit
3. Expected: Shows "continuing..." message
4. Result: ✅ PASS if response complete
```

---

## 🎯 Success Criteria

| Criteria | Target | How to Verify |
|----------|--------|--------------|
| Chat displays | 100% | Messages appear correctly |
| Streaming works | 100% | Tokens stream in real-time |
| First token time | <2 sec | Use browser dev tools timer |
| Sidebar updates | 100% | Check metrics after send |
| Cost accuracy | 100% | Verify math matches API |
| Mobile responsive | 100% | Test on iPhone/Android |

---

## 📝 File Structure

```
cag-estimate/
├── streamlit_app.py          # ← Main UI application
├── src/cag_estimate/
│   ├── config.py             # Settings (used by app)
│   ├── context/examples.py   # Context (displayed in sidebar)
│   └── main.py               # API (called from streamlit_app.py)
├── docker-compose.yml        # Run API: docker-compose up
└── DOCKER.md                 # Docker instructions
```

**To Run Locally:**
```bash
# Terminal 1: Start API
cd cag-estimate
docker-compose up

# Terminal 2: Start Streamlit UI
cd cag-estimate
streamlit run streamlit_app.py
```

---

## 🚀 Launch Checklist

- [ ] `streamlit_app.py` created and runs without errors
- [ ] Chat interface displays and accepts input
- [ ] Messages persist during session
- [ ] Streaming works (tokens appear in real-time)
- [ ] First token within 2 seconds
- [ ] Sidebar shows all 4 sections
- [ ] Metrics update in real-time
- [ ] Cost calculations accurate
- [ ] Mobile responsive design
- [ ] Error scenarios handled gracefully
- [ ] All 5 verification tests pass
- [ ] Code commented and clean
- [ ] Ready for code review

---

**Practice Focus:** UI, Streaming, Sidebar  
**Tech Lead Notes:** Clean, simple, focused on user experience  
**Business Value:** Demonstrates AI capability in familiar interface  
**Status:** Ready for Implementation

