# Session 5 notes: conversation memory

Summary of what was built against `practice5.md`, the deliberate decisions, and the known gaps. Details for users are in the "Conversation memory (sessions)" section of `../README.md`.

## Built, by practice step

| Step | Result |
|------|--------|
| 1. Session state | `services/sessions.py`: `ConversationHistory` (sliding window, system prompt never stored), `Session`, `SessionStore` (UUID4, LRU cap). `ProjectMetadata` lives in `schemas/session.py` so prompts do not import from services. Volatility is documented in the module docstring. |
| 2. Create sessions | `POST /api/v1/sessions` (201), plus `GET /api/v1/sessions/{id}` for a metadata snapshot. |
| 3. Attachments | `POST /api/v1/sessions/{id}/estimate`, multipart with `transcript` and `attachments`. Path B, separator `--- attachment: name ---`. |
| 4. Metadata in the prompt | Optional `<project_metadata>` block in the `v1` system template; empty metadata renders nothing. Heuristic extractor after each successful turn. |
| 5. Sliding window | `SESSION_MAX_TURNS` (default 6) pairs (evicted pairs feed anchors and an extractive summary, `services/session_compression.py`); `to_messages_list()` regenerates the system prompt from the current metadata. `LLMWrapper.complete_structured(history=...)` passes the window to the model. |
| 6. Client | Streamlit "Conversation (memory)" mode: lazy session, text area, multi-file uploader, `project_metadata` expander, "New conversation". |
| 7. Tests | `tests/test_session_integration.py` (three required tests) and `tests/test_sessions_endpoints.py`, all offline. |

## Audience tier and prompt v2

- **Problem (live evidence).** Follow-ups in a real session were estimated like separate small projects (182, 124, 126, 172 hours) instead of the updated full estimate.
- **Prompt v2 for sessions only.** `prompts/estimation/v2/system.j2` includes the `v1` system prompt and adds `<conversation_rules>` (history is context, metadata is the source of truth unless corrected, always return the updated FULL estimate, keep earlier tasks, stable `project_name`) and an `<audience>` block per tier. `v2/user.j2` adds a closing line asking for the full updated estimate. `v1` is untouched and pinned by a byte-level test; the non-session endpoints keep it. `SESSION_PROMPT_VERSION = "v2"` is a constant in `session_estimation.py` (nothing needs to switch it per deployment); the session store's system-prompt provider uses the same version.
- **Tier resolver.** `services/tier_resolver.py`: an ordered, data-driven rule chain (`override`, `executive_cues`, `pm_cues`, `developer_cues`, `technology_count`, then `default`); it returns the tier and the name of the deciding rule. It is a heuristic over vocabulary, pure and offline.
- **API.** Optional form field `tier` (invalid value: `422`); snapshot adds `last_tier` and `last_tier_rule`; the estimate response adds `resolved_tier` and `tier_rule`. State is set only after a successful turn.
- **UI.** "Audience" selector in conversation mode (Auto sends nothing); the memory panel shows the resolved tier.
- **Limit.** The prompt asks for the full estimate but the model can still drift; the live check is the evidence, not the unit tests (they use a fake LLM).

## Deliberate decisions

- **Path B for attachments.** Provider-agnostic with the Anthropic to OpenAI fallback, testable offline, and the guardrails see the document text.
- **Heuristic metadata extraction, not a second LLM call.** No extra cost, latency or failure mode per turn; deterministic tests.
- **Session estimates bypass the caches.** The answer depends on history and metadata, not only on the text.
- **`v1` prompt kept.** The metadata block is optional, so non-session prompts are byte-identical to before.
- **Compact user turn in history.** Only a truncated transcript (4,000 characters) and the attachment names are stored; the full text is sent once.
- **Failed turns leave the session untouched**, and requests on one session are serialized by a lock.
- **Structured output limit raised** (`LLM_STRUCTURED_MAX_TOKENS`, default 8192). A live test showed a larger follow-up estimate being truncated at 4000 tokens and failing with 502; this also affected the non-session endpoint.
- **Compose passes the new variables** (`SESSION_*`, `ATTACHMENT_*`) to the `api` service.

## Known gaps

- Sessions are not persisted: they are lost on restart and not shared across workers; the store is capped and evicts the least recently used session. Sessions can be deleted (`DELETE /api/v1/sessions/{id}`) and optionally expire after `SESSION_IDLE_TTL_SECONDS` of inactivity (default `0`, disabled); expiry is lazy and per process.
- The extractor only recognises technologies in its vocabulary (about 35), excludes ambiguous names such as `Go` and `R`, and `agreed_scope` is a truncated summary.
- History compression: anchors are regex heuristics and the summary is written by one short LLM call (`SESSION_SUMMARY_MODE=llm`, optional cheap `SESSION_SUMMARY_MODEL`; uncached, runs inside the session lock) and can lose detail; it falls back to the extractive summary if the call fails.
- Reviewed estimates (actor-critic) are available at `POST /api/v1/sessions/{id}/estimate-acb` and behind a Streamlit checkbox (critic, boss, `acb` schemas, `critic/v1` and `<critic_feedback>` prompts; offline tests). The session is changed once, from the clean final estimate, with the same history format as `/estimate`; caveats live only in `acb.caveats`. `ACB_MAX_ITERATIONS` (default 2, worst case 2 drafts + 2 reviews per request) and `CRITIC_MODEL` are the settings. Limits: the critic is an LLM and can be wrong or miss defects (a failed critic call accepts the draft and flags `critic_failed`), a reviewed request is slower and costs more calls, and the Rails UI does not use this endpoint yet.
- The Rails UI (`estimator-web/`) now has a conversation page over these endpoints (session mirror, multipart attachments, memory panel, audience selector, "New conversation"). It does not stream, and it recovers from a lost session (API restart) by starting a new conversation. Covered by WebMock tests; the paid estimate call was not exercised by the automated tests.
- No streaming in sessions: only the structured endpoint exists for them; the streaming mode of the UI remains single-shot.
- Scanned PDFs (no text layer) are rejected; there is no OCR.
- Word attachments: only `.docx` is read (paragraphs, then table rows); legacy `.doc` is rejected as unsupported (`415`).
- The session integration tests use a fake LLM, so they prove the plumbing (memory, attachment text and window reach the model), not the quality of a real model's answers. A manual live run against the real model was done earlier in the session.
- The Streamlit conversation mode was not exercised in a browser by the automated tests, and the multi-file uploader has no Streamlit `AppTest` coverage (the multipart request is tested at the client level).
- Three screenshots of the conversation with the metadata panel are in `session5/evidence/`. The optional GIF has not been produced.
