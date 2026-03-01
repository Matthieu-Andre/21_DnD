# Strands Integration Plan for Live STT Mood Tracking

## Objective

Build a live speech-to-text pipeline that keeps conversation state over time and assigns a scene mood that can be consumed by the music-selection component.

The design should stay responsive in the live audio path, keep state explicit and inspectable, and avoid unnecessary distributed complexity in the first version.

## Validated Architecture Decision

Use `Strands Agents` for mood classification and internal orchestration, but do not use `A2A` in the first version.

Use `MCP` only if another process or teammate needs a standard interface to read the current mood and conversation state. If the immediate integration is local and simple, a JSON state file is sufficient for v1.

### Why

- The live microphone and STT loop is latency-sensitive and should remain deterministic.
- The LLM should classify mood from finalized utterances, not control the audio stream.
- `A2A` is useful when a specialized agent lives in another service or process. That is not required yet.
- `MCP` is a good boundary if the music component needs a standard way to query state.

## Recommended v1 Design

### 1. Keep `stt.py` as the real-time owner

`stt.py` should remain responsible for:

- microphone capture
- streaming transcription
- buffering partial transcription text
- finalizing utterances
- persisting session state

The live STT path should not depend on an external agent service.

### 2. Introduce a session state model

Add a single in-memory and serialized session state structure with fields like:

- `allowed_moods`
- `current_mood`
- `partial_text`
- `utterances`
- `mood_history`
- `version`
- `started_at`
- `updated_at`

This state should be written to a JSON file so the rest of the system can inspect it easily.

### 3. Finalize utterances before mood classification

Streaming STT returns text deltas, not clean sentence boundaries. Add a segmentation layer:

- append deltas to `partial_text`
- finalize an utterance on punctuation when possible
- also flush after a short silence timeout
- keep finalized utterances as the authoritative transcript units

The mood classifier should only see finalized utterances.

### 4. Add a local specialized Strands agent for mood classification

Create a focused `MoodClassifierAgent` with:

- one clear system prompt
- one strict output format
- no authority over the audio loop

Inputs:

- current mood
- recent finalized utterances
- allowed mood list

Output:

```json
{
  "mood": "action",
  "confidence": 0.84,
  "reason": "Short explanation"
}
```

The Python controller remains the source of truth and applies the agent output to session state.

### 5. Start with a small fixed mood taxonomy

Suggested initial labels:

- `default`
- `tense`
- `action`
- `boss`
- `mystery`
- `victory`

This is a better v1 taxonomy than very broad emotional labels because it maps more directly to tabletop scene music.

### 6. Keep the agent boundary simple

The best first Strands pattern here is effectively a specialized local agent, optionally wrapped as a tool if we want clearer modularity.

Do not start with a multi-agent graph, swarm, or remote `A2A` topology unless there is a clear second autonomous service.

## Optional Strands Tool Layer

If we want a cleaner architecture inside the app, define a small set of local Strands tools around the state:

- `get_recent_context(limit)`
- `get_current_mood()`
- `persist_state_snapshot()`
- `list_allowed_moods()`

This is useful if the mood classifier agent or a future orchestrator needs structured access to state.

## MCP Plan

Use `MCP` if the music component should read state through a stable protocol rather than directly reading a file.

### MCP server responsibilities

Expose a minimal server with tools such as:

- `get_current_mood()`
- `get_session_state()`
- `get_recent_utterances(limit)`
- `get_state_since(version)`

### When MCP is worth it

Use it if:

- another teammate is building a separate consumer
- you want transport-agnostic integration
- you expect multiple clients to consume the same state

Skip it in v1 if:

- the music selector runs in the same environment
- a shared JSON file is enough

## A2A Plan

Do not use `A2A` in v1.

Add `A2A` only if one of these becomes true:

- the mood classifier moves to a separate service
- a remote orchestrator needs to call your mood agent
- multiple independent agent services need to collaborate over the network

At that point, the mood agent can be wrapped behind an A2A server and consumed by another Strands orchestrator or external system.

## Proposed Delivery Sequence

### Phase 1

- stabilize session state in `stt.py`
- finalize utterance segmentation
- persist state to JSON

### Phase 2

- integrate a local Strands mood classifier
- classify mood from the recent utterance window
- append decisions to `mood_history`

### Phase 3

- add a minimal MCP server if the music component needs it

### Phase 4

- consider `A2A` only if the architecture becomes distributed

## Non-Goals for v1

- remote agent orchestration
- complex multi-agent graph routing
- letting the LLM control audio capture or the transcription loop
- overfitting the mood taxonomy before real usage data exists

## Summary

The right first version is:

- local live STT in `stt.py`
- explicit session state
- local Strands mood classification
- JSON output by default
- optional MCP boundary for downstream consumers
- no A2A yet

This gives a clean and extensible architecture without adding network and orchestration complexity too early.
