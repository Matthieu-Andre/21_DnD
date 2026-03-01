# Lyria Integration Plan

## Goal

Integrate [lyria_eleven_child.py](/Users/matthieu/Documents/projects/hackatons/mistralai2026/21_DnD/lyria_eleven_child.py) into the existing FastAPI backend so the app can:

1. listen to the table conversation through the current browser-to-backend audio flow
2. derive a music steering prompt from the transcript and scene mood
3. generate live music with Lyria
4. stream that music back to the browser for playback

## What The Current Script Already Does

The script is already a good prototype for the music runtime. Its useful pieces are:

- `eleven_to_palette(prompt)`: converts a text prompt into a weighted prompt palette using ElevenLabs metadata.
- `raw_to_palette(prompt)`: sends the prompt directly to Lyria.
- `dev_to_palette(prompt)`: maps prompts into a fixed scene palette with Mistral.
- `palette_to_weighted_prompts(...)` and `interpolate_palettes(...)`: convert and crossfade prompt palettes.
- `live_dj(...)`: owns a long-lived Lyria realtime session, applies steering changes, and receives audio chunks continuously.
- `receive_audio(...)`: pulls audio chunks from Lyria.
- `steering_handler(...)`: updates the running music session when a new prompt arrives.

The parts that should not be carried into the backend as-is are:

- `sounddevice` playback
- terminal input via `input_reader(...)`
- the script-local microphone STT path in `voice_listener(...)`

Those are CLI concerns. The web app already has a browser mic path and a FastAPI session manager.

## Current App Constraints

The current web app already has:

- browser mic capture and chunk upload in [web/app.js](/Users/matthieu/Documents/projects/hackatons/mistralai2026/21_DnD/web/app.js)
- transcript + mood state in [api_server.py](/Users/matthieu/Documents/projects/hackatons/mistralai2026/21_DnD/api_server.py)
- one active session managed by `SessionManager`
- polling-based UI refresh

What it does not have yet:

- a backend music runtime
- a browser audio playback stream from the backend
- a prompt-planning layer that turns transcript and mood into music steering updates

## Target Architecture

```text
Browser mic
  -> FastAPI chunk upload
  -> transcript update
  -> mood update
  -> music prompt planner
  -> palette translator (raw / ElevenLabs / developer)
  -> persistent Lyria realtime session
  -> backend audio stream
  -> browser audio player
```

Core design choice:

- keep STT and music generation in the same session object
- keep Lyria as a long-lived backend connection
- stream generated audio to the browser over WebSocket, not polling

WebSocket is the right fit because the app needs low-latency, continuous binary audio. Polling or base64 JSON would add avoidable delay and overhead.

## Backend Plan

### 1. Extract The Reusable Lyria Runtime

Create a new module, for example:

- [lyria_service.py](/Users/matthieu/Documents/projects/hackatons/mistralai2026/21_DnD/lyria_service.py)

Move or adapt these responsibilities from [lyria_eleven_child.py](/Users/matthieu/Documents/projects/hackatons/mistralai2026/21_DnD/lyria_eleven_child.py):

- Lyria client setup
- prompt palette translation
- prompt interpolation / crossfade
- live Lyria session lifecycle
- inbound audio chunk handling

Do not move:

- `sounddevice` playback
- CLI printing
- terminal input thread
- script-owned realtime mic listener

### 2. Add A Music Runtime Object

Introduce a backend class such as `LiveMusicSession` with responsibility for:

- opening and owning the Lyria websocket
- storing the current palette
- receiving audio chunks from Lyria
- fanning those chunks out to connected frontend listeners
- applying prompt updates with crossfade
- exposing status for `/api/state`

Suggested internal shape:

- `start(initial_prompt, config)`
- `steer(prompt_text, source, metadata)`
- `attach_listener(listener_id)`
- `detach_listener(listener_id)`
- `close()`

### 3. Add A Music Prompt Planner

Do not feed every transcript chunk directly into Lyria. Add a planner layer first.

Suggested new module:

- [music_prompt_agent.py](/Users/matthieu/Documents/projects/hackatons/mistralai2026/21_DnD/music_prompt_agent.py)

Its job:

- read recent utterances
- read current mood
- produce a short music steering description
- decide whether the scene changed enough to justify a new prompt

Example output:

```json
{
  "prompt": "tense underground exploration with distant danger and low percussion",
  "reason": "The party is sneaking through a cave and expecting combat.",
  "should_update": true
}
```

This should be debounced. A good starting rule is:

- update when mood changes
- or when at least 2 new utterances were added and 6 to 10 seconds passed since the last steer

### 4. Extend The Existing Session Manager

Update `SessionManager` in [api_server.py](/Users/matthieu/Documents/projects/hackatons/mistralai2026/21_DnD/api_server.py) so a browser live session can optionally own a music session too.

Suggested additions to `StartSessionRequest`:

- `music_enabled: bool = true`
- `music_mode: "raw" | "eleven" | "developer"`
- `music_temperature`
- `music_guidance`
- `music_crossfade_seconds`
- `music_seed`

Suggested session changes:

- create `BrowserConversationSession.music_session`
- trigger the prompt planner after transcript registration and after mood classification
- include music status in `/api/state`

### 5. Add A Browser Audio Stream Endpoint

Add a websocket endpoint such as:

- `/api/session/{session_id}/music/stream`

Protocol:

- first message from client: JSON `{"type":"subscribe_music"}`
- server sends an initial JSON config packet with sample rate, channels, and codec
- server then sends binary PCM frames as they arrive
- server may also send JSON status events like `music_started`, `music_error`, `palette_updated`

Start with raw PCM `int16`, stereo, 48 kHz, because that matches the current Lyria runtime assumptions in [lyria_eleven_child.py](/Users/matthieu/Documents/projects/hackatons/mistralai2026/21_DnD/lyria_eleven_child.py).

### 6. Add Persistence Only For Metadata

Do not persist raw generated music chunks to disk in the first version.

Persist only:

- last applied music prompt
- last palette
- music mode
- music connection status
- recent steering events

That metadata can be included in `conversation_state.json` and the session artifact folder.

## Frontend Plan

### 1. Add A Music Playback Path

The frontend in [web/app.js](/Users/matthieu/Documents/projects/hackatons/mistralai2026/21_DnD/web/app.js) currently captures microphone audio but does not play backend-generated audio.

Add:

- a browser `AudioContext`
- a `WebSocket` connection to the backend music stream
- a PCM playback queue

Implementation recommendation:

- preferred: `AudioWorkletNode`
- acceptable first pass: `ScriptProcessorNode` if speed matters more than polish

The player should:

- start only after a user gesture
- accept 48 kHz stereo PCM chunks
- buffer a small amount of audio before playback
- handle underruns with silence rather than loud glitches

### 2. Add Music Controls To The UI

Extend [web/index.html](/Users/matthieu/Documents/projects/hackatons/mistralai2026/21_DnD/web/index.html) with:

- music on/off toggle
- music mode selector: `raw`, `eleven`, `developer`
- volume slider
- music status pill

Extend the dashboard with:

- current music prompt
- current palette summary
- last music update reason

### 3. Couple Music Lifecycle To Session Lifecycle

When the user presses `Start`:

- start the conversation session
- initialize the browser audio player
- connect the music websocket

When the user presses `Stop`:

- stop audio chunk upload
- close the music websocket
- suspend or close the `AudioContext`

### 4. Keep The Visualizer Independent

The circle visualizer should continue to reflect microphone activity, not generated music output.

That separation is useful:

- the ring shows table energy
- the player streams scene music

If needed later, we can add a second subtle music-reactive layer, but it should not replace the mic-driven ring.

## Recommended Implementation Order

### Phase 1. Refactor Lyria Into A Service

- extract reusable logic from [lyria_eleven_child.py](/Users/matthieu/Documents/projects/hackatons/mistralai2026/21_DnD/lyria_eleven_child.py)
- remove CLI-only concerns
- prove the backend can start a Lyria session and receive audio chunks

Success condition:

- a local test endpoint can create a music session and log palette changes

### Phase 2. Stream Music To The Browser

- add the websocket endpoint
- add frontend PCM playback
- manually steer the backend with a fixed prompt first

Success condition:

- the browser plays Lyria audio from the FastAPI backend

### Phase 3. Connect Transcript To Music Steering

- add the music prompt planner
- trigger updates from transcript and mood events
- add debouncing and crossfade

Success condition:

- live conversation changes the music without requiring manual prompt input

### Phase 4. Harden The Runtime

- add reconnect behavior
- surface errors in `/api/state`
- guard against multiple music listeners
- make music optional when `GOOGLE_API_KEY` is missing

Success condition:

- the app degrades cleanly instead of crashing

## Risks And Decisions

### 1. Railway Runtime Risk

This design depends on a long-lived Lyria connection and a long-lived browser audio websocket. That is operationally heavier than the current request/response chunk transcription path.

We should assume:

- one backend instance
- one active live session at a time in the first version

### 2. Browser Audio Constraints

Browser autoplay restrictions mean music playback must begin after a user interaction. The `Start` button is the right trigger for creating or resuming the `AudioContext`.

### 3. Prompt Stability

If prompts update too often, the soundtrack will thrash. The planner must debounce and prefer scene-level changes over literal transcript mirroring.

### 4. ElevenLabs Optionality

`eleven_to_palette(...)` is valuable, but it introduces another paid dependency and another failure point.

Recommendation:

- implement `raw` mode first
- add `eleven` mode second
- keep `developer` mode as a fallback/debug option

## Concrete First Slice

The lowest-risk integration slice is:

1. extract a `LiveMusicSession` from [lyria_eleven_child.py](/Users/matthieu/Documents/projects/hackatons/mistralai2026/21_DnD/lyria_eleven_child.py)
2. add `/api/session/{session_id}/music/stream`
3. add browser PCM playback
4. use a fixed prompt such as the current mood name for steering
5. replace that with a real prompt planner afterward

That gets end-to-end music working before we spend time on better prompt intelligence.
