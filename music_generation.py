"""
music_generation.py  —  Live AI DJ with Progressive Prompt Steering

Powered by Google Lyria RealTime + Mistral AI steering intelligence.

The DJ maintains a "prompt palette" — multiple weighted prompts mixed
simultaneously by Lyria. Each user input is interpreted by Mistral (fast,
low-latency model) which understands the full steering history and adjusts the
palette weights gradually rather than hard-resetting them.

Usage:
    python music_generation.py "techno rave"
    python music_generation.py "chill lo-fi beats" --temperature 0.9 --guidance 4.0 --crossfade 4.0

While running:
    • Type a steering instruction and press Enter (e.g. "more energy", "add some jazz", "slow it down")
    • Press Ctrl+C to stop

Architecture:
    One persistent WebSocket to Lyria RealTime.
    Mistral flash model interprets each instruction relative to full history.
    Crossfade loop sends interpolated weights over N steps for smooth morphing.
    Audio streamed via sounddevice (PortAudio).
"""

import argparse
import asyncio
import collections
import json
import os
import sys
import threading
import time
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mistralai import Mistral

try:
    import sounddevice as sd
except ImportError:
    print("❌  sounddevice is required:  pip install sounddevice")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config / env
# ---------------------------------------------------------------------------
load_dotenv()

# Enable ANSI escape codes on Windows (Windows 10 v1511+ supports VT100)
if sys.platform == "win32":
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# ---------------------------------------------------------------------------
# Terminal UI helpers — ANSI colours + line-safe print
# ---------------------------------------------------------------------------
class C:
    """ANSI colour codes (Windows 10+ / any modern terminal)."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    # Functional colours
    STATUS  = "\033[90m"          # dark grey  → status ticker
    VOICE   = "\033[96m"          # cyan        → voice / STT
    THINK   = "\033[93m"          # yellow      → Mistral thinking
    APPLIED = "\033[92m"          # green       → change applied!
    WARN    = "\033[91m"          # red         → errors / warnings
    INFO    = "\033[94m"          # blue        → info / startup
    DIM     = "\033[2m"           # dim         → secondary details

# Clears the \r status line then prints a full line — prevents overlap.
def ui_print(color: str, prefix: str, msg: str, **kw) -> None:
    print(f"\r{' ' * 100}\r{color}{prefix}{C.RESET} {msg}", **kw)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError("MISTRAL_API_KEY not found in .env")

MODEL = "models/lyria-realtime-exp"

# Fastest Mistral text model — minimises steering latency
MISTRAL_MODEL = "mistral-small-latest"

SAMPLE_RATE = 48_000
CHANNELS = 2
SAMPLE_WIDTH = 2
BYTES_PER_FRAME = CHANNELS * SAMPLE_WIDTH
BLOCK_SIZE = 2400

PRE_BUFFER_SECONDS = 0.5
PRE_BUFFER_BYTES = int(PRE_BUFFER_SECONDS * SAMPLE_RATE * BYTES_PER_FRAME)
BUFFER_CAPACITY = SAMPLE_RATE * BYTES_PER_FRAME * 10

lyria_client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options={"api_version": "v1alpha"},
)

mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# ---------------------------------------------------------------------------
# Lyria Reference Table
# Injected into the Mistral system prompt so the model knows exactly what
# knobs are available and what values make sense.
# ---------------------------------------------------------------------------
LYRIA_REFERENCE_TABLE = """
## Lyria RealTime — Controllable Parameters Reference

### Prompt palette (your primary tool)
Each prompt entry has:
  - text  : a short musical descriptor (genre, instrument, mood, tempo, etc.)
  - weight: float, typically 0.0 – 2.0
    • 0.0  = silent / removed
    • 0.5  = subtle influence
    • 1.0  = normal influence (reference)
    • 1.5  = strong emphasis
    • 2.0  = dominant influence

Effective text descriptors (non-exhaustive):
  Genres   : techno, house, trance, drum and bass, lo-fi hip-hop, jazz, blues,
              classical, ambient, pop, rock, metal, afrobeat, bossa nova,
              reggaeton, folk, cinematic, orchestral
  Tempo    : slow tempo, moderate tempo, fast tempo, very fast tempo, 160 BPM,
              130 BPM, half-time feel, double-time feel
  Energy   : low energy, medium energy, high energy, building tension,
              euphoric drop, breakdown, outro
  Mood     : melancholic, uplifting, dark, ethereal, aggressive, relaxing,
              mysterious, euphoric, nostalgic
  Texture  : sparse, dense, minimalist, layered, driving bassline, heavy kick,
              rolling bass, synth pads, arpeggiated synths
  Instruments: piano, guitar, violin, trumpet, synthesizer, drum machine,
               808 bass, acoustic drums, strings, brass section

### Generation config (set once at startup, not changed per-step)
  temperature : 0.5 – 2.0   → lower = more predictable, higher = more creative
  guidance    : 1.0 – 6.0   → how strictly the model follows prompts

### Steering rules
  1. NEVER set all weights to 0 simultaneously — music would cut out.
  2. Preserve the dominant existing prompt(s) unless the user explicitly asks
     to switch genre entirely.
  3. "More X" → increase weight of X-related prompts or add a new X entry.
  4. "Less X" → reduce weight of X-related entries (minimum 0.1 if it was active).
  5. "Remove X" → set weight to 0.0.
  6. "Only X" → set all other weights to 0.0, boost X to 1.5.
  7. Smooth changes: prefer delta of ±0.3–0.5 per instruction rather than jumps.
  8. Keep palette lean: 2–4 active prompts usually sounds best.
     Drop entries with weight < 0.05.
"""

# ---------------------------------------------------------------------------
# Thread-safe audio ring buffer
# ---------------------------------------------------------------------------
class AudioBuffer:
    def __init__(self, capacity: int = BUFFER_CAPACITY):
        self._chunks: collections.deque[bytes] = collections.deque()
        self._total_bytes = 0
        self._capacity = capacity
        self._lock = threading.Lock()
        self._total_received = 0

    def push(self, data: bytes) -> None:
        with self._lock:
            self._chunks.append(data)
            self._total_bytes += len(data)
            self._total_received += len(data)
            while self._total_bytes > self._capacity and self._chunks:
                old = self._chunks.popleft()
                self._total_bytes -= len(old)

    def pull(self, n: int) -> bytes:
        with self._lock:
            if self._total_bytes == 0:
                return b"\x00" * n
            result = bytearray()
            while len(result) < n and self._chunks:
                chunk = self._chunks[0]
                needed = n - len(result)
                if len(chunk) <= needed:
                    result.extend(chunk)
                    self._total_bytes -= len(chunk)
                    self._chunks.popleft()
                else:
                    result.extend(chunk[:needed])
                    self._chunks[0] = chunk[needed:]
                    self._total_bytes -= needed
            if len(result) < n:
                result.extend(b"\x00" * (n - len(result)))
            return bytes(result)

    @property
    def available(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def received_seconds(self) -> float:
        with self._lock:
            return self._total_received / (SAMPLE_RATE * BYTES_PER_FRAME)


# ---------------------------------------------------------------------------
# Terminal input reader (runs in a background thread)
# ---------------------------------------------------------------------------
def input_reader(prompt_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    try:
        while True:
            line = input()
            text = line.strip()
            if text:
                loop.call_soon_threadsafe(prompt_queue.put_nowait, text)
    except (EOFError, KeyboardInterrupt):
        pass


# ---------------------------------------------------------------------------
# Mistral Steering Engine
# ---------------------------------------------------------------------------
class SteeringEngine:
    """
    Wraps the Mistral API to interpret user instructions and update the
    prompt palette. Maintains a full conversation history so the model
    has context on how the session has evolved.
    """

    # Prompt used ONCE at startup to convert any scene/narrative description
    # into concrete Lyria-compatible musical descriptors.
    DNA_PROMPT = """You are a music expert. Translate any scene, image, mood, or narrative description
into a JSON array of musical descriptors that a music generation AI can use.

Rules:
- Use only concrete musical terms: genre, instruments, tempo, mood, texture.
- No narrative language (no 'knights', 'forest', 'battle' — translate those into music).
- Return 2-5 entries. Weights should sum to roughly 3.0.
- RAW JSON ONLY — no markdown, no explanation.

Examples:
Input: "a group of knights hiking in a peaceful field"
Output: [{"text": "medieval folk", "weight": 1.0}, {"text": "acoustic lute and flute", "weight": 0.8}, {"text": "moderate peaceful tempo", "weight": 0.7}, {"text": "pastoral mood", "weight": 0.5}]

Input: "spaceship launching into space"
Output: [{"text": "cinematic orchestral", "weight": 1.0}, {"text": "epic brass section", "weight": 0.8}, {"text": "fast building tempo", "weight": 0.7}, {"text": "dramatic tension", "weight": 0.5}]

Input: "techno rave"
Output: [{"text": "techno", "weight": 1.2}, {"text": "driving bassline", "weight": 0.9}, {"text": "fast tempo 140 BPM", "weight": 0.7}, {"text": "repetitive synth loop", "weight": 0.5}]
"""

    SYSTEM_PROMPT = f"""You are an expert AI DJ assistant controlling a real-time music generation system (Google Lyria RealTime).

{LYRIA_REFERENCE_TABLE}

## Your ONLY job
Given the CURRENT PALETTE, the DNA ANCHORS, and a USER INSTRUCTION,
return a JSON array of {{\"text\": str, \"weight\": float}} representing the ADJUSTED palette.
Respond with RAW JSON ONLY — no markdown, no explanation.

## THE GOLDEN RULE: DNA Anchors Must Survive
DNA ANCHORS are the core musical identity of this session. They are listed in every message.
They MUST appear in your output at their anchor weight (±0.3 is fine for evolution).
The ONLY exception: user EXPLICITLY says to stop/remove/change genre (Class C instruction).

## How to classify an instruction

### Class A — Narrative / scene evolution (most common)
The user describes what is HAPPENING. Stay in the same musical world, evolve energy/tempo.
→ Keep all DNA anchors at their weights. Add/adjust secondary prompts only.
→ Examples: "knight gets on a horse", "tension rises", "more energy", "they celebrate"

### Class B — Direct musical tweak
The user explicitly names a musical element to change.
→ Adjust that element; keep all DNA anchors.
→ Examples: "more bass", "add piano", "slower tempo", "louder drums"

### Class C — Explicit genre/style change
The user EXPLICITLY asks to switch genre or start fresh.
→ Reduce DNA anchor weights to 0.2 and boost new style.
→ Examples: "switch to jazz", "go full metal", "completely different vibe"

## Worked examples

EXAMPLE 1 — Class A (narrative)
DNA Anchors: [{{"text": "medieval folk", "weight": 1.0}}, {{"text": "acoustic lute", "weight": 0.8}}]
Current palette: [{{"text": "medieval folk", "weight": 1.0}}, {{"text": "acoustic lute", "weight": 0.8}}, {{"text": "pastoral mood", "weight": 0.5}}]
User: "the knight gets on his horse and starts galloping"
BAD: [{{"text": "galloping hooves", "weight": 1.0}}, {{"text": "battle drums", "weight": 0.8}}]  ← WRONG: wiped DNA
GOOD: [{{"text": "medieval folk", "weight": 1.0}}, {{"text": "acoustic lute", "weight": 0.7}}, {{"text": "faster tempo", "weight": 0.8}}, {{"text": "epic momentum", "weight": 0.6}}]

EXAMPLE 2 — Class B
DNA Anchors: [{{"text": "techno", "weight": 1.2}}, {{"text": "driving bassline", "weight": 0.9}}]
User: "more energy"
GOOD: [{{"text": "techno", "weight": 1.3}}, {{"text": "driving bassline", "weight": 1.0}}, {{"text": "high energy", "weight": 0.6}}]

EXAMPLE 3 — Class C
DNA Anchors: [{{"text": "techno", "weight": 1.2}}]
User: "switch to medieval folk now"
GOOD: [{{"text": "techno", "weight": 0.2}}, {{"text": "medieval folk", "weight": 1.2}}, {{"text": "acoustic instruments", "weight": 0.8}}]
"""

    def __init__(self, initial_palette: dict[str, float]):
        self.history: list[dict] = []
        self.current_palette: dict[str, float] = dict(initial_palette)
        # DNA anchors = core musical identity, never auto-dropped.
        # These are set once from the translated initial prompt and persist
        # across ALL rounds unless the user explicitly asks for a genre change.
        self._dna_anchors: dict[str, float] = dict(initial_palette)

    @classmethod
    def translate_scene_to_palette(
        cls, scene_description: str
    ) -> dict[str, float]:
        """
        Use Mistral to convert a raw user description (scene, mood, narrative)
        into concrete musical descriptors that Lyria understands.
        Called once at startup — synchronous.
        """
        print(f"    {C.INFO}🧬 Translating scene to musical DNA via Mistral…{C.RESET}", end=" ", flush=True)
        try:
            resp = mistral_client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": cls.DNA_PROMPT},
                    {"role": "user", "content": scene_description},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            entries = json.loads(raw.strip())
            palette = {
                e["text"]: float(e["weight"])
                for e in entries
                if e.get("text") and float(e.get("weight", 0)) > 0
            }
            if palette:
                print(f"{C.APPLIED}done{C.RESET}")
                return palette
        except Exception as exc:
            print(f"{C.WARN}failed ({exc}){C.RESET}")
        # Fallback: use raw description as single prompt
        return {scene_description: 1.0}

    def _palette_to_str(self) -> str:
        items = [{"text": t, "weight": w} for t, w in self.current_palette.items()]
        return json.dumps(items)

    def _dna_to_str(self) -> str:
        items = [{"text": t, "weight": w} for t, w in self._dna_anchors.items()]
        return json.dumps(items)

    def _enforce_dna(self, new_palette: dict[str, float]) -> dict[str, float]:
        """Always re-inject DNA anchors if Mistral silently dropped them."""
        result = dict(new_palette)
        for text, weight in self._dna_anchors.items():
            if text not in result:
                # Restore at 60 % — strong enough to be heard, won't dominate new direction
                result[text] = round(weight * 0.6, 3)
        return result

    async def steer(self, user_instruction: str) -> Optional[dict[str, float]]:
        """
        Ask Mistral to interpret the user instruction and return an updated palette.
        Returns None on failure (caller should skip the update gracefully).
        """
        user_content = (
            f"DNA Anchors (must survive): {self._dna_to_str()}\n"
            f"Current palette: {self._palette_to_str()}\n"
            f"User instruction: {user_instruction}"
        )

        # Append to conversation history (gives Mistral memory of the session)
        self.history.append({"role": "user", "content": user_content})

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + self.history

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: mistral_client.chat.complete(
                    model=MISTRAL_MODEL,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.2,  # Low temperature → consistent, fast, structured JSON
                ),
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if the model wraps its output
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            entries = json.loads(raw)
            new_palette = {
                e["text"]: float(e["weight"])
                for e in entries
                if e.get("text") and float(e.get("weight", 0)) > 0
            }

            if not new_palette:
                raise ValueError("Empty palette returned")

            # Always re-inject DNA anchors if Mistral silently dropped them
            new_palette = self._enforce_dna(new_palette)

            # Store the assistant response in history for context continuity
            self.history.append({"role": "assistant", "content": raw})

            return new_palette

        except Exception as exc:
            # Rollback the failed user message from history
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            ui_print(C.WARN, "⚠️ ", f"Mistral steering failed ({exc}), applying prompt directly.")
            return None


# ---------------------------------------------------------------------------
# Crossfade helper
# ---------------------------------------------------------------------------
def interpolate_palettes(
    src: dict[str, float],
    dst: dict[str, float],
    t: float,  # 0.0 → 1.0
) -> dict[str, float]:
    """Linear interpolation between two palettes at position t."""
    all_keys = set(src) | set(dst)
    result = {}
    for k in all_keys:
        w = src.get(k, 0.0) * (1 - t) + dst.get(k, 0.0) * t
        if w > 0.01:
            result[k] = round(w, 3)
    return result


def palette_to_weighted_prompts(palette: dict[str, float]) -> list[types.WeightedPrompt]:
    return [types.WeightedPrompt(text=t, weight=w) for t, w in palette.items()]


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------
def palette_display(palette: dict[str, float]) -> str:
    parts = [f'"{t}":{w:.2f}' for t, w in sorted(palette.items(), key=lambda x: -x[1])]
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Main async logic
# ---------------------------------------------------------------------------
async def live_dj(initial_prompt: str, args: argparse.Namespace) -> None:
    audio_buf = AudioBuffer()
    prompt_queue: asyncio.Queue[str] = asyncio.Queue()
    playback_started = threading.Event()
    stop_event = asyncio.Event()

    # Step 1: translate scene/narrative into musical DNA using Mistral
    initial_palette = SteeringEngine.translate_scene_to_palette(initial_prompt)

    # In ambient mode, blend in sparse/atmospheric anchors so Lyria
    # generates quiet background texture rather than a full structured song.
    if args.ambient:
        ambient_anchors = {
            "sparse ambient texture": 0.8,
            "slow atmospheric pads": 0.7,
            "minimal sparse arrangement": 0.6,
        }
        # Reduce scene palette weights slightly then merge ambient anchors
        initial_palette = {t: w * 0.75 for t, w in initial_palette.items()}
        initial_palette.update(ambient_anchors)
        # Ambient defaults: lower guidance if not overridden by user
        if args.guidance == 3.0:   # still at default — user didn’t touch it
            args.guidance = 2.0

    print(f"    {C.INFO}🎼 Musical DNA:{C.RESET} {palette_display(initial_palette)}")

    engine = SteeringEngine(initial_palette)
    # Shared palette (read by status display, written by steering handler)
    palette_lock = threading.Lock()
    current_palette: dict[str, float] = dict(initial_palette)

    def get_palette() -> dict[str, float]:
        with palette_lock:
            return dict(current_palette)

    def set_palette(p: dict[str, float]):
        nonlocal current_palette
        with palette_lock:
            current_palette = dict(p)

    # --- Sounddevice callback ---
    def audio_callback(outdata, frames, time_info, status):
        n_bytes = frames * BYTES_PER_FRAME
        data = audio_buf.pull(n_bytes)
        outdata[:] = memoryview(data).cast("B").cast("h", shape=(frames, CHANNELS))

    # --- Receive audio from Lyria ---
    async def receive_audio(session):
        buffering = True
        async for message in session.receive():
            if stop_event.is_set():
                return
            if not message.server_content or not message.server_content.audio_chunks:
                continue
            for chunk in message.server_content.audio_chunks:
                audio_buf.push(chunk.data)
            if buffering and audio_buf.available >= PRE_BUFFER_BYTES:
                buffering = False
                playback_started.set()

    # --- Steering handler ---
    async def steering_handler(session):
        while not stop_event.is_set():
            try:
                user_input = await asyncio.wait_for(prompt_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            ui_print(C.THINK, "🧠", f"Mistral analysing: \"{user_input}\" …", flush=True)
            t_start = time.perf_counter()

            src_palette = get_palette()

            # Ask Mistral for the target palette
            target = await engine.steer(user_input)

            if target is None:
                # Fallback: add the user input as a new prompt at weight 0.5
                target = dict(src_palette)
                target[user_input] = 0.5

            # Update the engine's internal palette record
            engine.current_palette = target

            latency_ms = (time.perf_counter() - t_start) * 1000
            ui_print(
                C.THINK, "🎚️",
                f"New palette → {palette_display(target)}  {C.DIM}[{latency_ms:.0f}ms]{C.RESET}",
                flush=True,
            )

            # Apply the target palette — instant by default (crossfade=0).
            # Lyria's model does its own internal smoothing, so snapping is fine.
            # Optional crossfade loop for those who want a slow morph.
            if args.crossfade > 0:
                n_steps = max(2, args.steps)
                step_sleep = args.crossfade / n_steps
                for i in range(1, n_steps + 1):
                    if stop_event.is_set():
                        break
                    t = i / n_steps
                    mid = interpolate_palettes(src_palette, target, t)
                    set_palette(mid)
                    await session.set_weighted_prompts(
                        prompts=palette_to_weighted_prompts(mid)
                    )
                    await asyncio.sleep(step_sleep)

            # Snap to final target (always runs — also acts as the single call when crossfade=0)
            set_palette(target)
            await session.set_weighted_prompts(
                prompts=palette_to_weighted_prompts(target)
            )
            # ✅ Confirm the change took effect
            ui_print(C.APPLIED, "✅", f"{C.BOLD}Applied!{C.RESET} Music is now being steered towards: {palette_display(target)}", flush=True)

    # --- Status display ---
    async def status_display():
        start_time = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            buf_secs = audio_buf.available / (SAMPLE_RATE * BYTES_PER_FRAME)
            mins, secs = divmod(int(elapsed), 60)
            pal = get_palette()
            line = (
                f"{C.STATUS}  ▶ [{mins:02d}:{secs:02d}]  "
                f"Buf:{buf_secs:.1f}s  "
                f"{C.DIM}Palette: {palette_display(pal)}{C.RESET}    "
            )
            # Pad to terminal width so previous longer lines are fully erased
            print(line, end="\r", flush=True)
            await asyncio.sleep(1.0)

    # --- Voice listener (Voxtral real-time STT) ---
    async def voice_listener():
        """
        Streams the microphone through Voxtral realtime transcription.
        Each completed utterance (after a silence) is pushed to prompt_queue,
        exactly as if the user had typed it in the terminal.
        """
        import traceback
        from mistralai.extra.realtime import UnknownRealtimeEvent
        from mistralai.models import (
            AudioFormat,
            RealtimeTranscriptionError,
            RealtimeTranscriptionSessionCreated,
            TranscriptionStreamDone,
            TranscriptionStreamTextDelta,
        )
        import pyaudio

        STT_SAMPLE_RATE   = 16_000
        STT_CHUNK_MS      = 480
        STT_MODEL         = "voxtral-mini-transcribe-realtime-2602"
        SILENCE_MIN_CHARS = 3

        # ── Startup mic test ──────────────────────────────────────────────
        ui_print(C.VOICE, "🎤", "Testing microphone…", flush=True)
        try:
            _p = pyaudio.PyAudio()
            _chunk = int(STT_SAMPLE_RATE * STT_CHUNK_MS / 1000)
            _st = _p.open(format=pyaudio.paInt16, channels=1,
                          rate=STT_SAMPLE_RATE, input=True,
                          frames_per_buffer=_chunk)
            _st.read(_chunk, exception_on_overflow=False)   # one test read
            _st.stop_stream(); _st.close(); _p.terminate()
            ui_print(C.APPLIED, "✅", "Microphone OK", flush=True)
        except Exception as exc:
            ui_print(C.WARN, "❌", f"Could not open microphone: {exc}", flush=True)
            ui_print(C.WARN, "  ", "Voice input disabled. Continue with keyboard only.", flush=True)
            return

        # ── Mic async generator ───────────────────────────────────────────
        async def iter_mic():
            p = pyaudio.PyAudio()
            chunk_samples = int(STT_SAMPLE_RATE * STT_CHUNK_MS / 1000)
            stream = p.open(
                format=pyaudio.paInt16, channels=1,
                rate=STT_SAMPLE_RATE, input=True,
                frames_per_buffer=chunk_samples,
            )
            loop = asyncio.get_running_loop()
            try:
                while not stop_event.is_set():
                    data = await loop.run_in_executor(
                        None, stream.read, chunk_samples, False
                    )
                    yield data
            finally:
                stream.stop_stream(); stream.close(); p.terminate()

        stt_client = Mistral(api_key=MISTRAL_API_KEY)
        audio_fmt  = AudioFormat(encoding="pcm_s16le", sample_rate=STT_SAMPLE_RATE)
        buf: list[str] = []
        last_delta_time: float = 0.0
        FLUSH_TIMEOUT = 2.5   # seconds without a new delta → force-flush

        async def flush_buf():
            """Send whatever is in buf to the steering queue, then clear it."""
            text = "".join(buf).strip()
            buf.clear()
            if len(text) >= SILENCE_MIN_CHARS:
                ui_print(C.VOICE, "🎤", f"{C.BOLD}Voice command:{C.RESET} \"{text}\"", flush=True)
                prompt_queue.put_nowait(text)
            else:
                print("", flush=True)

        ui_print(C.VOICE, "🎤", "Connecting to Voxtral realtime…", flush=True)
        try:
            async for event in stt_client.audio.realtime.transcribe_stream(
                audio_stream=iter_mic(),
                model=STT_MODEL,
                audio_format=audio_fmt,
            ):
                if stop_event.is_set():
                    break
                if isinstance(event, RealtimeTranscriptionSessionCreated):
                    ui_print(C.APPLIED, "📡", "Voxtral ready — speak to steer the music!", flush=True)
                elif isinstance(event, TranscriptionStreamTextDelta):
                    buf.append(event.text)
                    last_delta_time = time.monotonic()
                    joined = "".join(buf)
                    # Overwrite the same line while the user is still speaking
                    print(f"\r{C.VOICE}🎤 Hearing:{C.RESET} {joined[-80:]}   ", end="", flush=True)
                    # Flush immediately on sentence-ending punctuation
                    if joined.rstrip().endswith((".", "!", "?", "…")):
                        await flush_buf()
                elif isinstance(event, TranscriptionStreamDone):
                    await flush_buf()
                elif isinstance(event, RealtimeTranscriptionError):
                    ui_print(C.WARN, "⚠️ ", f"STT error: {event}", flush=True)
                elif isinstance(event, UnknownRealtimeEvent):
                    # Time-based flush: if we have text but no new delta for FLUSH_TIMEOUT
                    if buf and last_delta_time and (time.monotonic() - last_delta_time) > FLUSH_TIMEOUT:
                        await flush_buf()
        except Exception as exc:
            if not stop_event.is_set():
                ui_print(C.WARN, "❌", f"Voice listener crashed: {exc}", flush=True)
                traceback.print_exc()


    # --- Launch ---
    print(f"{C.INFO}{C.BOLD}🎵  Live AI DJ{C.RESET}  starting with: \"{initial_prompt}\"")
    print(f"{C.DIM}    Mistral model : {MISTRAL_MODEL}")
    print(f"    Mode          : {'Ambient / RPG landscape' if args.ambient else 'Full music'}")
    print(f"    Temperature   : {args.temperature}")
    print(f"    Guidance      : {args.guidance}")
    print(f"    Seed          : {args.seed if args.seed is not None else 'random'}")
    print(f"    Crossfade     : {args.crossfade}s over {args.steps} steps{C.RESET}")
    print(f"    {C.INFO}Connecting to Lyria RealTime …{C.RESET}\n")

    loop = asyncio.get_event_loop()
    input_thread = threading.Thread(
        target=input_reader, args=(prompt_queue, loop), daemon=True
    )
    input_thread.start()

    try:
        async with (
            lyria_client.aio.live.music.connect(model=MODEL) as session,
            asyncio.TaskGroup() as tg,
        ):
            # Set initial musical DNA palette
            await session.set_weighted_prompts(
                prompts=palette_to_weighted_prompts(initial_palette)
            )

            # Set generation config
            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(
                    temperature=args.temperature,
                    guidance=args.guidance,
                    seed=args.seed,
                    # Ambient mode: low density (sparse notes) + low brightness (soft timbre)
                    density=args.density,
                    brightness=args.brightness,
                )
            )

            await session.play()

            tg.create_task(receive_audio(session))
            tg.create_task(steering_handler(session))
            tg.create_task(status_display())
            if args.voice:
                tg.create_task(voice_listener())

            print(f"    {C.DIM}Buffering …{C.RESET}", end="\r", flush=True)
            while not playback_started.is_set():
                await asyncio.sleep(0.05)

            print(f"\n{C.APPLIED}{C.BOLD}    🔊 Playback started!{C.RESET}")
            print(f"{C.DIM}    Type a steering instruction and press Enter.")
            print(f"    Examples: \"more energy\" / \"add jazz piano\" / \"slow it down\" / \"only ambient\"")
            print(f"    Press Ctrl+C to stop.{C.RESET}\n")

            with sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=BLOCK_SIZE,
                callback=audio_callback,
            ):
                while True:
                    await asyncio.sleep(0.5)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop_event.set()
        print(f"\n\n{C.WARN}{C.BOLD}🛑  Stopped.{C.RESET} Thanks for listening!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live AI DJ with progressive prompt steering (Lyria + Mistral)"
    )
    parser.add_argument(
        "prompt",
        nargs="+",
        help="Initial music prompt (e.g. \"techno rave\")",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Lyria generation temperature 0.5–2.0 (default: 1.0)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=3.0,
        help="Lyria prompt guidance strength 1.0–6.0 (default: 3.0)",
    )
    parser.add_argument(
        "--crossfade",
        type=float,
        default=0.0,
        help="Seconds to crossfade between palette states (default: 0 = instant)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of interpolation steps in a crossfade (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42)",
    )
    parser.add_argument(
        "--ambient",
        action="store_true",
        help="Ambient / RPG landscape mode: sparse texture, minimal arrangement, low energy",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="Note density 0.0 (sparse) – 1.0 (dense). Ambient mode defaults to 0.3.",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=None,
        help="Timbre brightness 0.0 (dark/soft) – 1.0 (bright). Ambient mode defaults to 0.3.",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice input via Voxtral realtime STT (requires pyaudio)",
    )
    args = parser.parse_args()

    # Apply ambient defaults for density/brightness if not manually overridden
    if args.ambient:
        if args.density is None:
            args.density = 0.3
        if args.brightness is None:
            args.brightness = 0.3

    initial_prompt = " ".join(args.prompt)

    try:
        asyncio.run(live_dj(initial_prompt, args))
    except KeyboardInterrupt:
        print("\n🛑  Stopped. Thanks for listening!")


if __name__ == "__main__":
    main()
