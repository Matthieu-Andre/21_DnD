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

    SYSTEM_PROMPT = f"""You are an expert AI DJ assistant controlling a real-time music generation system (Google Lyria RealTime).

{LYRIA_REFERENCE_TABLE}

## Your ONLY job
Given the CURRENT PALETTE and a USER INSTRUCTION, return a JSON array of
{{\"text\": str, \"weight\": float}} representing the ADJUSTED palette.
Respond with RAW JSON ONLY — no markdown, no explanation.

## THE GOLDEN RULE: Musical DNA Must Survive
The current palette encodes the musical identity of the session (genre, instruments, mood).
Every steering instruction is an ADJUSTMENT, not a rewrite.

Dominating prompts (weight ≥ 0.5) are ANCHORS — they must appear in your output
unless the user EXPLICITLY says to stop, remove, or switch genre entirely.

## How to classify an instruction

### Class A — Narrative / scene evolution (most common)
The user describes what is HAPPENING in the scene. The music should evolve
to match the new energy/mood while staying in the same musical world.
→ Strategy: Keep all anchors. Adjust tempo/energy/intensity prompts only.
→ Examples: "knight gets on a horse", "the battle begins", "tension rises",
           "more energy", "he finds treasure", "they celebrate"

### Class B — Direct musical tweak
The user explicitly names a musical element to change.
→ Strategy: Adjust that element; keep everything else.
→ Examples: "more bass", "add piano", "slower tempo", "louder drums"

### Class C — Explicit genre/style change
The user EXPLICITLY asks to switch to a different genre or start fresh.
→ Strategy: Transition — reduce anchors to 0.3 while boosting the new style.
→ Examples: "switch to jazz", "make it electronic now", "go full metal",
           "stop the medieval stuff", "completely different vibe"

## Worked examples

EXAMPLE 1 — Narrative evolution (Class A)
Current palette: [{{"text": "medieval folk", "weight": 1.0}},
                  {{"text": "calm countryside", "weight": 0.8}}]
User: "the knight gets on his horse and starts galloping"
BAD output: [{{"text": "galloping hooves", "weight": 1.0}}, {{"text": "battle drums", "weight": 0.8}}]
  ← WRONG: wiped the medieval identity entirely
GOOD output: [{{"text": "medieval folk", "weight": 1.0}},
              {{"text": "calm countryside", "weight": 0.4}},
              {{"text": "faster tempo", "weight": 0.7}},
              {{"text": "epic orchestral swell", "weight": 0.6}}]
  ← RIGHT: medieval stays dominant, energy/tempo are added on top

EXAMPLE 2 — Direct tweak (Class B)
Current palette: [{{"text": "techno rave", "weight": 1.0}}, {{"text": "driving bassline", "weight": 0.8}}]
User: "more energy"
GOOD output: [{{"text": "techno rave", "weight": 1.2}},
              {{"text": "driving bassline", "weight": 1.0}},
              {{"text": "high energy", "weight": 0.6}}]

EXAMPLE 3 — Explicit change (Class C)
Current palette: [{{"text": "techno rave", "weight": 1.0}}]
User: "switch to medieval folk music now"
GOOD output: [{{"text": "techno rave", "weight": 0.3}},
              {{"text": "medieval folk", "weight": 1.2}},
              {{"text": "acoustic instruments", "weight": 0.8}}]
"""

    def __init__(self, initial_prompt: str):
        self.history: list[dict] = []
        self.current_palette: dict[str, float] = {initial_prompt: 1.0}
        # Anchors are the dominant prompts from the PREVIOUS step.
        # Used to enforce continuity if Mistral silently drops them.
        self._prev_anchors: dict[str, float] = {initial_prompt: 1.0}

    def _palette_to_str(self) -> str:
        items = [{"text": t, "weight": w} for t, w in self.current_palette.items()]
        return json.dumps(items)

    def _enforce_continuity(self, new_palette: dict[str, float]) -> dict[str, float]:
        """Safety net: restore any anchor (weight ≥ 0.5) silently dropped by Mistral."""
        result = dict(new_palette)
        for text, weight in self._prev_anchors.items():
            if text not in result:
                # Restore at 40 % of its former strength — enough to stay
                # musically present without dominating the new direction.
                result[text] = round(weight * 0.4, 3)
        return result

    async def steer(self, user_instruction: str) -> Optional[dict[str, float]]:
        """
        Ask Mistral to interpret the user instruction and return an updated palette.
        Returns None on failure (caller should skip the update gracefully).
        """
        user_content = (
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

            # Enforce continuity — restore silently dropped anchors
            new_palette = self._enforce_continuity(new_palette)

            # Update anchors to current dominant prompts for next round
            self._prev_anchors = {t: w for t, w in new_palette.items() if w >= 0.5}

            # Store the assistant response in history for context continuity
            self.history.append({"role": "assistant", "content": raw})

            return new_palette

        except Exception as exc:
            # Rollback the failed user message from history
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            print(f"\n⚠️  Mistral steering failed ({exc}), applying prompt directly.")
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

    engine = SteeringEngine(initial_prompt)
    # Shared palette (read by status display, written by steering handler)
    palette_lock = threading.Lock()
    current_palette: dict[str, float] = {initial_prompt: 1.0}

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

            print(f"\n🧠  Mistral analysing: \"{user_input}\" …", flush=True)
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
            print(
                f"🎚️  Steering → {palette_display(target)}  "
                f"[Mistral latency: {latency_ms:.0f}ms]",
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

    # --- Status display ---
    async def status_display():
        start_time = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            buf_secs = audio_buf.available / (SAMPLE_RATE * BYTES_PER_FRAME)
            mins, secs = divmod(int(elapsed), 60)
            pal = get_palette()
            print(
                f"  ▶ [{mins:02d}:{secs:02d}]  Buffer:{buf_secs:.1f}s  "
                f"Palette: {palette_display(pal)}    ",
                end="\r",
                flush=True,
            )
            await asyncio.sleep(1.0)

    # --- Launch ---
    print(f"🎵  Live AI DJ — Starting with: \"{initial_prompt}\"")
    print(f"    Mistral model : {MISTRAL_MODEL}")
    print(f"    Temperature   : {args.temperature}")
    print(f"    Guidance      : {args.guidance}")
    print(f"    Crossfade     : {args.crossfade}s over {args.steps} steps")
    print("    Connecting to Lyria RealTime …\n")

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
            # Set initial prompt
            await session.set_weighted_prompts(
                prompts=[types.WeightedPrompt(text=initial_prompt, weight=1.0)]
            )

            # Set generation config
            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(
                    temperature=args.temperature,
                    guidance=args.guidance,
                )
            )

            await session.play()

            tg.create_task(receive_audio(session))
            tg.create_task(steering_handler(session))
            tg.create_task(status_display())

            print("    Buffering …", end="\r", flush=True)
            while not playback_started.is_set():
                await asyncio.sleep(0.05)

            print("    🔊 Playback started!")
            print("    Type a steering instruction and press Enter.")
            print("    Examples: \"more energy\" / \"add jazz piano\" / \"slow it down\" / \"only ambient\"\n")
            print("    Press Ctrl+C to stop.\n")

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
        print("\n\n🛑  Stopped. Thanks for listening!")


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
    args = parser.parse_args()
    initial_prompt = " ".join(args.prompt)

    try:
        asyncio.run(live_dj(initial_prompt, args))
    except KeyboardInterrupt:
        print("\n🛑  Stopped. Thanks for listening!")


if __name__ == "__main__":
    main()
