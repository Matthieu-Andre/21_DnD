"""
lyria_eleven_child.py — ElevenLabs (fast metadata) → Lyria Live DJ

CONCEPT
-------
When the user types a prompt, we open an ElevenLabs compose_detailed()
HTTP stream and read chunks ONLY until the JSON metadata appears in the
multipart response. We then close the connection immediately, before
ElevenLabs has generated any audio. The rich compositionPlan is converted
into Lyria WeightedPrompts and pushed to a persistent Lyria WebSocket.

    User prompt
        → ElevenLabs multipart stream (JSON part only, connection closed early)
        → Lyria WeightedPrompts
        → live Lyria audio

Why this is faster than compose_detailed():
    The multipart response sends JSON FIRST, then audio.
    We close the stream as soon as we have the JSON,
    skipping audio generation + download entirely.

USAGE
-----
    python lyria_eleven_child.py "medieval tavern"
    python lyria_eleven_child.py "dark forest" --temperature 1.1 --guidance 3.5

WHILE RUNNING
-------------
    Type a new description + Enter → ElevenLabs metadata → Lyria steers
    Ctrl+C → stop

REQUIREMENTS
------------
    pip install google-genai elevenlabs python-dotenv sounddevice
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config / env
# ---------------------------------------------------------------------------
load_dotenv()

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env")

# ---------------------------------------------------------------------------
# Dependency imports
# ---------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌  pip install google-genai"); sys.exit(1)

# ElevenLabs is imported lazily inside eleven_to_palette() — only needed with --eleven

try:
    import sounddevice as sd
except ImportError:
    print("❌  pip install sounddevice"); sys.exit(1)

import collections

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    STATUS = "\033[90m"
    VOICE  = "\033[96m"
    THINK  = "\033[93m"
    APPLIED= "\033[92m"
    WARN   = "\033[91m"
    INFO   = "\033[94m"
    DIM    = "\033[2m"

def ui_print(color, prefix, msg, **kw):
    print(f"\r{' '*110}\r{color}{prefix}{C.RESET} {msg}", **kw)

# ---------------------------------------------------------------------------
# Lyria constants
# ---------------------------------------------------------------------------
LYRIA_MODEL  = "models/lyria-realtime-exp"
SAMPLE_RATE  = 48_000
CHANNELS     = 2
SAMPLE_WIDTH = 2
BYTES_PER_FRAME = CHANNELS * SAMPLE_WIDTH
BLOCK_SIZE   = 2400
PRE_BUFFER_BYTES = int(0.5 * SAMPLE_RATE * BYTES_PER_FRAME)
BUFFER_CAPACITY  = SAMPLE_RATE * BYTES_PER_FRAME * 10

lyria_client  = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options={"api_version": "v1alpha"},
)

# ---------------------------------------------------------------------------
# ElevenLabs → Lyria palette converter
# ---------------------------------------------------------------------------
def eleven_to_palette(prompt: str) -> dict[str, float]:
    """Call ElevenLabs compose_detailed() for rich musical DNA. Only used with --eleven."""
    try:
        from elevenlabs.client import ElevenLabs as _ElevenLabs
    except ImportError:
        ui_print(C.WARN, "⚠️ ", "elevenlabs not installed — pip install elevenlabs")
        return {prompt: 1.0}
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        ui_print(C.WARN, "⚠️ ", "ELEVENLABS_API_KEY not found in .env")
        return {prompt: 1.0}
    client = _ElevenLabs(api_key=api_key)
    ui_print(C.THINK, "🔬", f'ElevenLabs palette for "{prompt}"…', flush=True)
    try:
        response = client.music.compose_detailed(
            prompt=prompt,
            music_length_ms=3_000,
            force_instrumental=True,
            output_format="mp3_22050_32",
        )
    except Exception as exc:
        ui_print(C.WARN, "⚠️ ", f"ElevenLabs failed ({exc}) — fallback")
        return {prompt: 1.0}

    meta    = response.json
    palette: dict[str, float] = {}

    plan = meta.get("composition_plan", {})
    for style in plan.get("positive_global_styles", []):
        text = str(style).strip()
        if text:
            palette[text] = 1.2

    sections = plan.get("sections", [])
    if sections:
        for style in sections[0].get("positive_local_styles", []):
            text = str(style).strip()
            if text and text not in palette:
                palette[text] = 0.9

    song_meta = meta.get("song_metadata", {})
    for genre in song_meta.get("genres", []):
        text = str(genre).strip()
        if text and text not in palette:
            palette[text] = 1.0

    desc = song_meta.get("description", "")
    if desc:
        for part in str(desc).split(","):
            text = part.strip()
            if text and len(text) < 60 and text not in palette:
                palette[text] = 0.6

    if not palette:
        ui_print(C.WARN, "⚠️ ", "Empty palette — fallback")
        return {prompt: 1.0}

    return palette


def raw_to_palette(prompt: str) -> dict[str, float]:
    """Pass the prompt straight to Lyria as-is — no ElevenLabs call."""
    return {prompt: 1.0}


# ---------------------------------------------------------------------------
# Developer mode: 10 hardcoded palettes + Mistral scene selector
# ---------------------------------------------------------------------------
DEV_SCENES: dict[str, dict[str, float]] = {
    "peaceful_village": {
        "medieval folk": 1.2, "pastoral": 1.2, "acoustic": 1.2,
        "gentle flute melody": 0.9, "plucked lute strings": 0.9,
        "birdsong": 0.9, "soft hand drum": 0.9, "warm": 1.0,
        "relaxing": 1.0, "peaceful": 1.0, "moderate walking tempo": 0.9,
        "folk": 1.0, "instrumental": 1.0,
    },
    "mysterious_cave": {
        "dark ambient": 1.2, "cinematic": 1.2, "atmospheric": 1.2,
        "mysterious": 1.2, "eerie": 1.2, "soundscape": 1.2,
        "low resonant drone": 0.9, "dripping water sounds": 0.9,
        "subtle wind blowing": 0.9, "reverb": 0.9,
        "slowly building tension": 0.9, "sparse eerie piano": 0.9,
        "ambient": 1.0, "instrumental": 1.0,
    },
    "combat": {
        "orchestral action": 1.2, "battle music": 1.2, "intense": 1.2,
        "urgent": 1.0, "aggressive": 1.0, "driving rhythm": 1.0,
        "fast war drums": 0.9, "staccato strings": 0.9,
        "brass fanfare": 0.9, "percussion hits": 0.9,
        "high energy": 0.9, "fast tempo 150 BPM": 0.9,
        "cinematic": 1.0, "instrumental": 1.0,
    },
    "techno_party": {
        "techno": 1.2, "industrial": 1.2, "electronic": 1.2,
        "driving beat": 1.2, "rave": 1.2,
        "hard bass drop": 0.9, "distorted synth lead": 0.9,
        "repetitive kick drum": 0.9, "acid bassline": 0.9,
        "high energy": 1.0, "dark": 1.0,
        "fast 140 BPM": 0.9, "laser synth stab": 0.9,
        "instrumental": 1.0,
    },
    "victory": {
        "triumphant fanfare": 1.2, "celebratory": 1.2, "uplifting": 1.2,
        "joyful": 1.0, "bright": 1.0, "heroic": 1.0,
        "brass melody": 0.9, "orchestral swell": 0.9,
        "snare roll": 0.9, "harp glissando": 0.9,
        "chimes": 0.9, "major key": 0.9,
        "cinematic": 1.0, "instrumental": 1.0,
    },
    "puzzle": {
        "ambient": 1.2, "relaxing": 1.2, "lofi": 1.2,
        "chill": 1.1, "focus": 1.1, "study music": 1.1,
        "soft synths": 1.0, "subtle beat": 0.9,
        "calm": 1.0, "atmosphere": 1.0, "smooth": 0.9,
        "instrumental": 1.0, "gentle": 0.9,
    },
}

DEV_SCENE_NAMES = list(DEV_SCENES.keys())

DEV_MISTRAL_SYSTEM = f"""You map a user's free-form text to the single BEST matching scene from this list:
{', '.join(DEV_SCENE_NAMES)}

Reply with ONLY the scene name, nothing else. One word (or two with underscore).
If nothing matches well, pick the closest one anyway.

Scene descriptions:
  peaceful_village = peasants, farmers, calm, hiking, nature, peaceful, pastoral, medieval village, safe, resting, market
  busy_tavern      = tavern, pub, inn, drinking, festive, lively, medieval bar, ale, crowded, singing
  mysterious_cave  = cave, dungeon, dark, eerie, underground, exploring, tombs, ruins, spooky, shadows, creeping
  combat           = fight, battle, enemies, swords, attack, ambush, "an enemy appears", RPG combat, action, striking
  epic_boss        = boss fight, final battle, epic, massive, climax, dragon, demon lord, huge threat
  techno_party     = techno, rave, electronic, dance, party, bass, sci-fi club, cyberpunk bar
  victory          = won, victory, celebration, triumph, success, fanfare, defeated the boss, level up, loot
  paris_rain       = rain, night, city, jazz, noir, melancholy, sad, depressing, detective office, slow
  chase            = chase, pursuit, running, escape, fleeing, urgent, hurry, running away, chasing him
  puzzle           = puzzle, thinking, mystery, detective, riddle, investigation, coding, AI, artificial intelligence, hackathon, development, work, works, technical, intelligence, project"""


def dev_to_palette(prompt: str) -> dict[str, float]:
    """Developer mode: use Mistral to pick the closest hardcoded scene."""
    try:
        from mistralai import Mistral as _Mistral
    except ImportError:
        ui_print(C.WARN, "⚠️ ", "mistralai not installed — pip install mistralai")
        return {prompt: 1.0}
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        ui_print(C.WARN, "⚠️ ", "MISTRAL_API_KEY not found in .env")
        return {prompt: 1.0}

    ui_print(C.THINK, "⚡", f'Matching "{prompt}" to scene…', flush=True)
    client = _Mistral(api_key=api_key)
    try:
        resp = client.chat.complete(
            model="open-mistral-7b",
            messages=[
                {"role": "system", "content": DEV_MISTRAL_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=20,
            temperature=0.0,
        )
        scene_name = resp.choices[0].message.content.strip().lower().replace(" ", "_")
    except Exception as exc:
        ui_print(C.WARN, "⚠️ ", f"Mistral failed ({exc}) — defaulting to peaceful_village")
        scene_name = "peaceful_village"

    # Fuzzy match: find best matching key
    best = scene_name if scene_name in DEV_SCENES else None
    if not best:
        for key in DEV_SCENE_NAMES:
            if key in scene_name or scene_name in key:
                best = key
                break
    if not best:
        best = "peaceful_village"

    palette = DEV_SCENES[best]
    ui_print(C.APPLIED, "🎬", f'Scene: {C.BOLD}{best}{C.RESET}  ({len(palette)} prompts)')
    return dict(palette)


def palette_display(palette: dict[str, float]) -> str:
    parts = [f'"{t}":{w:.2f}' for t, w in sorted(palette.items(), key=lambda x: -x[1])]
    return "  ".join(parts)


def palette_to_weighted_prompts(palette: dict[str, float]) -> list[types.WeightedPrompt]:
    return [types.WeightedPrompt(text=t, weight=w) for t, w in palette.items()]


def interpolate_palettes(src: dict[str, float], dst: dict[str, float], t: float) -> dict[str, float]:
    """Linear crossfade between two palettes at position t (0→1)."""
    all_keys = set(src) | set(dst)
    result = {}
    for k in all_keys:
        w = src.get(k, 0.0) * (1 - t) + dst.get(k, 0.0) * t
        if w > 0.01:
            result[k] = round(w, 3)
    return result


# ---------------------------------------------------------------------------
# Ring buffer for Lyria audio
# ---------------------------------------------------------------------------
class AudioBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self._chunks: collections.deque[bytes] = collections.deque()
        self._total = 0
        self._capacity = capacity
        self._lock = threading.Lock()

    def push(self, data: bytes):
        with self._lock:
            self._chunks.append(data)
            self._total += len(data)
            while self._total > self._capacity and self._chunks:
                old = self._chunks.popleft()
                self._total -= len(old)

    def pull(self, n: int) -> bytes:
        with self._lock:
            if self._total == 0:
                return b"\x00" * n
            result = bytearray()
            while len(result) < n and self._chunks:
                chunk = self._chunks[0]
                needed = n - len(result)
                if len(chunk) <= needed:
                    result.extend(chunk)
                    self._total -= len(chunk)
                    self._chunks.popleft()
                else:
                    result.extend(chunk[:needed])
                    self._chunks[0] = chunk[needed:]
                    self._total -= needed
            if len(result) < n:
                result.extend(b"\x00" * (n - len(result)))
            return bytes(result)

    @property
    def available(self) -> int:
        with self._lock:
            return self._total


# ---------------------------------------------------------------------------
# Terminal input reader
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
# Voice listener (Voxtral real-time STT)
# ---------------------------------------------------------------------------
VOXTRAL_MODEL   = "voxtral-mini-transcribe-realtime-2602"
VOXTRAL_RATE    = 16_000
VOXTRAL_CHUNK_MS = 480


async def voice_listener(
    prompt_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    args: argparse.Namespace,
    current_scene_holder: list,     # mutable holder: [current_scene_name]
    voice_memory: list,             # accumulated speech segments
):
    """Listen to mic via Voxtral, feed utterances into the prompt_queue.

    Normal mode:   each completed utterance → prompt_queue as-is.
    Developer mode: utterances accumulate in voice_memory, Mistral decides
                    if the scene should change based on the full context.
    """
    import traceback
    from mistralai import Mistral as _Mistral
    from mistralai.extra.realtime import UnknownRealtimeEvent
    from mistralai.models import (
        AudioFormat,
        RealtimeTranscriptionError,
        RealtimeTranscriptionSessionCreated,
        TranscriptionStreamDone,
        TranscriptionStreamTextDelta,
    )
    import pyaudio

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        ui_print(C.WARN, "⚠️ ", "MISTRAL_API_KEY not found — voice disabled")
        return

    SILENCE_MIN_CHARS = 3

    # ── Startup mic test ──────────────────────────────────────────────
    ui_print(C.VOICE, "🎤", "Testing microphone…", flush=True)
    try:
        _p = pyaudio.PyAudio()
        _chunk = int(VOXTRAL_RATE * VOXTRAL_CHUNK_MS / 1000)
        _st = _p.open(format=pyaudio.paInt16, channels=1,
                      rate=VOXTRAL_RATE, input=True,
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
        chunk_samples = int(VOXTRAL_RATE * VOXTRAL_CHUNK_MS / 1000)
        stream = p.open(
            format=pyaudio.paInt16, channels=1,
            rate=VOXTRAL_RATE, input=True,
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

    client = _Mistral(api_key=api_key)
    audio_fmt = AudioFormat(encoding="pcm_s16le", sample_rate=VOXTRAL_RATE)
    buf: list[str] = []
    last_delta_time: float = 0.0
    FLUSH_TIMEOUT = 2.5   # seconds without a new delta → force-flush

    async def flush_buf():
        """Send whatever is in buf to the steering queue, then clear it."""
        text = "".join(buf).strip()
        buf.clear()
        if len(text) < SILENCE_MIN_CHARS:
            return

        ui_print(C.VOICE, "🎤", f'{C.BOLD}Voice command:{C.RESET} "{text}"', flush=True)

        if args.developer:
            # Accumulate memory and let Mistral decide
            voice_memory.append(text)
            if len(voice_memory) > 20:
                voice_memory[:] = voice_memory[-20:]

            loop = asyncio.get_event_loop()

            async def do_sfx_analysis():
                """Run SFX intent check concurrently."""
                sfx_prompt = await analyze_sfx(text)
                if sfx_prompt:
                    # Spawn playback in a true background thread so it doesn't block async event loop
                    threading.Thread(target=play_sfx, args=(sfx_prompt,), daemon=True).start()

            async def do_scene_analysis():
                new_scene = await loop.run_in_executor(
                    None, dev_memory_to_scene, voice_memory, current_scene_holder[0]
                )
                if new_scene and new_scene != current_scene_holder[0]:
                    ui_print(C.APPLIED, "🎬", f'Scene shift: {current_scene_holder[0]} → {C.BOLD}{new_scene}{C.RESET}')
                    current_scene_holder[0] = new_scene
                    prompt_queue.put_nowait(f"__DEV_SCENE__{new_scene}")
                else:
                    ui_print(C.DIM, "  ", f'Context unchanged — staying on {current_scene_holder[0]}')

            # Run both analyzes concurrently
            asyncio.create_task(do_sfx_analysis())
            asyncio.create_task(do_scene_analysis())

        else:
            # Normal mode: send utterance directly
            prompt_queue.put_nowait(text)

    ui_print(C.VOICE, "🎤", "Connecting to Voxtral realtime…", flush=True)
    try:
        async for event in client.audio.realtime.transcribe_stream(
            audio_stream=iter_mic(),
            model=VOXTRAL_MODEL,
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


def dev_memory_to_scene(memory: list[str], current_scene: str) -> str:
    """Ask Mistral if the accumulated voice memory warrants a scene change."""
    from mistralai import Mistral as _Mistral
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return current_scene
    client = _Mistral(api_key=api_key)

    if not memory:
        return current_scene

    # Use only the last 5 utterances to avoid dragging ancient history
    recent = memory[-5:]
    if len(recent) > 1:
        context_str = " | ".join(recent[:-1])
        now_str = recent[-1]
        user_prompt = f"Previous context: {context_str}\n\nJUST HAPPENED NOW: {now_str}"
    else:
        user_prompt = f"JUST HAPPENED NOW: {recent[0]}"

    system = f"""You are a scene selector for a live music system.
Available scenes: {', '.join(DEV_SCENE_NAMES)}

The CURRENT scene is: {current_scene}

Read the "Previous context" (if any) to understand the setting, but you MUST base your decision primarily on what "JUST HAPPENED NOW".
- If the narrative "JUST HAPPENED NOW" still fits the current scene, reply with EXACTLY: {current_scene}
- If the narrative has shifted to a different mood/scene (e.g., an enemy appeared -> combat), reply with the new scene name.

Reply with ONLY the scene name. Nothing else."""

    try:
        resp = client.chat.complete(
            model="open-mistral-7b",
            messages=[
                {"role": "system", "content": system + "\n\n" + DEV_MISTRAL_SYSTEM.split("Scene descriptions:")[1]},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=20,
            temperature=0.0,
        )
        scene = resp.choices[0].message.content.strip().lower().replace(" ", "_")
    except Exception:
        return current_scene

    # Fuzzy match
    if scene in DEV_SCENES:
        return scene
    for key in DEV_SCENE_NAMES:
        if key in scene or scene in key:
            return key
    return current_scene


# ---------------------------------------------------------------------------
# ElevenLabs SFX Integration
# ---------------------------------------------------------------------------
async def analyze_sfx(utterance: str) -> str | None:
    """Use Mistral JSON mode to decide if this utterance warrants a sound effect."""
    from mistralai import Mistral as _Mistral
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return None
        
    client = _Mistral(api_key=api_key)
    system = """You extract explicit sound effect triggers from speech.
Does the user's speech explicitly describe a specific event or action that makes a distinct sound? 
(e.g., "a loud explosion", "door creaks open", "dragon screams", "glass shattering").
Do NOT trigger on general moods, dialogue, abstract concepts, or continuous background noise.

Respond ONLY in valid JSON with two fields:
  "warrants_sfx" (boolean): true if a distinct sound effect must be played NOW.
  "sfx_prompt" (string | null): a short, descriptive prompt for sound generation (e.g. "heavy wooden door creaking")."""

    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.complete(
                model="open-mistral-7b",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": utterance},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
        )
        import json
        data = json.loads(resp.choices[0].message.content)
        if data.get("warrants_sfx") and data.get("sfx_prompt"):
            return str(data["sfx_prompt"])
    except Exception as e:
        ui_print(C.WARN, "⚠️ ", f"SFX analysis error: {e}")
    return None


def play_sfx(sfx_prompt: str):
    """Generate SFX with ElevenLabs, decode with pydub, and play on a new sounddevice stream."""
    try:
        import sounddevice as sd
        from pydub import AudioSegment
        import io
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        ui_print(C.WARN, "⚠️ ", f"Missing SFX dependency: {e} - pip install pydub sounddevice elevenlabs")
        return

    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        ui_print(C.WARN, "⚠️ ", "ELEVENLABS_API_KEY missing - cannot play SFX")
        return

    ui_print(C.THINK, "⚡", f'Generating SFX: "{sfx_prompt}"...')
    
    try:
        client = ElevenLabs(api_key=key)
        # 1. Generate SFX audio (generator of bytes)
        audio_generator = client.text_to_sound_effects.convert(
            text=sfx_prompt,
            duration_seconds=None, # auto-duration
            prompt_influence=0.3   # higher = closer to prompt
        )
        # Load all bytes
        audio_bytes = b"".join(list(audio_generator))
        
        # 2. Decode MP3 to raw PCM using pydub
        f = io.BytesIO(audio_bytes)
        seg = AudioSegment.from_file(f)
        
        # 3. Play stream
        ui_print(C.APPLIED, "🔊", f'Playing SFX: "{sfx_prompt}"')
        raw_data = seg.raw_data
        sample_rate = seg.frame_rate
        channels = seg.channels
        
        import numpy as np
        # Convert pydub string bytes to numpy array
        dt = np.int16 if seg.sample_width == 2 else np.int32
        audio_np = np.frombuffer(raw_data, dtype=dt)
        
        # Reshape for sounddevice (frames, channels)
        if channels > 1:
            audio_np = audio_np.reshape((-1, channels))
            
        # Play in a blocking call *inside this off-thread*
        sd.play(audio_np, samplerate=sample_rate)
        sd.wait()
        
    except Exception as e:
        ui_print(C.WARN, "⚠️ ", f"Failed playing SFX '{sfx_prompt}': {e}")


# ---------------------------------------------------------------------------
# Main async logic
# ---------------------------------------------------------------------------
async def live_dj(initial_prompt: str, args: argparse.Namespace) -> None:
    audio_buf = AudioBuffer()
    prompt_queue: asyncio.Queue[str] = asyncio.Queue()
    playback_started = threading.Event()
    stop_event = asyncio.Event()

    # ── Palette translator (respects --eleven / --developer flags) ──────
    # Track current scene for developer + voice mode
    current_scene_holder = ["peaceful_village"]  # mutable holder for scene name
    voice_memory: list[str] = []                 # accumulated voice utterances

    def to_palette(p: str) -> dict[str, float]:
        # Developer mode: check for internal scene override from voice
        if args.developer and p.startswith("__DEV_SCENE__"):
            scene = p.replace("__DEV_SCENE__", "")
            if scene in DEV_SCENES:
                current_scene_holder[0] = scene
                palette = DEV_SCENES[scene]
                ui_print(C.APPLIED, "🎬", f'Scene: {C.BOLD}{scene}{C.RESET}  ({len(palette)} prompts)')
                return dict(palette)
        if args.developer:
            result = dev_to_palette(p)
            # Track which scene was picked
            for key, val in DEV_SCENES.items():
                if result == val or result == dict(val):
                    current_scene_holder[0] = key
                    break
            return result
        elif args.eleven:
            return eleven_to_palette(p)
        else:
            return raw_to_palette(p)

    # ── Step 1: translate initial prompt ─────────────────────────────────
    loop = asyncio.get_event_loop()
    initial_palette = await loop.run_in_executor(None, to_palette, initial_prompt)
    ui_print(C.APPLIED, "🧬", f"Musical DNA: {palette_display(initial_palette)}")

    current_palette: dict[str, float] = dict(initial_palette)
    palette_lock = threading.Lock()

    def get_palette():
        with palette_lock:
            return dict(current_palette)

    def set_palette(p):
        nonlocal current_palette
        with palette_lock:
            current_palette = dict(p)

    # ── Sounddevice callback ──────────────────────────────────────────────
    def audio_callback(outdata, frames, time_info, status):
        n_bytes = frames * BYTES_PER_FRAME
        data = audio_buf.pull(n_bytes)
        outdata[:] = memoryview(data).cast("B").cast("h", shape=(frames, CHANNELS))

    # ── Receive audio from Lyria ──────────────────────────────────────────
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

    # ── Steering handler ───────────────────────────────────────────────────
    async def steering_handler(session):
        while not stop_event.is_set():
            try:
                user_input = await asyncio.wait_for(prompt_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            ui_print(C.THINK, "🔬", f'ElevenLabs fast-metadata for "{user_input}"…', flush=True)
            t0 = time.perf_counter()

            src_palette = get_palette()

            # Run in thread pool — non-blocking for asyncio
            target = await loop.run_in_executor(None, to_palette, user_input)

            latency_ms = (time.perf_counter() - t0) * 1000
            ui_print(
                C.THINK, "🎚️ ",
                f"New palette → {palette_display(target)}  {C.DIM}[{latency_ms:.0f}ms]{C.RESET}",
            )

            # Crossfade from current → target over args.crossfade seconds
            if args.crossfade > 0:
                n_steps = max(2, args.steps)
                step_sleep = args.crossfade / n_steps
                for i in range(1, n_steps + 1):
                    if stop_event.is_set():
                        break
                    t = i / n_steps
                    mid = interpolate_palettes(src_palette, target, t)
                    set_palette(mid)
                    await session.set_weighted_prompts(prompts=palette_to_weighted_prompts(mid))
                    await asyncio.sleep(step_sleep)

            set_palette(target)
            await session.set_weighted_prompts(prompts=palette_to_weighted_prompts(target))
            ui_print(C.APPLIED, "✅", f"{C.BOLD}Applied!{C.RESET} Lyria steering: {palette_display(target)}")

    # ── Status display ─────────────────────────────────────────────────────
    async def status_display():
        start = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start
            buf_s = audio_buf.available / (SAMPLE_RATE * BYTES_PER_FRAME)
            mins, secs = divmod(int(elapsed), 60)
            pal = get_palette()
            line = (
                f"{C.STATUS}  ▶ [{mins:02d}:{secs:02d}]  Buf:{buf_s:.1f}s  "
                f"{C.DIM}Palette: {palette_display(pal)}{C.RESET}    "
            )
            print(line, end="\r", flush=True)
            await asyncio.sleep(1.0)

    # ── Launch ──────────────────────────────────────────────────────────────
    if args.developer:
        mode_label = "Developer (10 preset scenes)"
    elif args.eleven:
        mode_label = "ElevenLabs DNA + Lyria"
    else:
        mode_label = "Lyria direct"
    print(f"\n{C.INFO}{C.BOLD}🎵  Lyria Live DJ  [{mode_label}]{C.RESET}")
    if args.developer:
        print(f"{C.DIM}    10 preset scenes, Mistral picks the best match")
        for name in DEV_SCENE_NAMES:
            print(f"      • {name}")
    elif args.eleven:
        print(f"{C.DIM}    ElevenLabs: compose_detailed() — rich musical DNA, audio discarded")
    else:
        print(f"{C.DIM}    Lyria direct: prompt used as-is")
    print(f"    Lyria     : live real-time audio generation")
    print(f"    Crossfade : {args.crossfade}s over {args.steps} steps")
    print(f"    Connecting to Lyria RealTime…{C.RESET}\n")

    _loop = asyncio.get_event_loop()
    t_input = threading.Thread(target=input_reader, args=(prompt_queue, _loop), daemon=True)
    t_input.start()

    # Optionally start voice listener
    voice_task = None

    try:
        async with (
            lyria_client.aio.live.music.connect(model=LYRIA_MODEL) as session,
            asyncio.TaskGroup() as tg,
        ):
            await session.set_weighted_prompts(
                prompts=palette_to_weighted_prompts(initial_palette)
            )
            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(
                    temperature=args.temperature,
                    guidance=args.guidance,
                    seed=args.seed,
                )
            )
            await session.play()

            tg.create_task(receive_audio(session))
            tg.create_task(steering_handler(session))
            tg.create_task(status_display())

            # Start voice listener if --voice
            if args.voice:
                tg.create_task(
                    voice_listener(
                        prompt_queue, stop_event, args,
                        current_scene_holder, voice_memory,
                    )
                )

            print(f"    {C.DIM}Buffering…{C.RESET}", end="\r", flush=True)
            while not playback_started.is_set():
                await asyncio.sleep(0.05)

            print(f"\n{C.APPLIED}{C.BOLD}    🔊 Playback started!{C.RESET}")
            if args.voice:
                print(f"{C.DIM}    🎤 Voice input active — just speak to steer.")
            print(f"{C.DIM}    Type a description and press Enter to steer the music.")
            if args.developer:
                print(f"    In developer mode, speech is accumulated and scene changes automatically.")
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
        print(f"\n\n{C.WARN}{C.BOLD}🛑  Stopped.{C.RESET}  Thanks for listening!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Lyria × ElevenLabs Hybrid DJ — ElevenLabs extracts musical DNA, Lyria plays live"
    )
    parser.add_argument("prompt", nargs="+", help='Initial prompt e.g. "medieval tavern"')
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Lyria temperature 0.5–2.0 (default: 1.0)")
    parser.add_argument("--guidance", type=float, default=4,
                        help="Lyria guidance 1.0–6.0 (default: 3.0)")
    parser.add_argument("--crossfade", type=float, default=4.0,
                        help="Crossfade seconds between palette changes (default: 4.0)")
    parser.add_argument("--steps", type=int, default=8,
                        help="Crossfade interpolation steps (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Lyria generation seed for reproducibility (default: 42)")
    parser.add_argument("--eleven", action="store_true",
                        help="Use ElevenLabs to enrich prompts into musical DNA")
    parser.add_argument("--developer", action="store_true",
                        help="Developer mode: 10 hardcoded scene palettes, Mistral picks the best match")
    parser.add_argument("--voice", action="store_true",
                        help="Enable voice input via Voxtral real-time STT (requires pyaudio)")
    args = parser.parse_args()

    initial_prompt = " ".join(args.prompt)
    asyncio.run(live_dj(initial_prompt, args))


if __name__ == "__main__":
    main()
