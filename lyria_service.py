from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    genai = None
    types = None

from mistralai import Mistral


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

LYRIA_MODEL = "models/lyria-realtime-exp"
SAMPLE_RATE = 48_000
CHANNELS = 2
SAMPLE_WIDTH = 2
DEV_SCENES: dict[str, dict[str, float]] = {
    "peaceful_village": {
        "medieval folk": 1.2,
        "pastoral": 1.2,
        "acoustic": 1.2,
        "gentle flute melody": 0.9,
        "plucked lute strings": 0.9,
        "birdsong": 0.9,
        "soft hand drum": 0.9,
        "warm": 1.0,
        "relaxing": 1.0,
        "peaceful": 1.0,
        "moderate walking tempo": 0.9,
        "folk": 1.0,
        "instrumental": 1.0,
    },
    "mysterious_cave": {
        "dark ambient": 1.2,
        "cinematic": 1.2,
        "atmospheric": 1.2,
        "mysterious": 1.2,
        "eerie": 1.2,
        "soundscape": 1.2,
        "low resonant drone": 0.9,
        "dripping water sounds": 0.9,
        "subtle wind blowing": 0.9,
        "reverb": 0.9,
        "slowly building tension": 0.9,
        "sparse eerie piano": 0.9,
        "ambient": 1.0,
        "instrumental": 1.0,
    },
    "combat": {
        "orchestral action": 1.2,
        "battle music": 1.2,
        "intense": 1.2,
        "urgent": 1.0,
        "aggressive": 1.0,
        "driving rhythm": 1.0,
        "fast war drums": 0.9,
        "staccato strings": 0.9,
        "brass fanfare": 0.9,
        "percussion hits": 0.9,
        "high energy": 0.9,
        "fast tempo 150 BPM": 0.9,
        "cinematic": 1.0,
        "instrumental": 1.0,
    },
    "techno_party": {
        "techno": 1.2,
        "industrial": 1.2,
        "electronic": 1.2,
        "driving beat": 1.2,
        "rave": 1.2,
        "hard bass drop": 0.9,
        "distorted synth lead": 0.9,
        "repetitive kick drum": 0.9,
        "acid bassline": 0.9,
        "high energy": 1.0,
        "dark": 1.0,
        "fast 140 BPM": 0.9,
        "laser synth stab": 0.9,
        "instrumental": 1.0,
    },
    "victory": {
        "triumphant fanfare": 1.2,
        "celebratory": 1.2,
        "uplifting": 1.2,
        "joyful": 1.0,
        "bright": 1.0,
        "heroic": 1.0,
        "brass melody": 0.9,
        "orchestral swell": 0.9,
        "snare roll": 0.9,
        "harp glissando": 0.9,
        "chimes": 0.9,
        "major key": 0.9,
        "cinematic": 1.0,
        "instrumental": 1.0,
    },
    "puzzle": {
        "mysterious": 1.2,
        "subtle": 1.2,
        "detective": 1.2,
        "investigation": 1.2,
        "thinking": 1.2,
        "light piano motif": 0.9,
        "subtle pizzicato strings": 0.9,
        "minimalist percussion": 0.9,
        "curiosity": 0.9,
        "a sense of quiet thought": 0.9,
        "sparse plucked notes": 0.9,
        "ambient": 1.0,
        "instrumental": 1.0,
    },
}
DEV_SCENE_NAMES = list(DEV_SCENES.keys())
DEV_MISTRAL_SYSTEM = f"""You map free-form scene descriptions to the closest single scene.
Available scenes: {', '.join(DEV_SCENE_NAMES)}

Reply with ONLY the scene name. If multiple fit, choose the closest one.

Scene descriptions:
  peaceful_village = calm, pastoral, village, market, resting, warm, safe, exploration
  mysterious_cave  = cave, dungeon, eerie, ruins, underground, shadows, suspense, mystery
  combat           = fight, battle, ambush, attack, enemies, urgent action, chaos
  techno_party     = rave, electronic, cyberpunk, party, club, bass, dancefloor
  victory          = celebration, triumph, success, heroic resolution, fanfare
  puzzle           = clues, investigation, thinking, riddles, quiet deduction
"""


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def palette_display(palette: dict[str, float]) -> str:
    if not palette:
        return "No palette"
    parts = [f'{text}:{weight:.2f}' for text, weight in sorted(palette.items(), key=lambda item: -item[1])]
    return " | ".join(parts[:6])


def palette_to_snapshot(palette: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"text": text, "weight": weight}
        for text, weight in sorted(palette.items(), key=lambda item: -item[1])
    ]


def palette_to_weighted_prompts(palette: dict[str, float]) -> list[Any]:
    if types is None:
        raise RuntimeError("google-genai is not installed.")
    return [types.WeightedPrompt(text=text, weight=weight) for text, weight in palette.items()]


def interpolate_palettes(src: dict[str, float], dst: dict[str, float], progress: float) -> dict[str, float]:
    all_keys = set(src) | set(dst)
    result: dict[str, float] = {}
    for key in all_keys:
        weight = src.get(key, 0.0) * (1 - progress) + dst.get(key, 0.0) * progress
        if weight > 0.01:
            result[key] = round(weight, 3)
    return result


def raw_to_palette(prompt: str) -> dict[str, float]:
    return {normalize_text(prompt) or "instrumental fantasy ambience": 1.0}


def eleven_to_palette(prompt: str, log_callback: Callable[[str], None] | None = None) -> dict[str, float]:
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        if log_callback:
            log_callback("ElevenLabs SDK missing. Falling back to raw prompt mode.")
        return raw_to_palette(prompt)

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        if log_callback:
            log_callback("ELEVENLABS_API_KEY missing. Falling back to raw prompt mode.")
        return raw_to_palette(prompt)

    try:
        client = ElevenLabs(api_key=api_key)
        response = client.music.compose_detailed(
            prompt=prompt,
            music_length_ms=3_000,
            force_instrumental=True,
            output_format="mp3_22050_32",
        )
    except Exception as exc:
        if log_callback:
            log_callback(f"ElevenLabs enrichment failed: {exc}. Falling back to raw prompt mode.")
        return raw_to_palette(prompt)

    meta = response.json
    palette: dict[str, float] = {}
    plan = meta.get("composition_plan", {})
    for style in plan.get("positive_global_styles", []):
        text = normalize_text(style)
        if text:
            palette[text] = 1.2

    sections = plan.get("sections", [])
    if sections:
        for style in sections[0].get("positive_local_styles", []):
            text = normalize_text(style)
            if text and text not in palette:
                palette[text] = 0.9

    song_meta = meta.get("song_metadata", {})
    for genre in song_meta.get("genres", []):
        text = normalize_text(genre)
        if text and text not in palette:
            palette[text] = 1.0

    description = normalize_text(song_meta.get("description", ""))
    if description:
        for part in description.split(","):
            text = normalize_text(part)
            if text and len(text) < 60 and text not in palette:
                palette[text] = 0.6

    return palette or raw_to_palette(prompt)


def dev_to_palette(prompt: str, log_callback: Callable[[str], None] | None = None) -> dict[str, float]:
    normalized_prompt = normalize_text(prompt).lower().replace(" ", "_")
    if normalized_prompt in DEV_SCENES:
        return dict(DEV_SCENES[normalized_prompt])

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return dict(DEV_SCENES["peaceful_village"])

    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="open-mistral-7b",
            temperature=0.0,
            max_tokens=20,
            messages=[
                {"role": "system", "content": DEV_MISTRAL_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        scene_name = normalize_text(response.choices[0].message.content).lower().replace(" ", "_")
    except Exception as exc:
        if log_callback:
            log_callback(f"Developer scene mapping failed: {exc}. Falling back to peaceful_village.")
        scene_name = "peaceful_village"

    best_match = scene_name if scene_name in DEV_SCENES else None
    if not best_match:
        for key in DEV_SCENE_NAMES:
            if key in scene_name or scene_name in key:
                best_match = key
                break
    return dict(DEV_SCENES[best_match or "peaceful_village"])


def translate_prompt_to_palette(
    *,
    mode: str,
    prompt: str,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, float]:
    if mode == "eleven":
        return eleven_to_palette(prompt, log_callback)
    if mode == "developer":
        return dev_to_palette(prompt, log_callback)
    return raw_to_palette(prompt)


@dataclass
class SteeringCommand:
    prompt: str
    source: str
    reason: str
    force: bool = False


class LiveMusicSession:
    def __init__(
        self,
        *,
        mode: str,
        temperature: float,
        guidance: float,
        crossfade_seconds: float,
        crossfade_steps: int,
        seed: int,
        log_callback: Callable[[str], None],
    ) -> None:
        self.mode = mode
        self.temperature = temperature
        self.guidance = guidance
        self.crossfade_seconds = max(0.0, crossfade_seconds)
        self.crossfade_steps = max(1, crossfade_steps)
        self.seed = seed
        self.log_callback = log_callback
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.active = False
        self.connected = False
        self.available = False
        self.error = ""
        self.current_prompt = ""
        self.current_palette: dict[str, float] = {}
        self.last_reason = ""
        self.last_source = ""
        self.last_update_at: float | None = None
        self.listener_queues: dict[str, asyncio.Queue[bytes]] = {}
        self._command_queue: asyncio.Queue[SteeringCommand] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._client = self._build_client()

    def _build_client(self) -> Any | None:
        if genai is None:
            self.error = "google-genai is not installed."
            return None
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            self.error = "GOOGLE_API_KEY is not set."
            return None
        self.available = True
        self.error = ""
        return genai.Client(
            api_key=google_api_key,
            http_options={"api_version": "v1alpha"},
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.active,
            "available": self.available,
            "connected": self.connected,
            "mode": self.mode,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "prompt": self.current_prompt,
            "palette": palette_to_snapshot(self.current_palette),
            "palette_summary": palette_display(self.current_palette),
            "last_reason": self.last_reason,
            "last_source": self.last_source,
            "last_update_at": self.last_update_at,
            "listener_count": len(self.listener_queues),
            "error": self.error,
        }

    async def start(self, *, initial_prompt: str, reason: str, source: str = "initial") -> None:
        if not self.available or self._client is None:
            return
        if self._task and not self._task.done():
            return
        self.active = True
        await self._command_queue.put(SteeringCommand(prompt=initial_prompt, reason=reason, source=source, force=True))
        self._task = asyncio.create_task(self._run())

    async def steer(self, *, prompt: str, source: str, reason: str, force: bool = False) -> None:
        if not self.available or not self.active:
            return
        await self._command_queue.put(
            SteeringCommand(
                prompt=normalize_text(prompt),
                source=source,
                reason=normalize_text(reason),
                force=force,
            )
        )

    async def close(self) -> None:
        self.active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.connected = False

    def attach_listener(self, listener_id: str) -> asyncio.Queue[bytes]:
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=48)
        self.listener_queues[listener_id] = queue
        return queue

    def detach_listener(self, listener_id: str) -> None:
        self.listener_queues.pop(listener_id, None)

    async def _run(self) -> None:
        if not self._client or not self.available:
            return

        try:
            async with self._client.aio.live.music.connect(model=LYRIA_MODEL) as session:
                self.connected = True
                self.log_callback("Lyria music session connected.")
                if types is None:
                    raise RuntimeError("google-genai types are unavailable.")
                await session.set_music_generation_config(
                    config=types.LiveMusicGenerationConfig(
                        temperature=self.temperature,
                        guidance=self.guidance,
                        seed=self.seed,
                    )
                )
                async with asyncio.TaskGroup() as group:
                    group.create_task(self._receive_audio(session))
                    group.create_task(self._apply_commands(session))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = str(exc)
            self.log_callback(f"Lyria session failed: {exc}")
        finally:
            self.connected = False
            self.active = False

    async def _receive_audio(self, session: Any) -> None:
        async for message in session.receive():
            if not self.active:
                return
            server_content = getattr(message, "server_content", None)
            audio_chunks = getattr(server_content, "audio_chunks", None) if server_content else None
            if not audio_chunks:
                continue
            for chunk in audio_chunks:
                await self._broadcast_audio(chunk.data)

    async def _broadcast_audio(self, data: bytes) -> None:
        stale: list[str] = []
        for listener_id, queue in self.listener_queues.items():
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                try:
                    _ = queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(data)
                except asyncio.QueueFull:
                    stale.append(listener_id)
        for listener_id in stale:
            self.listener_queues.pop(listener_id, None)

    async def _apply_commands(self, session: Any) -> None:
        has_started_playback = False
        loop = asyncio.get_running_loop()
        while self.active:
            command = await self._command_queue.get()
            if not command.prompt:
                continue
            target_palette = await loop.run_in_executor(
                None,
                lambda: translate_prompt_to_palette(
                    mode=self.mode,
                    prompt=command.prompt,
                    log_callback=self.log_callback,
                ),
            )

            if not has_started_playback:
                await session.set_weighted_prompts(
                    prompts=palette_to_weighted_prompts(target_palette)
                )
                await session.play()
                has_started_playback = True
            elif command.force or target_palette != self.current_palette:
                if self.crossfade_seconds > 0:
                    await self._crossfade_to_palette(session, target_palette)
                else:
                    await session.set_weighted_prompts(
                        prompts=palette_to_weighted_prompts(target_palette)
                    )

            self.current_prompt = command.prompt
            self.current_palette = dict(target_palette)
            self.last_reason = command.reason
            self.last_source = command.source
            self.last_update_at = time.time()
            self.error = ""
            self.log_callback(
                f"Music steer ({command.source}/{self.mode}) -> {command.prompt}"
            )

    async def _crossfade_to_palette(self, session: Any, target_palette: dict[str, float]) -> None:
        source_palette = dict(self.current_palette)
        if not source_palette:
            await session.set_weighted_prompts(
                prompts=palette_to_weighted_prompts(target_palette)
            )
            return

        step_sleep = self.crossfade_seconds / self.crossfade_steps
        for step in range(1, self.crossfade_steps + 1):
            progress = step / self.crossfade_steps
            middle = interpolate_palettes(source_palette, target_palette, progress)
            await session.set_weighted_prompts(
                prompts=palette_to_weighted_prompts(middle)
            )
            await asyncio.sleep(step_sleep)
