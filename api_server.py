from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mistralai import Mistral
from pydantic import BaseModel, Field

from mood_agent import StrandsMoodClassifier


ROOT_DIR = Path(__file__).parent.resolve()
WEB_DIR = ROOT_DIR / "web"
STATE_PATH = ROOT_DIR / "conversation_state.json"
HISTORY_DIR = ROOT_DIR / "live_sessions"
DEFAULT_TRANSCRIPTION_MODEL = "voxtral-mini-latest"
DEFAULT_MOOD_MODEL = "mistral-small-latest"
DEFAULT_LANGUAGE = "fr"
DEFAULT_MOOD_WINDOW = 6
DEFAULT_MOODS = [
    "default",
    "tense",
    "action",
    "boss",
    "mystery",
    "victory",
]
DEFAULT_MOOD_DESCRIPTIONS = {
    "default": "Exploration, planning, or neutral table talk.",
    "tense": "Danger, pressure, or mounting threat.",
    "action": "Combat, chase scenes, or immediate urgency.",
    "boss": "Climactic confrontation with a major enemy.",
    "mystery": "Stealth, investigation, omen, or discovery.",
    "victory": "Celebration, relief, or successful resolution.",
}


def load_api_key() -> str:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set.")
    return api_key


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def parse_mood_list(raw_moods: list[str] | None) -> list[str]:
    if not raw_moods:
        return list(DEFAULT_MOODS)

    moods: list[str] = []
    seen: set[str] = set()
    for item in raw_moods:
        normalized = normalize_text(str(item)).lower().replace(" ", "_")
        if normalized and normalized not in seen:
            moods.append(normalized)
            seen.add(normalized)
    return moods or list(DEFAULT_MOODS)


def build_demo_state() -> dict[str, Any]:
    return {
        "allowed_moods": list(DEFAULT_MOODS),
        "current_mood": DEFAULT_MOODS[0],
        "latest_mood_event": None,
        "utterances": [],
        "utterance_count": 0,
        "transcript": "",
        "partial_text": "",
        "mood_history": [],
        "started_at": None,
        "updated_at": None,
        "version": 0,
        "artifacts": {},
    }


def transcribe_audio_bytes(
    *,
    audio_bytes: bytes,
    file_name: str,
    language: str | None,
    model: str,
) -> str:
    client = Mistral(api_key=load_api_key())
    response = client.audio.transcriptions.complete(
        model=model,
        file={"content": audio_bytes, "file_name": file_name},
        language=language or None,
    )
    return normalize_text(response.text)


@dataclass
class ConversationState:
    allowed_moods: list[str]
    current_mood: str | None = None
    utterances: list[dict[str, Any]] = field(default_factory=list)
    mood_history: list[dict[str, Any]] = field(default_factory=list)
    partial_text: str = ""
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 0

    def __post_init__(self) -> None:
        if not self.allowed_moods:
            raise ValueError("ConversationState requires at least one mood.")
        if not self.current_mood:
            self.current_mood = self.allowed_moods[0]

    def add_utterance(self, text: str) -> bool:
        normalized = normalize_text(text)
        if not normalized:
            return False
        if self.utterances and self.utterances[-1]["text"] == normalized:
            self.updated_at = time.time()
            return False

        self.utterances.append({"text": normalized, "timestamp": time.time()})
        self.version += 1
        self.updated_at = time.time()
        return True

    def recent_utterances(self, limit: int) -> list[dict[str, Any]]:
        return self.utterances[-limit:]

    def transcript_text(self) -> str:
        return " ".join(item["text"] for item in self.utterances).strip()

    def to_payload(self) -> dict[str, Any]:
        latest_mood = self.mood_history[-1] if self.mood_history else None
        return {
            "allowed_moods": self.allowed_moods,
            "current_mood": self.current_mood,
            "latest_mood_event": latest_mood,
            "utterances": self.utterances,
            "utterance_count": len(self.utterances),
            "transcript": self.transcript_text(),
            "partial_text": normalize_text(self.partial_text),
            "mood_history": self.mood_history,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }


class BrowserSessionArtifacts:
    def __init__(self, *, latest_state_output: Path, history_dir: Path) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = timestamp
        self.session_dir = history_dir / timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir = self.session_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.latest_state_output = latest_state_output
        self.session_state_output = self.session_dir / "conversation_state.json"
        self.transcript_output = self.session_dir / "transcript.txt"
        self.moods_output = self.session_dir / "moods.json"
        self._chunk_index = 0

    def write_chunk(self, audio_bytes: bytes, suffix: str) -> Path:
        self._chunk_index += 1
        chunk_path = self.chunks_dir / f"chunk-{self._chunk_index:04d}{suffix}"
        chunk_path.write_bytes(audio_bytes)
        return chunk_path

    def build_artifact_paths(self) -> dict[str, str]:
        return {
            "session_dir": str(self.session_dir),
            "state_output": str(self.session_state_output),
            "transcript_output": str(self.transcript_output),
            "moods_output": str(self.moods_output),
            "chunks_dir": str(self.chunks_dir),
        }

    def persist(self, payload: dict[str, Any]) -> None:
        enriched_payload = dict(payload)
        enriched_payload["artifacts"] = self.build_artifact_paths()

        self.latest_state_output.parent.mkdir(parents=True, exist_ok=True)
        self.latest_state_output.write_text(
            json.dumps(enriched_payload, indent=2),
            encoding="utf-8",
        )
        self.session_state_output.write_text(
            json.dumps(enriched_payload, indent=2),
            encoding="utf-8",
        )
        transcript = str(payload.get("transcript", "")).strip()
        self.transcript_output.write_text(
            f"{transcript}\n" if transcript else "",
            encoding="utf-8",
        )
        self.moods_output.write_text(
            json.dumps(payload.get("mood_history", []), indent=2),
            encoding="utf-8",
        )


class BrowserConversationSession:
    def __init__(
        self,
        *,
        language: str,
        transcription_model: str,
        mood_model: str,
        allowed_moods: list[str],
        mood_window: int,
        mood_agent_enabled: bool,
        artifacts: BrowserSessionArtifacts,
        log_callback: Callable[[str], None],
    ) -> None:
        self.language = language
        self.transcription_model = transcription_model
        self.mood_model = mood_model
        self.mood_window = max(1, mood_window)
        self.mood_agent_enabled = mood_agent_enabled
        self.artifacts = artifacts
        self.log_callback = log_callback
        self.session_id = artifacts.session_id
        self.active = True
        self.chunk_count = 0
        self.last_chunk_at: float | None = None
        self.last_chunk_text = ""
        self.state = ConversationState(allowed_moods=allowed_moods)
        self._lock = Lock()
        self._requested_version = 0
        self._applied_version = 0
        self._classification_task: asyncio.Task[None] | None = None
        self._mood_classifier = (
            StrandsMoodClassifier(
                api_key=load_api_key(),
                model_id=mood_model,
                mood_descriptions={
                    mood: DEFAULT_MOOD_DESCRIPTIONS.get(
                        mood,
                        mood.replace("_", " "),
                    )
                    for mood in allowed_moods
                },
            )
            if mood_agent_enabled
            else None
        )
        self.persist()

    def persist(self) -> None:
        with self._lock:
            payload = self.state.to_payload()
        self.artifacts.persist(payload)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = self.state.to_payload()
        payload["artifacts"] = self.artifacts.build_artifact_paths()
        return payload

    def save_chunk(self, audio_bytes: bytes, file_name: str) -> Path:
        suffix = Path(file_name).suffix or ".wav"
        with self._lock:
            self.chunk_count += 1
            self.last_chunk_at = time.time()
            return self.artifacts.write_chunk(audio_bytes, suffix)

    def register_transcript(self, text: str) -> bool:
        normalized = normalize_text(text)
        with self._lock:
            self.last_chunk_text = normalized
            added = self.state.add_utterance(normalized)
            if added and self._mood_classifier:
                self._requested_version = self.state.version
        self.persist()
        if added:
            self._ensure_classification_task()
        return added

    def _ensure_classification_task(self) -> None:
        if not self._mood_classifier:
            return
        if self._classification_task is None or self._classification_task.done():
            self._classification_task = asyncio.create_task(self._run_classification_loop())

    async def _run_classification_loop(self) -> None:
        while True:
            with self._lock:
                if self._applied_version >= self._requested_version:
                    return
                target_version = self._requested_version
                recent_utterances = list(self.state.recent_utterances(self.mood_window))
                current_mood = self.state.current_mood or self.state.allowed_moods[0]
                allowed_moods = list(self.state.allowed_moods)

            try:
                decision = await asyncio.to_thread(
                    self._mood_classifier.classify_scene,
                    allowed_moods=allowed_moods,
                    recent_utterances=recent_utterances,
                    current_mood=current_mood,
                )
            except Exception as exc:
                self.log_callback(f"Mood classification failed: {exc}")
                return

            with self._lock:
                previous_mood = self.state.current_mood
                self.state.current_mood = decision.mood
                self.state.mood_history.append(
                    {
                        "mood": decision.mood,
                        "confidence": decision.confidence,
                        "reason": decision.reason,
                        "model": decision.model,
                        "timestamp": time.time(),
                        "utterance_count": len(self.state.utterances),
                        "changed": decision.mood != previous_mood,
                    }
                )
                self.state.updated_at = time.time()
                self._applied_version = target_version

            self.persist()
            if decision.mood != previous_mood:
                self.log_callback(
                    f"Mood -> {decision.mood} ({decision.confidence:.2f}) | {decision.reason}"
                )

    async def close(self) -> None:
        self.active = False
        if self._classification_task:
            try:
                await self._classification_task
            except asyncio.CancelledError:
                pass
        self.persist()


class StartSessionRequest(BaseModel):
    language: str = Field(default=DEFAULT_LANGUAGE, max_length=16)
    transcription_model: str = Field(default=DEFAULT_TRANSCRIPTION_MODEL)
    mood_model: str = Field(default=DEFAULT_MOOD_MODEL)
    moods: list[str] = Field(default_factory=lambda: list(DEFAULT_MOODS))
    mood_agent: bool = True
    mood_window: int = Field(default=DEFAULT_MOOD_WINDOW, ge=1, le=20)
    chunk_seconds: float = Field(default=2.5, ge=1.0, le=10.0)


class SessionManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._session: BrowserConversationSession | None = None
        self._logs: deque[str] = deque(maxlen=180)
        self._last_runtime: dict[str, Any] = {
            "running": False,
            "mode": "demo",
            "current_session_id": None,
            "config": {
                "language": DEFAULT_LANGUAGE,
                "transcription_model": DEFAULT_TRANSCRIPTION_MODEL,
                "mood_model": DEFAULT_MOOD_MODEL,
                "mood_agent": True,
                "mood_window": DEFAULT_MOOD_WINDOW,
                "chunk_seconds": 2.5,
            },
            "processing": False,
            "state_path": str(STATE_PATH),
        }

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        with self._lock:
            self._logs.append(f"[{timestamp}] {message}")

    def _load_last_state(self) -> dict[str, Any]:
        if not STATE_PATH.exists():
            return build_demo_state()
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return build_demo_state()

    def _runtime_payload(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._last_runtime)
            payload["logs"] = list(self._logs)[-20:]
        return payload

    def snapshot(self) -> dict[str, Any]:
        session = self._session
        session_payload = session.snapshot() if session else self._load_last_state()
        runtime = self._runtime_payload()
        if session and session.active:
            runtime["running"] = True
            runtime["mode"] = "live"
            runtime["current_session_id"] = session.session_id
        elif session_payload.get("utterance_count"):
            runtime["running"] = False
            runtime["mode"] = "snapshot"
        else:
            runtime["running"] = False
            runtime["mode"] = "demo"
        return {
            "runtime": runtime,
            "session": session_payload,
            "generated_at": time.time(),
        }

    def start(self, request: StartSessionRequest) -> dict[str, Any]:
        load_api_key()
        with self._lock:
            if self._session and self._session.active:
                raise RuntimeError("A browser live session is already running.")

        artifacts = BrowserSessionArtifacts(
            latest_state_output=STATE_PATH,
            history_dir=HISTORY_DIR,
        )
        session = BrowserConversationSession(
            language=request.language.strip() or DEFAULT_LANGUAGE,
            transcription_model=request.transcription_model,
            mood_model=request.mood_model,
            allowed_moods=parse_mood_list(request.moods),
            mood_window=request.mood_window,
            mood_agent_enabled=request.mood_agent,
            artifacts=artifacts,
            log_callback=self.log,
        )

        with self._lock:
            self._session = session
            self._last_runtime = {
                "running": True,
                "mode": "live",
                "current_session_id": session.session_id,
                "config": {
                    "language": session.language,
                    "transcription_model": session.transcription_model,
                    "mood_model": session.mood_model,
                    "mood_agent": session.mood_agent_enabled,
                    "mood_window": session.mood_window,
                    "chunk_seconds": request.chunk_seconds,
                },
                "processing": False,
                "state_path": str(STATE_PATH),
            }

        self.log("Browser live session started.")
        return self.snapshot()

    async def ingest_chunk(
        self,
        *,
        session_id: str,
        file_name: str,
        audio_bytes: bytes,
        sequence: int | None,
    ) -> dict[str, Any]:
        session = self._session
        if not session or not session.active or session.session_id != session_id:
            raise RuntimeError("No active session matches this upload.")

        chunk_path = session.save_chunk(audio_bytes, file_name)
        self.log(
            f"Received chunk {sequence if sequence is not None else session.chunk_count} -> {chunk_path.name}"
        )

        with self._lock:
            self._last_runtime["processing"] = True

        try:
            text = await asyncio.to_thread(
                transcribe_audio_bytes,
                audio_bytes=audio_bytes,
                file_name=file_name,
                language=session.language,
                model=session.transcription_model,
            )
        finally:
            with self._lock:
                self._last_runtime["processing"] = False

        if text:
            session.register_transcript(text)
            self.log(f"Transcript chunk -> {text}")
        else:
            self.log("Transcript chunk -> [no speech detected]")

        snapshot = self.snapshot()
        snapshot["chunk_text"] = text
        return snapshot

    async def stop(self, session_id: str) -> dict[str, Any]:
        session = self._session
        if not session or session.session_id != session_id:
            return self.snapshot()

        await session.close()
        with self._lock:
            self._session = None
            self._last_runtime["running"] = False
            self._last_runtime["mode"] = "snapshot"
            self._last_runtime["processing"] = False
            self._last_runtime["current_session_id"] = None
        self.log("Browser live session stopped.")
        return self.snapshot()


manager = SessionManager()
app = FastAPI(title="DnD Audio Scene Listener")

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    return manager.snapshot()


@app.post("/api/session/start")
def start_session(request: StartSessionRequest) -> dict[str, Any]:
    try:
        return manager.start(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/session/{session_id}/chunk")
async def upload_chunk(
    session_id: str,
    file: UploadFile = File(...),
    sequence: int | None = Form(default=None),
) -> dict[str, Any]:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio chunk is empty.")

    try:
        return await manager.ingest_chunk(
            session_id=session_id,
            file_name=file.filename or "chunk.wav",
            audio_bytes=audio_bytes,
            sequence=sequence,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        manager.log(f"Chunk processing failed: {exc}")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/session/{session_id}/stop")
async def stop_session(session_id: str) -> dict[str, Any]:
    return await manager.stop(session_id)


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
