from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from mistralai import Mistral


DEFAULT_MUSIC_PROMPT_MODEL = "mistral-small-latest"
DEFAULT_RECENT_UTTERANCE_LIMIT = 6
DEVELOPER_SCENE_NAMES = [
    "peaceful_village",
    "mysterious_cave",
    "combat",
    "techno_party",
    "victory",
    "puzzle",
]
DEVELOPER_SCENE_HINTS = {
    "default": "peaceful_village",
    "tense": "mysterious_cave",
    "action": "combat",
    "boss": "combat",
    "mystery": "puzzle",
    "victory": "victory",
}
DEVELOPER_SCENE_SYSTEM = f"""You are a scene selector for a live music engine.
Choose exactly one scene from this list:
{", ".join(DEVELOPER_SCENE_NAMES)}

Return strict JSON with keys scene and reason.
The scene must be one of the allowed values exactly.

Scene meanings:
- peaceful_village: calm exploration, tavern downtime, safe travel, market energy, warm fantasy ambience
- mysterious_cave: eerie ruins, suspense, stealth, dark discovery, ominous exploration
- combat: battle, ambush, chase, immediate danger, boss pressure, urgent action
- techno_party: rave, cyberpunk club, loud celebration, dancefloor energy
- victory: triumph, relief, success, treasure, celebratory resolution
- puzzle: investigation, riddles, clues, deduction, careful thinking
"""
MOOD_HINTS = {
    "default": "grounded exploration, warm ambient fantasy instrumentation, steady pace",
    "tense": "mounting danger, restrained percussion, low strings, pressure building",
    "action": "urgent conflict, driving rhythm, aggressive percussion, heroic momentum",
    "boss": "climactic confrontation, massive scale, dark brass, relentless percussion",
    "mystery": "uncertain discovery, eerie texture, sparse melody, hidden threat",
    "victory": "triumphant relief, brighter harmony, lifted energy, celebratory swells",
}


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def truncate_text(text: str, limit: int = 260) -> str:
    value = normalize_text(text)
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1].rstrip()}..."


def extract_message_text(content: object) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(part.strip() for part in parts if str(part).strip()).strip()

    return str(content or "").strip()


def extract_json_object(text: str) -> dict[str, Any]:
    value = normalize_text(text)
    if not value:
        return {}

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


@dataclass(frozen=True)
class MusicPromptDecision:
    prompt: str
    reason: str
    should_update: bool
    source: str
    model: str


class MusicPromptPlanner:
    def __init__(self, *, api_key: str, model_id: str = DEFAULT_MUSIC_PROMPT_MODEL) -> None:
        self.model_id = model_id
        self._client = Mistral(api_key=api_key)

    def plan(
        self,
        *,
        current_mood: str,
        recent_utterances: list[dict[str, Any]],
        previous_prompt: str,
        previous_mood: str | None,
        force: bool = False,
    ) -> MusicPromptDecision:
        fallback = self._heuristic_decision(
            current_mood=current_mood,
            recent_utterances=recent_utterances,
            previous_prompt=previous_prompt,
            previous_mood=previous_mood,
            force=force,
            reason="Fallback heuristic prompt.",
            model="heuristic",
        )

        transcript_lines = [
            normalize_text(item.get("text", ""))
            for item in recent_utterances[-DEFAULT_RECENT_UTTERANCE_LIMIT :]
            if normalize_text(item.get("text", ""))
        ]
        if not transcript_lines:
            return fallback

        system_prompt = (
            "You write short instrumental music steering prompts for a live D&D soundtrack engine. "
            "Return strict JSON with keys prompt and reason. "
            "The prompt must be short, vivid, and actionable for generative instrumental music. "
            "Do not mention dialogue, quotes, or exact player lines. "
            "Focus on mood, setting, motion, and texture."
        )
        user_prompt = json.dumps(
            {
                "current_mood": current_mood,
                "previous_mood": previous_mood,
                "previous_prompt": previous_prompt,
                "recent_utterances": transcript_lines,
            },
            ensure_ascii=True,
        )

        try:
            response = self._client.chat.complete(
                model=self.model_id,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            message = response.choices[0].message
            payload = extract_json_object(extract_message_text(message.content))
            prompt = truncate_text(payload.get("prompt", ""))
            reason = truncate_text(payload.get("reason", ""))
            if not prompt:
                return fallback
            should_update = force or self._should_update(
                prompt=prompt,
                previous_prompt=previous_prompt,
                current_mood=current_mood,
                previous_mood=previous_mood,
            )
            return MusicPromptDecision(
                prompt=prompt,
                reason=reason or "Music prompt updated from recent conversation context.",
                should_update=should_update,
                source="planner",
                model=self.model_id,
            )
        except Exception:
            return fallback

    def plan_developer_scene(
        self,
        *,
        current_mood: str,
        recent_utterances: list[dict[str, Any]],
        previous_prompt: str,
        previous_mood: str | None,
        force: bool = False,
    ) -> MusicPromptDecision:
        fallback_scene = self._heuristic_scene(current_mood)
        fallback = MusicPromptDecision(
            prompt=fallback_scene,
            reason="Fallback developer scene selection.",
            should_update=force
            or self._should_update(
                prompt=fallback_scene,
                previous_prompt=previous_prompt,
                current_mood=current_mood,
                previous_mood=previous_mood,
            ),
            source="developer_scene_selector",
            model="heuristic-scene",
        )

        transcript_lines = [
            normalize_text(item.get("text", ""))
            for item in recent_utterances[-DEFAULT_RECENT_UTTERANCE_LIMIT :]
            if normalize_text(item.get("text", ""))
        ]
        if not transcript_lines:
            return fallback

        user_prompt = json.dumps(
            {
                "current_mood": current_mood,
                "previous_mood": previous_mood,
                "previous_scene": previous_prompt,
                "recent_utterances": transcript_lines,
            },
            ensure_ascii=True,
        )

        try:
            response = self._client.chat.complete(
                model=self.model_id,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": DEVELOPER_SCENE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            )
            message = response.choices[0].message
            payload = extract_json_object(extract_message_text(message.content))
            scene = normalize_text(payload.get("scene", "")).lower().replace(" ", "_")
            if scene not in DEVELOPER_SCENE_NAMES:
                return fallback
            reason = truncate_text(payload.get("reason", "")) or "Developer scene updated from recent conversation context."
            return MusicPromptDecision(
                prompt=scene,
                reason=reason,
                should_update=force
                or self._should_update(
                    prompt=scene,
                    previous_prompt=previous_prompt,
                    current_mood=current_mood,
                    previous_mood=previous_mood,
                ),
                source="developer_scene_selector",
                model=self.model_id,
            )
        except Exception:
            return fallback

    def _heuristic_decision(
        self,
        *,
        current_mood: str,
        recent_utterances: list[dict[str, Any]],
        previous_prompt: str,
        previous_mood: str | None,
        force: bool,
        reason: str,
        model: str,
    ) -> MusicPromptDecision:
        hint = MOOD_HINTS.get(current_mood, MOOD_HINTS["default"])
        context_bits = [
            normalize_text(item.get("text", ""))
            for item in recent_utterances[-3:]
            if normalize_text(item.get("text", ""))
        ]
        context_suffix = ""
        if context_bits:
            merged = truncate_text(", ".join(context_bits), 140)
            context_suffix = f", inspired by {merged}"
        prompt = truncate_text(
            f"instrumental fantasy soundtrack, {hint}{context_suffix}",
            180,
        )
        return MusicPromptDecision(
            prompt=prompt,
            reason=reason,
            should_update=force
            or self._should_update(
                prompt=prompt,
                previous_prompt=previous_prompt,
                current_mood=current_mood,
                previous_mood=previous_mood,
            ),
            source="heuristic",
            model=model,
        )

    def _should_update(
        self,
        *,
        prompt: str,
        previous_prompt: str,
        current_mood: str,
        previous_mood: str | None,
    ) -> bool:
        if not previous_prompt:
            return True
        if current_mood != previous_mood:
            return True
        return normalize_text(prompt).casefold() != normalize_text(previous_prompt).casefold()

    def _heuristic_scene(self, current_mood: str) -> str:
        return DEVELOPER_SCENE_HINTS.get(current_mood, "peaceful_village")
