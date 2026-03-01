"""
mood_agent.py — Strands-based mood classification for live D&D sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.models.mistral import MistralModel
from strands.types.exceptions import MaxTokensReachedException


MOOD_CLASSIFIER_SYSTEM_PROMPT = """You are a live mood classifier for a Dungeons & Dragons session.
You classify the current scene mood for adaptive background music.

Rules:
- Always inspect the current scene using the available tools before deciding.
- Choose exactly one mood label from the allowed list.
- Focus on scene tone, pacing, tension, and stakes.
- Be conservative: if the scene is ambiguous, keep or return the safest neutral mood.
- Keep the reason short and concrete.
- Never narrate your plan to the user.
- Do not emit free-form text when you can return structured output directly.
"""


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def normalize_mood_label(label: str) -> str:
    return normalize_text(label).lower().replace(" ", "_")


@dataclass(frozen=True)
class MoodDecision:
    mood: str
    confidence: float
    reason: str
    model: str


class MoodClassification(BaseModel):
    mood: str = Field(description="One allowed mood label.")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )
    reason: str = Field(description="Short explanation for the chosen mood.")


class MoodContextTools:
    def __init__(
        self,
        *,
        current_mood: str,
        allowed_moods: list[str],
        mood_descriptions: dict[str, str],
        recent_utterances: list[dict[str, Any]],
    ) -> None:
        self.current_mood = current_mood
        self.allowed_moods = allowed_moods
        self.mood_descriptions = mood_descriptions
        self.recent_utterances = recent_utterances

    @tool
    def get_current_scene(self) -> dict[str, Any]:
        """Return the current mood and recent finalized utterances for the active scene."""
        return {
            "current_mood": self.current_mood,
            "recent_utterances": self.recent_utterances,
            "utterance_count": len(self.recent_utterances),
        }

    @tool
    def get_allowed_moods(self) -> dict[str, Any]:
        """Return the allowed mood labels with descriptions. Use this before selecting the final mood."""
        return {
            "allowed_moods": [
                {
                    "label": mood,
                    "description": self.mood_descriptions.get(
                        mood,
                        "Use the plain-English meaning of this label.",
                    ),
                }
                for mood in self.allowed_moods
            ]
        }


class StrandsMoodClassifier:
    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        mood_descriptions: dict[str, str],
    ) -> None:
        self.model_id = model_id
        self.mood_descriptions = mood_descriptions
        self.model = MistralModel(
            api_key=api_key,
            model_id=model_id,
            temperature=0.1,
            max_tokens=480,
            stream=True,
        )

    def _build_tool_agent(self, tools: MoodContextTools) -> Agent:
        return Agent(
            model=self.model,
            name="DnD Mood Classifier",
            description="Classifies the current D&D scene mood from recent utterances.",
            system_prompt=MOOD_CLASSIFIER_SYSTEM_PROMPT,
            tools=[tools.get_current_scene, tools.get_allowed_moods],
            callback_handler=None,
        )

    def _build_inline_agent(self) -> Agent:
        return Agent(
            model=self.model,
            name="DnD Mood Classifier",
            description="Classifies the current D&D scene mood from recent utterances.",
            system_prompt=MOOD_CLASSIFIER_SYSTEM_PROMPT,
            callback_handler=None,
        )

    def _result_to_decision(
        self,
        *,
        result: Any,
        allowed_moods: list[str],
        current_mood: str,
    ) -> MoodDecision:
        if not result.structured_output:
            raise ValueError("The mood classifier returned no structured output.")

        output = result.structured_output
        normalized_mood = normalize_mood_label(output.mood)
        mood = normalized_mood if normalized_mood in allowed_moods else current_mood
        reason = normalize_text(output.reason) or "No reason provided."

        return MoodDecision(
            mood=mood,
            confidence=max(0.0, min(1.0, float(output.confidence))),
            reason=reason,
            model=self.model_id,
        )

    def _classify_with_tools(
        self,
        *,
        allowed_moods: list[str],
        recent_utterances: list[dict[str, Any]],
        current_mood: str,
    ) -> MoodDecision:
        tools = MoodContextTools(
            current_mood=current_mood,
            allowed_moods=allowed_moods,
            mood_descriptions=self.mood_descriptions,
            recent_utterances=recent_utterances,
        )

        agent = self._build_tool_agent(tools)
        result = agent(
            (
                "Classify the current scene mood for adaptive music. "
                "Use the tools immediately, avoid free-form explanation, and return the final answer as structured output."
            ),
            structured_output_model=MoodClassification,
            structured_output_prompt=(
                "Return the final answer now using only the structured output tool. "
                "Do not add any plain text."
            ),
        )
        return self._result_to_decision(
            result=result,
            allowed_moods=allowed_moods,
            current_mood=current_mood,
        )

    def _classify_inline(
        self,
        *,
        allowed_moods: list[str],
        recent_utterances: list[dict[str, Any]],
        current_mood: str,
    ) -> MoodDecision:
        agent = self._build_inline_agent()
        payload = {
            "current_mood": current_mood,
            "allowed_moods": [
                {
                    "label": mood,
                    "description": self.mood_descriptions.get(
                        mood,
                        "Use the plain-English meaning of this label.",
                    ),
                }
                for mood in allowed_moods
            ],
            "recent_utterances": recent_utterances,
        }
        result = agent(
            (
                "Classify the current D&D scene mood for adaptive music from this JSON context.\n\n"
                f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
            ),
            structured_output_model=MoodClassification,
            structured_output_prompt=(
                "Return the final answer now using only the structured output tool. "
                "Do not add any plain text."
            ),
        )
        return self._result_to_decision(
            result=result,
            allowed_moods=allowed_moods,
            current_mood=current_mood,
        )

    def classify_scene(
        self,
        *,
        allowed_moods: list[str],
        recent_utterances: list[dict[str, Any]],
        current_mood: str,
    ) -> MoodDecision:
        if not recent_utterances:
            return MoodDecision(
                mood=current_mood,
                confidence=0.0,
                reason="No finalized utterance yet.",
                model=self.model_id,
            )

        try:
            return self._classify_with_tools(
                allowed_moods=allowed_moods,
                recent_utterances=recent_utterances,
                current_mood=current_mood,
            )
        except (MaxTokensReachedException, TypeError, ValueError) as exc:
            message = str(exc)
            if (
                not isinstance(exc, MaxTokensReachedException)
                and "can only concatenate str (not \"list\") to str" not in message
                and "returned no structured output" not in message
            ):
                raise

        return self._classify_inline(
            allowed_moods=allowed_moods,
            recent_utterances=recent_utterances,
            current_mood=current_mood,
        )
