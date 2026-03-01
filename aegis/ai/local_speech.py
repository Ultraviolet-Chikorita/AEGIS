from __future__ import annotations

import json
from typing import Any

from openai import OpenAI


class LocalSpeechClient:
    def __init__(self, base_url: str, api_key: str, model_id: str) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id

    def generate_speech(self, reasoning_output: dict[str, Any]) -> dict[str, Any]:
        speech_input = {
            "goals": reasoning_output.get("goals", []),
            "key_points": reasoning_output.get("key_points", []),
            "tone_progression": reasoning_output.get("tone_progression", []),
            "voice_cluster": reasoning_output.get("voice_cluster", "guarded_suspicious"),
            "spoken_words": reasoning_output.get("spoken_words", ""),
        }

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return only JSON with shape: "
                            "{'segments':[{'text':str,'tone':str,'pacing':str,'pause_after_ms':int}]}"
                        ),
                    },
                    {"role": "user", "content": json.dumps(speech_input)},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Speech model returned empty content")
            return json.loads(content)
        except Exception:
            fallback = reasoning_output.get("spoken_words") or "..."
            return {
                "segments": [
                    {
                        "text": fallback,
                        "tone": "neutral",
                        "pacing": "medium",
                        "pause_after_ms": 0,
                    }
                ]
            }
