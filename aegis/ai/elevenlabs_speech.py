from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests


class ElevenLabsSpeechClient:
    def __init__(self, api_key: str, voice_id: str, model_id: str) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.base_url = "https://api.elevenlabs.io/v1"
        self.output_dir = Path("artifacts/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_speech(self, reasoning_output: dict[str, Any]) -> dict[str, Any]:
        spoken_words = str(reasoning_output.get("spoken_words", "")).strip()
        if not spoken_words:
            key_points = reasoning_output.get("key_points", [])
            spoken_words = ". ".join(str(point) for point in key_points if point).strip() or "..."

        segment = {
            "text": spoken_words,
            "tone": _tone_from_reasoning(reasoning_output),
            "pacing": "medium",
            "pause_after_ms": 0,
        }

        audio_file = self._synthesize(spoken_words)
        result: dict[str, Any] = {"segments": [segment]}
        if audio_file:
            result["audio_file"] = audio_file
        return result

    def _synthesize(self, text: str) -> str | None:
        if not self.api_key.strip():
            return None

        url = f"{self.base_url}/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
        except requests.RequestException:
            return None

        filename = f"speech_{int(time.time() * 1000)}.mp3"
        path = self.output_dir / filename
        path.write_bytes(response.content)
        return os.fspath(path)


def _tone_from_reasoning(reasoning_output: dict[str, Any]) -> str:
    progression = reasoning_output.get("tone_progression", [])
    if progression and isinstance(progression, list):
        first = progression[0]
        if isinstance(first, dict):
            tone = first.get("tone")
            if isinstance(tone, str) and tone.strip():
                return tone
    return "neutral"
