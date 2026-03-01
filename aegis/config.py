from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    log_level: str
    reasoning_provider: str
    interface_mode: str
    web_host: str
    web_port: int
    aws_region: str
    bedrock_reasoning_model_id: str
    nvidia_api_key: str
    nvidia_base_url: str
    nvidia_reasoning_model_id: str
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    elevenlabs_model_id: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    simulation_tick_hz: int
    voice_input_model: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()
        return cls(
            log_level=_get_env("LOG_LEVEL", "INFO").upper(),
            reasoning_provider=_get_env("REASONING_PROVIDER", "bedrock").lower(),
            interface_mode=_get_env("INTERFACE_MODE", "web").lower(),
            web_host=_get_env("WEB_HOST", "127.0.0.1"),
            web_port=int(_get_env("WEB_PORT", "8765")),
            aws_region=_get_env("AWS_REGION", "us-east-1"),
            bedrock_reasoning_model_id=_get_env(
                "BEDROCK_REASONING_MODEL_ID", "mistral.mistral-large-2407-v1:0"
            ),
            nvidia_api_key=_get_env("NVIDIA_API_KEY", ""),
            nvidia_base_url=_get_env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            nvidia_reasoning_model_id=_get_env(
                "NVIDIA_REASONING_MODEL_ID", "mistralai/mistral-7b-instruct-v0.3"
            ),
            elevenlabs_api_key=_get_env("ELEVENLABS_API_KEY", ""),
            elevenlabs_voice_id=_get_env("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            elevenlabs_model_id=_get_env("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
            neo4j_uri=_get_required_env("NEO4J_URI"),
            neo4j_user=_get_required_env("NEO4J_USER"),
            neo4j_password=_get_required_env("NEO4J_PASSWORD"),
            simulation_tick_hz=int(_get_env("SIMULATION_TICK_HZ", "10")),
            voice_input_model=_get_env("VOICE_INPUT_MODEL", "base"),
        )


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value
