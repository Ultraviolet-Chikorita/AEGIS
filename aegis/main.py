from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
from typing import Any

from aegis.ai.bedrock import BedrockReasoningClient
from aegis.ai.elevenlabs_speech import ElevenLabsSpeechClient
from aegis.ai.nvidia_reasoning import NvidiaReasoningClient
from aegis.config import AppConfig
from aegis.graph.connection import Neo4jConnection
from aegis.graph.tools import GraphTools
from aegis.interface.text_ui import TextInterface
from aegis.interface.voice_input import VoiceInput
from aegis.interface.web_ui import WebInterface
from aegis.simulation.engine import SimulationEngine


logger = logging.getLogger(__name__)


class _LocalSpeechClient:
    def generate_speech(self, reasoning_output: dict[str, Any]) -> dict[str, Any]:
        spoken_words = str(reasoning_output.get("spoken_words", "")).strip()
        if not spoken_words:
            key_points = reasoning_output.get("key_points", [])
            spoken_words = ". ".join(str(point) for point in key_points if point).strip() or "..."
        return {
            "segments": [
                {
                    "text": spoken_words,
                    "tone": "neutral",
                    "pacing": "medium",
                    "pause_after_ms": 0,
                }
            ]
        }


async def _keep_alive_loop(graph: Neo4jConnection, interval_seconds: int = 3600) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await graph.query("RETURN 1 AS ok")
            logger.debug("Neo4j keep-alive ping succeeded")
        except Exception:
            logger.exception("Neo4j keep-alive ping failed")


async def _run() -> None:
    config = AppConfig.from_env()
    _configure_logging(config.log_level)
    logger.info("AEGIS boot starting")
    logger.info(
        "Runtime config: interface=%s provider=%s tick_hz=%s web=%s:%s",
        config.interface_mode,
        config.reasoning_provider,
        config.simulation_tick_hz,
        config.web_host,
        config.web_port,
    )

    graph = Neo4jConnection(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
    )
    logger.info("Connecting to Neo4j at %s", config.neo4j_uri)
    await graph.connect()
    logger.info("Neo4j connectivity verified")

    reasoning = _build_reasoning_client(config)
    if config.elevenlabs_api_key.strip():
        speech = ElevenLabsSpeechClient(
            api_key=config.elevenlabs_api_key,
            voice_id=config.elevenlabs_voice_id,
            model_id=config.elevenlabs_model_id,
        )
        logger.info("Speech mode: ElevenLabs TTS enabled (voice_id=%s)", config.elevenlabs_voice_id)
    else:
        speech = _LocalSpeechClient()
        logger.info("Speech mode: local text-only (ElevenLabs disabled; no outbound TTS calls)")
    ui = _build_interface(config)
    logger.info("Interface initialized: %s", type(ui).__name__)
    voice = VoiceInput(model_name=config.voice_input_model)
    logger.info("Voice input initialized (model=%s)", config.voice_input_model)
    graph_tools = GraphTools(graph)

    engine = SimulationEngine(
        graph=graph,
        reasoning=reasoning,
        speech=speech,
        ui=ui,
        voice=voice,
        graph_tools=graph_tools,
        tick_hz=config.simulation_tick_hz,
    )

    ui.stream_narrative("The presence stirs. Somewhere, a host awaits.")

    ui_task: asyncio.Task[Any] | None = None
    keep_alive_task: asyncio.Task[Any] | None = None
    if hasattr(ui, "start"):
        logger.info("Starting web interface task")
        ui_task = asyncio.create_task(ui.start())
    keep_alive_task = asyncio.create_task(_keep_alive_loop(graph))
    logger.info("Neo4j keep-alive task started")

    try:
        logger.info("Simulation loop started")
        await engine.run()
    finally:
        logger.info("Simulation shutdown initiated")
        if keep_alive_task is not None:
            keep_alive_task.cancel()
            with suppress(asyncio.CancelledError):
                await keep_alive_task
            logger.info("Neo4j keep-alive task stopped")
        if ui_task is not None:
            ui_task.cancel()
            with suppress(asyncio.CancelledError):
                await ui_task
            logger.info("Web interface task stopped")
        await graph.close()
        logger.info("Neo4j connection closed")


def main() -> None:
    asyncio.run(_run())


def _configure_logging(log_level: str) -> None:
    resolved_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _build_reasoning_client(config: AppConfig):
    if config.reasoning_provider == "nvidia":
        logger.info("Using NVIDIA Serverless reasoning provider")
        return NvidiaReasoningClient(
            api_key=config.nvidia_api_key,
            base_url=config.nvidia_base_url,
            model_id=config.nvidia_reasoning_model_id,
        )

    logger.info("Using Bedrock reasoning provider")
    return BedrockReasoningClient(
        model_id=config.bedrock_reasoning_model_id,
        region=config.aws_region,
    )


def _build_interface(config: AppConfig):
    if config.interface_mode == "web":
        return WebInterface(host=config.web_host, port=config.web_port, log_level=config.log_level)
    return TextInterface()


if __name__ == "__main__":
    main()
